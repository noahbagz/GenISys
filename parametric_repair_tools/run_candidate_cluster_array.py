#!/usr/bin/env python3
"""Run one restartable SLURM-array shard of candidate-index repairs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from .cluster_configuration import (
    candidate_batch_root,
    cluster_repair_configuration,
)
from .consolidated_storage import (
    checkpoint_task_rows,
    consolidated_repair_row,
    read_task_shard,
    task_shard_path,
)
from .pipeline import RepairPipeline


def _task_interval(
    row_count: int,
    task_id: int,
    task_count: int,
) -> tuple[int, int]:
    """Return one balanced, half-open contiguous selection interval."""

    start = int(task_id)*int(row_count)//int(task_count)
    stop = (int(task_id) + 1)*int(row_count)//int(task_count)
    return start, stop


def _task_scratch_root(task_id: int) -> Path:
    """Return a task-private node-local directory for transient artifacts."""

    scratch_parent = Path(
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or tempfile.gettempdir()
    ).resolve()
    job_id = str(os.environ.get("SLURM_ARRAY_JOB_ID", "local"))
    scratch_root = (
        scratch_parent
        / f"genisys_v1p1_repair_{job_id}_task_{int(task_id):04d}"
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    return scratch_root


def _validate_saved_rows(
    saved: pd.DataFrame,
    owned: pd.DataFrame,
    task_id: int,
) -> None:
    """Prove that a restart checkpoint belongs to this immutable task."""

    if saved.empty:
        return

    required = {
        "Task_ID",
        "Selection_Order",
        "Parent_Index",
        "Alternative_Number",
        "Index",
        "Final_Status",
        "Fully_Feasible",
    }
    missing = sorted(required - set(saved.columns))

    if missing:
        raise RuntimeError(
            f"Task shard is missing required columns: {missing}."
        )

    if saved["Selection_Order"].duplicated().any():
        raise RuntimeError("Task shard contains duplicate selection rows.")

    if not saved["Task_ID"].astype(int).eq(int(task_id)).all():
        raise RuntimeError("Task shard contains rows owned by another task.")

    expected = owned.set_index("Selection_Order")

    for row in saved.itertuples(index=False):
        selection_order = int(row.Selection_Order)

        if selection_order not in expected.index:
            raise RuntimeError(
                "Task shard contains a selection outside its frozen interval: "
                f"{selection_order}."
            )

        selected = expected.loc[selection_order]

        if (
            int(row.Parent_Index) != int(selected["Index"])
            or int(row.Alternative_Number)
            != int(selected["Alternative_Number"])
        ):
            raise RuntimeError(
                "Task shard identity differs from the frozen selection at "
                f"Selection_Order={selection_order}."
            )


def parse_arguments() -> argparse.Namespace:
    """Read the SLURM array coordinates explicitly."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-count", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    """Repair and evaluate this task's immutable selection interval."""

    arguments = parse_arguments()
    task_id = int(arguments.task_id)
    task_count = int(arguments.task_count)

    if task_count < 1 or not 0 <= task_id < task_count:
        raise ValueError(
            "--task-id must lie in [0, --task-count), and task count must "
            "be positive."
        )

    configuration = cluster_repair_configuration()
    batch_root = candidate_batch_root(configuration)
    selection_path = batch_root / "selected_source_designs.csv"
    run_manifest_path = batch_root / "repair_run_manifest.json"

    if not selection_path.exists() or not run_manifest_path.exists():
        raise FileNotFoundError(
            "The candidate repair batch has not been prepared. Run "
            "prepare_candidate_cluster_batch before submitting the array."
        )

    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    if int(manifest["array_task_count"]) != task_count:
        raise ValueError(
            "The submitted array task count differs from the frozen repair "
            "manifest."
        )

    if manifest["configuration_sha256"] != configuration.fingerprint():
        raise ValueError(
            "The cluster repair environment differs from the frozen "
            "configuration."
        )

    selection = pd.read_csv(selection_path).sort_values(
        "Selection_Order"
    ).reset_index(drop=True)
    start, stop = _task_interval(len(selection), task_id, task_count)
    owned = selection.iloc[start:stop].copy()
    pipeline = RepairPipeline(configuration)
    output_path = task_shard_path(batch_root, task_id)
    saved = read_task_shard(output_path)
    _validate_saved_rows(saved, owned, task_id)
    completed_orders = (
        set(saved["Selection_Order"].astype(int))
        if not saved.empty
        else set()
    )
    reused_count = len(completed_orders)
    scratch_root = _task_scratch_root(task_id)

    print(f"Repair array task: {task_id}/{task_count - 1}")
    print(f"Selection interval: [{start}, {stop})")
    print(f"Owned repaired evaluations: {len(owned)}")
    print(f"Consolidated task shard: {output_path}")

    for _, selected in owned.iterrows():
        parent_index = int(selected["Index"])
        alternative_number = int(selected["Alternative_Number"])
        selection_order = int(selected["Selection_Order"])

        if selection_order in completed_orders:
            continue

        # The ordinary evaluator still writes its established reduced-data
        # family, but it does so only inside this node-local directory.  Once
        # the result is committed as one task-row checkpoint, the temporary
        # files are removed and never consume shared-filesystem inodes.
        selection_root = scratch_root / f"selection_{selection_order:06d}"

        if selection_root.exists():
            shutil.rmtree(selection_root)

        transient_batch_root = selection_root / "candidate_indices_batch"
        summary = pipeline.repair_one(
            selected,
            "candidate_indices",
            batch_root_override=transient_batch_root,
        )
        design_root = Path(summary.result_directory)
        row = consolidated_repair_row(
            design_root,
            summary,
            task_id=task_id,
            parameter_names=pipeline.parameters.names,
            performance_columns=pipeline.dataset.performance.columns,
            constraint_catalog=pipeline.dataset.constraint_catalog,
        )
        saved = pd.concat(
            [saved, pd.DataFrame([row])],
            ignore_index=True,
            sort=False,
        )
        checkpoint_task_rows(saved, output_path)
        completed_orders.add(selection_order)
        shutil.rmtree(selection_root)

    # Re-read the committed file rather than reporting the in-memory table so
    # the final counts describe exactly what a requeued worker will observe.
    saved = read_task_shard(output_path)
    completed_count = int(
        saved["Final_Status"].astype(str).str.lower().eq("completed").sum()
    )
    feasible_count = int(
        saved["Fully_Feasible"]
        .map(lambda value: str(value).strip().lower() in {"true", "1"})
        .sum()
    )

    if len(saved) != len(owned):
        raise RuntimeError(
            f"Task shard contains {len(saved)} rows; expected {len(owned)}."
        )

    print(f"Reused completed alternatives: {reused_count}")
    print(f"Completed evaluations in task: {completed_count}/{len(saved)}")
    print(
        "Fully feasible alternatives in task: "
        f"{feasible_count}/{len(saved)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
