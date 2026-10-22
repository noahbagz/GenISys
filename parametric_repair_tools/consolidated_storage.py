"""Store many repaired designs as rows in one task-owned CSV shard.

The ordinary GenISys evaluator is intentionally file-oriented.  Cluster
repair instead runs that evaluator in node-local scratch, converts the useful
reduced-data products into one wide row, and checkpoints all completed rows in
one gzip-compressed CSV per SLURM task.  This preserves the design vector,
performance metrics, constraint results, failures, and repair provenance
without leaving dozens of shared-filesystem objects per repaired design.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .dataset import write_csv_atomic
from .pipeline import RepairDesignSummary, _read_csv_allow_empty


STORAGE_SCHEMA_VERSION = 1
TASK_SHARD_DIRECTORY = "task_results"


def task_shard_path(batch_root: str | Path, task_id: int) -> Path:
    """Return the single consolidated output path owned by one array task."""

    return (
        Path(batch_root)
        / TASK_SHARD_DIRECTORY
        / f"Repaired_Designs_Task_{int(task_id):04d}.csv.gz"
    )


def _table_json(path: Path) -> str:
    """Serialize one compact evaluator table into a JSON-valued CSV cell."""

    if not path.exists():
        return "[]"

    table = _read_csv_allow_empty(path)

    if table.empty:
        return "[]"

    return table.to_json(
        orient="records",
        double_precision=15,
        date_format="iso",
    )


def _json_file(path: Path) -> str:
    """Return one JSON artifact as stable compact JSON text."""

    if not path.exists():
        return "{}"

    payload = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _constraint_bit(value) -> bool:
    """Apply the project's convention that an empty/N/A bit is passing."""

    if pd.isna(value):
        return True

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {"true", "1"}


def _constraint_key(row: pd.Series) -> str:
    """Return the same stable requirement/name identity used by repair."""

    return (
        f"{str(row.get('Requirement', ''))}::"
        f"{str(row.get('Name', ''))}"
    )


def _constraint_catalog_keys(catalog: pd.DataFrame) -> tuple[str, ...]:
    """Return the complete, de-duplicated wide constraint schema."""

    if "Constraint_Key" in catalog:
        values = catalog["Constraint_Key"].dropna().astype(str)
    else:
        values = (
            catalog["Requirement"].fillna("").astype(str)
            + "::"
            + catalog["Name"].fillna("").astype(str)
        )

    # Sorting makes every task write an identical CSV header even when
    # conditional constraint rows are absent from its particular designs.
    return tuple(sorted(set(values)))


def _constraint_classes(catalog: pd.DataFrame) -> tuple[str, ...]:
    """Return every named subsystem/class represented by the rule catalog."""

    if "Class" not in catalog:
        return ()

    return tuple(sorted(set(catalog["Class"].dropna().astype(str))))


def consolidated_repair_row(
    design_root: str | Path,
    summary: RepairDesignSummary,
    *,
    task_id: int,
    parameter_names: Iterable[str],
    performance_columns: Iterable[str],
    constraint_catalog: pd.DataFrame,
) -> dict:
    """Convert one completed repair directory into one tabular data row.

    The fixed-width portions needed by the DDPM remain ordinary columns.
    Variable-length evidence remains losslessly represented as JSON text in
    individual cells.  Exact constraint rows are retained in JSON while a
    stable wide pass/residual view supports ordinary dataframe analysis.
    """

    design_root = Path(design_root)
    parameter_names = tuple(parameter_names)
    performance_columns = tuple(performance_columns)
    constraint_keys = _constraint_catalog_keys(constraint_catalog)
    constraint_classes = _constraint_classes(constraint_catalog)

    final_vector = _read_csv_allow_empty(
        design_root / "final_189_parameter_vector.csv.gz"
    )
    constraints = _read_csv_allow_empty(
        design_root / "final_constraint_evaluations.csv.gz"
    )
    performance = _read_csv_allow_empty(
        design_root / "final_performance_metrics.csv.gz"
    )

    row = {
        "Storage_Schema_Version": STORAGE_SCHEMA_VERSION,
        "Task_ID": int(task_id),
        "Index": int(summary.repaired_index),
        "Parent_Index": int(summary.parent_index),
        "Alternative_Number": int(summary.alternative_number),
        "Selection_Order": int(summary.selection_order),
        "Final_Status": str(summary.final_status),
        "Fully_Feasible": bool(summary.fully_feasible),
        "Repaired_Structure_Index": int(summary.repaired_structure_index),
    }

    # Retain every summary value as a scalar column.  The normalized names
    # above remain the public loader contract; these columns preserve the
    # complete one-pass repair bookkeeping without another summary file.
    for name, value in asdict(summary).items():
        if name == "result_directory":
            # The ordinary result directory exists only in node-local scratch
            # and is removed after this consolidated row is committed.
            value = ""

        row[f"Summary::{name}"] = value

    # A failed evaluator may not commit an updated-parameter row.  The repair
    # pipeline still writes the submitted final vector, so exactly one vector
    # row must always be available here.
    if len(final_vector) != 1:
        raise ValueError(
            "A consolidated repair row requires exactly one final vector: "
            f"{design_root}."
        )

    vector_row = final_vector.iloc[0]

    for name in parameter_names:
        row[str(name)] = vector_row.get(name, np.nan)

    # Performance outputs are prefixed to prevent names such as Index from
    # colliding with the repaired design identity or parameter coordinates.
    performance_row = performance.iloc[0] if len(performance) == 1 else None

    for name in performance_columns:
        value = np.nan if performance_row is None else performance_row.get(name)
        row[f"Performance::{name}"] = value

    # Initialize the complete catalog.  A missing conditional rule is N/A and
    # therefore passes under the approved GenISys convention.  An evaluator
    # failure remains globally infeasible through Final_Status.
    for key in constraint_keys:
        row[f"Constraint::{key}::Pass"] = True
        row[f"Constraint::{key}::Residual"] = np.nan

    for class_name in constraint_classes:
        row[f"Feasible::{class_name}"] = (
            str(summary.final_status).lower() == "completed"
        )

    if not constraints.empty:
        keyed = constraints.copy()
        keyed["__Constraint_Key"] = keyed.apply(_constraint_key, axis=1)
        keyed["__Passed"] = keyed["Constraint Bit"].map(_constraint_bit)
        keyed["__Residual"] = pd.to_numeric(
            keyed.get("Constraint Eval"),
            errors="coerce",
        )

        for key, group in keyed.groupby("__Constraint_Key", sort=False):
            if key not in constraint_keys:
                # Runtime-only rows remain present in the exact JSON payload.
                # They must not make task CSV headers differ from one another.
                continue

            row[f"Constraint::{key}::Pass"] = bool(group["__Passed"].all())
            residuals = group["__Residual"].dropna()
            row[f"Constraint::{key}::Residual"] = (
                float(residuals.max()) if not residuals.empty else np.nan
            )

        if str(summary.final_status).lower() == "completed":
            for class_name, group in keyed.groupby("Class", sort=False):
                if str(class_name) not in constraint_classes:
                    continue

                row[f"Feasible::{class_name}"] = bool(
                    group["__Passed"].all()
                )

    # These JSON cells preserve the variable-length evidence that cannot be
    # represented naturally as one scalar per design.  Structural elements
    # and large numerical arrays are intentionally not copied: the final 189
    # coordinates recreate them, and reduced-mode DDPM training does not read
    # those transient evaluator products.
    row.update(
        {
            "Constraint_Evaluations_JSON": _table_json(
                design_root / "final_constraint_evaluations.csv.gz"
            ),
            "Evaluation_Failures_JSON": _table_json(
                design_root / "final_evaluation_failures.csv.gz"
            ),
            "Repair_Actions_JSON": _table_json(
                design_root / "repair_actions.csv.gz"
            ),
            "Source_Constraint_Evaluations_JSON": _table_json(
                design_root / "source_constraint_evaluations.csv.gz"
            ),
            "Source_Vector_JSON": _table_json(
                design_root / "source_189_parameter_vector.csv.gz"
            ),
            "Proposed_Vector_JSON": _table_json(
                design_root / "proposed_189_parameter_vector.csv.gz"
            ),
            "Sampled_Repair_Decisions_JSON": _table_json(
                design_root / "sampled_repair_decisions.csv"
            ),
            "Unresolved_Source_Constraints_JSON": _table_json(
                design_root / "unresolved_source_constraints.csv.gz"
            ),
            "Final_Unresolved_Constraints_JSON": _table_json(
                design_root / "final_unresolved_constraints.csv.gz"
            ),
            "Proposed_Bound_Exceedances_JSON": _table_json(
                design_root / "proposed_sampling_bound_exceedances.csv"
            ),
            "Final_Bound_Exceedances_JSON": _table_json(
                design_root / "final_sampling_bound_exceedances.csv"
            ),
            "Evaluation_Failure_JSON": _json_file(
                design_root / "evaluation_failure.json"
            ),
        }
    )
    return row


def read_task_shard(path: str | Path) -> pd.DataFrame:
    """Read an existing restart checkpoint or return an empty table."""

    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def checkpoint_task_rows(rows: pd.DataFrame, path: str | Path) -> Path:
    """Atomically replace one task shard after a completed design commit."""

    if rows.empty:
        raise ValueError("A task checkpoint must contain at least one row.")

    if rows["Selection_Order"].duplicated().any():
        raise ValueError("A task checkpoint contains duplicate selection rows.")

    ordered = rows.sort_values("Selection_Order").reset_index(drop=True)
    return write_csv_atomic(ordered, Path(path))


__all__ = [
    "STORAGE_SCHEMA_VERSION",
    "TASK_SHARD_DIRECTORY",
    "checkpoint_task_rows",
    "consolidated_repair_row",
    "read_task_shard",
    "task_shard_path",
]
