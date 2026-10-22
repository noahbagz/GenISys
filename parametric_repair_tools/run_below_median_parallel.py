#!/usr/bin/env python3
"""Repair the complete below-median source population in parallel.

The main process freezes the source selection and creates all shared batch
metadata before any worker starts.  Each worker owns complete, independent
repair alternatives.  Workers never update a common summary file; the main
process writes the ordered batch summary only after every repair returns.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .configuration import RepairConfiguration
from .dataset import (
    freeze_below_median_alternatives,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)
from .pipeline import (
    RepairPipeline,
    _build_batch_output_manifest,
    _implementation_paths,
    _json_safe,
    _prepare_design_root,
    _runtime_provenance,
)


_WORKER_PIPELINE: RepairPipeline | None = None


def _initialize_worker(configuration: RepairConfiguration) -> None:
    """Load the immutable source dataset once in each worker process."""

    global _WORKER_PIPELINE
    _WORKER_PIPELINE = RepairPipeline(configuration)


def _repair_selection_row(row: dict) -> dict:
    """Repair and evaluate one alternative assigned by the process pool."""

    if _WORKER_PIPELINE is None:
        raise RuntimeError("The parallel repair worker was not initialized.")

    return asdict(
        _WORKER_PIPELINE.repair_one(
            pd.Series(row),
            "below_median",
        )
    )


def _prepare_result_directories(
    batch_root: Path,
    selection: pd.DataFrame,
) -> None:
    """Fail before dispatch if any alternative would overwrite evidence."""

    for _, row in selection.iterrows():
        design_root = (
            batch_root
            / "results"
            / f"parent_{int(row['Index']):06d}"
            / f"alternative_{int(row['Alternative_Number']):02d}"
        )
        _prepare_design_root(design_root)


def _write_batch_inputs(
    pipeline: RepairPipeline,
    batch_root: Path,
    selection: pd.DataFrame,
    workers: int,
    alternatives_per_source: int,
) -> None:
    """Write shared registries and provenance before parallel evaluation."""

    project_root = Path(__file__).resolve().parents[1]
    selected_parents = (
        selection["Index"].astype(int).drop_duplicates().tolist()
    )

    write_json_atomic(
        _json_safe(
            {
                "schema_version": 5,
                "method": "one_pass_stochastic_equation_repair",
                "execution": "parallel_process_pool",
                "worker_processes": int(workers),
                "configuration": pipeline.configuration.to_dict(),
                "configuration_sha256": (
                    pipeline.configuration.fingerprint()
                ),
                "batch_kind": "below_median",
                "selected_parent_indices": selected_parents,
                "repair_alternative_count": int(len(selection)),
                "runtime_provenance": _runtime_provenance(),
                "repair_contract": {
                    "source_evaluations": (
                        "saved V1.1 rows from the immutable 0-399 pool"
                    ),
                    "repair_passes_per_alternative": 1,
                    "repaired_evaluations_per_alternative": 1,
                    "alternatives_per_source": int(
                        alternatives_per_source
                    ),
                    "post_evaluation_repair": False,
                    "performance_metrics_used_for_repair": False,
                    "parallel_assignment": (
                        "each selection row is evaluated exactly once by one "
                        "ProcessPoolExecutor worker"
                    ),
                },
                "implementation_fingerprints": {
                    str(path.relative_to(project_root)): {
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in _implementation_paths(project_root)
                },
            }
        ),
        batch_root / "repair_run_manifest.json",
    )
    write_csv_atomic(
        pipeline.parameters.role_table(),
        batch_root / "parameter_repair_registry.csv",
    )
    write_csv_atomic(
        pipeline.constraint_registry.export_runtime_registry(
            pipeline.dataset.constraint_catalog
        ),
        batch_root / "constraint_repair_registry.csv",
    )


def _write_batch_outputs(
    batch_root: Path,
    summaries: list[dict],
    workers: int,
    alternatives_per_source: int,
) -> pd.DataFrame:
    """Compile worker returns into one ordered batch result."""

    table = pd.DataFrame(summaries).sort_values(
        "selection_order"
    ).reset_index(drop=True)
    write_csv_atomic(table, batch_root / "batch_repair_summary.csv")

    feasible_count = int(table["fully_feasible"].sum())
    write_json_atomic(
        _json_safe(
            {
                "batch_kind": "below_median",
                "execution": "parallel_process_pool",
                "worker_processes": int(workers),
                "source_design_count": int(
                    table["parent_index"].nunique()
                ),
                "alternatives_per_source": int(
                    alternatives_per_source
                ),
                "design_count": int(len(table)),
                "fully_feasible_count": feasible_count,
                "success_rate": float(table["fully_feasible"].mean()),
                "target": "report all below-median repair outcomes",
                "target_met": None,
            }
        ),
        batch_root / "batch_outcome.json",
    )
    write_json_atomic(
        _json_safe(_build_batch_output_manifest(batch_root)),
        batch_root / "batch_output_manifest.json",
    )
    return table


def parse_arguments() -> argparse.Namespace:
    """Read the laptop process count without changing repair equations."""

    parser = argparse.ArgumentParser(
        description=(
            "Create independently seeded one-pass repair alternatives for "
            "every completed "
            "partially feasible source below the median failure count."
        )
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of independent repair/evaluation worker processes.",
    )
    return parser.parse_args()


def main() -> int:
    """Freeze, parallelize, and compile the below-median repair batch."""

    arguments = parse_arguments()

    if int(arguments.workers) < 1:
        raise ValueError("--workers must be at least one.")

    configuration = RepairConfiguration()
    pipeline = RepairPipeline(configuration)
    batch_root = Path(configuration.experiment_root) / "below_median_batch"
    selection = freeze_below_median_alternatives(
        pipeline.dataset,
        batch_root,
        seed=configuration.below_median_batch_seed,
        alternatives_per_source=(
            configuration.alternatives_per_below_median_source
        ),
        batch_kind="below_median",
    ).sort_values("Selection_Order").reset_index(drop=True)

    _prepare_result_directories(batch_root, selection)
    _write_batch_inputs(
        pipeline,
        batch_root,
        selection,
        int(arguments.workers),
        configuration.alternatives_per_below_median_source,
    )

    records = selection.to_dict(orient="records")

    # Executor.map returns results in input order while dynamically assigning
    # independent alternatives to the six workers.  Any worker exception
    # aborts compilation, leaving the partial evidence available for review.
    with ProcessPoolExecutor(
        max_workers=int(arguments.workers),
        initializer=_initialize_worker,
        initargs=(configuration,),
    ) as executor:
        summaries = list(
            executor.map(_repair_selection_row, records, chunksize=1)
        )

    table = _write_batch_outputs(
        batch_root,
        summaries,
        int(arguments.workers),
        configuration.alternatives_per_below_median_source,
    )
    feasible = int(table["fully_feasible"].sum())
    print(
        f"Repaired alternatives: {len(table)}; "
        f"fully feasible: {feasible}; workers: {int(arguments.workers)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
