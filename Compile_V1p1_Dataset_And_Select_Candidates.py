"""Compile the production V1.1 dataset and select repair candidates.

This command is intentionally a thin orchestration layer.  It reuses the
existing V1.1 compiler, compiler acceptance checks, and repair-dataset median
selection instead of maintaining a second implementation of those rules.

The resulting ``compiled/Candidate_Indices.csv`` contains the indices of all
completed evaluations whose canonical V1.1 constraint-failure count is
strictly less than the median among completed evaluations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Compile_V1p1_Dataset import (
    DEFAULT_OUTPUT_ROOT,
    NUMBER_OF_STRUCTURES,
    compile_dataset,
)
from Validate_V1p1_Pipeline import PipelineValidationError
from parametric_repair_tools.dataset import (
    SourceDataset,
    read_candidate_indices,
    write_csv_atomic,
)
from tools.LoadingConditionLoads import (
    LOAD_CLOSURE_RELATIVE_TOLERANCE,
    MASS_CLOSURE_RELATIVE_TOLERANCE,
)


# These checks describe individual evaluations that cannot be admitted to the
# repair-candidate population.  They do not mean that the compiled population
# is incomplete, so they are reported as warnings instead of preventing every
# other valid, completed design from becoming a repair candidate.
CANDIDATE_SELECTION_WARNING_CHECKS = frozenset(
    {
        "no_numerical_or_implementation_failures",
        "named_condition_longitudinal_equilibrium_solved",
        "named_condition_relative_load_closure",
    }
)


def _validate_for_candidate_selection(
    dataset_root: Path,
) -> dict[str, object]:
    """Require all compilation checks except documented design-level checks."""

    checks_path = dataset_root / "compiled" / "Compilation_Checks.csv"

    if not checks_path.is_file():
        raise PipelineValidationError(
            f"Missing compiler checks: {checks_path}"
        )

    checks = pd.read_csv(checks_path)
    normalized = checks["Passed"].astype(str).str.strip().str.lower()

    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise PipelineValidationError(
            "Compiler checks contain an unrecognized Passed value."
        )

    passed = normalized.isin({"true", "1"})
    failed_checks = set(checks.loc[~passed, "Check"].astype(str))
    blocking_checks = sorted(
        failed_checks - CANDIDATE_SELECTION_WARNING_CHECKS
    )

    if blocking_checks:
        raise PipelineValidationError(
            "Compiler integrity checks failed: " + str(blocking_checks)
        )

    return {
        "compiler_checks": len(checks),
        "passed_compiler_checks": int(passed.sum()),
        "warning_checks": sorted(
            failed_checks & CANDIDATE_SELECTION_WARNING_CHECKS
        ),
    }


def _invalid_named_condition_indices(dataset_root: Path) -> set[int]:
    """Find completed-system indices with invalid named-condition solutions."""

    shard_root = dataset_root / "evaluations" / "named_condition_loads"
    shard_paths = sorted(shard_root.glob("Named_Condition_Loads_*.csv"))
    shard_paths += sorted(shard_root.glob("Named_Condition_Loads_*.csv.gz"))

    if not shard_paths:
        raise FileNotFoundError(
            f"No named-condition load shards were found under {shard_root}."
        )

    load_closure_columns = [
        "Net_Vertical_Force_Residual_Fraction",
        "Net_Vertical_First_Moment_Residual_Fraction",
        "Aft_Shear_Force_Residual_Fraction",
        "Aft_Bending_Moment_Residual_Fraction",
    ]
    required_columns = [
        "Index",
        "Equilibrium_Solved",
        "Mass_Residual_Fraction",
        *load_closure_columns,
    ]
    invalid_indices: set[int] = set()

    # Read one shard at a time so the production candidate-selection command
    # does not retain the complete named-condition table in memory.
    for shard_path in shard_paths:
        table = pd.read_csv(shard_path, usecols=required_columns)
        solved = (
            table["Equilibrium_Solved"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1"})
        )
        mass_residual = pd.to_numeric(
            table["Mass_Residual_Fraction"], errors="coerce"
        ).to_numpy(dtype=float)
        load_residuals = table[load_closure_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)

        invalid_mass_closure = (
            (~np.isfinite(mass_residual))
            | (np.abs(mass_residual) > MASS_CLOSURE_RELATIVE_TOLERANCE)
        )
        invalid_load_closure = (
            (~np.isfinite(load_residuals))
            | (
                np.abs(load_residuals)
                > LOAD_CLOSURE_RELATIVE_TOLERANCE
            )
        ).any(axis=1)
        invalid_rows = (~solved.to_numpy()) | invalid_mass_closure
        invalid_rows |= invalid_load_closure

        invalid_indices.update(
            table.loc[invalid_rows, "Index"].astype(int).tolist()
        )

    return invalid_indices


def compile_and_select_candidates(
    dataset_root: str | Path,
    *,
    number_of_structures: int = NUMBER_OF_STRUCTURES,
    start_structure: int = 0,
    save_mode: str = "reduced",
) -> Path:
    """Compile one dataset and write its below-median candidate indices."""

    dataset_root = Path(dataset_root).resolve()

    # Use the production compiler so this command creates the same compiled
    # tables, manifest, constraint identities, and acceptance checks as the
    # standalone compilation command.
    compile_dataset(
        output_root=dataset_root,
        number_of_structures=number_of_structures,
        save_mode=save_mode,
        start_structure=start_structure,
    )

    # Population-integrity failures remain fatal.  Individual design-level
    # failures are retained in the compiled data as labeled outcomes and are
    # reported as warnings because candidate selection only admits valid,
    # completed evaluations.
    compilation_summary = _validate_for_candidate_selection(dataset_root)

    # A small number of evaluations can have a completed scalar row while a
    # named loading condition did not solve.  Dynamically exclude those system
    # indices rather than hard-coding the indices observed in one dataset.
    invalid_named_condition_indices = _invalid_named_condition_indices(
        dataset_root
    )

    # Reuse the repair package's established population definition:
    # completed evaluations with failure counts strictly below the completed-
    # population median.  No second median formula is introduced here.
    source_dataset = SourceDataset(dataset_root)
    candidates, median_failures = source_dataset.completed_below_median(
        excluded_indices=invalid_named_condition_indices
    )

    candidate_path = dataset_root / "compiled" / "Candidate_Indices.csv"
    candidate_table = pd.DataFrame(
        {"Index": candidates["Index"].astype(int)}
    )
    write_csv_atomic(candidate_table, candidate_path)

    # Read the committed file through the production repair parser.  This
    # confirms that the single-column artifact is suitable for the later
    # candidate-index repair workflow.
    committed_indices = read_candidate_indices(candidate_path)

    print(f"Dataset root: {dataset_root}")
    print(
        "Compilation checks passed: "
        f"{compilation_summary['passed_compiler_checks']}/"
        f"{compilation_summary['compiler_checks']}"
    )
    print(
        "Non-blocking compilation warnings: "
        f"{compilation_summary['warning_checks']}"
    )
    print(
        "Invalid named-condition system indices excluded: "
        f"{sorted(invalid_named_condition_indices)}"
    )
    print(f"Completed-population median failure count: {median_failures:g}")
    print(f"Below-median candidate count: {len(committed_indices)}")
    print(f"Candidate index file: {candidate_path}")

    return candidate_path


def _parse_arguments() -> argparse.Namespace:
    """Parse the production defaults and optional smaller-test overrides."""

    parser = argparse.ArgumentParser(
        description=(
            "Compile a GenISys V1.1 dataset, validate the compilation, and "
            "write completed below-median design indices."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=(
            "Dataset root containing structures, systems, evaluations, and "
            "manifests. Defaults to Ship_Systems_Designs_v1p1/Dataset_100k."
        ),
    )
    parser.add_argument(
        "--number-of-structures",
        type=int,
        default=NUMBER_OF_STRUCTURES,
        help="Expected structure count; the production default is 5000.",
    )
    parser.add_argument(
        "--start-structure",
        type=int,
        default=0,
        help="First expected structure index; the production default is 0.",
    )
    parser.add_argument(
        "--save-mode",
        choices=("all", "reduced"),
        default="reduced",
        help="Compiled-data mode; production defaults to reduced.",
    )
    return parser.parse_args()


def main() -> None:
    """Run compilation, acceptance validation, and candidate selection."""

    arguments = _parse_arguments()
    compile_and_select_candidates(
        arguments.dataset_root,
        number_of_structures=arguments.number_of_structures,
        start_structure=arguments.start_structure,
        save_mode=arguments.save_mode,
    )


if __name__ == "__main__":
    main()
