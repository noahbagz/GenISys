#!/usr/bin/env python3
"""Validate one-pass repair evidence without rerunning a ship design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from .configuration import RepairConfiguration
from .dataset import sha256_file
from .evaluation import (
    canonical_evaluated_constraint_count,
    canonical_failed_constraint_rows,
    normalized_violation_score,
    vector_hash,
)
from .parameters import ParameterRegistry
from .pipeline import ACTION_OUTPUT_COLUMNS


BATCH_OUTPUT_FILENAMES = {
    "selected_source_designs.csv",
    "original_189_parameter_vectors.csv.gz",
    "original_constraint_results.csv.gz",
    "original_performance_metrics.csv.gz",
    "original_evaluation_failures.csv.gz",
    "selection_manifest.json",
    "repair_run_manifest.json",
    "parameter_repair_registry.csv",
    "constraint_repair_registry.csv",
    "batch_repair_summary.csv",
    "batch_outcome.json",
}


def _read_csv_allow_empty(path: Path) -> pd.DataFrame:
    """Read an evaluator table that may be empty after a failed evaluation."""

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _verify_hash(path: Path, record: dict, label: str) -> None:
    """Verify one path against its saved size and SHA-256."""

    if not path.exists():
        raise AssertionError(f"Missing recorded artifact {label}: {path}")

    if int(path.stat().st_size) != int(record["size_bytes"]):
        raise AssertionError(f"Recorded artifact size changed: {label}")

    if sha256_file(path) != record["sha256"]:
        raise AssertionError(f"Recorded artifact hash changed: {label}")


def _require_manifest_labels(
    manifest: dict,
    required_labels: set[str],
    context: str,
) -> None:
    """Reject a manifest that omits expected one-pass evidence."""

    missing = required_labels-set(manifest)

    if missing:
        raise AssertionError(
            f"{context} manifest omits required labels: {sorted(missing)}"
        )


def _read_vector(
    path: Path,
    parameters: ParameterRegistry,
) -> tuple[pd.DataFrame, object]:
    """Read one saved design row and reconstruct its 189-value vector."""

    table = pd.read_csv(path)

    if len(table) != 1:
        raise AssertionError(f"Expected exactly one vector row in {path}.")

    vector = parameters.vector_from_row(table.iloc[0])

    if len(vector) != 189:
        raise AssertionError(f"Expected 189 parameters in {path}.")

    return table, vector


def _validate_design(
    design_root: Path,
    selected: pd.Series,
    parameters: ParameterRegistry,
    configuration: RepairConfiguration,
) -> dict:
    """Validate one source/proposal/final record and its hashes."""

    parent_index = int(selected["Index"])
    summary_path = design_root/"design_summary.json"

    if not summary_path.exists():
        raise AssertionError(
            f"Parent {parent_index}: missing design_summary.json."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if summary.get("repair_method") != "one_pass_stochastic_equation_repair":
        raise AssertionError(f"Parent {parent_index}: wrong repair method.")

    if int(summary.get("repair_passes", 0)) != 1:
        raise AssertionError(f"Parent {parent_index}: repair pass count is not one.")

    if int(summary.get("repaired_design_evaluations", 0)) != 1:
        raise AssertionError(
            f"Parent {parent_index}: repaired evaluation count is not one."
        )

    if summary.get("repair_configuration_sha256") != configuration.fingerprint():
        raise AssertionError(
            f"Parent {parent_index}: repair configuration fingerprint changed."
        )

    _, source_vector = _read_vector(
        design_root/"source_189_parameter_vector.csv.gz",
        parameters,
    )
    _, proposed_vector = _read_vector(
        design_root/"proposed_189_parameter_vector.csv.gz",
        parameters,
    )
    _, final_vector = _read_vector(
        design_root/"final_189_parameter_vector.csv.gz",
        parameters,
    )

    expected_hashes = {
        "original_vector_sha256": vector_hash(source_vector),
        "proposed_vector_sha256": vector_hash(proposed_vector),
        "final_vector_sha256": vector_hash(final_vector),
    }

    for field, expected in expected_hashes.items():
        if summary.get(field) != expected:
            raise AssertionError(
                f"Parent {parent_index}: {field} does not match saved vector."
            )

    source_constraints = _read_csv_allow_empty(
        design_root/"source_constraint_evaluations.csv.gz"
    )
    final_constraints = _read_csv_allow_empty(
        design_root/"final_constraint_evaluations.csv.gz"
    )
    final_failures = _read_csv_allow_empty(
        design_root/"final_evaluation_failures.csv.gz"
    )

    source_failure_count = len(
        canonical_failed_constraint_rows(source_constraints)
    )
    final_failure_count = len(
        canonical_failed_constraint_rows(final_constraints)
    )
    final_evaluated_count = canonical_evaluated_constraint_count(
        final_constraints
    )
    fully_feasible = bool(
        summary.get("final_status") == "completed"
        and final_evaluated_count > 0
        and final_failure_count == 0
    )

    checks = {
        "source_canonical_failure_count": source_failure_count,
        "final_failure_count": final_failure_count,
        "final_evaluated_constraint_count": final_evaluated_count,
        "fully_feasible": fully_feasible,
    }

    for field, expected in checks.items():
        if summary.get(field) != expected:
            raise AssertionError(
                f"Parent {parent_index}: {field} does not match saved evidence."
            )

    saved_score = float(summary.get("final_violation_score", 0.0))
    actual_score = float(normalized_violation_score(final_constraints))

    if abs(saved_score-actual_score) > 1.0e-12:
        raise AssertionError(
            f"Parent {parent_index}: final violation score is inconsistent."
        )

    actions = _read_csv_allow_empty(design_root/"repair_actions.csv.gz")

    if list(actions.columns) != ACTION_OUTPUT_COLUMNS:
        raise AssertionError(
            f"Parent {parent_index}: repair action schema changed."
        )

    if list(design_root.glob("iterations")):
        raise AssertionError(
            f"Parent {parent_index}: obsolete iterative artifacts are present."
        )

    output_manifest_path = design_root/"final_output_manifest.json"
    output_manifest = json.loads(
        output_manifest_path.read_text(encoding="utf-8")
    )
    required_local = {
        "source_189_parameter_vector.csv.gz",
        "source_constraint_evaluations.csv.gz",
        "proposed_189_parameter_vector.csv.gz",
        "unresolved_source_constraints.csv.gz",
        "proposed_sampling_bound_exceedances.csv",
        "sampled_repair_decisions.csv",
        "final_189_parameter_vector.csv.gz",
        "final_constraint_evaluations.csv.gz",
        "final_evaluation_failures.csv.gz",
        "final_performance_metrics.csv.gz",
        "final_unresolved_constraints.csv.gz",
        "final_sampling_bound_exceedances.csv",
        "repair_actions.csv.gz",
        "design_summary.json",
    }
    required_labels = {
        f"design_artifact::{filename}" for filename in required_local
    }

    if summary.get("final_status") == "completed":
        required_labels.add("final_structural_elements")

    _require_manifest_labels(
        output_manifest,
        required_labels,
        f"Parent {parent_index}",
    )

    for label, record in output_manifest.items():
        _verify_hash(Path(record["path"]), record, label)

    # A structured rejection must have a saved evaluator failure. A completed
    # design must not silently retain one.
    if summary.get("final_status") == "completed" and not final_failures.empty:
        raise AssertionError(
            f"Parent {parent_index}: completed result has evaluator failures."
        )

    return summary


def validate_batch(
    batch_kind: str,
    configuration: RepairConfiguration | None = None,
) -> pd.DataFrame:
    """Validate every completed one-pass result in a frozen batch."""

    if batch_kind not in {"development", "validation"}:
        raise ValueError("batch_kind must be development or validation.")

    configuration = configuration or RepairConfiguration()
    parameters = ParameterRegistry(configuration)
    batch_root = (
        Path(configuration.experiment_root)/f"{batch_kind}_batch"
    )
    selection_path = batch_root/"selected_source_designs.csv"

    if not selection_path.exists():
        raise FileNotFoundError(selection_path)

    selection = pd.read_csv(selection_path).sort_values("Selection_Order")
    run_manifest_path = batch_root/"repair_run_manifest.json"

    if not run_manifest_path.exists():
        raise AssertionError("Completed repair records lack a run manifest.")

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    if run_manifest.get("method") != "one_pass_stochastic_equation_repair":
        raise AssertionError("Run manifest does not describe the one-pass method.")

    contract = run_manifest.get("repair_contract", {})

    if (
        int(contract.get("repair_passes_per_design", 0)) != 1
        or int(contract.get("repaired_evaluations_per_design", 0)) != 1
        or bool(contract.get("post_evaluation_repair", True))
    ):
        raise AssertionError("Run manifest violates the one-pass contract.")

    rows = []

    for _, selected in selection.iterrows():
        parent_index = int(selected["Index"])
        design_root = (
            batch_root/"results"/f"parent_{parent_index:06d}"
        )

        if not (design_root/"design_summary.json").exists():
            continue

        rows.append(
            _validate_design(
                design_root,
                selected,
                parameters,
                configuration,
            )
        )

    if not rows:
        raise AssertionError(
            f"The requested {batch_kind} batch has no repair records."
        )

    batch_manifest_path = batch_root/"batch_output_manifest.json"

    if batch_manifest_path.exists():
        batch_manifest = json.loads(
            batch_manifest_path.read_text(encoding="utf-8")
        )
        required = {
            f"batch_artifact::{filename}"
            for filename in BATCH_OUTPUT_FILENAMES
        }
        required.update(
            f"design_manifest::parent_{int(row['parent_index']):06d}"
            for row in rows
        )
        _require_manifest_labels(batch_manifest, required, "Batch output")

        for label, record in batch_manifest.items():
            _verify_hash(Path(record["path"]), record, label)

    return pd.DataFrame(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        choices=("development", "validation"),
        default="development",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    try:
        table = validate_batch(arguments.batch)
    except (AssertionError, FileNotFoundError, ValueError) as error:
        print(f"Repair-output validation failed: {error}", file=sys.stderr)
        return 1

    print(table.to_string(index=False))
    feasible = int(table["fully_feasible"].sum())
    print(f"Validated one-pass results: {len(table)}; feasible: {feasible}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
