"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


from concurrent.futures import ProcessPoolExecutor, as_completed


from dataclasses import replace


from pathlib import Path


import numpy as np


import pandas as pd


from parametric_repair_tools.configuration import RepairConfiguration


from parametric_repair_tools.evaluation import evaluate_repaired_vector


from parametric_repair_tools.parameters import ParameterRegistry


GENERATED_INDEX_BASE = 80_000_000


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _evaluate_one(payload: tuple) -> dict:
    (
        sample_index,
        vector,
        output_root,
        run_flooding_diagnostic,
        save_mode,
    ) = payload
    generated_index = GENERATED_INDEX_BASE + int(sample_index)
    configuration = replace(
        RepairConfiguration(),
        experiment_root=str(output_root),
        run_flooding_diagnostic=bool(run_flooding_diagnostic),
        save_mode=str(save_mode),
    )
    parameters = ParameterRegistry(configuration)
    identifiers = {
        "Hull_Variant_Index": 0,
        "Operating_Condition_Index": 0,
        "Hull_Candidate_Index": generated_index,
        "Structure_Sample_Attempt": 0,
    }

    try:
        paths, submitted, structural_elements = evaluate_repaired_vector(
            np.asarray(vector, dtype=float),
            output_root=output_root,
            repaired_index=generated_index,
            repaired_structure_index=generated_index,
            source_identifiers=identifiers,
            configuration=configuration,
            parameters=parameters,
        )
        return {
            "Sample_Index": int(sample_index),
            "Generated_Index": generated_index,
            "Worker_Status": "completed",
            "Worker_Error_Type": "",
            "Worker_Error_Message": "",
            "Submitted_Vector": submitted,
            "Structural_Elements_Path": str(structural_elements),
            "Paths": {key: str(value) for key, value in paths.items()},
        }
    except Exception as error:
        # Generation failures occur before evaluate_system_shard can create its
        # normal labeled-failure row.  Retain them here so useful yield always
        # uses the requested sample count as its denominator.
        return {
            "Sample_Index": int(sample_index),
            "Generated_Index": generated_index,
            "Worker_Status": "failed_before_evaluator_output",
            "Worker_Error_Type": type(error).__name__,
            "Worker_Error_Message": str(error),
            "Submitted_Vector": np.asarray(vector, dtype=float),
            "Structural_Elements_Path": "",
            "Paths": {},
        }


def evaluate_generated_samples(
    full_vectors: np.ndarray,
    output_root: str | Path,
    *,
    workers: int = 1,
    run_flooding_diagnostic: bool = False,
    save_mode: str = "reduced",
) -> dict[str, Path]:
    """Evaluate every generated vector once and compile retained outcomes."""

    full_vectors = np.asarray(full_vectors, dtype=float)
    if full_vectors.ndim != 2 or full_vectors.shape[1] != 189:
        raise ValueError("Generated evaluator inputs must have shape [n, 189].")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    compiled_root = output_root / "compiled"
    compiled_root.mkdir(parents=True, exist_ok=True)
    status_path = compiled_root / "Sample_Status.csv.gz"
    scalar_path = compiled_root / "Performance.csv.gz"
    constraint_path = compiled_root / "Constraint_Evaluations.csv.gz"
    failure_path = compiled_root / "Evaluation_Failures.csv.gz"
    submitted_path = compiled_root / "Submitted_189_Parameter_Vectors.csv.gz"

    targets = (status_path, scalar_path, constraint_path, failure_path, submitted_path)
    if any(path.exists() for path in targets):
        raise FileExistsError(
            f"Compiled generated-design outputs already exist in {compiled_root}."
        )

    payloads = [
        (
            sample_index,
            full_vectors[sample_index],
            output_root,
            run_flooding_diagnostic,
            save_mode,
        )
        for sample_index in range(len(full_vectors))
    ]

    if workers == 1:
        records = [_evaluate_one(payload) for payload in payloads]
    else:
        records = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_evaluate_one, payload): payload[0]
                for payload in payloads
            }
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda record: record["Sample_Index"])

    scalar_tables = []
    constraint_tables = []
    failure_tables = []
    status_rows = []
    submitted_rows = []

    for record in records:
        sample_index = record["Sample_Index"]
        generated_index = record["Generated_Index"]
        paths = record["Paths"]
        scalar = (
            _read_table(Path(paths["scalar_results"]))
            if "scalar_results" in paths
            else pd.DataFrame()
        )
        constraints = (
            _read_table(Path(paths["expanded_constraints_v1p1"]))
            if "expanded_constraints_v1p1" in paths
            else pd.DataFrame()
        )
        failures = (
            _read_table(Path(paths["failures"]))
            if "failures" in paths
            else pd.DataFrame()
        )

        for table in (scalar, constraints, failures):
            if not table.empty:
                table.insert(0, "Sample_Index", sample_index)

        if not scalar.empty:
            scalar_tables.append(scalar)
        if not constraints.empty:
            constraint_tables.append(constraints)
        if not failures.empty:
            failure_tables.append(failures)

        completed = not scalar.empty and str(
            scalar.iloc[0].get("Evaluation_Status", "completed")
        ).lower() == "completed"
        feasible = bool(
            scalar.iloc[0].get("Global_V1p1_Feasible", False)
        ) if completed else False
        violation_count = (
            scalar.iloc[0].get("V1p1_Constraint_Fail_Count", np.nan)
            if completed
            else np.nan
        )
        status_rows.append(
            {
                "Sample_Index": sample_index,
                "Generated_Index": generated_index,
                "Worker_Status": record["Worker_Status"],
                "Evaluator_Completed": completed,
                "Global_V1p1_Feasible": feasible,
                "Useful_Yield_Bit": bool(completed and feasible),
                "V1p1_Constraint_Fail_Count": violation_count,
                "Worker_Error_Type": record["Worker_Error_Type"],
                "Worker_Error_Message": record["Worker_Error_Message"],
                "Structural_Elements_Path": record["Structural_Elements_Path"],
            }
        )
        submitted_rows.append(
            {
                "Sample_Index": sample_index,
                "Generated_Index": generated_index,
                **{
                    f"Parameter_{position:03d}": value
                    for position, value in enumerate(record["Submitted_Vector"])
                },
            }
        )

    pd.DataFrame(status_rows).to_csv(status_path, index=False, compression="gzip")
    pd.DataFrame(submitted_rows).to_csv(
        submitted_path,
        index=False,
        compression="gzip",
    )
    (
        pd.concat(scalar_tables, ignore_index=True)
        if scalar_tables
        else pd.DataFrame()
    ).to_csv(scalar_path, index=False, compression="gzip")
    (
        pd.concat(constraint_tables, ignore_index=True)
        if constraint_tables
        else pd.DataFrame()
    ).to_csv(constraint_path, index=False, compression="gzip")
    (
        pd.concat(failure_tables, ignore_index=True)
        if failure_tables
        else pd.DataFrame()
    ).to_csv(failure_path, index=False, compression="gzip")

    return {
        "sample_status": status_path,
        "performance": scalar_path,
        "constraints": constraint_path,
        "failures": failure_path,
        "submitted_vectors": submitted_path,
    }


__all__ = ["GENERATED_INDEX_BASE", "evaluate_generated_samples"]


