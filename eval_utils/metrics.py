"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import numpy as np


import pandas as pd


OBJECTIVE_COLUMNS = (
    "Required_Freight_Rate_pounds_per_t",
    "Engine_Power_kW",
    "Annual_Cargo_t",
)


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    """Return the minimization nondominated mask for finite objective rows."""

    objectives = np.asarray(objectives, dtype=float)
    finite = np.isfinite(objectives).all(axis=1)
    result = np.zeros(len(objectives), dtype=bool)
    finite_indices = np.flatnonzero(finite)
    values = objectives[finite]

    for local_index, point in enumerate(values):
        dominated = np.any(
            np.all(values <= point, axis=1)
            & np.any(values < point, axis=1)
        )
        result[finite_indices[local_index]] = not dominated
    return result


def exact_duplicate_mask(vectors: np.ndarray) -> np.ndarray:
    """Mark every occurrence after the first exact postprocessed vector."""

    frame = pd.DataFrame(np.asarray(vectors, dtype=float))
    return frame.duplicated(keep="first").to_numpy()


def summarize_run(
    sample_status: pd.DataFrame,
    *,
    performance: pd.DataFrame | None = None,
    postprocessed_vectors: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute the primary denominator-safe experiment responses."""

    requested = len(sample_status)
    completed = sample_status["Evaluator_Completed"].fillna(False).astype(bool)
    feasible = sample_status["Global_V1p1_Feasible"].fillna(False).astype(bool)
    useful = completed & feasible
    violations = pd.to_numeric(
        sample_status["V1p1_Constraint_Fail_Count"],
        errors="coerce",
    )

    result = {
        "Requested_Sample_Count": requested,
        "Evaluator_Completed_Count": int(completed.sum()),
        "Verified_Feasible_Count": int(useful.sum()),
        "Evaluator_Validity_Rate": float(completed.mean()) if requested else np.nan,
        "Feasibility_Given_Valid": (
            float(feasible[completed].mean()) if bool(completed.any()) else np.nan
        ),
        "Useful_Yield": float(useful.mean()) if requested else np.nan,
        "Mean_Violation_Count_Given_Valid": (
            float(violations[completed].mean()) if bool(completed.any()) else np.nan
        ),
        "Median_Violation_Count_Given_Valid": (
            float(violations[completed].median()) if bool(completed.any()) else np.nan
        ),
    }

    if postprocessed_vectors is not None:
        duplicates = exact_duplicate_mask(postprocessed_vectors)
        result["Exact_Duplicate_Count"] = int(duplicates.sum())
        result["Exact_Duplicate_Rate"] = float(duplicates.mean())

    if performance is not None and not performance.empty:
        feasible_performance = performance.merge(
            sample_status.loc[
                useful,
                ["Sample_Index"],
            ],
            on="Sample_Index",
            how="inner",
        )
        if not feasible_performance.empty:
            objectives = feasible_performance.loc[
                :,
                OBJECTIVE_COLUMNS,
            ].to_numpy(dtype=float)
            objectives[:, 2] *= -1.0
            result["Feasible_Nondominated_Count"] = int(
                nondominated_mask(objectives).sum()
            )
        else:
            result["Feasible_Nondominated_Count"] = 0

    return pd.DataFrame([result])


def constraint_feasibility_table(constraints: pd.DataFrame) -> pd.DataFrame:
    """Report evaluated-pass frequency without converting N/A rows to failure."""

    if constraints.empty:
        return pd.DataFrame(
            columns=(
                "Class",
                "Requirement",
                "Name",
                "Evaluated_Count",
                "Pass_Count",
                "Pass_Rate_Given_Evaluated",
            )
        )
    rows = constraints.loc[constraints["Constraint Bit"].notna()].copy()
    normalized = rows["Constraint Bit"].astype(str).str.lower()
    rows["Passed"] = normalized.isin({"true", "1"})
    group_columns = ["Class", "Requirement", "Name"]
    result = rows.groupby(group_columns, dropna=False).agg(
        Evaluated_Count=("Passed", "size"),
        Pass_Count=("Passed", "sum"),
    ).reset_index()
    result["Pass_Rate_Given_Evaluated"] = (
        result["Pass_Count"] / result["Evaluated_Count"]
    )
    return result.sort_values(group_columns).reset_index(drop=True)


__all__ = [
    "constraint_feasibility_table",
    "exact_duplicate_mask",
    "nondominated_mask",
    "summarize_run",
]


