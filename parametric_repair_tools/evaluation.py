"""One-pass GenISys repair evaluation helpers.

The repair method reads an already-evaluated source design, changes its
parameters once, and sends the resulting vector through the ordinary V1.1
generation/evaluation pipeline once. This module deliberately contains no
candidate cache, ranking function, retry state, or optimization loop.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from cad_query_tools.generate import generate_structure_files
from Run_V1p1_System_Evaluation_Batch import evaluate_system_shard
from tools.ConstraintIdentity import (
    canonical_constraint_bits,
    collapse_constraint_aliases,
)

from .configuration import RepairConfiguration
from .dataset import write_csv_atomic
from .parameters import ParameterRegistry


def vector_hash(vector: np.ndarray) -> str:
    """Hash a vector's exact float64 representation for provenance."""

    values = np.asarray(vector, dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _constraint_bits(table: pd.DataFrame) -> pd.Series:
    """Return Boolean values only for applicable pass/fail result rows."""

    if table.empty or "Constraint Bit" not in table:
        return pd.Series(dtype=bool)

    nonmissing = table["Constraint Bit"].notna()
    values = table.loc[nonmissing, "Constraint Bit"]

    if values.dtype == bool:
        return values.astype(bool)

    normalized = values.astype(str).str.strip().str.lower()
    recognized = normalized.isin({"true", "false", "1", "0"})

    if not bool(recognized.all()):
        unknown = sorted(normalized.loc[~recognized].unique())
        raise ValueError(f"Unrecognized constraint-bit values: {unknown}")

    return normalized.isin({"true", "1"})


def failed_constraint_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Return applicable failed rows; metrics and N/A rows are excluded."""

    bits = _constraint_bits(table)
    return table.loc[bits.index[~bits]].copy()


def canonical_failed_constraint_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Return one failed row per reviewed physical constraint identity."""

    collapsed = collapse_constraint_aliases(table)

    if collapsed.empty:
        return collapsed

    bits = _constraint_bits(collapsed)
    return collapsed.loc[bits.index[~bits]].copy()


def canonical_evaluated_constraint_count(table: pd.DataFrame) -> int:
    """Count applicable constraints after reviewed alias consolidation."""

    return int(len(canonical_constraint_bits(table)))


def normalized_violation_score(table: pd.DataFrame) -> float:
    """Return a scale-neutral description of the remaining violations."""

    failed = canonical_failed_constraint_rows(table)

    if failed.empty:
        return 0.0

    residual = pd.to_numeric(
        failed["Constraint Eval"], errors="coerce"
    ).to_numpy(dtype=float)
    threshold = pd.to_numeric(
        failed["Threshold Value"], errors="coerce"
    ).to_numpy(dtype=float)
    offered = pd.to_numeric(
        failed["Constraint Value"], errors="coerce"
    ).to_numpy(dtype=float)
    scale = np.maximum.reduce(
        [np.ones_like(residual), np.abs(threshold), np.abs(offered)]
    )
    terms = np.where(
        np.isfinite(residual),
        np.maximum(residual, 0.0)/scale,
        1.0,
    )
    return float(np.sum(terms))


def evaluate_repaired_vector(
    vector: np.ndarray,
    *,
    output_root: str | Path,
    repaired_index: int,
    repaired_structure_index: int,
    source_identifiers: dict[str, int],
    configuration: RepairConfiguration,
    parameters: ParameterRegistry,
) -> tuple[dict[str, Path], np.ndarray, Path]:
    """Generate and evaluate one repaired vector exactly once.

    CadQuery remains the authority for its cleaned structural coordinates.
    The cleaned vector is written to the ordinary one-design V1.1 input shard,
    and evaluate_system_shard creates the reduced dataset output family. The
    returned submitted vector is retained if evaluation fails before writing
    an updated parameter row.
    """

    output_root = Path(output_root)
    canonical, _ = parameters.canonicalize(np.asarray(vector, dtype=float))

    element_root = output_root / "structures" / "elements"
    generation = generate_structure_files(
        canonical[69:].copy(),
        output_directory=element_root,
        ship_id=f"Structure_{int(repaired_structure_index):06d}",
        clean_parameters=False,
        parameter_names=parameters.names[69:],
        artifact_mode="elements_only",
    )

    submitted_vector = canonical.copy()
    submitted_vector[69:] = np.asarray(
        generation["cleaned_parameters"],
        dtype=float,
    )
    submitted_vector, _ = parameters.canonicalize(submitted_vector)
    structural_elements_path = Path(generation["structural_elements_csv"])

    identifiers = {
        "Index": int(repaired_index),
        "Structure_Index": int(repaired_structure_index),
        "Hull_Variant_Index": int(source_identifiers["Hull_Variant_Index"]),
        "Operating_Condition_Index": int(
            source_identifiers["Operating_Condition_Index"]
        ),
        "Hull_Candidate_Index": int(source_identifiers["Hull_Candidate_Index"]),
        "Structure_Sample_Attempt": int(
            source_identifiers["Structure_Sample_Attempt"]
        ),
    }
    input_row = {
        **identifiers,
        **parameters.row_from_vector(submitted_vector),
    }
    input_path = (
        output_root
        / "repair_inputs"
        / f"System_Designs_{int(repaired_index):06d}_{int(repaired_index):06d}.csv.gz"
    )
    write_csv_atomic(pd.DataFrame([input_row]), input_path)

    paths = evaluate_system_shard(
        input_csv=input_path,
        output_root=output_root,
        run_flooding_diagnostic=configuration.run_flooding_diagnostic,
        flooding_step=configuration.flooding_step,
        flooding_permeabilities=configuration.flooding_permeabilities,
        include_legacy_constraints=False,
        write_per_design_review_files=False,
        save_mode=configuration.save_mode,
    )
    return paths, submitted_vector, structural_elements_path


__all__ = [
    "canonical_evaluated_constraint_count",
    "canonical_failed_constraint_rows",
    "evaluate_repaired_vector",
    "failed_constraint_rows",
    "normalized_violation_score",
    "vector_hash",
]
