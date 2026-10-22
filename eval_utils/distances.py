"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


from pathlib import Path


import numpy as np


import pandas as pd


from scipy.spatial.distance import cdist


from training_utils.schema import V1p1ParameterSchema


PROJECT_ROOT = Path(__file__).resolve().parents[1]


UNIFIED_ROOT = PROJECT_ROOT / "Ship_Systems_Designs_v1p1/Unified_Training_Dataset"


def load_feasible_training_reference(
    unified_root: Path = UNIFIED_ROOT,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Read the frozen feasible train split and its complete physical vectors."""

    schema = V1p1ParameterSchema()
    labels = pd.read_csv(
        unified_root / "Granular_20_Subsystem_Labels.csv.gz",
        usecols=["Population", "Index", "Labels_Known", "Feasible::Global"],
    )
    split = pd.read_csv(
        unified_root / "Shared_Split_Rows.csv.gz",
        usecols=["Population", "Index", "Split"],
    )
    selected = labels.merge(split, on=["Population", "Index"], validate="one_to_one")
    selected = selected.loc[
        selected["Split"].eq("train")
        & selected["Labels_Known"].eq(True)
        & selected["Feasible::Global"].eq(True),
        ["Population", "Index"],
    ]
    training = pd.read_csv(
        unified_root / "Training_Rows.csv.gz",
        usecols=["Population", "Index", *schema.full_names],
    )
    reference = selected.merge(training, on=["Population", "Index"], validate="one_to_one")
    vectors = reference.loc[:, schema.full_names].to_numpy(dtype=np.float64)
    lower = vectors.min(axis=0)
    upper = vectors.max(axis=0)
    return schema.full_names, vectors, lower, upper


def scale_cleaned_189(
    vectors: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Scale with feasible-reference extrema; fixed dimensions remain zero."""

    constant = np.isclose(upper, lower, rtol=0.0, atol=1.0e-12)
    if not np.allclose(vectors[:, constant], lower[constant], rtol=0.0, atol=1.0e-9):
        raise ValueError("A generated 189-vector changed a feasible-reference constant coordinate.")
    widths = np.where(constant, 1.0, upper - lower)
    scaled = (vectors - lower) / widths
    scaled[:, constant] = 0.0
    return scaled


def distance_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float | int]:
    """Exact Euclidean Chamfer directions and leave-self-out sample MNN."""

    generated_minima = np.full(len(generated), np.inf)
    reference_minima = np.full(len(reference), np.inf)

    # Bound pairwise working memory while keeping every reference design.
    for start in range(0, len(reference), 512):
        distances = cdist(reference[start:start + 512], generated, metric="euclidean")
        reference_minima[start:start + 512] = distances.min(axis=1)
        generated_minima = np.minimum(generated_minima, distances.min(axis=0))

    pairwise = cdist(generated, generated, metric="euclidean")
    np.fill_diagonal(pairwise, np.inf)
    return {
        "Generated_To_Reference_Distance": float(generated_minima.mean()),
        "Reference_To_Generated_Distance": float(reference_minima.mean()),
        "Bidirectional_Chamfer_Distance": float(
            generated_minima.mean() + reference_minima.mean()
        ),
        "Mean_Nearest_Neighbor_Distance": float(pairwise.min(axis=1).mean()),
        "Scaled_Coordinates_Below_Zero": int((generated < 0.0).sum()),
        "Scaled_Coordinates_Above_One": int((generated > 1.0).sum()),
    }


