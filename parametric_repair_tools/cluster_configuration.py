"""Shared production-cluster configuration for candidate-index repair."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from .configuration import PROJECT_ROOT, RepairConfiguration


DEFAULT_CLUSTER_SOURCE_ROOT = (
    PROJECT_ROOT / "Ship_Systems_Designs_v1p1" / "Dataset_100k"
)
DEFAULT_CLUSTER_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "Ship_Systems_Designs_v1p1"
    / "repair_experiments"
    / "Dataset_100k_Repaired"
)


def cluster_repair_configuration() -> RepairConfiguration:
    """Resolve the production paths without changing repair equations."""

    source_root = Path(
        os.environ.get(
            "GENISYS_V1P1_REPAIR_SOURCE_ROOT",
            str(DEFAULT_CLUSTER_SOURCE_ROOT),
        )
    )
    experiment_root = Path(
        os.environ.get(
            "GENISYS_V1P1_REPAIR_OUTPUT_ROOT",
            str(DEFAULT_CLUSTER_EXPERIMENT_ROOT),
        )
    )
    alternatives = int(
        os.environ.get("GENISYS_V1P1_REPAIR_ALTERNATIVES", "3")
    )
    batch_seed = int(
        os.environ.get("GENISYS_V1P1_REPAIR_SEED", "20260915")
    )

    if alternatives < 1:
        raise ValueError(
            "GENISYS_V1P1_REPAIR_ALTERNATIVES must be at least one."
        )

    return replace(
        RepairConfiguration(),
        source_root=str(source_root),
        experiment_root=str(experiment_root),
        below_median_batch_seed=batch_seed,
        alternatives_per_below_median_source=alternatives,
        save_mode="reduced",
    )


def candidate_indices_path(configuration: RepairConfiguration) -> Path:
    """Return the user-supplied production candidate list."""

    return Path(configuration.source_root) / "compiled" / "Candidate_Indices.csv"


def candidate_batch_root(configuration: RepairConfiguration) -> Path:
    """Return the shared root for selection and array-task evidence."""

    return Path(configuration.experiment_root) / "candidate_indices_batch"


__all__ = [
    "DEFAULT_CLUSTER_EXPERIMENT_ROOT",
    "DEFAULT_CLUSTER_SOURCE_ROOT",
    "candidate_batch_root",
    "candidate_indices_path",
    "cluster_repair_configuration",
]
