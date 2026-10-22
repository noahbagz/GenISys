#!/usr/bin/env python3
"""Freeze the user-supplied 100k repair population before array dispatch."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .cluster_configuration import (
    candidate_batch_root,
    candidate_indices_path,
    cluster_repair_configuration,
)
from .dataset import (
    SourceDataset,
    freeze_candidate_index_alternatives,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)
from .parameters import ParameterRegistry
from .pipeline import (
    _implementation_paths,
    _json_safe,
    _runtime_provenance,
)
from .registry import ConstraintRepairRegistry


def parse_arguments() -> argparse.Namespace:
    """Read only the array partition count from the submission workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Freeze three repair alternatives for each design listed in "
            "compiled/Candidate_Indices.csv."
        )
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=250,
        help="Number of SLURM array tasks that will consume the selection.",
    )
    parser.add_argument(
        "--target-project-root",
        default=None,
        help=(
            "Optional absolute GenISys root on the execution cluster. When "
            "provided, prepare the data locally but write portable manifest "
            "paths and the configuration fingerprint for that cluster root."
        ),
    )
    return parser.parse_args()


def _relocate_manifest_paths(
    value,
    *,
    source_root: str,
    target_root: str,
):
    """Recursively replace only paths rooted in the local project directory."""

    if isinstance(value, dict):
        return {
            key: _relocate_manifest_paths(
                item,
                source_root=source_root,
                target_root=target_root,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _relocate_manifest_paths(
                item,
                source_root=source_root,
                target_root=target_root,
            )
            for item in value
        ]

    if isinstance(value, str) and value.startswith(source_root):
        return target_root + value[len(source_root):]

    return value


def main() -> int:
    """Validate the compiled source and commit one immutable selection."""

    arguments = parse_arguments()

    if int(arguments.task_count) < 1:
        raise ValueError("--task-count must be at least one.")

    configuration = cluster_repair_configuration()
    runtime_configuration = configuration
    parameters = ParameterRegistry(configuration)
    dataset = SourceDataset(configuration.source_root, parameters)
    batch_root = candidate_batch_root(configuration)
    candidate_path = candidate_indices_path(configuration)
    alternatives = int(
        configuration.alternatives_per_below_median_source
    )

    selection = freeze_candidate_index_alternatives(
        dataset,
        candidate_path,
        batch_root,
        seed=configuration.below_median_batch_seed,
        alternatives_per_source=alternatives,
        batch_kind="candidate_indices",
    )

    # The expensive source-data scan may be performed on a workstation before
    # upload.  Relocate only manifest paths while retaining the exact file
    # hashes and frozen repair seeds produced from the local source dataset.
    if arguments.target_project_root is not None:
        local_project_root = Path(__file__).resolve().parents[1]
        target_project_root = Path(arguments.target_project_root)

        if not target_project_root.is_absolute():
            raise ValueError("--target-project-root must be an absolute path.")

        runtime_configuration = replace(
            configuration,
            source_root=str(
                target_project_root
                / "Ship_Systems_Designs_v1p1"
                / "Dataset_100k"
            ),
            experiment_root=str(
                target_project_root
                / "Ship_Systems_Designs_v1p1"
                / "repair_experiments"
                / "Dataset_100k_Repaired"
            ),
        )

        selection_manifest_path = batch_root / "selection_manifest.json"
        selection_manifest = json.loads(
            selection_manifest_path.read_text(encoding="utf-8")
        )
        relocated_manifest = _relocate_manifest_paths(
            selection_manifest,
            source_root=str(local_project_root),
            target_root=str(target_project_root),
        )
        write_json_atomic(relocated_manifest, selection_manifest_path)

    # Save the same repair registries used by the laptop experiment so the
    # production batch remains traceable to exact parameter ownership and
    # constraint-to-repair mappings.
    write_csv_atomic(
        parameters.role_table(),
        batch_root / "parameter_repair_registry.csv",
    )
    write_csv_atomic(
        ConstraintRepairRegistry().export_runtime_registry(
            dataset.constraint_catalog
        ),
        batch_root / "constraint_repair_registry.csv",
    )

    project_root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": 1,
        "method": "one_pass_stochastic_equation_repair",
        "execution": "slurm_array",
        "batch_kind": "candidate_indices",
        "array_task_count": int(arguments.task_count),
        "source_design_count": int(selection["Index"].nunique()),
        "repair_design_count": int(len(selection)),
        "alternatives_per_source": alternatives,
        "configuration": runtime_configuration.to_dict(),
        "configuration_sha256": runtime_configuration.fingerprint(),
        "candidate_indices_path": str(
            Path(runtime_configuration.source_root)
            / "compiled"
            / "Candidate_Indices.csv"
        ),
        "candidate_indices_sha256": sha256_file(candidate_path),
        "runtime_provenance": _runtime_provenance(),
        "repair_contract": {
            "selection_authority": "compiled/Candidate_Indices.csv",
            "repair_passes_per_alternative": 1,
            "repaired_evaluations_per_alternative": 1,
            "post_evaluation_repair": False,
            "performance_metrics_used_for_repair": False,
            "save_mode": "reduced",
            "shared_output_storage": (
                "one wide repaired-design row per alternative in one "
                "gzip CSV shard per SLURM array task"
            ),
            "transient_evaluator_storage": "node-local scratch",
            "array_partition": (
                "contiguous balanced Selection_Order intervals"
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
    write_json_atomic(
        _json_safe(manifest),
        batch_root / "repair_run_manifest.json",
    )

    print(f"Candidate parents: {selection['Index'].nunique()}")
    print(f"Alternatives per parent: {alternatives}")
    print(f"Total repaired evaluations: {len(selection)}")
    print(f"SLURM array tasks: {int(arguments.task_count)}")
    print(f"Frozen selection: {batch_root / 'selected_source_designs.csv'}")
    print(
        "Execution project root: "
        f"{Path(runtime_configuration.source_root).parents[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
