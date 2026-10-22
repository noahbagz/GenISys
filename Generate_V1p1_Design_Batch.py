"""Generate one restartable V1.1 structure and system-design shard.

This script performs data generation only.  It does not evaluate ship-system
performance or constraints.  Every accepted structure writes one structural-
elements CSV and twenty system rows: four valid hull variants with five
legacy-style draft/Froude/range operating samples per hull.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import pandas as pd

from cad_query_tools.generate import generate_structure_files
from tools.V1p1Dataset import (
    DEFAULT_BASE_SEED,
    HULL_CANDIDATES_PER_STRUCTURE,
    HULLS_PER_STRUCTURE,
    HULL_STREAM_TAG,
    OPERATING_CONDITIONS_PER_HULL,
    OPERATING_STREAM_TAG,
    STRUCTURE_STREAM_TAG,
    cleaned_structural_bound_violations,
    expand_operating_conditions,
    generate_valid_hull_variants,
    rng_for,
    sample_structural_parameter_row,
    structural_parameter_catalog,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "Ship_Systems_Designs_v1p1" / "Dataset_100k"
)

# A 5,000-structure production population exposed a low-probability tail that
# the original 20-attempt laptop sample did not contain.  Keep one shared
# default for direct, laptop, and cluster generation so all three paths use
# the same deterministic retry sequence.
DEFAULT_MAXIMUM_STRUCTURE_ATTEMPTS = 100


def _write_csv_atomic(table: pd.DataFrame, output_path: Path) -> None:
    """Replace one completed CSV without exposing a partial shard."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    compression = (
        {"method": "gzip", "compresslevel": 6, "mtime": 0}
        if output_path.suffix == ".gz"
        else None
    )
    table.to_csv(temporary_path, index=False, compression=compression)
    temporary_path.replace(output_path)


def _csv_shard_suffix(save_mode: str) -> str:
    """Use compressed canonical shards only in reduced-storage mode."""

    if save_mode not in {"all", "reduced"}:
        raise ValueError("save_mode must be either 'all' or 'reduced'.")

    return ".csv.gz" if save_mode == "reduced" else ".csv"


def _write_json_atomic(payload: dict, output_path: Path) -> None:
    """Replace one complete provenance JSON without exposing a partial file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _file_provenance(path: str | Path) -> dict:
    """Return path, size, and SHA-256 for one generation input/source file."""

    path = Path(path).resolve()
    digest = hashlib.sha256()
    size_bytes = 0

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size_bytes += len(block)

    try:
        displayed_path = str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        displayed_path = str(path)

    return {
        "path": displayed_path,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _generation_invocation_payload(
    start_structure: int,
    end_structure: int,
    output_root: str | Path,
    base_seed: int,
    maximum_structure_attempts: int,
    hull_candidate_count: int,
    save_mode: str = "all",
    raise_on_failed_structure: bool = True,
) -> dict:
    """Describe the resolved generator configuration for one owned shard.

    The payload is deliberately data-only and deterministic.  It records the
    actual function inputs rather than trying to infer later which launcher or
    default produced a shard.
    """

    return {
        "schema_version": 1,
        "stage": "generation",
        "entry_point": _file_provenance(Path(__file__)),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
        },
        "configuration": {
            "start_structure": int(start_structure),
            "end_structure_exclusive": int(end_structure),
            "output_root": str(Path(output_root).resolve()),
            "base_seed": int(base_seed),
            "maximum_structure_attempts": int(maximum_structure_attempts),
            "raise_on_failed_structure": bool(raise_on_failed_structure),
            "hull_candidate_count": int(hull_candidate_count),
            "hulls_per_structure": int(HULLS_PER_STRUCTURE),
            "operating_conditions_per_hull": int(
                OPERATING_CONDITIONS_PER_HULL
            ),
            "cad_artifact_mode": "elements_only",
            "clean_parameters": False,
            "save_mode": save_mode,
        },
    }


def _flatten_timing(timing: dict) -> dict:
    """Flatten builder-stage timings into scalar CSV columns."""

    flat = {
        key: value
        for key, value in timing.items()
        if key != "builder_stage_seconds"
    }

    for stage_name, seconds in timing.get("builder_stage_seconds", {}).items():
        flat[f"builder_{stage_name}_seconds"] = seconds

    return flat


def generate_design_batch(
    start_structure: int,
    end_structure: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    base_seed: int = DEFAULT_BASE_SEED,
    maximum_structure_attempts: int = DEFAULT_MAXIMUM_STRUCTURE_ATTEMPTS,
    hull_candidate_count: int = HULL_CANDIDATES_PER_STRUCTURE,
    save_mode: str = "all",
    raise_on_failed_structure: bool = True,
) -> dict[str, Path]:
    """Generate the half-open structure range and its accepted system rows.

    The production cluster sets ``raise_on_failed_structure=False`` so one
    exhausted structure is recorded and skipped without suppressing the
    evaluation of independent accepted siblings in the same task. Standalone
    and laptop generation retain the fail-fast default because their explicit
    generation acceptance gate expects a complete requested population.
    """

    output_root = Path(output_root)
    shard_suffix = _csv_shard_suffix(save_mode)
    elements_directory = output_root / "structures" / "elements"
    parameter_directory = output_root / "structures" / "parameters"
    failure_directory = output_root / "structures" / "failures"
    system_directory = output_root / "systems" / "shards"

    for directory in (
        elements_directory,
        parameter_directory,
        failure_directory,
        system_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    structural_names = structural_parameter_catalog()["Name"].tolist()
    structure_rows = []
    system_rows = []
    status_rows = []
    failure_rows = []

    for structure_index in range(int(start_structure), int(end_structure)):
        accepted = False

        for attempt in range(int(maximum_structure_attempts)):
            stage = "sample_structure_parameters"

            try:
                structure_rng = rng_for(
                    base_seed,
                    structure_index,
                    STRUCTURE_STREAM_TAG,
                    attempt,
                )
                structural_parameters = sample_structural_parameter_row(
                    structure_rng
                )

                stage = "build_cadquery_structure"
                ship_id = f"Structure_{structure_index:06d}"
                generation = generate_structure_files(
                    structural_parameters,
                    output_directory=elements_directory,
                    ship_id=ship_id,
                    clean_parameters=False,
                    parameter_names=structural_names,
                    artifact_mode="elements_only",
                )

                # CadQuery interpretation intentionally snaps and aligns some
                # sampled member dimensions.  The returned corrected row is the
                # authoritative structure vector used by all twenty systems.
                corrected_parameters = generation["cleaned_parameters"].copy()

                # CadQuery cleanup deliberately snaps several dimensions to
                # supporting members.  The corrected vector, rather than the
                # original proposal, is the design used downstream; reject and
                # resample if that final geometry leaves a declared bound.
                structural_bound_errors = cleaned_structural_bound_violations(
                    corrected_parameters,
                )

                if structural_bound_errors:
                    first_error = structural_bound_errors[0]
                    raise ValueError(
                        "Corrected structural parameter is outside the "
                        "published V1.1 bounds: "
                        f"{first_error['Parameter']}={first_error['Value']} "
                        f"not in [{first_error['Lower_Bound']}, "
                        f"{first_error['Upper_Bound']}]."
                    )

                stage = "sample_valid_hulls"
                hull_rng = rng_for(
                    base_seed,
                    structure_index,
                    HULL_STREAM_TAG,
                    attempt,
                )
                valid_hulls = generate_valid_hull_variants(
                    corrected_parameters,
                    hull_rng,
                    candidate_count=hull_candidate_count,
                )

                if len(valid_hulls) != HULLS_PER_STRUCTURE:
                    raise RuntimeError(
                        "The legacy 500-candidate search did not produce four "
                        "valid hull variants for this structural sample."
                    )

                stage = "expand_operating_conditions"
                operating_rng = rng_for(
                    base_seed,
                    structure_index,
                    OPERATING_STREAM_TAG,
                    attempt,
                )
                generated_systems = expand_operating_conditions(
                    structure_index,
                    valid_hulls,
                    operating_rng,
                    structure_sample_attempt=attempt,
                )

                structure_row = {
                    "Structure_Index": structure_index,
                    "Structure_Sample_Attempt": attempt,
                    "Base_Seed": base_seed,
                    "Structural_Elements_File": str(
                        generation["structural_elements_csv"]
                    ),
                }
                structure_row.update(
                    dict(zip(structural_names, corrected_parameters, strict=True))
                )
                structure_rows.append(structure_row)
                system_rows.extend(generated_systems)

                status_rows.append(
                    {
                        "Structure_Index": structure_index,
                        "Status": "completed",
                        "Structure_Sample_Attempt": attempt,
                        "Valid_Hull_Count": len(valid_hulls),
                        "System_Row_Count": len(generated_systems),
                        **_flatten_timing(generation["timing"]),
                    }
                )
                accepted = True
                break

            except Exception as error:
                failure_rows.append(
                    {
                        "Structure_Index": structure_index,
                        "Structure_Sample_Attempt": attempt,
                        "Stage": stage,
                        "Error_Type": type(error).__name__,
                        "Error_Message": str(error),
                        "Traceback": traceback.format_exc(),
                    }
                )

        if not accepted:
            # A structure is part of the V1.1 population only when CadQuery
            # generation and the four-hull search both succeed.  Remove a
            # possible element file from the last rejected attempt so the
            # output directory cannot imply that the failed index was accepted.
            rejected_elements = (
                elements_directory
                / f"Structure_{structure_index:06d}_Structural_Elements.csv"
            )
            rejected_elements.unlink(missing_ok=True)

            status_rows.append(
                {
                    "Structure_Index": structure_index,
                    "Status": "failed",
                    "Structure_Sample_Attempt": maximum_structure_attempts - 1,
                    "Valid_Hull_Count": 0,
                    "System_Row_Count": 0,
                }
            )

    range_label = f"{int(start_structure):06d}_{int(end_structure) - 1:06d}"
    structure_path = parameter_directory / (
        f"Structure_Parameters_{range_label}{shard_suffix}"
    )
    status_path = parameter_directory / (
        f"Structure_Status_{range_label}{shard_suffix}"
    )
    failure_path = failure_directory / (
        f"Structure_Failures_{range_label}{shard_suffix}"
    )
    system_path = system_directory / (
        f"System_Designs_{range_label}{shard_suffix}"
    )

    _write_csv_atomic(pd.DataFrame(structure_rows), structure_path)
    _write_csv_atomic(pd.DataFrame(status_rows), status_path)
    _write_csv_atomic(
        pd.DataFrame(
            failure_rows,
            columns=[
                "Structure_Index",
                "Structure_Sample_Attempt",
                "Stage",
                "Error_Type",
                "Error_Message",
                "Traceback",
            ],
        ),
        failure_path,
    )
    _write_csv_atomic(pd.DataFrame(system_rows), system_path)

    # Save the actual resolved invocation next to the dataset manifests.  The
    # compiler later embeds this JSON, its hash, and its size in the combined
    # manifest.  This closes the ambiguity between a source default and the
    # explicit arguments used for one particular shard.
    invocation_path = (
        output_root
        / "manifests"
        / "invocations"
        / "generation"
        / f"Generation_{range_label}.json"
    )
    invocation_payload = _generation_invocation_payload(
        start_structure=start_structure,
        end_structure=end_structure,
        output_root=output_root,
        base_seed=base_seed,
        maximum_structure_attempts=maximum_structure_attempts,
        hull_candidate_count=hull_candidate_count,
        save_mode=save_mode,
        raise_on_failed_structure=raise_on_failed_structure,
    )
    invocation_payload["output_shards"] = {
        "structure_parameters": _file_provenance(structure_path),
        "structure_status": _file_provenance(status_path),
        "structure_failures": _file_provenance(failure_path),
        "system_designs": _file_provenance(system_path),
    }
    _write_json_atomic(invocation_payload, invocation_path)

    failed_structure_count = sum(
        row["Status"] == "failed" for row in status_rows
    )

    if failed_structure_count and raise_on_failed_structure:
        raise RuntimeError(
            "One or more Structure_Index values exhausted all sample attempts. "
            f"See {failure_path}."
        )

    if failed_structure_count:
        print(
            f"Skipped {failed_structure_count} exhausted structure(s); "
            "accepted sibling systems remain available for evaluation."
        )

    return {
        "structure_parameters": structure_path,
        "structure_status": status_path,
        "structure_failures": failure_path,
        "system_designs": system_path,
        "invocation_provenance": invocation_path,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one V1.1 structure range and its 4 x 5 system rows."
        )
    )
    parser.add_argument("--start-structure", required=True, type=int)
    parser.add_argument("--end-structure", required=True, type=int)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--maximum-structure-attempts",
        type=int,
        default=DEFAULT_MAXIMUM_STRUCTURE_ATTEMPTS,
    )
    parser.add_argument(
        "--hull-candidate-count",
        type=int,
        default=HULL_CANDIDATES_PER_STRUCTURE,
    )
    parser.add_argument(
        "--save-mode",
        choices=("all", "reduced"),
        default="all",
        help=(
            "Write ordinary CSV shards ('all') or gzip-compressed canonical "
            "shards ('reduced'). Structural-element and per-design "
            "constraint files remain ordinary CSVs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    outputs = generate_design_batch(
        start_structure=args.start_structure,
        end_structure=args.end_structure,
        output_root=args.output_root,
        base_seed=args.base_seed,
        maximum_structure_attempts=args.maximum_structure_attempts,
        hull_candidate_count=args.hull_candidate_count,
        save_mode=args.save_mode,
    )

    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
