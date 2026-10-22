"""Command-line and Python entry points for Rhino-independent generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .builder import build_structure
from .exporters import save_structure
from .parameters import ParametricStructureDefinition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHIP_SYSTEM_PARAMETER_SCHEMA = PROJECT_ROOT / "tools" / "ShipSystemDesignParameters.csv"


def select_structural_parameter_table(table: pd.DataFrame) -> pd.DataFrame:
    """Select the ordered 120 structure columns from full-system or direct input."""

    parameter_catalog = pd.read_csv(SHIP_SYSTEM_PARAMETER_SCHEMA)
    structural_names = parameter_catalog.loc[
        parameter_catalog["Class"] == "Structure",
        "Name",
    ].tolist()

    if all(name in table.columns for name in structural_names):
        return table.loc[:, structural_names].copy()

    if table.shape[1] == len(structural_names):
        return table.copy()

    raise ValueError(
        "Input must contain the 120 ordered structural columns or the full "
        "ship-system table containing their names"
    )


def generate_structure_files(
    parameters: np.ndarray,
    output_directory: str | Path,
    ship_id: str,
    file_format: str = "igs",
    clean_parameters: bool = False,
    parameter_names: list[str] | None = None,
    artifact_mode: str = "full",
) -> dict[str, Any]:
    """Build one design and write the requested persistent artifacts.

    ``artifact_mode='full'`` preserves the established validation and
    visualization behavior: structural CAD, mesh CAD, the structural-element
    CSV, a geometry manifest, the corrected parameter row, and timing JSON are
    all written.

    ``artifact_mode='elements_only'`` is the V1.1 production-data path.  The
    complete CadQuery structure is still constructed in memory so every area,
    volume, centroid, and section property is calculated by the same builder.
    Only the downstream structural-element CSV is serialized.  Batch-level
    parameter, timing, and failure tables are written by the caller, avoiding
    three unnecessary sidecar files and two large CAD files per structure.
    """

    if artifact_mode not in {"full", "elements_only"}:
        raise ValueError(
            "artifact_mode must be either 'full' or 'elements_only'"
        )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    parameter_row = np.asarray(parameters, dtype=float).copy()

    parameter_start = time.perf_counter()

    if clean_parameters:
        ParametricStructureDefinition.clean_struct_params(
            parameter_row.reshape(1, -1)
        )

    definition = ParametricStructureDefinition(parameter_row)
    parameter_seconds = time.perf_counter() - parameter_start

    build_start = time.perf_counter()
    structure = build_structure(definition, ship_id=ship_id)
    build_seconds = time.perf_counter() - build_start

    structural_files = []
    mesh_files = []
    export_seconds = 0.0

    if artifact_mode == "full":
        export_start = time.perf_counter()
        structural_files = save_structure(
            structure,
            output / ship_id,
            file_format=file_format,
        )

        # Preserve the separate mesh artifact used by the current gmsh
        # workflow.  The V1.1 elements-only dataset path deliberately skips it.
        mesh_files = save_structure(
            structure,
            output / f"{ship_id}_MeshElements",
            file_format=file_format,
            mesh=True,
        )
        export_seconds = time.perf_counter() - export_start

    structural_csv = structure.write_structural_elements_csv(
        output / f"{ship_id}_Structural_Elements.csv"
    )
    manifest = None
    cleaned_parameters = None

    if artifact_mode == "full":
        manifest = structure.write_manifest(
            output / f"{ship_id}_Geometry_Manifest.csv",
            exported_files=structural_files + mesh_files,
        )

        if parameter_names is None:
            parameter_names = (
                ParametricStructureDefinition.parameter_schema["name"].tolist()
            )
        cleaned_parameters = output / f"{ship_id}_Parameters_Updated.csv"
        pd.DataFrame([definition.params], columns=parameter_names).to_csv(
            cleaned_parameters,
            index=False,
        )

    timing = {
        "ship_id": ship_id,
        "parameter_interpretation_seconds": parameter_seconds,
        "cadquery_build_seconds": build_seconds,
        "cadquery_export_seconds": export_seconds,
        "cadquery_total_seconds": parameter_seconds + build_seconds + export_seconds,
        "artifact_mode": artifact_mode,
        "builder_stage_seconds": structure.timings,
        # Timing is recorded for comparison only and is not an acceptance test.
        "timing_is_acceptance_criterion": False,
    }
    timing_path = None

    if artifact_mode == "full":
        timing_path = output / f"{ship_id}_CadQuery_Timing.json"
        timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")

    return {
        "structure": structure,
        "structural_cad": structural_files,
        "mesh_cad": mesh_files,
        "structural_elements_csv": structural_csv,
        "geometry_manifest": manifest,
        "updated_parameters_csv": cleaned_parameters,
        "cleaned_parameters": np.asarray(definition.params, dtype=float).copy(),
        "timing_json": timing_path,
        "timing": timing,
    }


def generate_csv_row(
    input_csv: str | Path,
    row_index: int,
    output_directory: str | Path,
    ship_id: str | None = None,
    file_format: str = "igs",
    clean_parameters: bool = False,
    artifact_mode: str = "full",
) -> dict[str, Any]:
    """Generate one design from a row of a 120-column parameter CSV."""

    table = select_structural_parameter_table(pd.read_csv(input_csv))
    parameters = table.iloc[int(row_index)].to_numpy(dtype=float, copy=True)
    design_id = ship_id or f"CadQuery_Design_{int(row_index)}"

    return generate_structure_files(
        parameters,
        output_directory=output_directory,
        ship_id=design_id,
        file_format=file_format,
        clean_parameters=clean_parameters,
        parameter_names=table.columns.tolist(),
        artifact_mode=artifact_mode,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one ship structure with CadQuery and no Rhino license."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--row-index", required=True, type=int)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--ship-id")
    parser.add_argument(
        "--file-format",
        choices=("igs", "iges", "step", "stp", "both"),
        default="igs",
    )
    parser.add_argument(
        "--clean-parameters",
        action="store_true",
        help=(
            "Apply the legacy random-sample cleanup first. Leave this off for "
            "persisted Rhino reference rows, where cleanup is not idempotent."
        ),
    )
    parser.add_argument(
        "--elements-only",
        action="store_true",
        help=(
            "Build the structure and write only its structural-elements CSV. "
            "Do not serialize structural CAD, mesh CAD, or per-design sidecars."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    result = generate_csv_row(
        input_csv=args.input_csv,
        row_index=args.row_index,
        output_directory=args.output_directory,
        ship_id=args.ship_id,
        file_format=args.file_format,
        clean_parameters=args.clean_parameters,
        artifact_mode=("elements_only" if args.elements_only else "full"),
    )

    summary = {
        "structural_cad": [str(path) for path in result["structural_cad"]],
        "mesh_cad": [str(path) for path in result["mesh_cad"]],
        "structural_elements_csv": str(result["structural_elements_csv"]),
        "geometry_manifest": (
            str(result["geometry_manifest"])
            if result["geometry_manifest"] is not None
            else None
        ),
        "updated_parameters_csv": (
            str(result["updated_parameters_csv"])
            if result["updated_parameters_csv"] is not None
            else None
        ),
        "timing_json": (
            str(result["timing_json"])
            if result["timing_json"] is not None
            else None
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
