"""Build repeatable CAD scenes from unified or imported evaluated designs.

The scene is a display artifact.  It does not change the half-ship structure,
the structural-elements table, or any system mass and center-of-gravity input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cadquery as cq
import numpy as np
import pandas as pd

from .builder import build_structure
from .generate import select_structural_parameter_table
from .parameters import ParametricStructureDefinition
from .system_scene_render import render_and_export_scene
from .system_visualization_data import (
    load_visualization_design, load_evaluated_visualization_design,
)
from .system_visualization_geometry import build_visualization_parts

from tools.HullParameterization import Hull_Parameterization


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMETER_CATALOG = PROJECT_ROOT / "tools" / "ShipSystemDesignParameters.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cad_visualizations"
EXAMPLE_INDICES = (99, 170, 178)


def _hull_parameters(system_row: pd.Series) -> np.ndarray:
    """Read the existing ordered hull vector from the project catalog."""

    parameter_catalog = pd.read_csv(PARAMETER_CATALOG, encoding="utf-8-sig")
    names = parameter_catalog.loc[
        parameter_catalog["Class"] == "Hull", "Name"
    ].tolist()

    return system_row.loc[names].to_numpy(dtype=float)


def _structural_parameters(system_row: pd.Series) -> np.ndarray:
    """Use the evaluated design's structure-coupled coordinates."""

    table = select_structural_parameter_table(
        pd.DataFrame([system_row.to_dict()])
    )

    return table.iloc[0].to_numpy(dtype=float).copy()


def _write_step(parts: list[dict], destination: Path) -> None:
    """Export separately named, metre-coordinate CadQuery display parts."""

    assembly = cq.Assembly(name=destination.stem)

    for part in parts:
        assembly.add(
            part["shape"],
            name=part["name"],
            color=cq.Color(*part["color"]),
        )

    assembly.export(
        str(destination),
        exportType="STEP",
        unit="M",
        outputUnit="M",
    )


def create_system_visualization(
    *,
    population: str,
    index: int,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    styles: tuple[str, ...] = ("shaded", "wireframe"),
    views: tuple[str, ...] = ("isometric", "top", "side", "longitudinal_cutaway", "transverse_cutaway"),
    hull_waterlines: int = 51,
    hull_points_per_waterline: int = 151,
    evaluation_root: Path | None = None,
    publication: bool = False,
) -> dict[str, Path]:
    """Regenerate a saved design's display CAD, mesh, manifest, and images.

    Set ``evaluation_root`` for an imported generated-design evaluation;
    otherwise load the unified training dataset. Publication mode changes
    image presentation only, preserving the established geometry contract.
    """

    # Keep each design's generated artifacts together and identify the source
    # population so an older review folder with the same Index cannot be used.
    output_directory = Path(output_root) / f"{population}_{index:06d}"
    output_directory.mkdir(parents=True, exist_ok=True)

    design = (
        load_visualization_design(PROJECT_ROOT, population, index)
        if evaluation_root is None else
        load_evaluated_visualization_design(evaluation_root, population, index)
    )

    # Build the actual sampled full hull. The open deck exposes the empty
    # cargo holds and structure in section, matching the hull STL convention.
    hull = Hull_Parameterization(_hull_parameters(design.system_row))
    hull_stl = output_directory / "hull.stl"
    hull.gen_stl(
        NUM_WL=hull_waterlines,
        PointsPerWL=hull_points_per_waterline,
        bit_AddTransom=1,
        bit_AddDeckLid=0,
        bit_RefineBowAndStern=1,
        namepath=str(hull_stl.with_suffix("")),
    )

    # Rebuild the existing neutral-surface structure from the matching
    # system row.  Its original half-model remains untouched by scene placement.
    definition = ParametricStructureDefinition(
        _structural_parameters(design.system_row)
    )
    structure = build_structure(
        definition,
        ship_id=f"Unified_{population}_{index}",
    )

    parts, component_rows, placement = build_visualization_parts(
        hull,
        structure,
        design.bulkheads,
        design.hatches,
        design.compartments,
    )

    # STEP retains the CAD surfaces and named display components.  The full
    # hydro hull remains an STL; the GLB combines its mesh with the CAD parts.
    cad_step = output_directory / "system_assembly.step"
    _write_step(parts, cad_step)

    rendered = render_and_export_scene(
        parts,
        hull_stl,
        output_directory,
        waterline_z=float(design.system_row["Draft_frac"]) * float(hull.Dd),
        styles=styles,
        views=views,
        transverse_x=0.5 * sum(placement["selected_cargo_boundaries_x_m"][1:3]),
        publication=publication,
    )

    component_manifest = output_directory / "components.csv"
    pd.DataFrame(component_rows).to_csv(component_manifest, index=False)

    # Record every visualization approximation, source table, and transform
    # without asserting that display solids carry engineering mass properties.
    manifest = {
        "Population": population,
        "Index": index,
        "Structure_Index": int(design.training_row["Structure_Index"]),
        "Evaluation_Status": str(design.training_row["Evaluation_Status"]),
        "Global_V1p1_Feasible": bool(
            design.training_row["Global_V1p1_Feasible"]
        ),
        "Coordinate_Units": "metres",
        "Hull_STL": str(hull_stl),
        "CAD_STEP": str(cad_step),
        "Component_Manifest": str(component_manifest),
        "Scene_Outputs": {key: str(value) for key, value in rendered.items()},
        "Provenance": {
            key: str(value) for key, value in design.provenance_paths.items()
        },
        "Structural_Elements_Source": str(design.structural_elements_path),
        "Placement": placement,
        "Hull_Mesh_Resolution": {
            "Waterlines": hull_waterlines,
            "Points_Per_Waterline": hull_points_per_waterline,
            "Deck_Lid": False,
        },
        "Waterline_Z_m": float(design.system_row["Draft_frac"]) * float(hull.Dd),
        "Cutaway_Views": {
            "Longitudinal": "starboard shell window, 8% to 92% of hull length and above 13% of hull depth",
            "Transverse_X_m": 0.5 * sum(placement["selected_cargo_boundaries_x_m"][1:3]),
            "Cutaway_GLBS": ["longitudinal_cutaway.glb", "transverse_cutaway.glb"],
        },
        "Superstructure_Display_Assumptions": {
            "Breadth_Source": "Hatch_Covers.Cover_Width_m",
            "Longitudinal_Span": "engine_room_forward_bulkhead to midpoint between engine_room_aft_bulkhead and aft_peak_bulkhead",
            "Height_Rule": "2 * engine room bulkhead span",
            "Mass_Model_Activated": False,
        },
        "Display_Only": True,
        "Publication_Rendering": publication,
    }
    manifest_path = output_directory / "scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "hull_stl": hull_stl,
        "cad_step": cad_step,
        "component_manifest": component_manifest,
        "scene_manifest": manifest_path,
        **rendered,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", default="random")
    parser.add_argument("--indices", type=int, nargs="+", default=EXAMPLE_INDICES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--style", choices=("shaded", "wireframe", "both"), default="both"
    )
    parser.add_argument("--hull-waterlines", type=int, default=51)
    parser.add_argument("--hull-points-per-waterline", type=int, default=151)
    parser.add_argument(
        "--publication", action="store_true",
        help="Write tightly framed, white-background CAD PNGs at 600 dpi.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    styles = (
        ("shaded", "wireframe")
        if args.style == "both"
        else (args.style,)
    )

    for index in args.indices:
        outputs = create_system_visualization(
            population=args.population,
            index=index,
            output_root=args.output_root,
            styles=styles,
            hull_waterlines=args.hull_waterlines,
            hull_points_per_waterline=args.hull_points_per_waterline,
            publication=args.publication,
        )
        print(f"{args.population}:{index} -> {outputs['scene_manifest']}")


if __name__ == "__main__":
    main()
