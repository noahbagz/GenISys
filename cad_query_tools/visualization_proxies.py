"""Optional CadQuery solids for system-level design visualization.

The production dataset intentionally stores structural-element CSV files, not
CAD for every design.  These helpers let a selected design rebuild lightweight
visual proxies from already calculated component tables.  Solids created here
are display envelopes only: their OpenCascade volume and material density must
never be used to replace the assigned mass and CG metadata.

Hatch-cover placement follows approved assumptions HDF-ASM-021 through
HDF-ASM-025.  A deckhouse is created only from an explicit parameterized mass
row; this module supplies no default dimensions or location.
"""

from __future__ import annotations

import cadquery as cq
import pandas as pd

from .geometry import rounded_rectangle_face_xy
from .model import ShipStructure


def _rounded_cover_shape(row: pd.Series) -> cq.Shape:
    """Create one closed rounded-prism cover from its approved dimensions."""

    face = rounded_rectangle_face_xy(
        float(row["LCG"]),
        float(row["TCG"]),
        float(row["Cover_Length_m"]),
        float(row["Cover_Width_m"]),
        z=float(row["Cover_Bottom_Z_m"]),
        radius=float(row["Cover_Corner_Radius_m"]),
    )
    return cq.Solid.extrudeLinear(
        face.outerWire(),
        face.innerWires(),
        (0.0, 0.0, float(row["Cover_Envelope_Depth_m"])),
    )


def build_hatch_cover_visualization_proxies(
    hatch_cover_components: pd.DataFrame,
) -> tuple[list[cq.Shape], pd.DataFrame]:
    """Return cover solids and a mass-aware visualization manifest."""

    shapes = []
    rows = []

    for proxy_index, (_, cover) in enumerate(
        hatch_cover_components.reset_index(drop=True).iterrows()
    ):
        shapes.append(_rounded_cover_shape(cover))
        rows.append(
            {
                "Proxy_Index": proxy_index,
                "Component_ID": cover["Component_ID"],
                "Proxy_Type": "hatch_cover_closed_envelope",
                "Assigned_Mass_kg": float(cover["Total_Weight_kg"]),
                "Assigned_LCG_m": float(cover["LCG"]),
                "Assigned_TCG_m": float(cover["TCG"]),
                "Assigned_VCG_m": float(cover["VCG"]),
                "Geometry_Is_Mass_Solid": False,
                "Status": "approved_visual_proxy_not_certification_geometry",
                "Assumption_IDs": cover["Assumption_IDs"],
            }
        )

    return shapes, pd.DataFrame(rows)


def build_deckhouse_visualization_proxy(
    deckhouse_component: pd.Series | dict,
) -> tuple[cq.Shape, dict]:
    """Create one box envelope from an explicitly supplied deckhouse row."""

    row = pd.Series(deckhouse_component)
    shape = (
        cq.Workplane("XY")
        .rect(float(row["Length_m"]), float(row["Breadth_m"]))
        .extrude(float(row["Height_m"]))
        .translate(
            (
                float(row["Base_X_m"]) + 0.5*float(row["Length_m"]),
                float(row["Centerline_Y_m"]),
                float(row["Base_Z_m"]),
            )
        )
        .val()
    )
    manifest = {
        "Component_ID": "simple_deckhouse_000",
        "Proxy_Type": "simple_deckhouse_closed_envelope",
        "Assigned_Mass_kg": float(row["Total_Weight_kg"]),
        "Assigned_LCG_m": float(row["LCG"]),
        "Assigned_TCG_m": float(row["TCG"]),
        "Assigned_VCG_m": float(row["VCG"]),
        "Geometry_Is_Mass_Solid": False,
        "Status": "explicit_visual_proxy_not_spatially_validated",
        "Assumption_IDs": "explicit project scenario required",
    }

    return shape, manifest


def build_system_visualization_proxy(
    *,
    hatch_cover_components: pd.DataFrame,
    structure: ShipStructure | cq.Shape | None = None,
    deckhouse_components: pd.DataFrame | None = None,
) -> tuple[cq.Shape, pd.DataFrame]:
    """Combine existing structure and optional above-deck proxy geometry.

    ``structure`` may be the regenerated CadQuery structural model or its
    existing compound.  This helper does not create a new hydro hull or infer
    how a deckhouse should fit around hatches and machinery spaces.
    """

    if isinstance(structure, ShipStructure):
        shapes = [structure.shape()]
    elif isinstance(structure, cq.Shape):
        shapes = [structure]
    elif structure is None:
        shapes = []
    else:
        raise TypeError("structure must be ShipStructure, cq.Shape, or None")

    cover_shapes, cover_manifest = (
        build_hatch_cover_visualization_proxies(hatch_cover_components)
    )
    shapes.extend(cover_shapes)
    manifest_tables = [cover_manifest]

    if deckhouse_components is not None and not deckhouse_components.empty:
        deckhouse_rows = []

        for proxy_index, (_, deckhouse) in enumerate(
            deckhouse_components.reset_index(drop=True).iterrows(),
            start=len(cover_manifest),
        ):
            shape, row = build_deckhouse_visualization_proxy(deckhouse)
            shapes.append(shape)
            row["Proxy_Index"] = proxy_index
            deckhouse_rows.append(row)

        manifest_tables.append(pd.DataFrame(deckhouse_rows))

    if not shapes:
        raise ValueError("No structural or proxy geometry was supplied.")

    manifest = pd.concat(manifest_tables, ignore_index=True)

    return cq.Compound.makeCompound(shapes), manifest
