"""Display geometry for one unified GenISys system design.

This module places a copy of the existing three-hold structural half-model in
the full hull. Fuel-tank display shapes are sampled illustrations rather than
the watertight boundaries or mass solids used by engineering calculations.
Cargo holds are left empty so the cutaway exposes their structure.
"""

from __future__ import annotations

import cadquery as cq
import numpy as np
import pandas as pd

from tools.HullParameterization import Hull_Parameterization

from .model import ShipStructure
from .visualization_proxies import build_hatch_cover_visualization_proxies


Color = tuple[float, float, float]


def _bounds(shape: cq.Shape) -> dict[str, float]:
    """Record bounds in metres for the component manifest."""

    box = shape.BoundingBox()

    return {
        "xmin": float(box.xmin),
        "xmax": float(box.xmax),
        "ymin": float(box.ymin),
        "ymax": float(box.ymax),
        "zmin": float(box.zmin),
        "zmax": float(box.zmax),
    }


def _add_part(
    parts: list[dict],
    manifest: list[dict],
    *,
    name: str,
    shape: cq.Shape,
    color: Color,
    opacity: float,
    kind: str,
    source_table: str,
    source_identifier: str,
    source_index: int | None,
    display_approximation: bool,
    notes: str,
) -> None:
    """Keep rendered parts and manifest entries in matching order."""

    parts.append(
        {
            "name": name,
            "shape": shape,
            "color": color,
            "opacity": opacity,
        }
    )
    manifest.append(
        {
            "name": name,
            "kind": kind,
            "source_table": source_table,
            "source_identifier": source_identifier,
            "source_index": source_index,
            "bounds_m": _bounds(shape),
            "geometry_is_mass_solid": False,
            "display_approximation": display_approximation,
            "notes": notes,
        }
    )


def _half_breadth(hull: Hull_Parameterization, x: float, z: float) -> float:
    """Evaluate the non-bulb hull half-breadth at one station and height.

    The hull STL may include bow/stern bulbs.  Subdivision planes represent
    only the main hull envelope, which is the relevant interior approximation.
    """

    mid_breadth = float(hull.halfBeam_MidBody(z))

    if hull.Lb <= x <= hull.Lb + hull.Lm:
        return max(0.0, mid_breadth)

    if x < hull.Lb:
        bow_start = float(hull.bow_profile(z))
        full_section = float(hull.delta_bow(z))

        if x <= bow_start:
            return 0.0

        if x >= full_section:
            return max(0.0, mid_breadth)

        profile = hull.solve_waterline_bow(z)
        return max(0.0, float(hull.halfBeam_Bow([x], profile)[0]))

    full_section = float(hull.delta_stern(z))
    stern_end = float(hull.stern_profile(z))

    if x <= full_section:
        return max(0.0, mid_breadth)

    if x >= stern_end:
        return 0.0

    profile = hull.solve_waterline_stern(z)
    return max(0.0, float(hull.halfBeam_Stern([x], profile)[0]))


def _section_wire(
    hull: Hull_Parameterization,
    x: float,
    top_z: float,
) -> cq.Wire:
    """Sample one full-width interior section in the YZ plane."""

    # Find the first height where the main hull has usable interior breadth.
    # The station-dependent floor follows bow/stern rise approximately.
    search_z = np.linspace(0.0, top_z, 81)
    search_y = [_half_breadth(hull, x, float(z)) for z in search_z]
    first_open = next(
        (i for i, y in enumerate(search_y) if y > 1.0e-5),
        None,
    )

    if first_open is None:
        raise ValueError(f"No main-hull interior section at x={x:.6f} m")

    bottom_z = float(search_z[max(0, first_open - 1)])

    # Keep a common number of vertices at each station for a ruled loft.
    levels = np.linspace(bottom_z, top_z, 17)
    starboard = [
        (x, _half_breadth(hull, x, float(z)), float(z))
        for z in levels
    ]
    port = [(px, -py, pz) for px, py, pz in reversed(starboard)]

    # A rising keel has one centerline vertex rather than a zero-length edge.
    if starboard[0][1] <= 1.0e-8:
        port.pop()

    return cq.Wire.makePolygon(starboard + port, close=True)


def _section_face(
    hull: Hull_Parameterization,
    x: float,
) -> tuple[cq.Face, bool]:
    """Make a sampled hull plane, or a marker if the station lies beyond it."""

    sample_z = np.linspace(0.0, float(hull.Dd), 81)
    has_interior = any(
        _half_breadth(hull, x, float(z)) > 1.0e-5 for z in sample_z
    )

    if has_interior:
        return cq.Face.makeFromWires(_section_wire(hull, x, float(hull.Dd))), False

    # One historical detail row can place the aft-peak bulkhead beyond the
    # main-hull LOA after hydrostatic-grid snapping.  Keep its recorded X
    # visible with a narrow centerline marker and expose that mismatch.
    half_marker_width = 0.01 * float(hull.Bd)
    marker = cq.Wire.makePolygon(
        [
            (x, -half_marker_width, 0.0),
            (x, half_marker_width, 0.0),
            (x, half_marker_width, float(hull.Dd)),
            (x, -half_marker_width, float(hull.Dd)),
        ],
        close=True,
    )

    return cq.Face.makeFromWires(marker), True


def _compartment_shape(
    hull: Hull_Parameterization,
    start_x: float,
    end_x: float,
    top_z: float,
) -> cq.Solid:
    """Loft five sampled main-hull sections into an illustrative envelope."""

    stations = np.linspace(start_x, end_x, 5)
    wires = [_section_wire(hull, float(x), top_z) for x in stations]

    return cq.Solid.makeLoft(wires, ruled=True)


def _selected_cargo_holds(
    cargo: pd.DataFrame,
    midbody_start: float,
    midbody_end: float,
) -> pd.DataFrame:
    """Choose the central eligible run of three consecutive cargo holds."""

    ordered = cargo.sort_values("Compartment_Number")
    eligible = ordered.loc[
        (ordered["X_Start_m"] >= midbody_start - 1.0e-6)
        & (ordered["X_End_m"] <= midbody_end + 1.0e-6)
    ]
    rows = list(eligible.iterrows())
    choices = []
    midbody_center = 0.5 * (midbody_start + midbody_end)

    for start in range(len(rows) - 2):
        trio = rows[start : start + 3]
        numbers = [int(row["Compartment_Number"]) for _, row in trio]

        if numbers[1] != numbers[0] + 1 or numbers[2] != numbers[1] + 1:
            continue

        outer_start = float(trio[0][1]["X_Start_m"])
        outer_end = float(trio[-1][1]["X_End_m"])
        center = 0.5 * (outer_start + outer_end)
        choices.append((abs(center - midbody_center), numbers[0], trio))

    if not choices:
        raise ValueError("No three consecutive cargo holds fit inside the parallel midbody")

    _, _, selected_rows = min(choices, key=lambda item: (item[0], item[1]))

    return pd.DataFrame([row for _, row in selected_rows])


def build_visualization_parts(
    hull: Hull_Parameterization,
    structure: ShipStructure,
    bulkheads: pd.DataFrame,
    hatches: pd.DataFrame,
    compartments: pd.DataFrame,
) -> tuple[list[dict], list[dict], dict]:
    """Build named display geometry from one unified-dataset design.

    ``structure`` is never modified.  Its positive-y half-model is translated
    without scaling and mirrored only in the returned display parts.  The
    manifest marks every returned shape as separate from engineering mass/CG.
    """

    parts: list[dict] = []
    manifest: list[dict] = []
    midbody_start = float(hull.Lb)
    midbody_end = float(hull.Lb + hull.Lm)
    depth = float(hull.Dd)

    cargo = compartments.loc[compartments["Compartment_Type"] == "cargo_hold"]
    selected = _selected_cargo_holds(cargo, midbody_start, midbody_end)
    selected_start = float(selected.iloc[0]["X_Start_m"])
    selected_end = float(selected.iloc[-1]["X_End_m"])
    selected_center = 0.5 * (selected_start + selected_end)

    # Locate the existing four hold-bounding structural planes rather than
    # assuming the builder's full x span equals its hold-bounded span.
    structure_stations = sorted(
        {
            float(element.metadata["x_loc"])
            for element in structure.elements
            if element.metadata.get("Class") == "Transverse_Bulkhead"
        }
    )

    if len(structure_stations) != 4:
        raise ValueError("The three-hold structure must have four bulkhead stations")

    structure_center = 0.5 * (structure_stations[0] + structure_stations[-1])
    translate_x = selected_center - structure_center
    starboard = structure.shape().translate((translate_x, 0.0, 0.0))
    structure_bounds = _bounds(starboard)

    # Separate the near-side plates and upper deck from the rest of the
    # structural half-model. They remain in the complete CAD assembly, while
    # cutaways omit them to expose hold framing and inner decks.
    deck_heights = [
        float(element.metadata["z_loc"])
        for element in structure.elements
        if element.metadata.get("Class") == "Deck"
    ]
    upper_deck_z = max(deck_heights, default=float("inf"))
    skin_shapes = []
    interior_shapes = []

    for element in structure.elements:
        element_class = element.metadata.get("Class")
        is_upper_deck = (
            element_class == "Deck"
            and abs(float(element.metadata["z_loc"]) - upper_deck_z) < 1.0e-6
        )
        target = (
            skin_shapes
            if element_class in {"Side_Shell", "Inner_Side_Shell"} or is_upper_deck
            else interior_shapes
        )
        target.append(element.shape)

    starboard_interior = cq.Compound.makeCompound(interior_shapes).translate(
        (translate_x, 0.0, 0.0)
    )
    starboard_skin = cq.Compound.makeCompound(skin_shapes).translate(
        (translate_x, 0.0, 0.0)
    ) if skin_shapes else None

    if (
        structure_bounds["xmin"] < midbody_start - 1.0e-6
        or structure_bounds["xmax"] > midbody_end + 1.0e-6
    ):
        raise ValueError("The translated three-hold module exceeds the parallel midbody")

    _add_part(
        parts,
        manifest,
        name="structure_starboard_interior",
        shape=starboard_interior,
        color=(0.72, 0.76, 0.82),
        opacity=0.9,
        kind="representative_three_hold_structure_interior",
        source_table="source_structure_parameters",
        source_identifier=structure.ship_id,
        source_index=None,
        display_approximation=True,
        notes="Unscaled positive-y three-hold structure without its separately exported near-side shell plates and upper deck.",
    )
    if starboard_skin is not None:
        _add_part(
            parts,
            manifest,
            name="structure_starboard_cutaway_skin",
            shape=starboard_skin,
            color=(0.67, 0.72, 0.78),
            opacity=0.9,
            kind="representative_three_hold_cutaway_skin",
            source_table="source_structure_parameters",
            source_identifier=structure.ship_id,
            source_index=None,
            display_approximation=True,
            notes="Unscaled side shell, inner side shell, and upper deck of the original positive-y structure; omitted from cutaway views only.",
        )
    _add_part(
        parts,
        manifest,
        name="structure_port_display_copy",
        shape=starboard_interior.mirror("XZ"),
        color=(0.60, 0.66, 0.74),
        opacity=0.68,
        kind="mirrored_representative_structure_interior",
        source_table="source_structure_parameters",
        source_identifier=structure.ship_id,
        source_index=None,
        display_approximation=True,
        notes="Mirrored interior and framing across y=0 for display; not added to the half-model mass/CG.",
    )
    if starboard_skin is not None:
        _add_part(
            parts,
            manifest,
            name="structure_port_cutaway_skin",
            shape=starboard_skin.mirror("XZ"),
            color=(0.60, 0.66, 0.74),
            opacity=0.68,
            kind="mirrored_representative_structure_skin",
            source_table="source_structure_parameters",
            source_identifier=structure.ship_id,
            source_index=None,
            display_approximation=True,
            notes="Mirrored port side shell, inner side shell, and upper deck; retained in the complete assembly but omitted from the cutaway.",
        )

    # Every full-ship subdivision row receives one main-hull plane or an
    # explicitly identified marker if its X is beyond the available hull.
    out_of_hull_bulkheads = []

    for _, row in bulkheads.sort_values("Bulkhead_Number").iterrows():
        number = int(row["Bulkhead_Number"])
        boundary_type = str(row["Boundary_Type"])
        x = float(row["X_m"])
        shape, is_station_marker = _section_face(hull, x)

        if is_station_marker:
            out_of_hull_bulkheads.append(
                {"bulkhead_number": number, "boundary_type": boundary_type, "x_m": x}
            )

        _add_part(
            parts,
            manifest,
            name=f"bulkhead_{number:02d}_{boundary_type}",
            shape=shape,
            color=(0.94, 0.70, 0.27),
            opacity=0.42,
            kind=(
                "out_of_hull_bulkhead_station_marker"
                if is_station_marker else "full_ship_bulkhead_plane"
            ),
            source_table="Bulkhead_Locations",
            source_identifier=boundary_type,
            source_index=number,
            display_approximation=True,
            notes=(
                "Recorded X has no main-hull section; narrow centerline station "
                "marker is shown, not a hull-bounded bulkhead."
                if is_station_marker else
                "Sampled non-bulb hull section; plane has no engineering thickness."
            ),
        )

    # Keep the evaluated cover records intact. Taper only a display copy where
    # the recorded rectangular cover extends beyond the local deck breadth.
    sorted_hatches = hatches.sort_values("Hold_Number").reset_index(drop=True)
    cargo_by_number = cargo.set_index("Compartment_Number")
    cover_shapes, _ = build_hatch_cover_visualization_proxies(sorted_hatches)
    display_min_width_factors = {}

    for (_, row), shape in zip(sorted_hatches.iterrows(), cover_shapes, strict=True):
        hold_number = int(row["Hold_Number"])
        hold = cargo_by_number.loc[hold_number]
        cover_start = float(row["LCG"]) - 0.5 * float(row["Cover_Length_m"])
        cover_end = float(row["LCG"]) + 0.5 * float(row["Cover_Length_m"])

        if (
            cover_start < float(hold["X_Start_m"]) - 1.0e-6
            or cover_end > float(hold["X_End_m"]) + 1.0e-6
        ):
            raise ValueError(f"Hatch cover {hold_number} exceeds its cargo-hold bounds")

        # The hull can taper inside a hold even when its bounding bulkheads
        # have ample beam. Sample the entire cover, then trim only the visual
        # planform to leave a small visible deck margin on each side.
        stations = np.linspace(cover_start, cover_end, 17)
        half_width = 0.5 * float(row["Cover_Width_m"])
        deck_half_breadths = np.array([
            _half_breadth(hull, float(x), depth) for x in stations
        ])
        tapered = bool(np.any(half_width > 0.9 * deck_half_breadths))
        display_min_width_factors[str(hold_number)] = float(min(
            1.0, 0.9 * deck_half_breadths.min() / half_width
        ))

        if tapered:
            side_widths = np.minimum(half_width, 0.9 * deck_half_breadths)
            bottom_z = float(row["Cover_Bottom_Z_m"])
            port_edge = [
                (float(x), -float(y), bottom_z)
                for x, y in zip(stations, side_widths)
            ]
            starboard_edge = [
                (float(x), float(y), bottom_z)
                for x, y in zip(stations[::-1], side_widths[::-1])
            ]
            wire = cq.Wire.makePolygon(port_edge + starboard_edge, close=True)
            shape = cq.Solid.extrudeLinear(
                wire, [], (0.0, 0.0, float(row["Cover_Envelope_Depth_m"]))
            )

        _add_part(
            parts,
            manifest,
            name=f"hatch_{row['Component_ID']}",
            shape=shape,
            color=(0.20, 0.39, 0.51),
            opacity=1.0,
            kind="hatch_cover_envelope",
            source_table="Hatch_Covers",
            source_identifier=str(row["Component_ID"]),
            source_index=hold_number,
            display_approximation=True,
            notes=(
                "Display cover tapers to at most 90% of sampled local deck "
                "breadth; evaluated cover dimensions and mass are unchanged."
                if tapered else
                "Existing rounded cover visualization proxy; assigned mass is metadata only."
            ),
        )

    # The source STL intentionally has no deck lid. A separate display plane
    # supplies a neutral deck under the hatch covers in exterior views; the
    # cutaways omit it so the hold contents remain inspectable.
    deck_stations = np.linspace(0.0, float(hull.LOA), 101)
    deck_display_z = depth - 0.05
    deck_edge = [
        (float(x), _half_breadth(hull, float(x), depth), deck_display_z)
        for x in deck_stations
    ]
    deck_edge = [point for point in deck_edge if point[1] > 1.0e-3]
    deck_panels = []

    for (x0, y0, z0), (x1, y1, z1) in zip(deck_edge[:-1], deck_edge[1:]):
        # Short panels keep software depth ordering behind nearby covers.
        panel_wire = cq.Wire.makePolygon(
            [(x0, -y0, z0), (x1, -y1, z1),
             (x1, y1, z1), (x0, y0, z0)],
            close=True,
        )
        deck_panels.append(cq.Face.makeFromWires(panel_wire))

    deck_surface = cq.Compound.makeCompound(deck_panels)

    _add_part(
        parts,
        manifest,
        name="deck_display_plane",
        shape=deck_surface,
        color=(0.34, 0.39, 0.43),
        opacity=1.0,
        kind="deck_display_surface",
        source_table="Hull_Parameters",
        source_identifier="Dd",
        source_index=None,
        display_approximation=True,
        notes="Exterior-only open-deck cover 0.05 m below hull depth to avoid hatch overlap; omitted from cutaway views and not structural geometry.",
    )

    # Show the evaluated inner-bottom elevation along the entire main hull.
    # This is a system-view surface; the engineering structural half-model
    # remains limited to its native three-hold length.
    inner_bottom_z = float(structure.source_parameters["Db"]) / 1000.0
    inner_bottom_stations = np.linspace(
        float(hull.bow_profile(inner_bottom_z)),
        float(hull.stern_profile(inner_bottom_z)),
        151,
    )
    inner_bottom_edges = [
        (float(x), _half_breadth(hull, float(x), inner_bottom_z))
        for x in inner_bottom_stations
    ]
    inner_bottom_panels = []

    for (x0, y0), (x1, y1) in zip(
        inner_bottom_edges[:-1], inner_bottom_edges[1:]
    ):
        if y0 <= 1.0e-5:
            outline = [(x0, 0.0, inner_bottom_z),
                       (x1, -y1, inner_bottom_z),
                       (x1, y1, inner_bottom_z)]
        elif y1 <= 1.0e-5:
            outline = [(x0, -y0, inner_bottom_z),
                       (x1, 0.0, inner_bottom_z),
                       (x0, y0, inner_bottom_z)]
        else:
            outline = [(x0, -y0, inner_bottom_z),
                       (x1, -y1, inner_bottom_z),
                       (x1, y1, inner_bottom_z),
                       (x0, y0, inner_bottom_z)]
        inner_bottom_panels.append(cq.Face.makeFromWires(
            cq.Wire.makePolygon(outline, close=True)
        ))

    _add_part(
        parts,
        manifest,
        name="inner_bottom_display_plane",
        shape=cq.Compound.makeCompound(inner_bottom_panels),
        color=(0.70, 0.75, 0.75),
        opacity=1.0,
        kind="full_length_inner_bottom_display_surface",
        source_table="source_structure_parameters",
        source_identifier="Db",
        source_index=None,
        display_approximation=True,
        notes="Hull-bounded full-length surface at the evaluated Db elevation; display only, not a full-length structural plate or added mass.",
    )

    for _, row in compartments.iterrows():
        compartment_type = str(row["Compartment_Type"])

        # Keep cargo holds physically empty in every CAD export and view.
        # Loading remains represented in the separate 2D loading diagram.
        if compartment_type != "fuel_tank":
            continue

        compartment_id = str(row["Compartment_ID"])
        start_x = float(row["X_Start_m"])
        end_x = float(row["X_End_m"])
        envelope = _compartment_shape(hull, start_x, end_x, depth)

        _add_part(
            parts,
            manifest,
            name=f"compartment_{compartment_id}_envelope",
            shape=envelope,
            color=(0.91, 0.56, 0.18),
            opacity=0.12,
            kind=f"{compartment_type}_envelope",
            source_table="Compartment_Loads",
            source_identifier=compartment_id,
            source_index=int(row["Compartment_State_Index"]),
            display_approximation=True,
            notes="Five-station sampled main-hull envelope; interior tank/hold surfaces are unavailable.",
        )

    engine_forward = float(
        bulkheads.loc[
            bulkheads["Boundary_Type"] == "engine_room_forward_bulkhead",
            "X_m",
        ].iloc[0]
    )
    engine_aft = float(
        bulkheads.loc[
            bulkheads["Boundary_Type"] == "engine_room_aft_bulkhead",
            "X_m",
        ].iloc[0]
    )
    aft_peak = float(
        bulkheads.loc[
            bulkheads["Boundary_Type"] == "aft_peak_bulkhead",
            "X_m",
        ].iloc[0]
    )
    engine = _compartment_shape(hull, engine_forward, engine_aft, depth)

    _add_part(
        parts,
        manifest,
        name="compartment_engine_room_envelope",
        shape=engine,
        color=(0.61, 0.39, 0.70),
        opacity=0.18,
        kind="engine_room_envelope",
        source_table="Bulkhead_Locations",
        source_identifier="engine_room_forward_bulkhead:engine_room_aft_bulkhead",
        source_index=None,
        display_approximation=True,
        notes="Main-hull sampled envelope between the engine-room bulkheads; machinery layout unavailable.",
    )

    # Keep the display deckhouse as one rectangular block. It starts at the
    # engine-room/fuel-tank boundary and reaches halfway from the aft
    # engine-room bulkhead toward the aft-peak bulkhead.
    superstructure_breadth = float(sorted_hatches.iloc[0]["Cover_Width_m"])
    engine_length = engine_aft - engine_forward
    superstructure_height = 2.0 * engine_length
    house_start = engine_forward
    house_end = 0.5 * (engine_aft + aft_peak)
    house_length = house_end - house_start
    superstructure = cq.Solid.makeBox(
        house_length, superstructure_breadth, superstructure_height,
        cq.Vector(house_start, -0.5 * superstructure_breadth, depth),
    )

    _add_part(
        parts,
        manifest,
        name="superstructure_display_box",
        shape=superstructure,
        color=(0.98, 0.98, 0.97),
        opacity=1.0,
        kind="superstructure_box",
        source_table="Bulkhead_Locations",
        source_identifier="engine_room_forward_bulkhead:midpoint_from_engine_room_aft_to_aft_peak",
        source_index=None,
        display_approximation=True,
        notes="Illustrative single-block deckhouse from the engine-room/fuel-tank boundary to the midpoint between the aft engine-room and aft-peak bulkheads; width is hatch-cover width and total height is twice the engine-room span. No mass is assigned.",
    )

    placement = {
        "coordinate_system": "x bow-to-stern, y starboard-positive, z keel-to-deck; metres",
        "midbody_x_m": [midbody_start, midbody_end],
        "selected_cargo_hold_numbers": selected["Compartment_Number"].astype(int).tolist(),
        "selected_cargo_boundaries_x_m": [
            selected_start,
            *selected["X_End_m"].astype(float).tolist(),
        ],
        "structure_bulkhead_x_m_original": structure_stations,
        "structure_bulkhead_x_m_placed": [
            float(x + translate_x) for x in structure_stations
        ],
        "structure_translation_x_m": float(translate_x),
        "structure_scale": 1.0,
        "structure_display_mirror_plane": "XZ (y=0)",
        "structure_display_bounds_m": structure_bounds,
        "out_of_hull_bulkhead_stations": out_of_hull_bulkheads,
        "hatch_display_min_width_factor_by_hold": display_min_width_factors,
        "superstructure": {
            "x_start_m": house_start,
            "x_end_m": house_end,
            "breadth_m": superstructure_breadth,
            "base_z_m": depth,
            "height_m": superstructure_height,
            "breadth_source": "Hatch_Covers.Cover_Width_m",
            "height_rule": "2 * (engine_room_aft_bulkhead.X_m - engine_room_forward_bulkhead.X_m)",
        },
    }

    # The detail tables use a source-system Index.  Attach it to each logical
    # component row so a sidecar manifest can locate its exact shard records.
    source_design_index = int(bulkheads.iloc[0]["Index"])

    for row in manifest:
        row["source_design_index"] = source_design_index

    return parts, manifest, placement
