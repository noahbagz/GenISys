"""CadQuery surface constructors matching the legacy Rhino half-ship geometry.

The legacy generator in :mod:`Rhino_tools.rhino_StructGen` models only the
positive-y half of a prismatic three-hold hull.  Its outer section consists of
a flat bottom, a quarter-circle bilge, and a vertical side.  All shapes in this
module follow the same convention and remain zero-thickness CAD faces.

Public coordinates are in metres.  Dictionary values documented by the Rhino
implementation as millimetres (``Db``, bracket vectors, panel offsets, shell
strake height, and legacy hatch dimensions) are converted at their point of
use.  Plate thickness is metadata and is therefore intentionally not applied
to these mid-surface shapes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import asin, cos, pi, sin, sqrt
from numbers import Real
from typing import Any

import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.GeomAbs import GeomAbs_C1
from OCP.GeomConvert import GeomConvert_ApproxCurve
from OCP.gp import gp_Vec


METRES_PER_MILLIMETRE = 0.001
GEOMETRY_TOLERANCE = 1.0e-10


def millimetres_to_metres(value: Real) -> float:
    """Convert one scalar Rhino parameter from millimetres to metres."""

    return float(value) * METRES_PER_MILLIMETRE


# Keep the short spelling convenient at the geometry boundary.
mm_to_m = millimetres_to_metres


def line_edge(start: Sequence[Real], end: Sequence[Real]) -> cq.Edge:
    """Create a straight edge between two metre-coordinate points."""

    return cq.Edge.makeLine(tuple(map(float, start)), tuple(map(float, end)))


def arc_edge(
    start: Sequence[Real],
    midpoint: Sequence[Real],
    end: Sequence[Real],
) -> cq.Edge:
    """Create a circular arc through three metre-coordinate points."""

    return cq.Edge.makeThreePointArc(
        tuple(map(float, start)),
        tuple(map(float, midpoint)),
        tuple(map(float, end)),
    )


def planar_face(points: Sequence[Sequence[Real]]) -> cq.Face:
    """Create one planar face from ordered boundary vertices in metres."""

    vertices = [tuple(map(float, point)) for point in points]
    wire = cq.Wire.makePolygon(vertices, close=True)

    return cq.Face.makeFromWires(wire)


def extrude_edge_to_face(
    edge: cq.Edge,
    vector: Sequence[Real],
) -> cq.Face:
    """Extrude an open edge along ``vector`` to make a zero-thickness face."""

    direction = tuple(map(float, vector))
    prism = BRepPrimAPI_MakePrism(edge.wrapped, gp_Vec(*direction)).Shape()
    face = cq.Shape.cast(prism)

    # Extruding one open edge is expected to produce exactly one face.
    if isinstance(face, cq.Face):
        return face

    return face.Faces()[0]


def rounded_rectangle_face_xy(
    center_x: Real,
    center_y: Real,
    length: Real,
    width: Real,
    z: Real = 0.0,
    radius: Real = 0.0,
) -> cq.Face:
    """Create an XY-plane rectangular face, optionally with rounded corners."""

    center_x = float(center_x)
    center_y = float(center_y)
    length = float(length)
    width = float(width)
    z = float(z)
    radius = float(radius)

    x_min = center_x - length / 2.0
    x_max = center_x + length / 2.0
    y_min = center_y - width / 2.0
    y_max = center_y + width / 2.0

    if radius <= GEOMETRY_TOLERANCE:
        return planar_face(
            [
                (x_min, y_min, z),
                (x_max, y_min, z),
                (x_max, y_max, z),
                (x_min, y_max, z),
            ]
        )

    if radius >= min(length, width) / 2.0:
        raise ValueError("Rounded-rectangle radius must be less than half its width")

    diagonal = radius / sqrt(2.0)
    edges = [
        line_edge((x_min + radius, y_min, z), (x_max - radius, y_min, z)),
        arc_edge(
            (x_max - radius, y_min, z),
            (x_max - radius + diagonal, y_min + radius - diagonal, z),
            (x_max, y_min + radius, z),
        ),
        line_edge((x_max, y_min + radius, z), (x_max, y_max - radius, z)),
        arc_edge(
            (x_max, y_max - radius, z),
            (x_max - radius + diagonal, y_max - radius + diagonal, z),
            (x_max - radius, y_max, z),
        ),
        line_edge((x_max - radius, y_max, z), (x_min + radius, y_max, z)),
        arc_edge(
            (x_min + radius, y_max, z),
            (x_min + radius - diagonal, y_max - radius + diagonal, z),
            (x_min, y_max - radius, z),
        ),
        line_edge((x_min, y_max - radius, z), (x_min, y_min + radius, z)),
        arc_edge(
            (x_min, y_min + radius, z),
            (x_min + radius - diagonal, y_min + radius - diagonal, z),
            (x_min + radius, y_min, z),
        ),
    ]
    wire = cq.Wire.assembleEdges(edges)

    return cq.Face.makeFromWires(wire)


def _hull_dimensions(hull: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Return length, half-breadth, depth, and bilge radius in metres."""

    length = float(hull["L_3h"])
    half_breadth = float(hull["B"]) / 2.0
    depth = float(hull["D"])
    radius = float(hull["R_b"])

    # These are the geometric limits of the analytical quarter-circle section.
    if radius <= 0.0 or radius > min(half_breadth, depth):
        raise ValueError("R_b must be positive and no larger than B/2 or D")

    return length, half_breadth, depth, radius


def _bilge_point(
    half_breadth: float,
    radius: float,
    theta: float,
    x: float,
) -> tuple[float, float, float]:
    """Return a point on the bilge for ``theta`` from -pi/2 to zero."""

    center_y = half_breadth - radius
    y = center_y + radius * cos(theta)
    z = radius + radius * sin(theta)

    return (x, y, z)


def _bilge_theta_at_z(radius: float, z: float) -> float:
    """Map an elevation on the quarter-circle bilge to its angle."""

    value = max(-1.0, min(0.0, (z - radius) / radius))

    return asin(value)


def half_hull_section_edges(
    hull: Mapping[str, Any],
    x: Real = 0.0,
) -> dict[str, cq.Edge]:
    """Build the ordered bottom, bilge, and side edges of one half-section."""

    _, half_breadth, depth, radius = _hull_dimensions(hull)
    x = float(x)
    tangent_y = half_breadth - radius

    bottom = line_edge((x, 0.0, 0.0), (x, tangent_y, 0.0))
    bilge = arc_edge(
        (x, tangent_y, 0.0),
        _bilge_point(half_breadth, radius, -pi / 4.0, x),
        (x, half_breadth, radius),
    )
    side = line_edge(
        (x, half_breadth, radius),
        (x, half_breadth, depth),
    )

    return {"bottom": bottom, "bilge": bilge, "side": side}


def half_breadth_at_z(hull: Mapping[str, Any], z: Real) -> float:
    """Return the positive-y shell coordinate at elevation ``z`` in metres."""

    _, half_breadth, depth, radius = _hull_dimensions(hull)
    z = float(z)

    if z < -GEOMETRY_TOLERANCE or z > depth + GEOMETRY_TOLERANCE:
        raise ValueError("z must lie between the keel and moulded depth")

    if z <= radius:
        center_y = half_breadth - radius
        radial_y = sqrt(max(0.0, radius**2 - (z - radius) ** 2))

        return center_y + radial_y

    return half_breadth


def shell_height_at_y(hull: Mapping[str, Any], y: Real) -> float:
    """Return the lower outer-shell elevation at a positive-y coordinate."""

    _, half_breadth, _, radius = _hull_dimensions(hull)
    y = float(y)

    if y < -GEOMETRY_TOLERANCE or y > half_breadth + GEOMETRY_TOLERANCE:
        raise ValueError("y must lie between the centerline and half-breadth")

    center_y = half_breadth - radius

    if y <= center_y:
        return 0.0

    radial_z = sqrt(max(0.0, radius**2 - (y - center_y) ** 2))

    return radius - radial_z


def _section_edges_between_z(
    hull: Mapping[str, Any],
    z_min: float,
    z_max: float,
    *,
    x: float = 0.0,
    include_bottom: bool = False,
) -> list[cq.Edge]:
    """Clip the analytical outer-section curve between two elevations."""

    _, half_breadth, depth, radius = _hull_dimensions(hull)
    lower = max(0.0, float(z_min))
    upper = min(depth, float(z_max))

    if upper < lower - GEOMETRY_TOLERANCE:
        return []

    edges: list[cq.Edge] = []
    tangent_y = half_breadth - radius

    if include_bottom and lower <= GEOMETRY_TOLERANCE:
        edges.append(line_edge((x, 0.0, 0.0), (x, tangent_y, 0.0)))

    arc_lower = max(lower, 0.0)
    arc_upper = min(upper, radius)

    if arc_upper - arc_lower > GEOMETRY_TOLERANCE:
        theta_lower = _bilge_theta_at_z(radius, arc_lower)
        theta_upper = _bilge_theta_at_z(radius, arc_upper)
        theta_midpoint = (theta_lower + theta_upper) / 2.0
        edges.append(
            arc_edge(
                _bilge_point(half_breadth, radius, theta_lower, x),
                _bilge_point(half_breadth, radius, theta_midpoint, x),
                _bilge_point(half_breadth, radius, theta_upper, x),
            )
        )

    side_lower = max(lower, radius)
    side_upper = min(upper, depth)

    if side_upper - side_lower > GEOMETRY_TOLERANCE:
        edges.append(
            line_edge(
                (x, half_breadth, side_lower),
                (x, half_breadth, side_upper),
            )
        )

    return edges


def _compound(shapes: Iterable[cq.Shape]) -> cq.Compound:
    """Create a compound after materializing a one-pass shape iterable."""

    return cq.Compound.makeCompound(list(shapes))


def join_edges_as_single_curve(edges: Sequence[cq.Edge]) -> cq.Edge:
    """Approximate a tangent edge chain as one curve, matching Rhino JoinCurves.

    The Rhino hull starts from a joined bottom/bilge/side polycurve and its
    extrusion is exported as one IGES face. Keeping separate analytical OCC
    edges is geometrically equivalent but changes face topology. This helper
    retains the analytical path to sub-micrometre tolerance while restoring
    one joined physical face.
    """

    edge_list = list(edges)

    if len(edge_list) == 1:
        return edge_list[0]

    wire = cq.Wire.assembleEdges(edge_list)
    approximation = GeomConvert_ApproxCurve(
        wire._geomAdaptor(),
        1.0e-10,
        GeomAbs_C1,
        1000,
        8,
    )
    curve = approximation.Curve()

    return cq.Edge(BRepBuilderAPI_MakeEdge(curve).Edge())


def make_half_hull_shell(hull: Mapping[str, Any]) -> cq.Compound:
    """Create the analytical flat-bottom/bilge/side half-hull shell.

    This helper intentionally retains the three exact section segments.  The
    exported bottom and side shell plates are joined separately by
    :func:`make_shell_plating`, where matching Rhino's one-face-per-plate
    topology matters.  Keeping this analytical helper exact avoids the small
    surface-area loss introduced when OCC approximates all three segments as
    one spline.
    """

    length, _, _, _ = _hull_dimensions(hull)
    section = half_hull_section_edges(hull)

    return _compound(
        extrude_edge_to_face(edge, (length, 0.0, 0.0))
        for edge in section.values()
    )


def make_shell_plating(
    hull: Mapping[str, Any],
    structure: Mapping[str, Any],
) -> dict[str, cq.Face]:
    """Split the half-hull into bottom and side shell at ``Db``.

    ``structure['Db']`` follows the Rhino contract and is interpreted as
    millimetres.  The lower group includes the flat bottom and every shell
    segment below the double-bottom elevation; the upper group contains the
    remaining bilge/side shell.
    """

    length, _, depth, _ = _hull_dimensions(hull)
    double_bottom = mm_to_m(structure["Db"])

    if double_bottom <= 0.0 or double_bottom >= depth:
        raise ValueError("Db must lie above the keel and below moulded depth")

    bottom_edges = _section_edges_between_z(
        hull,
        0.0,
        double_bottom,
        include_bottom=True,
    )
    side_edges = _section_edges_between_z(
        hull,
        double_bottom,
        depth,
    )

    return {
        "bottom_shell": extrude_edge_to_face(
            join_edges_as_single_curve(bottom_edges),
            (length, 0.0, 0.0),
        ),
        "side_shell": extrude_edge_to_face(
            join_edges_as_single_curve(side_edges),
            (length, 0.0, 0.0),
        ),
    }


def _as_values(value: Any) -> list[Any]:
    """Normalize a scalar or one-dimensional iterable without NumPy coupling."""

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return list(value)

    return [value]


def _legacy_hatches(
    hull: Mapping[str, Any],
    lightening_holes: Sequence[Mapping[str, Any]],
) -> list[dict[str, float]]:
    """Translate Rhino deck ``LH_Dict`` entries into centered half-hull cuts."""

    length = float(hull["L_3h"])
    hatches: list[dict[str, float]] = []

    for hole in lightening_holes:
        hatch_length = mm_to_m(hole["l"])
        half_breadth = mm_to_m(hole["h"]) / 2.0
        radius = mm_to_m(hole.get("r", 0.0))

        for x_fraction in _as_values(hole["x1"]):
            hatches.append(
                {
                    "center_x": float(x_fraction) * length,
                    "length": hatch_length,
                    "half_breadth": half_breadth,
                    "radius": radius,
                }
            )

    return hatches


def _normalise_hatches(
    hull: Mapping[str, Any],
    deck: Mapping[str, Any],
) -> list[dict[str, float]]:
    """Accept either direct metre hatch data or the legacy ``LH_Dict`` form."""

    if "hatches" in deck:
        normalized = []

        for hatch in deck["hatches"]:
            if "center_x" in hatch:
                center_x = float(hatch["center_x"])
                length = float(hatch["length"])
            else:
                x_start = float(hatch["x_start"])
                x_end = float(hatch["x_end"])
                center_x = (x_start + x_end) / 2.0
                length = x_end - x_start

            normalized.append(
                {
                    "center_x": center_x,
                    "length": length,
                    "half_breadth": float(hatch["half_breadth"]),
                    "radius": float(hatch.get("radius", 0.0)),
                }
            )

        return normalized

    if "LH_Dict" in deck:
        return _legacy_hatches(hull, deck["LH_Dict"])

    return []


def make_deck_face(
    hull: Mapping[str, Any],
    deck: Mapping[str, Any],
) -> cq.Shape:
    """Create a deck or inner-bottom face with optional centered hatch notches.

    ``deck['z']`` is in metres.  Direct ``deck['hatches']`` entries also use
    metres and contain ``center_x``/``length``/``half_breadth`` plus an
    optional ``radius``.  The original Rhino ``LH_Dict`` representation is
    accepted as well; its center positions are length fractions and its
    ``l``, ``h``, and ``r`` values are millimetres.
    """

    length, _, _, _ = _hull_dimensions(hull)
    z = float(deck["z"])
    breadth = half_breadth_at_z(hull, z)
    plate: cq.Shape = planar_face(
        [
            (0.0, 0.0, z),
            (length, 0.0, z),
            (length, breadth, z),
            (0.0, breadth, z),
        ]
    )

    # A full-width centered hatch intersects this positive-y model as a notch
    # extending from the centerline to the hatch half-breadth.
    for hatch in _normalise_hatches(hull, deck):
        cutter = rounded_rectangle_face_xy(
            center_x=hatch["center_x"],
            center_y=0.0,
            length=hatch["length"],
            width=2.0 * hatch["half_breadth"],
            z=z,
            radius=hatch["radius"],
        )
        plate = plate.cut(cutter)

    return plate


def make_transverse_bulkhead_face(
    hull: Mapping[str, Any],
    x: Real,
) -> cq.Face:
    """Create one complete positive-y transverse bulkhead at metre ``x``."""

    _, half_breadth, depth, _ = _hull_dimensions(hull)
    x = float(x)
    section = half_hull_section_edges(hull, x=x)
    edges = [
        section["bottom"],
        section["bilge"],
        section["side"],
        line_edge((x, half_breadth, depth), (x, 0.0, depth)),
        line_edge((x, 0.0, depth), (x, 0.0, 0.0)),
    ]

    return cq.Face.makeFromWires(cq.Wire.assembleEdges(edges))


def make_transverse_bulkheads(
    hull: Mapping[str, Any],
    bulkhead: Mapping[str, Any],
) -> list[cq.Face]:
    """Create Rhino-compatible transverse bulkheads from fractional x values."""

    length = float(hull["L_3h"])

    return [
        make_transverse_bulkhead_face(hull, float(x_fraction) * length)
        for x_fraction in _as_values(bulkhead["x"])
    ]


def make_longitudinal_bulkhead_face(
    hull: Mapping[str, Any],
    y: Real,
) -> cq.Face:
    """Create one longitudinal bulkhead at a positive-y metre coordinate."""

    length, _, depth, _ = _hull_dimensions(hull)
    y = float(y)
    bottom = shell_height_at_y(hull, y)

    return planar_face(
        [
            (0.0, y, bottom),
            (length, y, bottom),
            (length, y, depth),
            (0.0, y, depth),
        ]
    )


def make_longitudinal_bulkheads(
    hull: Mapping[str, Any],
    bulkhead: Mapping[str, Any],
) -> list[cq.Face]:
    """Create longitudinal bulkheads from Rhino half-breadth fractions."""

    half_breadth = float(hull["B"]) / 2.0

    return [
        make_longitudinal_bulkhead_face(hull, float(y_fraction) * half_breadth)
        for y_fraction in _as_values(bulkhead["y"])
    ]


def make_shell_strake(
    hull: Mapping[str, Any],
    strake: Mapping[str, Any],
) -> cq.Face:
    """Create a shell band from metre ``z`` and millimetre ``h`` values."""

    length, _, depth, _ = _hull_dimensions(hull)
    z_lower = float(strake["z"])
    z_upper = min(depth, z_lower + mm_to_m(strake["h"]))
    edges = _section_edges_between_z(hull, z_lower, z_upper)

    if not edges:
        raise ValueError("Shell strake does not intersect the half-hull shell")

    joined_edges = join_edges_as_single_curve(edges)

    return extrude_edge_to_face(joined_edges, (length, 0.0, 0.0))


def make_transverse_floor_face(
    hull: Mapping[str, Any],
    structure: Mapping[str, Any],
    x: Real,
) -> cq.Face:
    """Create the double-bottom transverse floor bounded by the outer shell."""

    _, _, depth, _ = _hull_dimensions(hull)
    double_bottom = mm_to_m(structure["Db"])
    x = float(x)

    if double_bottom <= 0.0 or double_bottom >= depth:
        raise ValueError("Db must lie above the keel and below moulded depth")

    shell_edges = _section_edges_between_z(
        hull,
        0.0,
        double_bottom,
        x=x,
        include_bottom=True,
    )
    breadth = half_breadth_at_z(hull, double_bottom)
    edges = [
        *shell_edges,
        line_edge((x, breadth, double_bottom), (x, 0.0, double_bottom)),
        line_edge((x, 0.0, double_bottom), (x, 0.0, 0.0)),
    ]

    return cq.Face.makeFromWires(cq.Wire.assembleEdges(edges))


def make_bracket_face(
    vertex: Sequence[Real],
    l1_mm: Sequence[Real],
    l2_mm: Sequence[Real],
    num_sides: int = 3,
) -> cq.Face:
    """Create one Rhino-compatible triangular or rectangular bracket face."""

    origin = tuple(map(float, vertex))
    l1 = tuple(mm_to_m(value) for value in l1_mm)
    l2 = tuple(mm_to_m(value) for value in l2_mm)

    point_1 = tuple(origin[i] + l1[i] for i in range(3))
    point_2 = tuple(origin[i] + l2[i] for i in range(3))

    if num_sides == 3:
        return planar_face([origin, point_1, point_2])

    if num_sides == 4:
        point_3 = tuple(origin[i] + l1[i] + l2[i] for i in range(3))

        return planar_face([origin, point_1, point_3, point_2])

    raise ValueError("Only three- and four-sided brackets are supported")


def make_bracket_faces(bracket: Mapping[str, Any]) -> list[cq.Face]:
    """Create all faces described by one legacy Rhino bracket dictionary."""

    return [
        make_bracket_face(
            vertex,
            bracket["L1"],
            bracket["L2"],
            int(bracket["num_sides"]),
        )
        for vertex in bracket["Vertex"]
    ]


def make_panel_face(
    hull: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> cq.Face:
    """Create one longitudinal hopper/container panel mid-surface.

    ``panel['start']`` is expressed in metres.  The legacy ``L1`` (y offset)
    and ``L2`` (z offset) values are expressed in millimetres.  The resulting
    face spans ``L_3h`` in x exactly as ``make_Pannel_Surface`` does in Rhino.
    """

    length = float(hull["L_3h"])
    start = tuple(map(float, panel["start"]))
    offset_y = mm_to_m(panel["L1"])
    offset_z = mm_to_m(panel["L2"])
    end = (start[0] + length, start[1], start[2])

    return planar_face(
        [
            start,
            end,
            (end[0], end[1] + offset_y, end[2] + offset_z),
            (start[0], start[1] + offset_y, start[2] + offset_z),
        ]
    )
