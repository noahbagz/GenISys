"""CadQuery profile transforms and surface sweeps for stiffened structure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cadquery as cq
from OCP.BRepFill import BRepFill

from .geometry import join_edges_as_single_curve
from .identifiers import IdentifierFactory
from .model import ShipStructure, StructuralElement


def _as_wire(path: cq.Edge | cq.Wire) -> cq.Wire:
    """Return a path wire while preserving the caller's edge orientation."""

    if isinstance(path, cq.Wire):
        return path

    if isinstance(path, cq.Edge):
        return cq.Wire.assembleEdges([path])

    raise TypeError("A stiffener path must be a cadquery.Edge or cadquery.Wire")


def _rotated_profile_edge(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    rotation: float,
    direction: tuple[int, int, int],
) -> cq.Edge:
    """Apply the Rhino profile transformation order to one local edge."""

    edge = cq.Edge.makeLine(start, end)

    # The local profile first rotates in its XY plane about global Z.
    edge = edge.rotate((0, 0, 0), (0, 0, 1), float(rotation))

    # Rhino then rotates the whole profile into the plane normal to the path.
    if direction == (1, 0, 0):
        edge = edge.rotate((0, 0, 0), (0, 1, 0), -90.0)
    elif direction == (0, 1, 0):
        edge = edge.rotate((0, 0, 0), (1, 0, 0), 90.0)
    elif direction != (0, 0, 1):
        raise ValueError("Stiffener direction must be a positive Cartesian axis")

    return edge


def make_stiffener_profile(
    stiffener: Mapping[str, Any],
    buffer: float = 0.0,
) -> list[cq.Edge]:
    """Build the web and optional flange profile edges at the origin.

    The source dictionaries store ``h``, ``t``, and ``w`` in millimetres.
    Geometry is returned in metres. Thickness does not alter the neutral
    surface profile; it remains structural metadata exactly as in Rhino.
    """

    height = float(stiffener["h"]) / 1000.0
    width = float(stiffener["w"]) / 1000.0
    rotation = float(stiffener["rot"])
    direction = tuple(int(value) for value in stiffener["dir"])

    edges = [
        _rotated_profile_edge(
            (-float(buffer), 0.0, 0.0),
            (height, 0.0, 0.0),
            rotation,
            direction,
        )
    ]

    if width > 0.0:
        if bool(stiffener["bit_TorC"]):
            flange_start = (height, -width / 2.0, 0.0)
            flange_end = (height, width / 2.0, 0.0)
        else:
            flange_start = (height, 0.0, 0.0)
            flange_end = (height, width, 0.0)

        edges.append(
            _rotated_profile_edge(
                flange_start,
                flange_end,
                rotation,
                direction,
            )
        )

    return edges


def sweep_profile_edge(profile: cq.Edge, path: cq.Edge | cq.Wire) -> cq.Shape:
    """Extrude one profile edge along a path with a fixed global orientation.

    Rhino ``ExtrudeCurve(profile, path)`` translates the profile without
    rotating it around a curved bilge rail. That detail is visible in the
    source comment noting that transverse frames are not kept flush to shell
    curvature. A default OpenCascade pipe uses a moving trihedron and changes
    web areas by roughly three percent, so the two translated rail curves are
    joined directly as a ruled face instead.
    """

    path_wire = _as_wire(path)
    path_edges = path_wire.Edges()

    if len(path_edges) == 1:
        rail = path_edges[0]
    else:
        # Rhino's joined polycurve extrudes as one surface. Approximate the
        # compound path as one B-spline edge so IGES face counts retain the
        # one-physical-member/one-face contract.
        rail = join_edges_as_single_curve(path_edges)

    first_rail = rail.translate(profile.startPoint())
    second_rail = rail.translate(profile.endPoint())
    face = BRepFill.Face_s(first_rail.wrapped, second_rail.wrapped)

    return cq.Face(face)


def sweep_stiffener(
    path: cq.Edge | cq.Wire,
    stiffener: Mapping[str, Any],
) -> list[cq.Shape]:
    """Return web and optional flange surfaces along one path."""

    return [
        sweep_profile_edge(profile, path)
        for profile in make_stiffener_profile(stiffener)
    ]


def _sweep_profile_edge_for_properties(
    profile: cq.Edge,
    path: cq.Edge | cq.Wire,
) -> cq.Shape:
    """Sweep every analytical rail segment for Rhino-compatible properties.

    The exported surface remains one joined face. This segmented companion is
    used only for area and centroid calculations, matching Rhino's property
    calculation before its IGES writer approximates the joined polycurve.
    """

    path_edges = _as_wire(path).Edges()

    if len(path_edges) == 1:
        return sweep_profile_edge(profile, path_edges[0])

    faces = []

    for rail in path_edges:
        first_rail = rail.translate(profile.startPoint())
        second_rail = rail.translate(profile.endPoint())
        faces.append(
            cq.Face(BRepFill.Face_s(first_rail.wrapped, second_rail.wrapped))
        )

    return cq.Compound.makeCompound(faces)


def add_stiffener(
    structure: ShipStructure,
    identifiers: IdentifierFactory,
    path: cq.Edge | cq.Wire,
    stiffener: Mapping[str, Any],
    parent_id: str = "",
    object_id_root: str | None = None,
    side: str = "starboard-half",
    color: object = None,
) -> list[StructuralElement]:
    """Sweep a member and append downstream-compatible web/flange records."""

    path_wire = _as_wire(path)
    start = path_wire.startPoint()
    end = path_wire.endPoint()
    surfaces = sweep_stiffener(path_wire, stiffener)
    profiles = make_stiffener_profile(stiffener)

    if len(path_wire.Edges()) == 1:
        # Straight and single-arc members need no companion topology. Reuse
        # their exported faces and avoid doubling the dominant build work.
        property_surfaces = surfaces
    else:
        property_surfaces = [
            _sweep_profile_edge_for_properties(profile, path_wire)
            for profile in profiles
        ]

    element_class = str(stiffener["Class"])
    direction = tuple(int(value) for value in stiffener["dir"])
    projected_length = max(
        abs(end.x - start.x),
        abs(end.y - start.y),
        abs(end.z - start.z),
    )
    dimensions = [float(stiffener["h"]) / 1000.0, float(stiffener["w"]) / 1000.0]
    rotations = [float(stiffener["rot"]), float(stiffener["rot"]) + 90.0]
    classes = [element_class, "Stiffener_Flange"]
    save_as_1d = [bool(stiffener["1D_element"]), True]

    # The private section key links web/flange calculations without relying on
    # dataframe adjacency. It is deliberately excluded from the CSV schema.
    section_key = f"{structure.ship_id}:{element_class}:{len(structure.elements)}"
    added: list[StructuralElement] = []

    # Issue one logical member ID. Existing evaluation code finds its flange
    # by appending ``_Flange`` to this exact value, so the deterministic
    # replacement preserves that relationship instead of assigning the
    # flange an unrelated class counter.
    if object_id_root is None:
        # Keep the standalone helper usable for callers that do not provide
        # ship-structure context.  The complete builder always supplies the
        # coordinate-derived Rhino naming root.
        web_object_id, web_sequence = identifiers.make(
            element_class,
            start.x,
            side=side,
        )
    else:
        web_object_id, web_sequence = identifiers.make_rhino(object_id_root)

    for profile_index, surface in enumerate(surfaces):
        profile_start = profiles[profile_index].startPoint().add(start)
        location = [profile_start.x, profile_start.y, profile_start.z]

        # Preserve the legacy projected-location convention: the coordinate
        # parallel to the member path is always stored as zero.
        for axis_index, component in enumerate(direction):
            if component == 1:
                location[axis_index] = 0.0

        if profile_index == 0:
            object_id = web_object_id
        else:
            object_id = f"{web_object_id}_Flange"

        sequence = web_sequence

        mesh_anchor = (start.x, start.y, start.z)

        if profile_index == 1 and not bool(stiffener["1D_element"]):
            web_tip = profiles[0].endPoint().add(start)
            mesh_anchor = (web_tip.x, web_tip.y, web_tip.z)

        metadata = {
            "Color": color,
            "x_loc": location[0],
            "y_loc": location[1],
            "z_loc": location[2],
            "x_dir": direction[0],
            "y_dir": direction[1],
            "z_dir": direction[2],
            "rot": rotations[profile_index],
            "L1": projected_length,
            "L2": dimensions[profile_index],
            "Thickness": float(stiffener["t"]) / 1000.0,
            "Lightening_Holes": (
                str(stiffener.get("LH_Dict")) if profile_index == 0 and "LH_Dict" in stiffener else None
            ),
            "bit_1D_element": save_as_1d[profile_index],
            "Class": classes[profile_index],
            "Type": "Stiffener",
            "Parent_Struct": parent_id,
            "_section_group": section_key,
            "_mesh_anchor": mesh_anchor,
            "_web_is_1d": bool(stiffener["1D_element"]),
        }

        # Minor members use the centre path as their mesh representation. Major
        # web surfaces retain their own surface while the flange gets a line.
        if profile_index == 0:
            mesh_shape = path_wire if bool(stiffener["1D_element"]) else surface
        elif bool(stiffener["1D_element"]):
            mesh_shape = None
        else:
            mesh_shape = path_wire.translate(profiles[0].endPoint())

        element = StructuralElement(
            object_id=object_id,
            shape=surface,
            mesh_shape=mesh_shape,
            property_shape=property_surfaces[profile_index],
            metadata=metadata,
            side=side,
            sequence=sequence,
        )
        structure.add(element)
        added.append(element)

    return added
