"""Build complete Rhino-compatible three-hold structures with CadQuery.

The legacy :class:`Rhino_tools.Parametric_Structure_V2.Structure_3H` class
contains both parameter interpretation and document-oriented Rhino calls.
This module supplies the equivalent orchestration layer for the independent
CadQuery primitives.  It intentionally retains the existing half-model,
class names, projected metadata, and parent relationships used by the
evaluation scripts.

All CAD coordinates are metres.  Plate and profile values originating in the
120-value parameter row remain millimetres until they cross into geometry.
The generated structure contains zero-thickness mid-surfaces on the positive-y
half of the hull; physical thickness is retained in element metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from math import atan2, degrees, hypot
from time import perf_counter
from typing import Any

import cadquery as cq
import numpy as np

from .geometry import (
    GEOMETRY_TOLERANCE,
    _section_edges_between_z,
    extrude_edge_to_face,
    half_breadth_at_z,
    line_edge,
    make_bracket_faces,
    make_deck_face,
    make_longitudinal_bulkhead_face,
    make_panel_face,
    make_shell_plating,
    make_shell_strake,
    make_transverse_bulkhead_face,
    make_transverse_floor_face,
    millimetres_to_metres,
    shell_height_at_y,
)
from .identifiers import IdentifierFactory
from .model import ShipStructure, StructuralElement
from .parameters import ParametricStructureDefinition
from .properties import calculate_properties
from .stiffeners import add_stiffener


PATH_TOLERANCE = 1.0e-8


class CadQueryStructureBuilder:
    """Generate one complete parametric three-hold ship structure.

    Parameters
    ----------
    params_or_definition:
        Either the legacy mutable 120-value parameter row or an already
        interpreted :class:`ParametricStructureDefinition`.
    ship_id:
        Stable design label used as the root of deterministic logical IDs.

    Notes
    -----
    ``build`` starts from an empty in-memory model each time.  Rebuilding the
    same interpreted definition therefore produces the same IDs and element
    order while recording fresh execution timings.
    """

    def __init__(
        self,
        params_or_definition: Any,
        ship_id: str,
    ):
        parameter_start = perf_counter()

        if isinstance(params_or_definition, ParametricStructureDefinition):
            self.definition = params_or_definition
        else:
            self.definition = ParametricStructureDefinition(params_or_definition)

        self.parameter_interpretation_time = perf_counter() - parameter_start
        self.ship_id = str(ship_id)
        self.structure = ShipStructure(ship_id=self.ship_id)
        self.identifiers = IdentifierFactory(self.ship_id)

    @contextmanager
    def _stage(self, name: str):
        """Record the elapsed wall time for one deterministic build stage."""

        start = perf_counter()

        try:
            yield
        finally:
            self.structure.timings[name] = perf_counter() - start

    @property
    def hull(self) -> dict[str, Any]:
        """Return the interpreted legacy hull dictionary."""

        return self.definition.Hull_Dict

    @property
    def structural_parameters(self) -> dict[str, Any]:
        """Return the interpreted plate and double-bottom dictionary."""

        return self.definition.Struct_Params

    @property
    def length(self) -> float:
        return float(self.hull["L_3h"])

    @property
    def half_breadth(self) -> float:
        return float(self.hull["B"]) / 2.0

    @property
    def depth(self) -> float:
        return float(self.hull["D"])

    @property
    def double_bottom(self) -> float:
        return millimetres_to_metres(self.structural_parameters["Db"])

    def _add_surface(
        self,
        shape: cq.Shape,
        *,
        element_class: str,
        element_type: str,
        parent_id: str,
        x_loc: float,
        y_loc: float,
        z_loc: float,
        direction: Sequence[int],
        rotation: float,
        length_1: float,
        length_2: float,
        thickness_mm: float,
        lightening_holes: object = None,
        bit_1d_element: bool = False,
        object_id: str | None = None,
        mesh_shape: cq.Shape | None = None,
        property_shape: cq.Shape | None = None,
    ) -> StructuralElement:
        """Append one plate-like surface with the established CSV vocabulary."""

        if object_id is None:
            object_id, sequence = self.identifiers.make(
                element_class,
                float(x_loc),
            )
        else:
            # Register every semantic root through one collision-aware path.
            # The first entity keeps the exact Rhino name; only repeated
            # physical pieces receive a deterministic ``_part_n`` suffix.
            object_id, sequence = self.identifiers.make_rhino(object_id)

        metadata = {
            "Color": None,
            "x_loc": float(x_loc),
            "y_loc": float(y_loc),
            "z_loc": float(z_loc),
            "x_dir": int(direction[0]),
            "y_dir": int(direction[1]),
            "z_dir": int(direction[2]),
            "rot": float(rotation),
            "L1": float(length_1),
            "L2": float(length_2),
            "Thickness": millimetres_to_metres(thickness_mm),
            "Lightening_Holes": lightening_holes,
            "bit_1D_element": bool(bit_1d_element),
            "Class": element_class,
            "Type": element_type,
            "Parent_Struct": parent_id,
        }

        element = StructuralElement(
            object_id=object_id,
            shape=shape,
            mesh_shape=shape if mesh_shape is None else mesh_shape,
            property_shape=property_shape,
            metadata=metadata,
            sequence=sequence,
        )
        self.structure.add(element)

        return element

    @staticmethod
    def _edge_sort_key(edge: cq.Edge) -> tuple[float, ...]:
        """Sort disjoint intersection paths in stable coordinate order."""

        bounds = edge.BoundingBox()

        return (
            bounds.xmin,
            bounds.ymin,
            bounds.zmin,
            bounds.xmax,
            bounds.ymax,
            bounds.zmax,
        )

    @staticmethod
    def _orient_edge(edge: cq.Edge, axis: int) -> cq.Edge:
        """Orient a straight intersection edge along its positive path axis."""

        start = edge.startPoint().toTuple()
        end = edge.endPoint().toTuple()

        if end[axis] < start[axis]:
            return edge.reverse()

        return edge

    def _plate_line_paths(
        self,
        plate: cq.Shape,
        start: Sequence[float],
        end: Sequence[float],
        *,
        axis: int,
    ) -> list[cq.Edge]:
        """Return surviving line pieces after clipping against a plate face.

        This is what makes deck longitudinals and transverse beams stop at
        hatch boundaries.  A centerline hatch can produce several independent
        longitudinal pieces, each of which remains a physical member record.
        """

        candidate = line_edge(start, end)
        intersection = plate.intersect(candidate)
        paths = [
            self._orient_edge(edge, axis)
            for edge in intersection.Edges()
            if edge.Length() > PATH_TOLERANCE
        ]

        return sorted(paths, key=self._edge_sort_key)

    @staticmethod
    def _wire_from_edges(edges: Sequence[cq.Edge]) -> cq.Wire:
        """Join ordered analytical shell edges into one physical frame path."""

        return cq.Wire.assembleEdges(list(edges))

    def _add_shell_plating(self) -> None:
        """Add the two shell groups created during Rhino object initialization."""

        plating = make_shell_plating(self.hull, self.structural_parameters)
        bottom_property_shape = cq.Compound.makeCompound(
            [
                extrude_edge_to_face(edge, (self.length, 0.0, 0.0))
                for edge in _section_edges_between_z(
                    self.hull,
                    0.0,
                    self.double_bottom,
                    include_bottom=True,
                )
            ]
        )
        side_property_shape = cq.Compound.makeCompound(
            [
                extrude_edge_to_face(edge, (self.length, 0.0, 0.0))
                for edge in _section_edges_between_z(
                    self.hull,
                    self.double_bottom,
                    self.depth,
                )
            ]
        )

        # These exact IDs are consumed directly by StructureCalcs, Weights,
        # and Parametric_Structure_Eval.
        self._add_surface(
            plating["side_shell"],
            element_class="Side_Shell",
            element_type="Side_Shell",
            parent_id="Hull",
            x_loc=0.0,
            y_loc=self.half_breadth,
            z_loc=self.double_bottom,
            direction=(1, 0, 0),
            rotation=0.0,
            length_1=self.length,
            length_2=self.depth - self.double_bottom,
            thickness_mm=self.structural_parameters["Side_Shell_Thickness"],
            object_id="Side_Shell",
            property_shape=side_property_shape,
        )
        self._add_surface(
            plating["bottom_shell"],
            element_class="Bottom_Shell",
            element_type="Bottom_Shell",
            parent_id="Hull",
            x_loc=0.0,
            y_loc=0.0,
            z_loc=0.0,
            direction=(1, 0, 0),
            rotation=90.0,
            length_1=self.length,
            length_2=self.half_breadth,
            thickness_mm=self.structural_parameters["Bottom_Shell_Thickness"],
            object_id="Bottom_Shell",
            property_shape=bottom_property_shape,
        )

    def _add_deck(
        self,
        deck_parameters: dict[str, Any],
        longitudinal_members: Sequence[dict[str, Any]],
        transverse_members: Sequence[dict[str, Any]],
    ) -> StructuralElement:
        """Add one deck plate and every member clipped to its actual pieces."""

        elevation = float(deck_parameters["z"])
        breadth = half_breadth_at_z(self.hull, elevation)
        deck_shape = make_deck_face(self.hull, deck_parameters)
        deck = self._add_surface(
            deck_shape,
            element_class="Deck",
            element_type="Deck",
            parent_id="Hull",
            x_loc=0.0,
            y_loc=0.0,
            z_loc=elevation,
            direction=(1, 0, 0),
            rotation=90.0,
            length_1=self.length,
            # Rhino retained nominal half-beam metadata even where a low
            # inner bottom met the curved bilge before B/2.
            length_2=self.half_breadth,
            thickness_mm=deck_parameters["t"],
            lightening_holes=(
                str(deck_parameters["LH_Dict"])
                if "LH_Dict" in deck_parameters
                else None
            ),
            object_id=f"Deck_Z_{elevation}",
        )

        # Longitudinal girders inside a hatch opening become multiple physical
        # pieces.  Intersecting the plate itself also respects corner radii.
        for member in longitudinal_members:
            y_positions = np.atleast_1d(member["y"]).astype(float)

            for y_fraction in y_positions:
                y = float(y_fraction) * self.half_breadth
                paths = self._plate_line_paths(
                    deck_shape,
                    (0.0, y, elevation),
                    (self.length, y, elevation),
                    axis=0,
                )

                for path in paths:
                    add_stiffener(
                        self.structure,
                        self.identifiers,
                        path,
                        member,
                        parent_id=deck.object_id,
                        object_id_root=(
                            f"{deck.object_id}_Long_Stiffener_Y_{y}"
                        ),
                    )

        # Transverse beams stop at a hatch edge when their station falls inside
        # an opening, and otherwise retain the complete positive-y span.
        for member in transverse_members:
            x_positions = np.atleast_1d(member["x"]).astype(float)

            for x_fraction in x_positions:
                x = float(x_fraction) * self.length
                paths = self._plate_line_paths(
                    deck_shape,
                    (x, 0.0, elevation),
                    (x, breadth, elevation),
                    axis=1,
                )

                for path in paths:
                    add_stiffener(
                        self.structure,
                        self.identifiers,
                        path,
                        member,
                        parent_id=deck.object_id,
                        object_id_root=(
                            f"{deck.object_id}_Trans_Stiffener_X_{x}"
                        ),
                    )

        return deck

    def _add_container_top_deck(self) -> None:
        """Add the container ship's narrow top-deck strip as a swept surface."""

        member = self.definition.Container_Deck_Dict
        elevation = float(np.atleast_1d(member["z"])[0]) * self.depth
        y = half_breadth_at_z(self.hull, elevation)
        path = line_edge((0.0, y, elevation), (self.length, y, elevation))

        add_stiffener(
            self.structure,
            self.identifiers,
            path,
            member,
            parent_id="Side_Shell",
            object_id_root=f"Side_Shell_Side_Stiffener_Z_{elevation}",
        )

    def _add_decks_and_members(self) -> None:
        """Build the class-specific top deck and the common inner bottom."""

        definition = self.definition

        if int(definition.Ship_Class[1]) == 0:
            top_deck = {
                "z": self.depth,
                "t": self.structural_parameters["Top_Deck_Thickness"],
            }

            if int(definition.bit_Hatch) == 1:
                top_deck["LH_Dict"] = definition.Deck_Lightening_Hole_Dict

            self._add_deck(
                top_deck,
                definition.Deck_Long_Dict,
                definition.Deck_Beam_Dict,
            )
        else:
            self._add_container_top_deck()

        inner_bottom = {
            "z": self.double_bottom,
            "t": self.structural_parameters["Inner_Bottom_Thickness"],
        }
        self._add_deck(
            inner_bottom,
            definition.Inner_Bottom_Stiffener_Dict,
            definition.Inner_Bottom_Trans_Frame_Dict,
        )

    def _add_bottom_longitudinals(self) -> None:
        """Add bottom girders and stiffeners, capped at the inner bottom."""

        for member in (
            self.definition.Bottom_Girder_Dict,
            self.definition.Bottom_Stiffener_Dict,
        ):
            y_positions = np.atleast_1d(member["y"]).astype(float)

            for y_fraction in y_positions:
                y = float(y_fraction) * self.half_breadth
                shell_z = shell_height_at_y(self.hull, y)

                # Preserve the Rhino correction that prevents a bottom member
                # from passing through the inner-bottom plate near the bilge.
                corrected = dict(member)
                corrected["h"] = min(
                    float(member["h"]),
                    1000.0 * (self.double_bottom - shell_z),
                )
                path = line_edge(
                    (0.0, y, shell_z),
                    (self.length, y, shell_z),
                )
                add_stiffener(
                    self.structure,
                    self.identifiers,
                    path,
                    corrected,
                    parent_id="Bottom_Shell",
                    object_id_root=f"Bottom_Shell_Long_Stiffener_Y_{y}",
                )

    def _add_transverse_bulkheads(self) -> None:
        """Add four hold bulkheads and both orthogonal stiffener families."""

        thickness = self.structural_parameters["Bulkhead_Thickness"]

        for x_fraction in np.atleast_1d(self.definition.x_bulkheads):
            x = float(x_fraction) * self.length
            face = make_transverse_bulkhead_face(self.hull, x)
            bulkhead = self._add_surface(
                face,
                element_class="Transverse_Bulkhead",
                element_type="Transverse_Bulkhead",
                parent_id="Hull",
                x_loc=x,
                y_loc=0.0,
                z_loc=0.0,
                direction=(0, 1, 0),
                rotation=90.0,
                length_1=self.half_breadth,
                length_2=self.depth,
                thickness_mm=thickness,
                object_id=f"Trans_Bulkhead_X_{x}",
            )

            for member in self.definition.BLKHD_Vert_Stiff_Dict:
                for y_fraction in np.atleast_1d(member["y"]):
                    y = float(y_fraction) * self.half_breadth
                    bottom = shell_height_at_y(self.hull, y)
                    path = line_edge((x, y, bottom), (x, y, self.depth))
                    add_stiffener(
                        self.structure,
                        self.identifiers,
                        path,
                        member,
                        parent_id=bulkhead.object_id,
                        object_id_root=(
                            f"{bulkhead.object_id}_Vert_Stiffener_Y_{y}"
                        ),
                    )

            for member in self.definition.BLKHD_Trans_Stiff_Dict:
                for z_fraction in np.atleast_1d(member["z"]):
                    z = float(z_fraction) * self.depth
                    breadth = half_breadth_at_z(self.hull, z)
                    path = line_edge((x, 0.0, z), (x, breadth, z))
                    add_stiffener(
                        self.structure,
                        self.identifiers,
                        path,
                        member,
                        parent_id=bulkhead.object_id,
                        object_id_root=(
                            f"{bulkhead.object_id}_Trans_Stiffener_Z_{z}"
                        ),
                    )

    def _add_side_shell_longitudinals(self) -> None:
        """Add longitudinal members directly on the analytical outer shell."""

        member = self.definition.Side_Shell_Stiff_Dict

        for z_fraction in np.atleast_1d(member["z"]):
            z = float(z_fraction) * self.depth
            y = half_breadth_at_z(self.hull, z)
            path = line_edge((0.0, y, z), (self.length, y, z))
            add_stiffener(
                self.structure,
                self.identifiers,
                path,
                member,
                parent_id="Side_Shell",
                object_id_root=f"Side_Shell_Side_Stiffener_Z_{z}",
            )

    def _add_longitudinal_bulkhead(
        self,
        y_fraction: float,
        thickness_mm: float,
        longitudinal_members: Sequence[dict[str, Any]],
        vertical_members: Sequence[dict[str, Any]],
        *,
        plate_class: str = "Longitudinal_Bulkhead",
        object_id_root: str = "Long_Bulkhead",
    ) -> StructuralElement:
        """Add one longitudinal plate and its two member directions.

        The inner-side shell and a cargo-space longitudinal bulkhead use the
        same geometric constructor, but they are not the same Rule member.
        Keeping the plate role explicit here prevents the generated inner
        hull from incorrectly selecting the MVR 3-2-1/3.9.4 bulkhead branch.
        """

        y = float(y_fraction) * self.half_breadth
        bottom = shell_height_at_y(self.hull, y)
        face = make_longitudinal_bulkhead_face(self.hull, y)
        bulkhead = self._add_surface(
            face,
            element_class=plate_class,
            element_type=plate_class,
            parent_id="Hull",
            x_loc=0.0,
            y_loc=y,
            z_loc=0.0,
            direction=(1, 0, 0),
            rotation=0.0,
            length_1=self.length,
            length_2=self.depth,
            thickness_mm=thickness_mm,
            object_id=f"{object_id_root}_Y_{y}",
        )

        for member in vertical_members:
            for x_fraction in np.atleast_1d(member["x"]):
                x = float(x_fraction) * self.length
                path = line_edge((x, y, bottom), (x, y, self.depth))
                add_stiffener(
                    self.structure,
                    self.identifiers,
                    path,
                    member,
                    parent_id=bulkhead.object_id,
                    object_id_root=(
                        f"{bulkhead.object_id}_Vert_Stiffener_X_{x}"
                    ),
                )

        for member in longitudinal_members:
            for z_fraction in np.atleast_1d(member["z"]):
                z = float(z_fraction) * self.depth

                if z < bottom - GEOMETRY_TOLERANCE:
                    continue

                path = line_edge((0.0, y, z), (self.length, y, z))
                add_stiffener(
                    self.structure,
                    self.identifiers,
                    path,
                    member,
                    parent_id=bulkhead.object_id,
                    object_id_root=(
                        f"{bulkhead.object_id}_Long_Stiffener_Z_{z}"
                    ),
                )

        return bulkhead

    def _add_internal_longitudinal_structure(self) -> None:
        """Add the required inner-side shell and any explicit bulkheads."""

        definition = self.definition
        inner_side_thickness = float(
            self.structural_parameters["Inner_Side_Shell_Thickness"]
        )

        if inner_side_thickness > 0.0:
            inner_side_y_fraction = (
                self.half_breadth
                - millimetres_to_metres(
                    definition.Web_Frame_Dict["Side_Dict"]["h"]
                )
            ) / self.half_breadth
            self._add_longitudinal_bulkhead(
                inner_side_y_fraction,
                inner_side_thickness,
                definition.Inner_Side_Shell_Long_Stiff_Dict,
                definition.Inner_Side_Shell_Vert_Stiff_Dict,
                plate_class="Inner_Side_Shell",
                object_id_root="Inner_Side_Shell",
            )

        if int(definition.bit_Long_Bulkheads) == 1:
            for y_fraction in np.atleast_1d(
                definition.Long_Bulkhead_Params["y"]
            ):
                self._add_longitudinal_bulkhead(
                    float(y_fraction),
                    definition.Long_Bulkhead_Params["t"],
                    definition.LONG_BLKHD_Long_Stiff_Dict,
                    definition.LONG_BLKHD_Vert_Stiff_Dict,
                )

    def _add_shell_strake(self) -> None:
        """Add the upper side-shell shear strake."""

        strake_parameters = {
            "z": self.depth
            - millimetres_to_metres(self.definition.Shear_Strake_Dict["h"]),
            "t": self.definition.Shear_Strake_Dict["t"],
            "h": self.definition.Shear_Strake_Dict["h"],
        }
        shape = make_shell_strake(self.hull, strake_parameters)
        self._add_surface(
            shape,
            element_class="Shell_Strake",
            element_type="Shell_Strake",
            parent_id="Side_Shell",
            x_loc=0.0,
            y_loc=self.half_breadth,
            z_loc=strake_parameters["z"],
            direction=(1, 0, 0),
            rotation=0.0,
            length_1=self.length,
            length_2=millimetres_to_metres(strake_parameters["h"]),
            thickness_mm=strake_parameters["t"],
            object_id="Shell_Strake",
        )

    def _bottom_transverse_path(self, x: float) -> cq.Wire:
        """Return one unsplit physical path over the complete bottom shell."""

        edges = _section_edges_between_z(
            self.hull,
            0.0,
            self.double_bottom,
            x=x,
            include_bottom=True,
        )

        return self._wire_from_edges(edges)

    def _side_transverse_path(self, x: float) -> cq.Wire:
        """Return the side-shell path from double bottom to top deck."""

        edges = _section_edges_between_z(
            self.hull,
            self.double_bottom,
            self.depth,
            x=x,
        )

        return self._wire_from_edges(edges)

    def _add_transverse_frames(self) -> None:
        """Add web-frame floors plus major and intermediate shell frames."""

        web_frames = self.definition.Web_Frame_Dict

        for x_fraction in np.atleast_1d(web_frames["x"]):
            x = float(x_fraction) * self.length
            floor_shape = make_transverse_floor_face(
                self.hull,
                self.structural_parameters,
                x,
            )
            self._add_surface(
                floor_shape,
                element_class="Transverse_Floor",
                element_type="Transverse_Floor",
                parent_id="Bottom_Shell",
                x_loc=x,
                y_loc=0.0,
                z_loc=0.0,
                direction=(0, 1, 0),
                rotation=90.0,
                length_1=self.half_breadth,
                length_2=self.double_bottom,
                thickness_mm=web_frames["Bottom_Dict"]["t"],
                object_id=f"Trans_Bottom_X_{x}",
            )
            add_stiffener(
                self.structure,
                self.identifiers,
                self._side_transverse_path(x),
                web_frames["Side_Dict"],
                parent_id="Side_Shell",
                object_id_root=f"Side_Shell_frame_X_{x}",
            )

        # Keep one row and one joined mesh wire for each physical bottom frame.
        # The Rhino post-process split that wire and duplicated its CSV row;
        # CadQuery retains the physical-member identity requested by this port.
        for x_fraction in np.atleast_1d(self.definition.x_trans_frames):
            x = float(x_fraction) * self.length
            add_stiffener(
                self.structure,
                self.identifiers,
                self._side_transverse_path(x),
                self.definition.Trans_Frame_Dict,
                parent_id="Side_Shell",
                object_id_root=f"Side_Shell_frame_X_{x}",
            )
            add_stiffener(
                self.structure,
                self.identifiers,
                self._bottom_transverse_path(x),
                self.definition.Trans_Frame_Bottom_Dict,
                parent_id="Bottom_Shell",
                object_id_root=f"Bottom_Shell_frame_X_{x}",
            )

    def _add_brackets(self) -> None:
        """Add every class-specific floor/deck bracket at web-frame stations."""

        for bracket in self.definition.Bracket_Dict:
            faces = make_bracket_faces(bracket)
            l1 = np.asarray(bracket["L1"], dtype=float) / 1000.0
            l2 = np.asarray(bracket["L2"], dtype=float) / 1000.0
            l1_norm = float(np.linalg.norm(l1))
            l2_norm = float(np.linalg.norm(l2))
            direction = np.abs(l1 / l1_norm).astype(int)
            num_sides = int(bracket["num_sides"])

            for face, vertex in zip(faces, bracket["Vertex"], strict=True):
                self._add_surface(
                    face,
                    element_class=str(bracket["Class"]),
                    element_type=f"Bracket_{num_sides}_sides",
                    parent_id="Bracket",
                    x_loc=float(vertex[0]),
                    y_loc=float(vertex[1]),
                    z_loc=float(vertex[2]),
                    direction=direction,
                    rotation=0.0,
                    length_1=l1_norm,
                    length_2=l2_norm,
                    thickness_mm=bracket["t"],
                    object_id=(
                        f"Bracket_{num_sides}_sides_Vertex_"
                        f"X_{float(vertex[0])}_Y_{float(vertex[1])}_"
                        f"Z_{float(vertex[2])}"
                    ),
                )

    def _panel_stiffener_positions(
        self,
        panel: dict[str, Any],
        *,
        threshold: float = 0.05,
    ) -> list[tuple[float, float]]:
        """Place panel longitudinals with the effective legacy snap behavior.

        The Rhino helper calculates provisional y and z snaps, but its final
        array assignment overwrites every z-only adjustment.  Its caller then
        intersects the panel with a plane at the returned y coordinate, so a
        y snap moves the point along the panel while a z-only snap has no
        effect on generated geometry.  Preserve that quirk for exact reference
        parity; it can be corrected later as a separate design-rule change.
        """

        count = int(panel["num_stiffeners"])
        start = np.asarray(panel["start"], dtype=float)
        delta_y = millimetres_to_metres(panel["L1"])
        delta_z = millimetres_to_metres(panel["L2"])
        y_bulkhead_members = (
            np.asarray(self.definition.y_vert_stiffeners, dtype=float)
            * self.half_breadth
        )
        positions: list[tuple[float, float]] = []

        # Use the same NumPy linspace construction as Rhino.  Algebraically
        # equivalent fraction arithmetic can differ in the final binary digit,
        # which would then leak into the unformatted coordinate-based ID.
        y_positions = np.linspace(
            float(start[1]),
            float(start[1] + delta_y),
            count + 2,
        )[1:-1]
        z_positions = np.linspace(
            float(start[2]),
            float(start[2] + delta_z),
            count + 2,
        )[1:-1]

        for y_position, z_position in zip(
            y_positions,
            z_positions,
            strict=True,
        ):
            y = float(y_position)
            z = float(z_position)

            # Only y alignment reaches the Rhino-generated intersection path.
            # In particular, do not preserve a nearby z-only snap here.
            if abs(delta_y) > GEOMETRY_TOLERANCE:
                nearest_y = float(
                    y_bulkhead_members[np.argmin(np.abs(y_bulkhead_members - y))]
                )

                if abs(nearest_y - y) < threshold:
                    y = nearest_y
                    fraction = (y - float(start[1])) / delta_y
                    z = float(start[2] + fraction * delta_z)

            positions.append((y, z))

        return positions

    def _add_panels(self, panels: Sequence[dict[str, Any]]) -> None:
        """Add container/hopper panels and their longitudinal stiffeners."""

        for panel in panels:
            start = np.asarray(panel["start"], dtype=float)
            delta_y_mm = float(panel["L1"])
            delta_z_mm = float(panel["L2"])
            panel_depth = millimetres_to_metres(
                hypot(delta_y_mm, delta_z_mm)
            )
            panel_rotation = degrees(atan2(delta_y_mm, delta_z_mm))
            face = make_panel_face(self.hull, panel)
            panel_element = self._add_surface(
                face,
                element_class="Hopper Pannel",
                element_type="Stiffener",
                parent_id="Hull",
                x_loc=0.0,
                y_loc=float(start[1]),
                z_loc=float(start[2]),
                direction=(1, 0, 0),
                rotation=panel_rotation,
                length_1=self.length,
                length_2=panel_depth,
                thickness_mm=panel["t_pan"],
                bit_1d_element=False,
                object_id=(
                    f"Hopper_Pannel_Y_{float(start[1])}_Z_{float(start[2])}"
                ),
            )
            stiffener = {
                "h": panel["h"],
                "t": panel["t"],
                "w": panel["w"],
                "rot": panel_rotation + float(panel["rot"]),
                "dir": [1, 0, 0],
                "bit_TorC": panel["bit_TorC"],
                "1D_element": panel["1D_element"],
                "Class": "Hopper_Pannel_Longitudinal_Stiffener",
            }

            for y, z in self._panel_stiffener_positions(panel):
                path = line_edge((0.0, y, z), (self.length, y, z))

                if abs(delta_y_mm) > GEOMETRY_TOLERANCE:
                    stiffener_root = f"{panel_element.object_id}_Stiffener_Y_{y}"
                else:
                    stiffener_root = f"{panel_element.object_id}_Stiffener_Z_{z}"

                add_stiffener(
                    self.structure,
                    self.identifiers,
                    path,
                    stiffener,
                    parent_id=panel_element.object_id,
                    object_id_root=stiffener_root,
                )

    def _add_class_specific_panels(self) -> None:
        """Dispatch to the panel family selected by the one-hot ship class."""

        if int(self.definition.Ship_Class[1]) == 1:
            self._add_panels(self.definition.ContainerShip_Pannel_Dict)
        elif int(self.definition.Ship_Class[2]) == 1:
            self._add_panels(self.definition.BulkCarrier_Hopper_Dict)

    def build(self) -> ShipStructure:
        """Build all structure families, derive properties, and return the model."""

        build_start = perf_counter()

        # Reset mutable build state so an explicit rebuild remains deterministic.
        self.structure = ShipStructure(
            ship_id=self.ship_id,
            source_parameters=self.definition.parameters_by_name(),
        )
        self.identifiers = IdentifierFactory(self.ship_id)
        self.structure.timings["parameter_interpretation"] = (
            self.parameter_interpretation_time
        )

        with self._stage("shell_plating"):
            self._add_shell_plating()

        with self._stage("decks_and_deck_members"):
            self._add_decks_and_members()

        with self._stage("bottom_longitudinals"):
            self._add_bottom_longitudinals()

        with self._stage("transverse_bulkheads"):
            self._add_transverse_bulkheads()

        with self._stage("side_shell_longitudinals"):
            self._add_side_shell_longitudinals()

        with self._stage("internal_longitudinal_structure"):
            self._add_internal_longitudinal_structure()

        with self._stage("shell_strake"):
            self._add_shell_strake()

        with self._stage("transverse_frames"):
            self._add_transverse_frames()

        with self._stage("brackets"):
            self._add_brackets()

        with self._stage("class_specific_panels"):
            self._add_class_specific_panels()

        with self._stage("derived_properties"):
            calculate_properties(self.structure)

        self.structure.timings["total_build"] = perf_counter() - build_start

        return self.structure


def build_structure(
    params_or_definition: Any,
    ship_id: str,
) -> ShipStructure:
    """Build a complete structure from a legacy row or interpreted definition."""

    return CadQueryStructureBuilder(params_or_definition, ship_id).build()


__all__ = ["CadQueryStructureBuilder", "build_structure"]
