"""ABS MVR longitudinal-bulkhead shear-flow calculation for GenISys.

This module implements the thin-walled, multicell calculation selected for
ABS MVR Part 3, Chapter 2, Section 1/3.9.4.  The governing equations are the
Appendix 3-2-A1 relations:

* ``N = F*m/I + N_i`` for shear flow along the section; and
* simultaneous zero-twist compatibility for every closed cell.

The generated structural-elements file is a starboard half-model.  This
module completes the port side, retains a centerline member only once, uses
the actual generated plate ordinates, and solves the complete connected wall
network by minimum complementary energy.  That weighted network solution is
the graph form of the Appendix Bredt zero-twist equations: equilibrium fixes
the open-section flow and minimization of ``integral(N**2/t ds)`` supplies the
otherwise-indeterminate cell constants.

The solver is intentionally a thin-walled beam calculation, not FEA.  It is
appropriate to the detailed early-design scope recorded in
``docs/ABS_Structural_Rule_Assumptions.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, pi, sin

import numpy as np
import pandas as pd


POINT_TOLERANCE_M = 1.0e-7
MINIMUM_PLATE_THICKNESS_M = 1.0e-9

# These are the generated plating surfaces that transmit longitudinal shear
# around and between the hull-girder cells.  Shell strakes and all longitudinal
# stiffeners/girders remain effective as concentrated boom area, but they do
# not create a second wall on top of their parent plating.
SHEAR_WALL_CLASSES = frozenset(
    {
        'Deck',
        'Bottom_Shell',
        'Side_Shell',
        'Inner_Side_Shell',
        'Longitudinal_Bulkhead',
        'Hopper Pannel',
    }
)


@dataclass(frozen=True)
class _RawWall:
    """One straight piece of a generated thin-walled section."""

    start: tuple[float, float]
    end: tuple[float, float]
    thickness_m: float
    member_class: str
    source_key: str


def actual_longitudinal_bulkhead_count(structural_elements: pd.DataFrame) -> int:
    """Return the full-ship count represented by the half-model CAD table.

    A plate on ``y=0`` is one centerline bulkhead.  Every distinct positive-y
    plate has a corresponding port plate and therefore represents two actual
    bulkheads.  The supported generated arrangements consequently resolve to
    exactly one, two, or three longitudinal bulkheads.

    Archived Rhino structural-element files used ``Longitudinal_Bulkhead`` for
    the inner-side-shell plate even though the plate's attached stiffeners were
    correctly assigned ``Inner_Side_Shell_*`` classes.  Those parent plates
    are inner-side shells under ASM-019 and must not select the MVR 3.9.4
    longitudinal-bulkhead branch.  Current CadQuery files label the plate
    itself ``Inner_Side_Shell`` and therefore do not require this compatibility
    mapping.
    """

    if structural_elements.empty:
        return 0

    rows = structural_elements.loc[
        structural_elements['Class'].astype(str).eq(
            'Longitudinal_Bulkhead'
        )
    ]

    if rows.empty:
        return 0

    # Identify the legacy Rhino inner-side-shell signature by its explicit
    # parent/child relationship.  Do not infer the role from transverse
    # position because an off-center actual longitudinal bulkhead is a valid
    # MVR topology and must continue to be counted.
    if {'Class', 'Parent_Struct'}.issubset(structural_elements.columns):
        inner_side_children = structural_elements.loc[
            structural_elements['Class'].astype(str).str.startswith(
                'Inner_Side_Shell_'
            )
        ]
        inner_side_parent_ids = set(
            inner_side_children['Parent_Struct'].dropna().astype(str)
        )

        if inner_side_parent_ids:
            legacy_inner_side_mask = pd.Series(
                False,
                index=rows.index,
            )

            # Rhino child rows reference Object_ID.  Include struct_rsObject
            # as a second accepted identifier for compatible imported tables.
            for identifier_column in ('Object_ID', 'struct_rsObject'):
                if identifier_column in rows.columns:
                    legacy_inner_side_mask |= rows[
                        identifier_column
                    ].astype(str).isin(inner_side_parent_ids)

            rows = rows.loc[~legacy_inner_side_mask]

    if rows.empty:
        return 0

    ordinates_m = np.unique(
        np.round(rows['y_loc'].astype(float).to_numpy(), decimals=9)
    )

    return int(
        sum(1 if np.isclose(y_m, 0.0, atol=POINT_TOLERANCE_M) else 2
            for y_m in ordinates_m)
    )


def _shell_height_at_y(hull: dict, y_m: float) -> float:
    """Return the exact flat-bottom/quarter-bilge shell elevation."""

    half_breadth_m = 0.5*float(hull['B'])
    bilge_radius_m = float(hull['R_b'])
    tangent_y_m = half_breadth_m-bilge_radius_m
    y_m = abs(float(y_m))

    if y_m <= tangent_y_m:
        return 0.0

    radial_y_m = y_m-tangent_y_m
    radial_z_m = np.sqrt(
        max(0.0, bilge_radius_m**2.0-radial_y_m**2.0)
    )

    return bilge_radius_m-radial_z_m


def _source_key(row_index: object, member_class: str) -> str:
    return f'{row_index}:{member_class}'


def _outer_shell_walls(
    hull: dict,
    effective_members: pd.DataFrame,
) -> list[_RawWall]:
    """Discretize the exact generated outer contour at the double-bottom cut."""

    bottom = effective_members.loc[
        effective_members['Class'].astype(str).eq('Bottom_Shell')
    ].iloc[0]
    side = effective_members.loc[
        effective_members['Class'].astype(str).eq('Side_Shell')
    ].iloc[0]

    half_breadth_m = 0.5*float(hull['B'])
    depth_m = float(hull['D'])
    bilge_radius_m = float(hull['R_b'])
    tangent_y_m = half_breadth_m-bilge_radius_m

    # The structural generator splits bottom and side plating at the generated
    # inner-bottom elevation.  Recover that cut from the lowest Deck row.
    deck_elevations_m = effective_members.loc[
        effective_members['Class'].astype(str).eq('Deck'),
        'centroid_cx_11',
    ].astype(float)
    double_bottom_m = float(deck_elevations_m.min())

    points = [(0.0, 0.0), (tangent_y_m, 0.0)]

    # Sixteen chords preserve the quarter-circle geometry without introducing
    # a costly geometric dependency in every parallel constraint evaluation.
    arc_angles = list(np.linspace(-0.5*pi, 0.0, 17)[1:])

    if 0.0 < double_bottom_m < bilge_radius_m:
        split_angle = asin(
            (double_bottom_m-bilge_radius_m)/bilge_radius_m
        )
        arc_angles.append(split_angle)

    for theta in sorted(set(round(value, 14) for value in arc_angles)):
        points.append(
            (
                tangent_y_m+bilge_radius_m*cos(theta),
                bilge_radius_m+bilge_radius_m*sin(theta),
            )
        )

    if double_bottom_m > bilge_radius_m and double_bottom_m < depth_m:
        points.append((half_breadth_m, double_bottom_m))

    points.append((half_breadth_m, depth_m))
    points = sorted(
        set((round(y_m, 12), round(z_m, 12)) for y_m, z_m in points),
        key=lambda point: (point[1], point[0]),
    )

    walls = []

    for start, end in zip(points[:-1], points[1:]):
        midpoint_z_m = 0.5*(start[1]+end[1])
        source = bottom if midpoint_z_m < double_bottom_m else side
        member_class = str(source['Class'])
        walls.append(
            _RawWall(
                start=start,
                end=end,
                thickness_m=float(source['Thickness']),
                member_class=member_class,
                source_key=_source_key(source.name, member_class),
            )
        )

    return walls


def _straight_generated_walls(
    hull: dict,
    effective_members: pd.DataFrame,
) -> list[_RawWall]:
    """Recover every non-shell generated wall from its section metadata."""

    walls = []

    for row_index, row in effective_members.iterrows():
        member_class = str(row['Class'])

        if member_class not in SHEAR_WALL_CLASSES:
            continue

        if member_class in {'Bottom_Shell', 'Side_Shell'}:
            continue

        thickness_m = float(row['Thickness'])
        centroid_y_m = float(row['centroid_cx_22'])
        centroid_z_m = float(row['centroid_cx_11'])
        length_m = float(row['L2'])

        if member_class == 'Deck':
            # After the MVR hatch deduction, these centroid/length values
            # describe the actual surviving side-deck strip.
            start = (centroid_y_m-0.5*length_m, centroid_z_m)
            end = (centroid_y_m+0.5*length_m, centroid_z_m)
        elif member_class in {
            'Inner_Side_Shell',
            'Longitudinal_Bulkhead',
        }:
            start = (
                centroid_y_m,
                _shell_height_at_y(hull, centroid_y_m),
            )
            end = (centroid_y_m, float(hull['D']))
        else:
            # Hopper metadata retains the generated start point.  The exact
            # opposite endpoint follows from the calculated cross-section
            # centroid and does not depend on a rotation convention.
            start = (float(row['y_loc']), float(row['z_loc']))
            end = (
                2.0*centroid_y_m-start[0],
                2.0*centroid_z_m-start[1],
            )

        walls.append(
            _RawWall(
                start=start,
                end=end,
                thickness_m=thickness_m,
                member_class=member_class,
                source_key=_source_key(row_index, member_class),
            )
        )

    return walls


def _mirror_walls(half_walls: list[_RawWall]) -> list[_RawWall]:
    """Complete the port side while retaining a centerline wall once."""

    walls = list(half_walls)

    for wall in half_walls:
        if (
            abs(wall.start[0]) <= POINT_TOLERANCE_M
            and abs(wall.end[0]) <= POINT_TOLERANCE_M
        ):
            continue

        walls.append(
            _RawWall(
                start=(-wall.start[0], wall.start[1]),
                end=(-wall.end[0], wall.end[1]),
                thickness_m=wall.thickness_m,
                member_class=wall.member_class,
                source_key=wall.source_key,
            )
        )

    return walls


def _intersection_parameters(first: _RawWall, second: _RawWall):
    """Return line parameters at one proper segment intersection."""

    p = np.asarray(first.start, dtype=float)
    r = np.asarray(first.end, dtype=float)-p
    q = np.asarray(second.start, dtype=float)
    s = np.asarray(second.end, dtype=float)-q
    cross_rs = r[0]*s[1]-r[1]*s[0]

    if abs(cross_rs) <= POINT_TOLERANCE_M:
        return None

    q_minus_p = q-p
    first_parameter = (
        q_minus_p[0]*s[1]-q_minus_p[1]*s[0]
    )/cross_rs
    second_parameter = (
        q_minus_p[0]*r[1]-q_minus_p[1]*r[0]
    )/cross_rs

    if (
        -POINT_TOLERANCE_M <= first_parameter <= 1.0+POINT_TOLERANCE_M
        and -POINT_TOLERANCE_M <= second_parameter <= 1.0+POINT_TOLERANCE_M
    ):
        return (
            min(1.0, max(0.0, first_parameter)),
            min(1.0, max(0.0, second_parameter)),
        )

    return None


def _split_wall_network(walls: list[_RawWall]):
    """Split walls at all physical intersections and return graph arrays."""

    parameters = [{0.0, 1.0} for _ in walls]

    for first_index, first in enumerate(walls):
        for second_index in range(first_index+1, len(walls)):
            crossing = _intersection_parameters(first, walls[second_index])

            if crossing is None:
                continue

            parameters[first_index].add(crossing[0])
            parameters[second_index].add(crossing[1])

    node_lookup = {}
    node_coordinates = []
    edges = []

    def node_index(point):
        key = (
            round(float(point[0])/POINT_TOLERANCE_M),
            round(float(point[1])/POINT_TOLERANCE_M),
        )

        if key not in node_lookup:
            node_lookup[key] = len(node_coordinates)
            node_coordinates.append((float(point[0]), float(point[1])))

        return node_lookup[key]

    for wall, wall_parameters in zip(walls, parameters):
        start = np.asarray(wall.start, dtype=float)
        delta = np.asarray(wall.end, dtype=float)-start
        ordered = sorted(wall_parameters)

        for lower, upper in zip(ordered[:-1], ordered[1:]):
            if upper-lower <= POINT_TOLERANCE_M:
                continue

            first_point = start+lower*delta
            second_point = start+upper*delta
            length_m = float(np.linalg.norm(second_point-first_point))

            if length_m <= POINT_TOLERANCE_M:
                continue

            edges.append(
                {
                    'start': node_index(first_point),
                    'end': node_index(second_point),
                    'length_m': length_m,
                    'thickness_m': max(
                        wall.thickness_m,
                        MINIMUM_PLATE_THICKNESS_M,
                    ),
                    'member_class': wall.member_class,
                    'source_key': wall.source_key,
                }
            )

    return np.asarray(node_coordinates, dtype=float), edges


def _nearest_node(
    node_coordinates: np.ndarray,
    y_m: float,
    z_m: float,
) -> int:
    distances_squared = (
        (node_coordinates[:, 0]-float(y_m))**2.0
        +(node_coordinates[:, 1]-float(z_m))**2.0
    )

    return int(np.argmin(distances_squared))


def _add_symmetric_source(
    source: np.ndarray,
    node_coordinates: np.ndarray,
    y_m: float,
    z_m: float,
    value: float,
    mirror_factor: float,
) -> None:
    """Apply a full-section boom/correction without breaking symmetry."""

    if mirror_factor <= 1.0 or abs(float(y_m)) <= POINT_TOLERANCE_M:
        source[_nearest_node(node_coordinates, y_m, z_m)] += value
        return

    source[_nearest_node(node_coordinates, y_m, z_m)] += 0.5*value
    source[_nearest_node(node_coordinates, -float(y_m), z_m)] += 0.5*value


def appendix_a1_unit_shear_stresses(
    hull: dict,
    effective_members: pd.DataFrame,
    neutral_axis_m: float,
    inertia_m4: float,
) -> dict[str, float | int]:
    """Return MVR Appendix A1 stresses produced by one kN of total shear.

    Results are returned in ``kN/cm^2 per kN``.  Multiplying by the governing
    absolute still-water-plus-wave shear force produces the nominal stresses
    compared with the MVR 3-2-1/3.9.1 limit.
    """

    if inertia_m4 <= 0.0:
        raise ValueError('Hull-girder inertia must be positive.')

    half_walls = _outer_shell_walls(hull, effective_members)
    half_walls.extend(_straight_generated_walls(hull, effective_members))
    walls = _mirror_walls(half_walls)
    node_coordinates, edges = _split_wall_network(walls)

    node_count = len(node_coordinates)
    edge_count = len(edges)

    if node_count < 2 or edge_count < 1:
        raise ValueError('The generated shear-wall network is empty.')

    incidence = np.zeros((node_count, edge_count), dtype=float)
    source = np.zeros(node_count, dtype=float)

    # Integrate Z*t*ds on every generated wall segment.  Each source row is
    # subsequently corrected to the exact gross area and centroid retained in
    # StructureCalcs, so chord discretization cannot alter the section first
    # moment used by the Rule equation.
    geometric_first_moment_by_source = {}

    for edge_index, edge in enumerate(edges):
        start_index = edge['start']
        end_index = edge['end']
        incidence[start_index, edge_index] = -1.0
        incidence[end_index, edge_index] = 1.0

        average_z_from_neutral_axis_m = (
            0.5
            * (
                node_coordinates[start_index, 1]
                + node_coordinates[end_index, 1]
            )
            - neutral_axis_m
        )
        first_moment_m3 = (
            edge['thickness_m']
            * edge['length_m']
            * average_z_from_neutral_axis_m
        )
        geometric_first_moment_by_source[edge['source_key']] = (
            geometric_first_moment_by_source.get(edge['source_key'], 0.0)
            + first_moment_m3
        )

        # A consistent half-to-each-end lump preserves the exact integrated
        # first moment while allowing the flow to change along a long panel.
        source[start_index] += 0.5*first_moment_m3/inertia_m4
        source[end_index] += 0.5*first_moment_m3/inertia_m4

    # Match each wall row to its exact effective full-section first moment.
    # This also applies the correct one-versus-two mirror factor.
    wall_row_keys = set(edge['source_key'] for edge in edges)

    for row_index, row in effective_members.iterrows():
        member_class = str(row['Class'])
        row_key = _source_key(row_index, member_class)
        mirror_factor = float(row['MVR_Mirror_Factor'])
        exact_first_moment_m3 = (
            mirror_factor
            * float(row['area_cx'])
            * (float(row['centroid_cx_11'])-neutral_axis_m)
        )

        if row_key in wall_row_keys:
            correction_m3 = (
                exact_first_moment_m3
                - geometric_first_moment_by_source.get(row_key, 0.0)
            )
            _add_symmetric_source(
                source,
                node_coordinates,
                row['centroid_cx_22'],
                row['centroid_cx_11'],
                correction_m3/inertia_m4,
                mirror_factor,
            )
            continue

        # Longitudinal stiffeners, girders, flanges, and strakes contribute
        # Appendix A1 boom area but do not create independent shear walls.
        _add_symmetric_source(
            source,
            node_coordinates,
            row['centroid_cx_22'],
            row['centroid_cx_11'],
            exact_first_moment_m3/inertia_m4,
            mirror_factor,
        )

    # Numerical roundoff is the only permitted imbalance because the neutral
    # axis was calculated from these same effective areas.  Remove that tiny
    # residual at the reference node before dropping its redundant equation.
    source[0] -= source.sum()

    flexibility = np.asarray(
        [edge['length_m']/edge['thickness_m'] for edge in edges],
        dtype=float,
    )
    inverse_sqrt_flexibility = 1.0/np.sqrt(flexibility)

    # Solve the minimum-energy equilibrium flow.  The incidence matrix has
    # one redundant equation for a connected network, so the first node is
    # the reference.  ``lstsq`` returns the minimum-norm transformed flow,
    # which is exactly the simultaneous zero-twist multicell solution.
    reduced_incidence = incidence[1:, :]
    transformed_matrix = (
        reduced_incidence*inverse_sqrt_flexibility[np.newaxis, :]
    )
    transformed_flow, _, matrix_rank, _ = np.linalg.lstsq(
        transformed_matrix,
        -source[1:],
        rcond=None,
    )
    shear_flow_kN_per_m_per_kN = (
        inverse_sqrt_flexibility*transformed_flow
    )

    equilibrium_residual = (
        incidence@shear_flow_kN_per_m_per_kN+source
    )

    if (
        matrix_rank != node_count-1
        or np.max(np.abs(equilibrium_residual)) > 1.0e-8
    ):
        raise ValueError(
            'MVR Appendix A1 wall topology is disconnected or singular.'
        )

    stress_by_class = {}

    for edge, shear_flow in zip(edges, shear_flow_kN_per_m_per_kN):
        nominal_stress_kN_cm2_per_kN = (
            abs(shear_flow)/edge['thickness_m']/10000.0
        )
        member_class = edge['member_class']
        stress_by_class[member_class] = max(
            stress_by_class.get(member_class, 0.0),
            nominal_stress_kN_cm2_per_kN,
        )

    return {
        'Side_Shell_kN_cm2_per_kN': stress_by_class.get(
            'Side_Shell',
            0.0,
        ),
        'Longitudinal_Bulkhead_kN_cm2_per_kN': stress_by_class.get(
            'Longitudinal_Bulkhead',
            0.0,
        ),
        'Node_Count': node_count,
        'Wall_Segment_Count': edge_count,
        'Equilibrium_Residual_per_m': float(
            np.max(np.abs(equilibrium_residual))
        ),
        'Matrix_Rank': int(matrix_rank),
    }
