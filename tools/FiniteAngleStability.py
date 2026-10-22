"""Standalone finite-angle hydrostatics for a symmetric hull section grid.

The kernel consumes the same geometric contract used by the generated
GenISys hydrostatic grid:

* ``x_positions_m`` contains common longitudinal section positions;
* ``z_positions_m`` contains common vertical positions; and
* ``half_breadths_m[x_index, z_index]`` contains starboard offsets.

Each section is reflected about ``y = 0`` and closed at its lowest and highest
represented ordinates.  A heeled waterplane clips that polygon.  Section area
and first moments are then integrated longitudinally with the composite
trapezoidal rule, preserving the project's numerical-integration convention.

This module is not connected to :mod:`tools.ShipSystemDesign` or any dataset
runner.  It supplies the geometric KN/GZ foundation needed by HDF-005,
HDF-006, and HDF-085; it does not evaluate those regulatory constraints.
Free trim, finite-angle tank free-surface effects (HDF-092), deck-edge or
downflooding immersion, and intact-stability criteria remain outside this
kernel.
"""

from __future__ import annotations

import math

import numpy as np


def _validated_section_grid(
    x_positions_m,
    z_positions_m,
    half_breadths_m,
):
    """Return the section-grid arrays after checking the geometry contract."""

    x_positions = np.asarray(x_positions_m, dtype=float)
    z_positions = np.asarray(z_positions_m, dtype=float)
    half_breadths = np.asarray(half_breadths_m, dtype=float)

    # These checks enforce the physical section-grid contract.  They are not
    # project bounds and do not impose a particular hull size or resolution.
    if x_positions.ndim != 1 or len(x_positions) < 2:
        raise ValueError(
            "x_positions_m must contain at least two longitudinal stations."
        )

    if z_positions.ndim != 1 or len(z_positions) < 2:
        raise ValueError(
            "z_positions_m must contain at least two vertical ordinates."
        )

    if half_breadths.shape != (len(x_positions), len(z_positions)):
        raise ValueError(
            "half_breadths_m must have shape "
            "(len(x_positions_m), len(z_positions_m))."
        )

    if np.any(np.diff(x_positions) <= 0.0):
        raise ValueError("x_positions_m must be strictly increasing.")

    if np.any(np.diff(z_positions) <= 0.0):
        raise ValueError("z_positions_m must be strictly increasing.")

    if (
        not np.all(np.isfinite(x_positions))
        or not np.all(np.isfinite(z_positions))
        or not np.all(np.isfinite(half_breadths))
    ):
        raise ValueError("The symmetric hull section grid must be finite.")

    if np.any(half_breadths < 0.0):
        raise ValueError("Hull half-breadths cannot be negative.")

    return x_positions, z_positions, half_breadths


def _heel_angle_radians(heel_angle_deg: float) -> float:
    """Return a finite heel angle within the open interval (-90, 90)."""

    heel_angle_deg = float(heel_angle_deg)

    if not math.isfinite(heel_angle_deg) or abs(heel_angle_deg) >= 90.0:
        raise ValueError(
            "Heel angle must be finite and strictly between -90 and 90 deg."
        )

    return math.radians(heel_angle_deg)


def build_symmetric_section_polygon(
    z_positions_m,
    half_breadths_m,
) -> np.ndarray:
    """Build one closed full-breadth section polygon in ``(y, z)``.

    The starboard curve is traversed upward and the reflected port curve is
    traversed downward.  This produces a counter-clockwise polygon for an
    ordinary section.  Straight segments between grid ordinates are the same
    piecewise-linear geometry implied by trapezoidal section integration.
    """

    z_positions = np.asarray(z_positions_m, dtype=float)
    half_breadths = np.asarray(half_breadths_m, dtype=float)

    if (
        z_positions.ndim != 1
        or half_breadths.ndim != 1
        or len(z_positions) != len(half_breadths)
        or len(z_positions) < 2
    ):
        raise ValueError(
            "One section requires matching one-dimensional z and "
            "half-breadth arrays with at least two ordinates."
        )

    starboard = np.column_stack((half_breadths, z_positions))
    port = np.column_stack(
        (-half_breadths[::-1], z_positions[::-1])
    )

    return np.vstack((starboard, port))


def clip_polygon_below_heeled_waterplane(
    polygon_yz,
    heel_angle_deg: float,
    waterplane_offset_m: float,
) -> np.ndarray:
    """Clip a section polygon to the immersed side of a heeled waterplane.

    Positive heel means starboard (positive ``y``) down.  In vessel-fixed
    coordinates the unit upward vector is ``n = (0, -sin(phi), cos(phi))``.
    A point is immersed when:

    ``z*cos(phi) - y*sin(phi) <= waterplane_offset_m``.

    The normalized offset is the perpendicular height of the waterplane from
    the coordinate origin.  At zero heel it is the ordinary waterline height.
    """

    polygon = np.asarray(polygon_yz, dtype=float)
    heel_angle_rad = _heel_angle_radians(heel_angle_deg)
    waterplane_offset_m = float(waterplane_offset_m)

    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("A section polygon must contain (y, z) points.")

    if len(polygon) < 3:
        return np.empty((0, 2), dtype=float)

    if not math.isfinite(waterplane_offset_m):
        raise ValueError("Waterplane offset must be finite.")

    sin_heel = math.sin(heel_angle_rad)
    cos_heel = math.cos(heel_angle_rad)

    def signed_distance(point):

        # A non-positive result lies below or exactly on the waterplane.
        y_coordinate, z_coordinate = point

        return (
            z_coordinate*cos_heel
            - y_coordinate*sin_heel
            - waterplane_offset_m
        )

    clipped_points = []
    previous_point = polygon[-1]
    previous_distance = signed_distance(previous_point)
    previous_inside = previous_distance <= 0.0

    # Sutherland-Hodgman clipping against one linear half-plane preserves the
    # polygon order and inserts exact linear edge/waterplane intersections.
    for current_point in polygon:
        current_distance = signed_distance(current_point)
        current_inside = current_distance <= 0.0

        if current_inside != previous_inside:
            edge_fraction = (
                previous_distance
                / (previous_distance - current_distance)
            )
            intersection = previous_point + edge_fraction * (
                current_point - previous_point
            )
            clipped_points.append(intersection)

        if current_inside:
            clipped_points.append(current_point.copy())

        previous_point = current_point
        previous_distance = current_distance
        previous_inside = current_inside

    if not clipped_points:
        return np.empty((0, 2), dtype=float)

    return np.asarray(clipped_points, dtype=float)


def polygon_area_and_centroid(polygon_yz):
    """Return area and ``(y, z)`` centroid of a simple section polygon."""

    polygon = np.asarray(polygon_yz, dtype=float)

    if len(polygon) < 3:
        return 0.0, np.nan, np.nan

    y_coordinates = polygon[:, 0]
    z_coordinates = polygon[:, 1]
    next_y_coordinates = np.roll(y_coordinates, -1)
    next_z_coordinates = np.roll(z_coordinates, -1)

    cross_products = (
        y_coordinates*next_z_coordinates
        - next_y_coordinates*z_coordinates
    )
    signed_area = 0.5*float(np.sum(cross_products))

    # A clipped polygon with zero area is either dry or only tangent to the
    # waterplane.  Its buoyancy centroid is undefined and carries no moment.
    if signed_area == 0.0:
        return 0.0, np.nan, np.nan

    centroid_y_m = float(
        np.sum(
            (y_coordinates + next_y_coordinates) * cross_products
        )
        / (6.0*signed_area)
    )
    centroid_z_m = float(
        np.sum(
            (z_coordinates + next_z_coordinates) * cross_products
        )
        / (6.0*signed_area)
    )

    return abs(signed_area), centroid_y_m, centroid_z_m


def _build_section_polygons(z_positions, half_breadths):
    """Build every reflected section once for repeated equilibrium trials."""

    return [
        build_symmetric_section_polygon(
            z_positions,
            half_breadths[station_index],
        )
        for station_index in range(len(half_breadths))
    ]


def _integrate_prepared_sections(
    x_positions,
    section_polygons,
    heel_angle_deg,
    waterplane_offset_m,
):
    """Clip prepared sections and integrate their volume/first moments."""

    section_areas_m2 = np.zeros(len(x_positions), dtype=float)
    transverse_section_moments_m3 = np.zeros(
        len(x_positions),
        dtype=float,
    )
    vertical_section_moments_m3 = np.zeros(
        len(x_positions),
        dtype=float,
    )

    for station_index, section_polygon in enumerate(section_polygons):
        immersed_polygon = clip_polygon_below_heeled_waterplane(
            section_polygon,
            heel_angle_deg,
            waterplane_offset_m,
        )
        area_m2, centroid_y_m, centroid_z_m = polygon_area_and_centroid(
            immersed_polygon,
        )

        section_areas_m2[station_index] = area_m2

        if area_m2 > 0.0:
            transverse_section_moments_m3[station_index] = (
                area_m2*centroid_y_m
            )
            vertical_section_moments_m3[station_index] = (
                area_m2*centroid_z_m
            )

    # HDF-ASM-028: retain composite trapezoidal longitudinal integration for
    # volume and each first moment.  No higher-order quadrature is introduced
    # in this standalone finite-angle foundation.
    displacement_volume_m3 = float(
        np.trapezoid(section_areas_m2, x=x_positions)
    )

    if displacement_volume_m3 <= 0.0:
        return {
            "Displacement_Volume_m3": 0.0,
            "LCB_m": np.nan,
            "TCB_m": np.nan,
            "VCB_m": np.nan,
        }

    # As in Hydrostatics._integrate_section_properties, use the exact first
    # moment of the linearly varying area inside each trapezoidal volume cell.
    # Applying np.trapezoid directly to x*A would incorrectly concentrate the
    # area at cell endpoints and bias LCB for tapered sections.
    longitudinal_first_moment_m4 = 0.0

    for cell_index in range(1, len(x_positions)):
        x_0 = x_positions[cell_index - 1]
        x_1 = x_positions[cell_index]
        dx = x_1 - x_0
        area_0 = section_areas_m2[cell_index - 1]
        area_1 = section_areas_m2[cell_index]

        longitudinal_first_moment_m4 += dx/6.0 * (
            area_0*(2.0*x_0 + x_1)
            + area_1*(x_0 + 2.0*x_1)
        )

    transverse_first_moment_m4 = float(
        np.trapezoid(
            transverse_section_moments_m3,
            x=x_positions,
        )
    )
    vertical_first_moment_m4 = float(
        np.trapezoid(
            vertical_section_moments_m3,
            x=x_positions,
        )
    )

    return {
        "Displacement_Volume_m3": displacement_volume_m3,
        "LCB_m": float(
            longitudinal_first_moment_m4/displacement_volume_m3
        ),
        "TCB_m": transverse_first_moment_m4/displacement_volume_m3,
        "VCB_m": vertical_first_moment_m4/displacement_volume_m3,
    }


def calculate_immersed_hull_properties(
    *,
    x_positions_m,
    z_positions_m,
    half_breadths_m,
    heel_angle_deg: float,
    waterplane_offset_m: float,
) -> dict:
    """Return volume and center of buoyancy at a specified waterplane."""

    x_positions, z_positions, half_breadths = _validated_section_grid(
        x_positions_m,
        z_positions_m,
        half_breadths_m,
    )
    section_polygons = _build_section_polygons(
        z_positions,
        half_breadths,
    )

    properties = _integrate_prepared_sections(
        x_positions,
        section_polygons,
        heel_angle_deg,
        waterplane_offset_m,
    )
    properties.update(
        {
            "Heel_Angle_deg": float(heel_angle_deg),
            "Waterplane_Offset_m": float(waterplane_offset_m),
        }
    )

    return properties


def _waterplane_offset_bounds(
    z_positions,
    half_breadths,
    heel_angle_deg,
):
    """Return offsets that bracket dry through fully immersed geometry."""

    heel_angle_rad = _heel_angle_radians(heel_angle_deg)
    sin_heel = math.sin(heel_angle_rad)
    cos_heel = math.cos(heel_angle_rad)

    # Evaluate the normalized waterplane coordinate at both reflected shell
    # points.  The global extrema bracket every polygon vertex and therefore
    # the complete immersed-volume range represented by the section grid.
    starboard_projection = (
        z_positions[np.newaxis, :]*cos_heel
        - half_breadths*sin_heel
    )
    port_projection = (
        z_positions[np.newaxis, :]*cos_heel
        + half_breadths*sin_heel
    )

    lower_offset_m = float(
        min(
            np.min(starboard_projection),
            np.min(port_projection),
        )
    )
    upper_offset_m = float(
        max(
            np.max(starboard_projection),
            np.max(port_projection),
        )
    )

    return lower_offset_m, upper_offset_m


def _automatic_volume_tolerance_m3(
    target_volume_m3,
    full_volume_m3,
    x_count,
    z_count,
):
    """Return a floating-point-scale solve tolerance, not a policy margin."""

    volume_scale_m3 = max(
        1.0,
        abs(float(target_volume_m3)),
        abs(float(full_volume_m3)),
    )
    approximate_operation_count = max(
        1,
        (int(x_count) - 1)*(int(z_count) - 1),
    )

    return float(
        64.0
        * np.finfo(float).eps
        * volume_scale_m3
        * approximate_operation_count
    )


def solve_heeled_volume_equilibrium(
    *,
    x_positions_m,
    z_positions_m,
    half_breadths_m,
    target_displacement_volume_m3: float,
    heel_angle_deg: float,
    volume_tolerance_m3: float | None = None,
    max_iterations: int = 200,
) -> dict:
    """Solve heave at fixed heel and fixed trim by waterplane bisection.

    The only equilibrium equation in this function is displaced volume equal
    to target volume.  The waterplane remains parallel to the longitudinal
    axis, so longitudinal moment/free-trim equilibrium is not enforced.
    """

    x_positions, z_positions, half_breadths = _validated_section_grid(
        x_positions_m,
        z_positions_m,
        half_breadths_m,
    )
    heel_angle_rad = _heel_angle_radians(heel_angle_deg)
    target_displacement_volume_m3 = float(
        target_displacement_volume_m3
    )

    if (
        not math.isfinite(target_displacement_volume_m3)
        or target_displacement_volume_m3 <= 0.0
    ):
        raise ValueError(
            "Target displacement volume must be positive and finite."
        )

    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least one.")

    section_polygons = _build_section_polygons(
        z_positions,
        half_breadths,
    )
    lower_offset_m, upper_offset_m = _waterplane_offset_bounds(
        z_positions,
        half_breadths,
        heel_angle_deg,
    )
    full_properties = _integrate_prepared_sections(
        x_positions,
        section_polygons,
        heel_angle_deg,
        upper_offset_m,
    )
    full_volume_m3 = full_properties["Displacement_Volume_m3"]

    if volume_tolerance_m3 is None:
        volume_tolerance_m3 = _automatic_volume_tolerance_m3(
            target_displacement_volume_m3,
            full_volume_m3,
            len(x_positions),
            len(z_positions),
        )
    else:
        volume_tolerance_m3 = float(volume_tolerance_m3)

        if (
            not math.isfinite(volume_tolerance_m3)
            or volume_tolerance_m3 <= 0.0
        ):
            raise ValueError(
                "Volume tolerance must be positive and finite."
            )

    if target_displacement_volume_m3 > (
        full_volume_m3 + volume_tolerance_m3
    ):
        raise ValueError(
            "Target displacement exceeds the closed volume represented by "
            "the symmetric hull section grid."
        )

    # A target equal to the represented closed volume is already bracketed by
    # the fully immersed upper plane and needs no iteration.
    if abs(
        full_volume_m3 - target_displacement_volume_m3
    ) <= volume_tolerance_m3:
        equilibrium_properties = full_properties
        equilibrium_offset_m = upper_offset_m
        iteration_count = 0

    else:
        equilibrium_properties = None
        equilibrium_offset_m = np.nan
        iteration_count = 0

        for iteration_count in range(1, int(max_iterations) + 1):
            midpoint_offset_m = 0.5*(
                lower_offset_m + upper_offset_m
            )
            midpoint_properties = _integrate_prepared_sections(
                x_positions,
                section_polygons,
                heel_angle_deg,
                midpoint_offset_m,
            )
            volume_residual_m3 = (
                midpoint_properties["Displacement_Volume_m3"]
                - target_displacement_volume_m3
            )

            if abs(volume_residual_m3) <= volume_tolerance_m3:
                equilibrium_properties = midpoint_properties
                equilibrium_offset_m = midpoint_offset_m
                break

            if volume_residual_m3 < 0.0:
                lower_offset_m = midpoint_offset_m
            else:
                upper_offset_m = midpoint_offset_m

        if equilibrium_properties is None:
            raise RuntimeError(
                "Heeled displacement bisection did not converge within "
                f"{max_iterations} iterations."
            )

    displacement_volume_m3 = equilibrium_properties[
        "Displacement_Volume_m3"
    ]
    volume_residual_m3 = (
        displacement_volume_m3 - target_displacement_volume_m3
    )

    result = dict(equilibrium_properties)
    result.update(
        {
            "Target_Displacement_Volume_m3": (
                target_displacement_volume_m3
            ),
            "Full_Represented_Hull_Volume_m3": full_volume_m3,
            "Heel_Angle_deg": float(heel_angle_deg),
            "Waterplane_Offset_m": float(equilibrium_offset_m),
            "Centerline_Waterline_Height_m": float(
                equilibrium_offset_m/math.cos(heel_angle_rad)
            ),
            "Volume_Residual_m3": float(volume_residual_m3),
            "Volume_Tolerance_m3": float(volume_tolerance_m3),
            "Equilibrium_Iteration_Count": int(iteration_count),
            "Volume_Equilibrium_Solved": True,
            "Longitudinal_Equilibrium_Enforced": False,
            "Heel_Convention": (
                "positive_heel_is_starboard_positive_y_down"
            ),
        }
    )

    return result


def calculate_kn_and_gz(
    *,
    transverse_center_of_buoyancy_m: float,
    vertical_center_of_buoyancy_m: float,
    heel_angle_deg: float,
    vertical_center_of_gravity_m: float,
    transverse_center_of_gravity_m: float,
    keel_reference_z_m: float,
):
    """Return signed cross-curve ordinate ``KN`` and righting arm ``GZ``.

    With positive starboard-down heel, the buoyancy center normally shifts to
    positive ``y``.  The formulas are:

    ``KN = TCB*cos(phi) + (VCB - zK)*sin(phi)``

    ``GZ = KN - (KG - zK)*sin(phi) - TCG*cos(phi)``.

    For a centerline CG this reduces to the conventional
    ``GZ = KN - KG*sin(phi)`` when vertical dimensions are measured from K.
    """

    heel_angle_rad = _heel_angle_radians(heel_angle_deg)
    sin_heel = math.sin(heel_angle_rad)
    cos_heel = math.cos(heel_angle_rad)

    transverse_center_of_buoyancy_m = float(
        transverse_center_of_buoyancy_m
    )
    vertical_center_of_buoyancy_m = float(
        vertical_center_of_buoyancy_m
    )
    vertical_center_of_gravity_m = float(vertical_center_of_gravity_m)
    transverse_center_of_gravity_m = float(
        transverse_center_of_gravity_m
    )
    keel_reference_z_m = float(keel_reference_z_m)

    kn_m = (
        transverse_center_of_buoyancy_m*cos_heel
        + (vertical_center_of_buoyancy_m - keel_reference_z_m)*sin_heel
    )
    gz_m = (
        kn_m
        - (vertical_center_of_gravity_m - keel_reference_z_m)*sin_heel
        - transverse_center_of_gravity_m*cos_heel
    )

    return float(kn_m), float(gz_m)


def evaluate_finite_angle_stability_point(
    *,
    x_positions_m,
    z_positions_m,
    half_breadths_m,
    target_displacement_volume_m3: float,
    heel_angle_deg: float,
    vertical_center_of_gravity_m: float,
    transverse_center_of_gravity_m: float,
    keel_reference_z_m: float,
    volume_tolerance_m3: float | None = None,
    max_iterations: int = 200,
) -> dict:
    """Solve one fixed-trim cross-curve point and return KN and GZ."""

    equilibrium = solve_heeled_volume_equilibrium(
        x_positions_m=x_positions_m,
        z_positions_m=z_positions_m,
        half_breadths_m=half_breadths_m,
        target_displacement_volume_m3=target_displacement_volume_m3,
        heel_angle_deg=heel_angle_deg,
        volume_tolerance_m3=volume_tolerance_m3,
        max_iterations=max_iterations,
    )
    kn_m, gz_m = calculate_kn_and_gz(
        transverse_center_of_buoyancy_m=equilibrium["TCB_m"],
        vertical_center_of_buoyancy_m=equilibrium["VCB_m"],
        heel_angle_deg=heel_angle_deg,
        vertical_center_of_gravity_m=vertical_center_of_gravity_m,
        transverse_center_of_gravity_m=(
            transverse_center_of_gravity_m
        ),
        keel_reference_z_m=keel_reference_z_m,
    )

    equilibrium.update(
        {
            "Vertical_Center_Of_Gravity_m": float(
                vertical_center_of_gravity_m
            ),
            "Transverse_Center_Of_Gravity_m": float(
                transverse_center_of_gravity_m
            ),
            "Keel_Reference_Z_m": float(keel_reference_z_m),
            "Cross_Curve_KN_m": kn_m,
            "Righting_Arm_GZ_m": gz_m,
            "Finite_Angle_Free_Surface_Correction_Applied": False,
            "Downflooding_Limit_Evaluated": False,
            "HDF_085_Numerical_Foundation_Status": (
                "partial_fixed_trim_volume_equilibrium_only"
            ),
            "HDF_005_HDF_006_Constraint_Status": "not_evaluated",
            "Regulatory_Compliance_Status": (
                "not_determined_requires_free_trim_free_surface_"
                "downflooding_and_applicable_criteria"
            ),
        }
    )

    return equilibrium


def evaluate_finite_angle_stability_curve(
    *,
    x_positions_m,
    z_positions_m,
    half_breadths_m,
    target_displacement_volume_m3: float,
    heel_angles_deg,
    vertical_center_of_gravity_m: float,
    transverse_center_of_gravity_m: float,
    keel_reference_z_m: float,
    volume_tolerance_m3: float | None = None,
    max_iterations: int = 200,
) -> list[dict]:
    """Return independent fixed-trim KN/GZ points for requested heel angles."""

    return [
        evaluate_finite_angle_stability_point(
            x_positions_m=x_positions_m,
            z_positions_m=z_positions_m,
            half_breadths_m=half_breadths_m,
            target_displacement_volume_m3=(
                target_displacement_volume_m3
            ),
            heel_angle_deg=float(heel_angle_deg),
            vertical_center_of_gravity_m=(
                vertical_center_of_gravity_m
            ),
            transverse_center_of_gravity_m=(
                transverse_center_of_gravity_m
            ),
            keel_reference_z_m=keel_reference_z_m,
            volume_tolerance_m3=volume_tolerance_m3,
            max_iterations=max_iterations,
        )
        for heel_angle_deg in heel_angles_deg
    ]
