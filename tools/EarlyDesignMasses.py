"""Explicit preliminary mass models awaiting V1.1 design parameters.

Hatch-cover mass is already integrated from approved project assumptions.
Deckhouse weight and lightship growth margin are kept here as parameterized
calculations because their dimensions, factors, and centers have not yet been
approved. No function in this module supplies silent defaults. The active
``ShipSystemDesign`` path calls the component builders only when an explicit
dataset-scenario mapping supplies the corresponding input group. The only
optional coordinate defaults are transverse centers of zero, matching the
approved symmetric intact-loading assumption HDF-ASM-013.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _point_equivalent_distribution(
    x_coordinates,
    total_mass_kg: float,
    lcg_m: float,
) -> np.ndarray:
    """Represent a component with exact trapezoidal mass and first moment.

    The optional early-design items do not yet have an approved detailed
    longitudinal mass distribution.  Place their equivalent load at the
    explicitly supplied LCG using the two grid stations that bracket it.  The
    trapezoidal quadrature then recovers both scalar mass and scalar first
    moment exactly, so activating a component cannot corrupt the still-water
    load closure.
    """

    x_m = np.asarray(x_coordinates, dtype=float)
    mass_kg = float(total_mass_kg)
    lcg_m = float(lcg_m)

    if lcg_m < x_m[0] or lcg_m > x_m[-1]:
        raise ValueError(
            "An early-design component LCG lies outside the hydrostatic grid."
        )

    # These are the integration weights used by ``numpy.trapezoid`` on a
    # nonuniform grid.  Storing density = nodal_mass / quadrature_weight makes
    # the area beneath the discrete distribution equal the intended mass.
    quadrature_weights_m = np.empty_like(x_m)
    quadrature_weights_m[0] = 0.5*(x_m[1] - x_m[0])
    quadrature_weights_m[-1] = 0.5*(x_m[-1] - x_m[-2])
    quadrature_weights_m[1:-1] = 0.5*(x_m[2:] - x_m[:-2])
    distribution_kg_per_m = np.zeros_like(x_m)

    upper = int(np.searchsorted(x_m, lcg_m, side="left"))

    if upper == len(x_m):
        upper = len(x_m) - 1

    if x_m[upper] == lcg_m or upper == 0:
        distribution_kg_per_m[upper] = (
            mass_kg/quadrature_weights_m[upper]
        )
        return distribution_kg_per_m

    lower = upper - 1
    upper_fraction = (
        (lcg_m - x_m[lower])/(x_m[upper] - x_m[lower])
    )
    lower_mass_kg = mass_kg*(1.0 - upper_fraction)
    upper_mass_kg = mass_kg*upper_fraction
    distribution_kg_per_m[lower] = (
        lower_mass_kg/quadrature_weights_m[lower]
    )
    distribution_kg_per_m[upper] = (
        upper_mass_kg/quadrature_weights_m[upper]
    )

    return distribution_kg_per_m


def estimate_simple_deckhouse_mass(
    *,
    length_m: float,
    breadth_m: float,
    height_m: float,
    base_x_m: float,
    base_z_m: float,
    equivalent_areal_mass_kg_m2: float,
    centerline_y_m: float = 0.0,
) -> dict:
    """Estimate a thin-walled rectangular deckhouse and its mass centroid.

    The shell representation consists of four vertical walls and one roof.
    The deck below the house is excluded to avoid double counting existing
    deck plating.  ``equivalent_areal_mass_kg_m2`` must represent the selected
    preliminary allowance for plating, stiffening, internal structure, and
    outfit; GenISys does not currently prescribe that project value.
    """

    roof_area_m2 = length_m*breadth_m
    side_wall_area_m2 = 2.0*length_m*height_m
    end_wall_area_m2 = 2.0*breadth_m*height_m
    exposed_area_m2 = roof_area_m2 + side_wall_area_m2 + end_wall_area_m2
    total_mass_kg = exposed_area_m2*equivalent_areal_mass_kg_m2

    # All five rectangular surfaces are centered longitudinally and
    # transversely.  Their vertical surface centroids differ, so calculate the
    # combined VCG rather than assigning the half-height by inspection.
    wall_area_m2 = side_wall_area_m2 + end_wall_area_m2
    vcg_m = (
        roof_area_m2*(base_z_m + height_m)
        + wall_area_m2*(base_z_m + 0.5*height_m)
    )/exposed_area_m2

    return {
        "Name": "Simple Deckhouse",
        "Geometry_Status": "parameterized_not_active",
        "Length_m": float(length_m),
        "Breadth_m": float(breadth_m),
        "Height_m": float(height_m),
        "Base_X_m": float(base_x_m),
        "Base_Z_m": float(base_z_m),
        "Centerline_Y_m": float(centerline_y_m),
        "Exposed_Shell_Area_m2": float(exposed_area_m2),
        "Equivalent_Areal_Mass_kg_m2": float(
            equivalent_areal_mass_kg_m2
        ),
        "Total_Weight_kg": float(total_mass_kg),
        "LCG": float(base_x_m + 0.5*length_m),
        "TCG": float(centerline_y_m),
        "VCG": float(vcg_m),
    }


def calculate_lightship_margin_component(
    *,
    modeled_lightship_mass_kg: float,
    margin_fraction: float,
    margin_lcg_m: float,
    margin_vcg_m: float,
    margin_tcg_m: float = 0.0,
) -> dict:
    """Return an explicit early-design lightship growth allowance.

    The margin fraction and all three coordinates are deliberately required
    inputs.  They must be approved as a project scenario before this component
    can be integrated into loading conditions, displacement, or stability.
    """

    margin_mass_kg = modeled_lightship_mass_kg*margin_fraction

    return {
        "Name": "Lightship Growth Margin",
        "Geometry_Status": "parameterized_not_active",
        "Margin_Fraction": float(margin_fraction),
        "Total_Weight_kg": float(margin_mass_kg),
        "LCG": float(margin_lcg_m),
        "TCG": float(margin_tcg_m),
        "VCG": float(margin_vcg_m),
    }


def build_simple_deckhouse_component(
    x_coordinates,
    **explicit_inputs,
) -> pd.DataFrame:
    """Build one activatable deckhouse weight row from explicit inputs only.

    No dimensions or location are supplied here.  The caller must provide the
    complete argument set required by ``estimate_simple_deckhouse_mass``.
    Spatial fit against the hull/deck is intentionally not asserted until the
    project defines the intended deckhouse arrangement.
    """

    component = estimate_simple_deckhouse_mass(**explicit_inputs)
    component["Geometry_Status"] = (
        "explicit_input_not_spatially_validated"
    )
    component["Dist_Weight"] = _point_equivalent_distribution(
        x_coordinates,
        component["Total_Weight_kg"],
        component["LCG"],
    )

    return pd.DataFrame([component])


def build_lightship_margin_component(
    x_coordinates,
    *,
    modeled_lightship_mass_kg: float,
    **explicit_inputs,
) -> pd.DataFrame:
    """Build one activatable growth-margin row from explicit inputs only."""

    component = calculate_lightship_margin_component(
        modeled_lightship_mass_kg=modeled_lightship_mass_kg,
        **explicit_inputs,
    )
    component["Geometry_Status"] = "explicit_input_assigned_mass_center"
    component["Dist_Weight"] = _point_equivalent_distribution(
        x_coordinates,
        component["Total_Weight_kg"],
        component["LCG"],
    )

    return pd.DataFrame([component])
