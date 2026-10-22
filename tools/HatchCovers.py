"""Preliminary cargo-hatch cover mass properties for GenISys V1.1.

The cover geometry and mass model implement the user-approved early-design
assumptions HDF-ASM-021 through HDF-ASM-025.  These values support weight, CG,
windage, and branch-selection calculations; they are not a hatch-cover
strength, securing, drainage, or weathertightness certification analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


NOMINAL_AREAL_MASS_T_PER_M2 = 0.30
CONSERVATIVE_AREAL_MASS_T_PER_M2 = 0.35
COVER_EDGE_OFFSET_M = 0.20
COAMING_HEIGHT_M = 0.60
COVER_ENVELOPE_DEPTH_M = 0.80


def _structure_value(structure_evaluations: pd.DataFrame, name: str) -> float:
    """Return one named value from ``StructureCalcs.Struct_Dict``."""

    matches = structure_evaluations.loc[
        structure_evaluations["Parameter"] == name,
        "Value",
    ]

    if matches.empty:
        raise KeyError(f"Missing structural hatch parameter: {name}")

    return float(matches.iloc[0])


def rounded_rectangle_area_m2(
    length_m: float,
    width_m: float,
    corner_radius_m: float,
) -> float:
    """Return the plan area of a rectangle with four rounded corners."""

    return float(
        length_m*width_m
        - (4.0 - np.pi)*corner_radius_m**2
    )


def _uniform_interval_distribution(
    x_m: np.ndarray,
    center_m: float,
    length_m: float,
    total_mass_kg: float,
) -> np.ndarray:
    """Place one cover over its plan length and close mass by trapezoid.

    The hydrostatic grid is coarser than final hatch structure.  A uniform
    footprint is therefore used only as a longitudinal weight distribution;
    it does not replace the later rounded CAD proxy.
    """

    x_m = np.asarray(x_m, dtype=float)
    half_length_m = 0.5*float(length_m)
    distribution = np.where(
        np.abs(x_m - float(center_m)) <= half_length_m,
        1.0,
        0.0,
    )
    integrated_shape_m = float(np.trapezoid(distribution, x=x_m))

    # A cover shorter than one longitudinal grid interval may contain no
    # interior station.  In that case, place a nodal triangular proxy at the
    # nearest station and normalize it using the same integration convention.
    if integrated_shape_m <= 0.0:
        nearest = int(np.argmin(np.abs(x_m - float(center_m))))
        distribution[nearest] = 1.0
        integrated_shape_m = float(np.trapezoid(distribution, x=x_m))

    if integrated_shape_m <= 0.0:
        raise ValueError("The longitudinal grid cannot represent hatch mass.")

    return distribution*float(total_mass_kg)/integrated_shape_m


def build_hatch_cover_components(
    structure_evaluations: pd.DataFrame,
    cargo_holds,
    hull_depth_m: float,
) -> pd.DataFrame:
    """Return one weight-component row per generated cargo-hold hatch.

    Each actual cargo hold receives one cover using the common hatch dimensions
    calculated for the repeated structural module.  Longitudinal cover centers
    come from the actual generated cargo-hold boundaries.
    """

    opening_length_m = _structure_value(
        structure_evaluations,
        "Hatch_Length",
    )
    opening_width_m = _structure_value(
        structure_evaluations,
        "Hatch_Width",
    )
    opening_radius_m = (
        _structure_value(structure_evaluations, "Hatch_Corner_Radius")
        / 1000.0
    )

    cover_length_m = opening_length_m + 2.0*COVER_EDGE_OFFSET_M
    cover_width_m = opening_width_m + 2.0*COVER_EDGE_OFFSET_M
    cover_radius_m = opening_radius_m + COVER_EDGE_OFFSET_M
    cover_area_m2 = rounded_rectangle_area_m2(
        cover_length_m,
        cover_width_m,
        cover_radius_m,
    )

    if cover_area_m2 <= 0.0:
        raise ValueError("Approved hatch-cover dimensions produce no plan area.")

    nominal_mass_kg = (
        NOMINAL_AREAL_MASS_T_PER_M2*1000.0*cover_area_m2
    )
    conservative_mass_kg = (
        CONSERVATIVE_AREAL_MASS_T_PER_M2*1000.0*cover_area_m2
    )
    cover_vcg_m = (
        float(hull_depth_m)
        + COAMING_HEIGHT_M
        + 0.5*COVER_ENVELOPE_DEPTH_M
    )
    cover_bottom_z_m = float(hull_depth_m) + COAMING_HEIGHT_M
    cover_top_z_m = cover_bottom_z_m + COVER_ENVELOPE_DEPTH_M

    # HDF-ASM-021/HDF-ASM-025: the approved closed-box proxy has a directly
    # calculable side projection.  Save it now for later windage aggregation;
    # it is not by itself a CFR weather-criterion calculation.
    projected_lateral_area_m2 = cover_length_m*COVER_ENVELOPE_DEPTH_M

    number_of_holds = len(cargo_holds.CargoHold_Assignments) - 1
    hold_boundaries_m = np.asarray(
        cargo_holds.X_bulkheads[:number_of_holds + 1],
        dtype=float,
    )
    rows = []

    for hold_number in range(number_of_holds):
        lcg_m = 0.5*(
            hold_boundaries_m[hold_number]
            + hold_boundaries_m[hold_number + 1]
        )
        distribution = _uniform_interval_distribution(
            cargo_holds.X,
            lcg_m,
            cover_length_m,
            nominal_mass_kg,
        )
        rows.append(
            {
                "Name": "Hatch_Cover",
                "Component_ID": f"Hatch_Cover_{hold_number:02d}",
                "Hold_Number": hold_number,
                "Opening_Length_m": opening_length_m,
                "Opening_Width_m": opening_width_m,
                "Opening_Corner_Radius_m": opening_radius_m,
                "Cover_Length_m": cover_length_m,
                "Cover_Width_m": cover_width_m,
                "Cover_Corner_Radius_m": cover_radius_m,
                "Projected_Plan_Area_m2": cover_area_m2,
                "Projected_Lateral_Area_m2": projected_lateral_area_m2,
                "Lateral_Area_Centroid_X_m": lcg_m,
                "Lateral_Area_Centroid_Z_m": cover_vcg_m,
                "Cover_Bottom_Z_m": cover_bottom_z_m,
                "Cover_Top_Z_m": cover_top_z_m,
                "Cover_Envelope_Depth_m": COVER_ENVELOPE_DEPTH_M,
                "Coaming_Height_m": COAMING_HEIGHT_M,
                "Nominal_Areal_Mass_t_per_m2": (
                    NOMINAL_AREAL_MASS_T_PER_M2
                ),
                "Conservative_Areal_Mass_t_per_m2": (
                    CONSERVATIVE_AREAL_MASS_T_PER_M2
                ),
                "Conservative_Weight_kg": conservative_mass_kg,
                "Total_Weight_kg": nominal_mass_kg,
                "VCG": cover_vcg_m,
                "LCG": lcg_m,
                "TCG": 0.0,
                "Dist_Weight": distribution,
                "Assumption_IDs": (
                    "HDF-ASM-021; HDF-ASM-022; HDF-ASM-023; "
                    "HDF-ASM-024; HDF-ASM-025"
                ),
                "CAD_Proxy_Status": (
                    "approved_geometry_available_on_demand_not_serialized"
                ),
                "Regulatory_Status_Basis": (
                    "HDF-ASM-027 assumed qualifying closed weathertight cover; "
                    "not validated by this mass model"
                ),
            }
        )

    return pd.DataFrame(rows)
