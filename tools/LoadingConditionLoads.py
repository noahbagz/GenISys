"""Longitudinal still-water loads for named GenISys loading conditions.

The V1.1 dataset keeps the approved full-load departure and arrival states
independently from its five sampled draft/Froude/range points.  This module
evaluates both states on the same mass-conservative, composite-trapezoidal load
foundation as the primary full-cargo structural case.  It returns scalar
summaries plus in-memory arrays; the batch dataset publishes the summaries and
retains the existing primary arrays to avoid unnecessary NPZ growth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.Weights import (
    _scale_distribution_to_mass,
    solve_longitudinal_buoyancy_equilibrium,
)


EXPECTED_LOADING_CONDITION_IDS = {
    "full_cargo_departure",
    "full_cargo_arrival",
}

DEPARTURE_LOADING_CONDITION_IDS = {
    "full_cargo_departure",
}


# HDF-ASM-035: engineering closure limits are numerical-quality gates.  They
# do not replace or modify any ABS, CFR, SOLAS, or IMO design requirement.
MASS_CLOSURE_RELATIVE_TOLERANCE = 1.0e-8
LOAD_CLOSURE_RELATIVE_TOLERANCE = 1.0e-6


def _cumulative_trapezoid(values, x_coordinates):
    """Return a cumulative trapezoidal integral with a zero first value."""

    values = np.asarray(values, dtype=float)
    x_coordinates = np.asarray(x_coordinates, dtype=float)
    integrated = np.zeros(len(values), dtype=float)
    integrated[1:] = np.cumsum(
        0.5*(values[:-1] + values[1:])*np.diff(x_coordinates)
    )

    return integrated


def build_named_condition_still_water_envelope(
    condition_summary: pd.DataFrame,
    required_condition_ids: set[str] | None = None,
) -> dict:
    """Return still-water extrema across an explicitly selected case set.

    The ABS hull-girder routines consume one negative and one positive shear
    and bending value, then combine them with their rule wave terms.  The
    default preserves the complete two-condition helper contract used by
    existing callers and tests.  A caller may provide a smaller, explicit set
    only when project scope intentionally excludes another named condition
    from active structural loading; the helper still refuses missing,
    unexpected, or unsolved cases inside that selected set.
    """

    required_ids = (
        EXPECTED_LOADING_CONDITION_IDS
        if required_condition_ids is None
        else {str(value) for value in required_condition_ids}
    )

    if not required_ids:
        raise ValueError(
            "A still-water envelope requires at least one named condition."
        )

    observed_ids = set(
        condition_summary["Loading_Condition_ID"].astype(str)
    )

    if observed_ids != required_ids:
        missing = sorted(required_ids - observed_ids)
        unexpected = sorted(observed_ids - required_ids)
        raise ValueError(
            "Named still-water envelope requires exactly the selected "
            f"conditions; missing={missing}, unexpected={unexpected}."
        )

    unsolved = condition_summary.loc[
        ~condition_summary["Equilibrium_Solved"].astype(bool),
        "Loading_Condition_ID",
    ].astype(str).tolist()

    if unsolved:
        raise ValueError(
            "Named still-water envelope cannot omit unsolved equilibrium "
            f"conditions: {sorted(unsolved)}."
        )

    extrema = {
        "Fnsw": ("Minimum_Shear_Force_N", "minimum"),
        "Fpsw": ("Maximum_Shear_Force_N", "maximum"),
        "Mnsw": ("Minimum_Bending_Moment_Nm", "minimum"),
        "Mpsw": ("Maximum_Bending_Moment_Nm", "maximum"),
    }
    envelope = {
        "Load_Case_Count": len(condition_summary),
        "Load_Basis": (
            "; ".join(
                condition_summary["Loading_Condition_Name"].astype(str)
            )
            + "; simultaneous displacement and longitudinal-moment "
            "equilibrium; composite trapezoidal integration"
        ),
    }

    for output_name, (column, direction) in extrema.items():
        values = condition_summary[column].to_numpy(dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Named still-water envelope received nonfinite {column}."
            )

        governing_index = (
            int(np.argmin(values))
            if direction == "minimum"
            else int(np.argmax(values))
        )
        governing_row = condition_summary.iloc[governing_index]

        # The existing structural interface is kN/kNm.  Keep that contract
        # explicit at this one conversion boundary.
        envelope[output_name] = float(values[governing_index]/1000.0)
        envelope[f"{output_name}_Condition_ID"] = str(
            governing_row["Loading_Condition_ID"]
        )
        envelope[f"{output_name}_Condition_Name"] = str(
            governing_row["Loading_Condition_Name"]
        )

    return envelope


def evaluate_named_condition_longitudinal_loads(
    weights,
    component_states: pd.DataFrame,
    condition_specific_fuel_tanks: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Evaluate force/moment equilibrium and still-water curves by condition.

    ``component_states`` follows :mod:`tools.LoadingConditions`.  Component
    indices refer back to ``weights.Weights``.  The explicit fuel-state table
    is used when present so arrival loads carry the recalculated ten-percent
    fill distribution rather than a scaled full-tank shape.
    """

    x_coordinates = np.asarray(weights.X, dtype=float)
    base_components = weights.Weights.reset_index(drop=True)
    summaries = []
    arrays = {}

    for condition_id, condition in component_states.groupby(
        "Loading_Condition_ID",
        sort=False,
    ):
        condition_name = str(condition.iloc[0]["Loading_Condition_Name"])
        mass_distribution_kg_per_m = np.zeros(len(x_coordinates))

        for _, component_state in condition.iterrows():
            component_index = int(component_state["Component_Index"])
            component_name = str(component_state["Component_Name"])
            condition_mass_kg = float(
                component_state["Condition_Weight_kg"]
            )

            if condition_mass_kg == 0.0:
                continue

            distribution = None

            if (
                component_name == "Fuel_Tank"
                and condition_specific_fuel_tanks is not None
            ):
                matches = condition_specific_fuel_tanks.loc[
                    condition_specific_fuel_tanks[
                        "Loading_Condition_ID"
                    ].astype(str)
                    == str(condition_id)
                ]

                if not matches.empty:
                    distribution = np.asarray(
                        matches.iloc[0]["Dist_Weight"],
                        dtype=float,
                    )

            if distribution is None:
                base_component = base_components.iloc[component_index]
                normalized_base_distribution = _scale_distribution_to_mass(
                    base_component["Dist_Weight"],
                    x_coordinates,
                    float(base_component["Total_Weight_kg"]),
                )
                distribution = (
                    normalized_base_distribution
                    * float(component_state["Applied_Fraction"])
                )

            # Normalize at the condition boundary as an independent check that
            # each state carries the exact scalar mass used for displacement.
            mass_distribution_kg_per_m += _scale_distribution_to_mass(
                distribution,
                x_coordinates,
                condition_mass_kg,
            )

        total_mass_kg = float(condition["Condition_Weight_kg"].sum())
        target_lcg_m = float(
            (
                condition["Condition_Weight_kg"]
                * condition["LCG_m"]
            ).sum()/total_mass_kg
        )
        target_volume_m3 = total_mass_kg/float(weights.rho)
        equilibrium = solve_longitudinal_buoyancy_equilibrium(
            x_coordinates,
            weights.Z,
            weights.sectional_Areas,
            target_volume_m3,
            target_lcg_m,
            weights.design_draft,
        )

        distributed_weight_n_per_m = (
            -mass_distribution_kg_per_m*float(weights.g)
        )

        if equilibrium["Equilibrium_Solved"]:
            buoyancy_n_per_m = (
                float(weights.rho)
                * float(weights.g)
                * equilibrium["Local_Sectional_Areas_m2"]
            )
            net_load_n_per_m = (
                distributed_weight_n_per_m + buoyancy_n_per_m
            )
            shear_force_n = _cumulative_trapezoid(
                net_load_n_per_m,
                x_coordinates,
            )
            bending_moment_nm = _cumulative_trapezoid(
                shear_force_n,
                x_coordinates,
            )
        else:
            buoyancy_n_per_m = np.full(len(x_coordinates), np.nan)
            net_load_n_per_m = np.full(len(x_coordinates), np.nan)
            shear_force_n = np.full(len(x_coordinates), np.nan)
            bending_moment_nm = np.full(len(x_coordinates), np.nan)

        integrated_mass_kg = float(
            np.trapezoid(
                mass_distribution_kg_per_m,
                x=x_coordinates,
            )
        )
        mass_first_moment_kg_m = float(
            np.trapezoid(
                mass_distribution_kg_per_m*x_coordinates,
                x=x_coordinates,
            )
        )
        weight_first_moment_n_m = float(
            np.trapezoid(
                distributed_weight_n_per_m
                * (x_coordinates - x_coordinates[0]),
                x=x_coordinates,
            )
        )
        buoyancy_force_n = float(
            np.trapezoid(buoyancy_n_per_m, x=x_coordinates)
        )
        weight_force_n = float(
            np.trapezoid(
                distributed_weight_n_per_m,
                x=x_coordinates,
            )
        )
        buoyancy_first_moment_n_m = float(
            np.trapezoid(
                buoyancy_n_per_m*(x_coordinates - x_coordinates[0]),
                x=x_coordinates,
            )
        )

        # HDF-ASM-035 normalizes unlike closure quantities by the physical
        # scale of the loading condition.  This makes one approved tolerance
        # meaningful across the full GenISys size range.  Signed residuals are
        # retained for diagnosis; the constraint adapter compares magnitudes.
        modeled_length_m = float(x_coordinates[-1] - x_coordinates[0])
        displacement_weight_n = total_mass_kg*float(weights.g)
        force_scale_n = max(
            abs(displacement_weight_n),
            np.finfo(float).eps,
        )
        moment_scale_nm = max(
            abs(displacement_weight_n*modeled_length_m),
            np.finfo(float).eps,
        )
        mass_scale_kg = max(abs(total_mass_kg), np.finfo(float).eps)
        mass_residual_kg = integrated_mass_kg - total_mass_kg
        net_vertical_force_residual_n = (
            weight_force_n + buoyancy_force_n
        )
        net_vertical_first_moment_residual_nm = (
            weight_first_moment_n_m + buoyancy_first_moment_n_m
        )
        aft_shear_force_residual_n = float(shear_force_n[-1])
        aft_bending_moment_residual_nm = float(bending_moment_nm[-1])

        equilibrium_draft_m = 0.5*(
            equilibrium["Forward_Draft_m"]
            + equilibrium["Aft_Draft_m"]
        )
        signed_trim_m = (
            equilibrium["Aft_Draft_m"]
            - equilibrium["Forward_Draft_m"]
        )
        trim_limit_m = 0.01*equilibrium_draft_m
        trim_equality_allowance_m = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(signed_trim_m), abs(trim_limit_m))
        )

        if equilibrium["Equilibrium_Solved"]:
            minimum_shear_force_n = float(np.min(shear_force_n))
            maximum_shear_force_n = float(np.max(shear_force_n))
            minimum_bending_moment_nm = float(np.min(bending_moment_nm))
            maximum_bending_moment_nm = float(np.max(bending_moment_nm))
        else:
            minimum_shear_force_n = np.nan
            maximum_shear_force_n = np.nan
            minimum_bending_moment_nm = np.nan
            maximum_bending_moment_nm = np.nan

        summaries.append(
            {
                "Loading_Condition_ID": str(condition_id),
                "Loading_Condition_Name": condition_name,
                "Equilibrium_Solved": bool(
                    equilibrium["Equilibrium_Solved"]
                ),
                "Total_Mass_kg": total_mass_kg,
                "Integrated_Distributed_Mass_kg": integrated_mass_kg,
                "Mass_Residual_kg": mass_residual_kg,
                "Mass_Residual_Fraction": mass_residual_kg/mass_scale_kg,
                "Target_LCG_m": target_lcg_m,
                "Integrated_Distributed_LCG_m": (
                    mass_first_moment_kg_m/integrated_mass_kg
                ),
                "Forward_Draft_m": equilibrium["Forward_Draft_m"],
                "Aft_Draft_m": equilibrium["Aft_Draft_m"],
                "Equilibrium_Draft_m": equilibrium_draft_m,
                "Signed_Trim_Aft_Minus_Forward_m": signed_trim_m,
                "Trim_Limit_One_Percent_Draft_m": trim_limit_m,
                "Trim_Constraint_Passed": bool(
                    equilibrium["Equilibrium_Solved"]
                    and abs(signed_trim_m)
                    <= trim_limit_m + trim_equality_allowance_m
                ),
                "Normalized_Volume_Residual": equilibrium[
                    "Normalized_Volume_Residual"
                ],
                "Normalized_First_Moment_Residual": equilibrium[
                    "Normalized_First_Moment_Residual"
                ],
                "Net_Vertical_Force_Residual_N": (
                    net_vertical_force_residual_n
                ),
                "Net_Vertical_Force_Residual_Fraction": (
                    net_vertical_force_residual_n/force_scale_n
                ),
                "Net_Vertical_First_Moment_Residual_Nm": (
                    net_vertical_first_moment_residual_nm
                ),
                "Net_Vertical_First_Moment_Residual_Fraction": (
                    net_vertical_first_moment_residual_nm/moment_scale_nm
                ),
                "Aft_Shear_Force_Residual_N": aft_shear_force_residual_n,
                "Aft_Shear_Force_Residual_Fraction": (
                    aft_shear_force_residual_n/force_scale_n
                ),
                "Aft_Bending_Moment_Residual_Nm": (
                    aft_bending_moment_residual_nm
                ),
                "Aft_Bending_Moment_Residual_Fraction": (
                    aft_bending_moment_residual_nm/moment_scale_nm
                ),
                "Minimum_Shear_Force_N": minimum_shear_force_n,
                "Maximum_Shear_Force_N": maximum_shear_force_n,
                "Minimum_Bending_Moment_Nm": minimum_bending_moment_nm,
                "Maximum_Bending_Moment_Nm": maximum_bending_moment_nm,
                "Load_Model_Status": (
                    "approved full-load scope; condition-specific mass and "
                    "fuel fill; optimized per-hold cargo geometry"
                ),
            }
        )
        arrays[str(condition_id)] = {
            "X_m": x_coordinates.copy(),
            "Distributed_Weight_N_per_m": (
                distributed_weight_n_per_m
            ),
            "Buoyancy_N_per_m": buoyancy_n_per_m,
            "Net_Load_N_per_m": net_load_n_per_m,
            "Shear_Force_N": shear_force_n,
            "Bending_Moment_Nm": bending_moment_nm,
        }

    return pd.DataFrame(summaries), arrays
