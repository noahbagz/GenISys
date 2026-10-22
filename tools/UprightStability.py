"""Condition-specific upright stability and intact free-surface calculations.

This module is intentionally separate from the legacy stability and flooding
diagnostics in :mod:`tools.Hydrostatics`.  It evaluates intact displacement,
upright GM, and the approved linear trim model for every supplied loading
condition.  A corrected-GM predicate is emitted only when an explicit
condition-specific free-surface correction is available.

The corrected-GM predicate implements only the atomic 0.15 m requirement in
HDF-007.  It does not establish 2008 IS Code compliance because GenISys does
not yet calculate finite-heel GZ, GZ areas, the severe-wind-and-roll criterion,
or a verified downflooding angle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MINIMUM_CORRECTED_GM_M = 0.15


def rectangular_free_surface_moment_m4(
    tank_length_m: float,
    free_surface_breadth_m: float,
) -> float:
    """Return ``L*b^3/12`` for one rectangular tank free surface."""

    return float(tank_length_m * free_surface_breadth_m**3.0 / 12.0)


def integrate_free_surface_moment_m4(
    longitudinal_positions_m,
    free_surface_breadths_m,
) -> float:
    """Integrate one tank's upright transverse free-surface inertia.

    The free-surface breadth is assumed to vary linearly between longitudinal
    ordinates.  Integrating ``b(x)^3/12`` analytically in each cell is therefore
    consistent with the linear waterplane representation in ``Hydrostatics``.
    Each independent tank or connected tank group must be passed separately;
    this function does not infer transverse pairing or connectivity.
    """

    x = np.asarray(longitudinal_positions_m, dtype=float)
    breadth = np.asarray(free_surface_breadths_m, dtype=float)

    dx = np.diff(x)
    breadth_0 = breadth[:-1]
    breadth_1 = breadth[1:]

    cell_moments = dx/48.0 * (
        breadth_0**3.0
        + breadth_0**2.0*breadth_1
        + breadth_0*breadth_1**2.0
        + breadth_1**3.0
    )

    return float(np.sum(cell_moments))


def select_cfr_170_285_free_surface_groups(
    tank_group_states: pd.DataFrame,
) -> pd.DataFrame:
    """Select the minimum tank-group set required by 46 CFR 170.285.

    Each input row must already represent either one centerline tank, one
    transverse pair of wing tanks, or one individual non-consumable tank.  The
    row's upright moment must also already be the governing value for the fill
    range represented by that row.  This function cannot infer those facts
    from the current combined cargo/ballast volume model.

    Required columns are ``Loading_Condition_ID``, ``Free_Surface_Group_ID``,
    ``Liquid_Type``, ``Is_Consumable``, ``Is_Partially_Filled``,
    ``Liquid_Density_kg_m3``, and ``Upright_Free_Surface_Moment_m4``.
    """

    candidates = tank_group_states.copy()
    candidates["Density_Adjusted_Free_Surface_Moment_kg_m"] = (
        candidates["Liquid_Density_kg_m3"]
        * candidates["Upright_Free_Surface_Moment_m4"]
    )

    selected_tables = []

    for _, condition_candidates in candidates.groupby(
        "Loading_Condition_ID",
        sort=False,
    ):
        consumable_candidates = condition_candidates[
            condition_candidates["Is_Consumable"]
        ]

        if not consumable_candidates.empty:
            greatest_indices = consumable_candidates.groupby(
                "Liquid_Type",
                sort=False,
            )["Density_Adjusted_Free_Surface_Moment_kg_m"].idxmax()
            selected_consumables = candidates.loc[greatest_indices].copy()
            selected_consumables["CFR_170_285_Selection_Basis"] = (
                "greatest_group_for_consumable_liquid_type"
            )
            selected_tables.append(selected_consumables)

        selected_non_consumables = condition_candidates[
            (~condition_candidates["Is_Consumable"])
            & condition_candidates["Is_Partially_Filled"]
        ].copy()

        if not selected_non_consumables.empty:
            selected_non_consumables["CFR_170_285_Selection_Basis"] = (
                "each_partially_filled_non_consumable_tank"
            )
            selected_tables.append(selected_non_consumables)

    if not selected_tables:
        empty = candidates.iloc[0:0].copy()
        empty["CFR_170_285_Selection_Basis"] = pd.Series(dtype=str)

        return empty

    return pd.concat(selected_tables, ignore_index=True)


def calculate_cfr_170_285_free_surface_corrections(
    tank_group_states: pd.DataFrame,
    loading_conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Return one intact free-surface correction for each loading condition.

    The density-adjusted free-surface moment is ``rho_liquid * I`` in
    ``kg*m``.  Dividing its selected sum by vessel displacement mass in
    kilograms gives the virtual rise in ``G`` in metres.
    """

    selected = select_cfr_170_285_free_surface_groups(tank_group_states)
    registered_condition_ids = set(
        tank_group_states["Loading_Condition_ID"].astype(str)
    )
    rows = []

    for _, condition in loading_conditions.iterrows():
        condition_id = str(condition["Loading_Condition_ID"])
        condition_groups = selected[
            selected["Loading_Condition_ID"].astype(str) == condition_id
        ]

        if condition_id not in registered_condition_ids:
            total_density_adjusted_moment_kg_m = np.nan
            free_surface_correction_m = np.nan
            selected_group_ids = ""
            status = "missing_condition_tank_group_registry"
        else:
            total_density_adjusted_moment_kg_m = float(
                condition_groups[
                    "Density_Adjusted_Free_Surface_Moment_kg_m"
                ].sum()
            )
            free_surface_correction_m = (
                total_density_adjusted_moment_kg_m
                / float(condition["Total_Weight_kg"])
            )
            selected_group_ids = "; ".join(
                condition_groups["Free_Surface_Group_ID"].astype(str)
            )
            status = "calculated_cfr_170_285_selection"

        rows.append(
            {
                "Loading_Condition_ID": condition_id,
                "Selected_Free_Surface_Group_IDs": selected_group_ids,
                "Density_Adjusted_Free_Surface_Moment_kg_m": (
                    total_density_adjusted_moment_kg_m
                ),
                "Free_Surface_Correction_m": free_surface_correction_m,
                "CFR_170_285_Status": status,
                "Source_Citation": "46 CFR 170.285",
                "Missing_Data_ID": (
                    "" if status.startswith("calculated") else "HDF-DATA-003"
                ),
            }
        )

    return pd.DataFrame(rows)


def _unresolved_equilibrium_row(condition: pd.Series, displacement_volume_m3):
    """Return explicit missing results when the hydrostatic grid cannot float."""

    row = condition.to_dict()
    row.update(
        {
            "Displacement_Volume_m3": displacement_volume_m3,
            "Equilibrium_Draft_m": np.nan,
            "LCB_m": np.nan,
            "VCB_m": np.nan,
            "Waterplane_Area_m2": np.nan,
            "LCF_m": np.nan,
            "Transverse_Waterplane_Inertia_m4": np.nan,
            "Longitudinal_Waterplane_Inertia_m4": np.nan,
            "KM_T_m": np.nan,
            "GM_T_Uncorrected_m": np.nan,
            "Free_Surface_Correction_m": np.nan,
            "GM0_Corrected_m": np.nan,
            "HDF_007_GM_Residual_m": np.nan,
            "HDF_007_Minimum_Corrected_GM_Passed": pd.NA,
            "GM_L_m": np.nan,
            "MCT_1cm_N_m": np.nan,
            "Trim_cm": np.nan,
            "Draft_Forward_m": np.nan,
            "Draft_Aft_m": np.nan,
            "Equilibrium_Solved": False,
            "Upright_Stability_Status": (
                "blocked_displacement_outside_hydrostatic_grid"
            ),
            "Complete_Intact_Stability_Compliance": False,
            "Calculation_Assumption_IDs": (
                "HDF-ASM-013; HDF-ASM-030; HDF-ASM-031"
            ),
            "Source_Citations": (
                "46 CFR 170.165; 46 CFR 170.285; "
                "2008 IS Code Part A 2.2.1.5"
            ),
        }
    )

    return row


def evaluate_upright_loading_conditions(
    hydrostatics,
    loading_conditions: pd.DataFrame,
    free_surface_corrections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate intact upright equilibrium, GM, and trim for each condition.

    ``loading_conditions`` uses the summary contract from
    :func:`tools.LoadingConditions.summarize_loading_conditions`.
    ``free_surface_corrections`` may use the output of
    :func:`calculate_cfr_170_285_free_surface_corrections`.
    """

    volumes, lcb_curve, vcb_curve = hydrostatics.calc_Volumes()
    volumes = np.asarray(volumes, dtype=float)
    lcb_curve = np.asarray(lcb_curve, dtype=float)
    vcb_curve = np.asarray(vcb_curve, dtype=float)
    drafts = np.asarray(hydrostatics.Z, dtype=float)

    if free_surface_corrections is None:
        correction_by_condition = {}
    else:
        correction_by_condition = free_surface_corrections.set_index(
            "Loading_Condition_ID"
        )["Free_Surface_Correction_m"].to_dict()

    rows = []

    for _, condition in loading_conditions.iterrows():
        condition_id = str(condition["Loading_Condition_ID"])
        displacement_mass_kg = float(condition["Total_Weight_kg"])
        displacement_volume_m3 = displacement_mass_kg/hydrostatics.rho

        if (
            displacement_volume_m3 < float(np.min(volumes))
            or displacement_volume_m3 > float(np.max(volumes))
        ):
            rows.append(
                _unresolved_equilibrium_row(
                    condition,
                    displacement_volume_m3,
                )
            )
            continue

        # Solve the intact displacement condition on the existing upright
        # hydrostatic curves, then evaluate the waterplane at that draft.
        equilibrium_draft_m = float(
            np.interp(displacement_volume_m3, volumes, drafts)
        )
        lcb_m = float(np.interp(equilibrium_draft_m, drafts, lcb_curve))
        vcb_m = float(np.interp(equilibrium_draft_m, drafts, vcb_curve))
        waterplane_area_m2, lcf_m, i_transverse_m4, i_longitudinal_m4 = (
            hydrostatics.calc_WaterPlane(equilibrium_draft_m)
        )

        vcg_m = float(condition["VCG_m"])
        lcg_m = float(condition["LCG_m"])

        km_transverse_m = vcb_m + i_transverse_m4/displacement_volume_m3
        gm_transverse_uncorrected_m = km_transverse_m - vcg_m
        gm_longitudinal_m = (
            vcb_m + i_longitudinal_m4/displacement_volume_m3 - vcg_m
        )

        # HDF-ASM-030/HDF-ASM-031: this is the approved preliminary LBP trim
        # model.  It is deliberately not used as Load Line length.
        mct_1cm_n_m = (
            hydrostatics.rho
            * hydrostatics.g
            * displacement_volume_m3
            * gm_longitudinal_m
            * 0.01
            / hydrostatics.LBP
        )
        trim_moment_n_m = (
            (lcg_m - lcb_m)
            * hydrostatics.rho
            * hydrostatics.g
            * displacement_volume_m3
        )
        trim_cm = trim_moment_n_m/mct_1cm_n_m
        draft_forward_m = (
            equilibrium_draft_m
            - 0.01*trim_cm*lcf_m/hydrostatics.LBP
        )
        draft_aft_m = (
            equilibrium_draft_m
            + 0.01*trim_cm*(hydrostatics.LBP - lcf_m)/hydrostatics.LBP
        )

        free_surface_correction_m = correction_by_condition.get(
            condition_id,
            np.nan,
        )

        if pd.isna(free_surface_correction_m):
            corrected_gm_m = np.nan
            minimum_gm_residual_m = np.nan
            minimum_gm_passed = pd.NA
            status = "upright_foundation_only_missing_free_surface"
        else:
            free_surface_correction_m = float(free_surface_correction_m)
            corrected_gm_m = (
                gm_transverse_uncorrected_m - free_surface_correction_m
            )
            minimum_gm_residual_m = (
                corrected_gm_m - MINIMUM_CORRECTED_GM_M
            )
            minimum_gm_passed = corrected_gm_m >= MINIMUM_CORRECTED_GM_M
            status = "calculated_atomic_hdf_007_only"

        row = condition.to_dict()
        row.update(
            {
                "Displacement_Volume_m3": displacement_volume_m3,
                "Equilibrium_Draft_m": equilibrium_draft_m,
                "LCB_m": lcb_m,
                "VCB_m": vcb_m,
                "Waterplane_Area_m2": float(waterplane_area_m2),
                "LCF_m": float(lcf_m),
                "Transverse_Waterplane_Inertia_m4": float(i_transverse_m4),
                "Longitudinal_Waterplane_Inertia_m4": float(
                    i_longitudinal_m4
                ),
                "KM_T_m": km_transverse_m,
                "GM_T_Uncorrected_m": gm_transverse_uncorrected_m,
                "Free_Surface_Correction_m": free_surface_correction_m,
                "GM0_Corrected_m": corrected_gm_m,
                "HDF_007_GM_Residual_m": minimum_gm_residual_m,
                "HDF_007_Minimum_Corrected_GM_Passed": minimum_gm_passed,
                "GM_L_m": gm_longitudinal_m,
                "MCT_1cm_N_m": mct_1cm_n_m,
                "Trim_cm": trim_cm,
                "Draft_Forward_m": draft_forward_m,
                "Draft_Aft_m": draft_aft_m,
                "Equilibrium_Solved": True,
                "Upright_Stability_Status": status,
                "Complete_Intact_Stability_Compliance": False,
                "Calculation_Assumption_IDs": (
                    "HDF-ASM-013; HDF-ASM-030; HDF-ASM-031"
                ),
                "Source_Citations": (
                    "46 CFR 170.165; 46 CFR 170.285; "
                    "2008 IS Code Part A 2.2.1.5"
                ),
            }
        )
        rows.append(row)

    results = pd.DataFrame(rows)

    if "HDF_007_Minimum_Corrected_GM_Passed" in results:
        results["HDF_007_Minimum_Corrected_GM_Passed"] = results[
            "HDF_007_Minimum_Corrected_GM_Passed"
        ].astype("boolean")

    return results
