"""Atomic ordinary Type-B freeboard calculations for the GenISys audit.

The functions in this module implement the numerical pieces of 46 CFR
42.20-15 through 42.20-75 that are unambiguous for the approved ordinary,
unreduced Type-B branch.  They deliberately require formal Load Line inputs;
``LOA`` and the preliminary ``LBP`` are not substitutes for Load Line ``L``.

The aggregate result uses the approved GenISys assumptions that qualifying
position-1 hatch covers are present (HDF-ASM-027), that no superstructure or
trunk deduction is claimed, and that the sheer correction is zero
(HDF-ASM-017).  It is a numerical freeboard result, not proof that the opening
and other conditions of assignment in 46 CFR Part 42 are satisfied.
"""

from __future__ import annotations

import numpy as np


METRES_PER_FOOT = 0.3048
METRES_PER_INCH = 0.0254


# 46 CFR 42.20-15(b)(1) publishes the Type-B table in feet and inches.
# Keeping the source units here avoids adding rounded metric ordinates.
TYPE_B_LENGTHS_FT = np.arange(80.0, 1201.0, 10.0)
TYPE_B_FREEBOARDS_IN = np.asarray(
    [
        8.0, 8.9, 9.8, 10.8, 11.9, 13.0, 14.2, 15.5, 16.9, 18.3,
        19.8, 21.3, 22.9, 24.7, 26.6, 28.5, 30.4, 32.4, 34.4, 36.5,
        38.7, 41.0, 43.3, 45.7, 48.2, 50.7, 53.2, 55.7, 58.2, 60.7,
        63.2, 65.7, 68.2, 70.7, 73.2, 75.7, 78.2, 80.7, 83.1, 85.6,
        88.1, 90.6, 93.1, 95.6, 98.1, 100.6, 103.0, 105.4, 107.7,
        110.0, 112.3, 114.6, 116.8, 119.0, 121.1, 123.2, 125.3,
        127.3, 129.3, 131.3, 133.3, 135.3, 137.1, 139.0, 140.9,
        142.7, 144.5, 146.3, 148.1, 149.8, 151.5, 153.2, 154.8,
        156.4, 158.0, 159.6, 161.2, 162.8, 164.3, 165.9, 167.4,
        168.9, 170.4, 171.8, 173.3, 174.7, 176.1, 177.5, 178.9,
        180.3, 181.7, 183.1, 184.4, 185.8, 187.2, 188.5, 189.8,
        191.0, 192.3, 193.5, 194.8, 196.1, 197.3, 198.6, 199.9,
        201.2, 202.3, 203.5, 204.6, 205.8, 206.9, 208.1, 209.3,
    ],
    dtype=float,
)


def load_line_block_coefficient(
    molded_displacement_volume_m3: float,
    load_line_length_m: float,
    load_line_breadth_m: float,
    molded_draft_d1_m: float,
) -> float:
    """Return Load Line ``Cb`` using 46 CFR 42.13-15(g)."""

    return (
        molded_displacement_volume_m3
        / (
            load_line_length_m
            * load_line_breadth_m
            * molded_draft_d1_m
        )
    )


def type_b_tabular_freeboard_m(load_line_length_m: float) -> float:
    """Interpolate the 46 CFR 42.20-15(b)(1) Type-B table."""

    length_ft = load_line_length_m / METRES_PER_FOOT

    # The CFR directs individual determination above 1,200 ft.  The published
    # table begins at 80 ft, so this function does not extrapolate either end.
    if length_ft < TYPE_B_LENGTHS_FT[0] or length_ft > TYPE_B_LENGTHS_FT[-1]:
        raise ValueError(
            "46 CFR 42.20-15(b)(1) tabulation covers 80 to 1,200 ft; "
            "do not extrapolate the Type-B table."
        )

    freeboard_in = np.interp(
        length_ft,
        TYPE_B_LENGTHS_FT,
        TYPE_B_FREEBOARDS_IN,
    )

    return float(freeboard_in * METRES_PER_INCH)


def type_b_short_length_increase_m(
    load_line_length_m: float,
    effective_superstructure_length_m: float = 0.0,
) -> float:
    """Return the 46 CFR 42.20-20 Type-B increase below 328 ft.

    HDF-ASM-017 supplies an effective superstructure length of zero for the
    current conservative calculation.  The explicit argument remains useful
    for testing the source formula without awarding any project credit.
    """

    length_ft = load_line_length_m / METRES_PER_FOOT
    effective_length_ft = (
        effective_superstructure_length_m / METRES_PER_FOOT
    )

    if length_ft >= 328.0:
        return 0.0

    effective_length_ratio = effective_length_ft / length_ft

    if effective_length_ratio > 0.35:
        return 0.0

    increase_in = (
        0.09
        * (328.0 - length_ft)
        * (0.35 - effective_length_ratio)
    )

    return float(increase_in * METRES_PER_INCH)


def type_b_block_coefficient_factor(block_coefficient: float) -> float:
    """Return the multiplier prescribed by 46 CFR 42.20-25."""

    if block_coefficient <= 0.68:
        return 1.0

    return float((block_coefficient + 0.68) / 1.36)


def ordinary_type_b_depth_increase_m(
    load_line_length_m: float,
    depth_for_freeboard_m: float,
) -> float:
    """Return the 46 CFR 42.20-30 depth increase with no reduction credit.

    Under HDF-ASM-017, GenISys takes no enclosed-superstructure or complete-
    trunk credit.  Consequently, a depth below ``L/15`` produces zero change
    rather than the conditional reduction in paragraphs (b) and (c).
    """

    length_ft = load_line_length_m / METRES_PER_FOOT
    depth_ft = depth_for_freeboard_m / METRES_PER_FOOT
    excess_depth_ft = depth_ft - length_ft/15.0

    if excess_depth_ft <= 0.0:
        return 0.0

    if length_ft < 393.6:
        rate_in_per_ft = length_ft/131.2
    else:
        rate_in_per_ft = 3.0

    increase_in = excess_depth_ft * rate_in_per_ft

    return float(increase_in * METRES_PER_INCH)


def minimum_bow_height_m(
    load_line_length_m: float,
    block_coefficient: float,
) -> float:
    """Return the minimum bow height from 46 CFR 42.20-70(a)."""

    length_ft = load_line_length_m / METRES_PER_FOOT
    cb_for_bow_height = max(block_coefficient, 0.68)
    coefficient_factor = 1.36/(cb_for_bow_height + 0.68)

    if length_ft < 820.0:
        bow_height_in = (
            0.672
            * length_ft
            * (1.0 - length_ft/1640.0)
            * coefficient_factor
        )
    else:
        bow_height_in = 275.6 * coefficient_factor

    return float(bow_height_in * METRES_PER_INCH)


def calculate_standard_type_b_freeboard(
    load_line_length_m: float,
    depth_for_freeboard_m: float,
    block_coefficient: float,
    deck_line_correction_m: float,
    actual_bow_height_m: float | None = None,
) -> dict:
    """Calculate the approved ordinary Type-B numerical freeboard chain.

    ``deck_line_correction_m`` is positive when the actual depth to the upper
    edge of the deck line exceeds Load Line ``D``.  It must be supplied from
    the formal freeboard-deck/deck-line arrangement; the function does not
    assume that this correction is zero.

    ``actual_bow_height_m`` is optional because the current GenISys data model
    does not yet supply the regulation-defined bow height.  When it is absent,
    the arithmetic through summer freeboard is returned, but the full numeric
    HDF-097 chain is explicitly marked partial.
    """

    tabular_freeboard_m = type_b_tabular_freeboard_m(load_line_length_m)

    # HDF-ASM-027: qualifying position-1 covers mean the 42.20-5(b)
    # non-qualifying-cover surcharge is not applied.
    hatch_cover_increase_m = 0.0

    # HDF-ASM-017: take no effective-superstructure credit.  For the short-
    # vessel increase this is the conservative E = 0 source-formula case.
    short_length_increase_m = type_b_short_length_increase_m(
        load_line_length_m,
        effective_superstructure_length_m=0.0,
    )

    pre_cb_freeboard_m = (
        tabular_freeboard_m
        + hatch_cover_increase_m
        + short_length_increase_m
    )
    cb_factor = type_b_block_coefficient_factor(block_coefficient)
    after_cb_freeboard_m = pre_cb_freeboard_m * cb_factor

    depth_increase_m = ordinary_type_b_depth_increase_m(
        load_line_length_m,
        depth_for_freeboard_m,
    )

    # HDF-ASM-017 makes both later credits explicitly zero.
    superstructure_and_trunk_deduction_m = 0.0
    sheer_correction_m = 0.0

    pre_deck_line_freeboard_m = (
        after_cb_freeboard_m
        + depth_increase_m
        - superstructure_and_trunk_deduction_m
        + sheer_correction_m
    )

    # 46 CFR 42.20-75(a)(2) applies the two-inch floor without the deck-line
    # correction for the qualifying-cover branch selected by HDF-ASM-027.
    minimum_pre_deck_line_freeboard_m = 2.0 * METRES_PER_INCH
    minimum_freeboard_adjustment_m = max(
        minimum_pre_deck_line_freeboard_m - pre_deck_line_freeboard_m,
        0.0,
    )
    calculated_summer_freeboard_m = (
        pre_deck_line_freeboard_m
        + minimum_freeboard_adjustment_m
        + deck_line_correction_m
    )

    required_bow_height_m = minimum_bow_height_m(
        load_line_length_m,
        block_coefficient,
    )

    if actual_bow_height_m is None:
        bow_height_passed = None
        bow_height_deficiency_m = np.nan
        numeric_status = "partial_missing_regulation_defined_bow_height"
    else:
        bow_height_passed = actual_bow_height_m >= required_bow_height_m
        bow_height_deficiency_m = max(
            required_bow_height_m - actual_bow_height_m,
            0.0,
        )
        numeric_status = (
            "calculated_full_numeric_chain"
            if bow_height_passed
            else "calculated_bow_height_deficiency"
        )

    return {
        "Load_Line_Length_m": float(load_line_length_m),
        "Depth_For_Freeboard_m": float(depth_for_freeboard_m),
        "Load_Line_Block_Coefficient": float(block_coefficient),
        "Type_B_Tabular_Freeboard_m": tabular_freeboard_m,
        "Hatch_Cover_Increase_m": hatch_cover_increase_m,
        "Short_Length_Increase_m": short_length_increase_m,
        "Block_Coefficient_Factor": cb_factor,
        "Freeboard_After_Cb_Correction_m": after_cb_freeboard_m,
        "Depth_Increase_m": depth_increase_m,
        "Superstructure_And_Trunk_Deduction_m": (
            superstructure_and_trunk_deduction_m
        ),
        "Sheer_Correction_m": sheer_correction_m,
        "Minimum_Freeboard_Adjustment_m": minimum_freeboard_adjustment_m,
        "Deck_Line_Correction_m": float(deck_line_correction_m),
        "Calculated_Summer_Freeboard_m": calculated_summer_freeboard_m,
        "Required_Minimum_Bow_Height_m": required_bow_height_m,
        "Actual_Bow_Height_m": actual_bow_height_m,
        "Minimum_Bow_Height_Deficiency_m": bow_height_deficiency_m,
        "Minimum_Bow_Height_Passed": bow_height_passed,
        "HDF_097_Numerical_Chain_Complete": actual_bow_height_m is not None,
        "HDF_097_Numerical_Status": numeric_status,
        "Part_42_Compliance_Status": (
            "not_determined_opening_and_conditions_of_assignment_out_of_scope"
        ),
        "Source_Citations": (
            "46 CFR 42.13-15; 42.20-5; 42.20-15; 42.20-20; "
            "42.20-25; 42.20-30; 42.20-35; 42.20-60; "
            "42.20-65; 42.20-70; 42.20-75"
        ),
        "Assumption_IDs": "HDF-ASM-006; HDF-ASM-017; HDF-ASM-027",
    }
