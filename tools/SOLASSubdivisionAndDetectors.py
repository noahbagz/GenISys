"""Standalone SOLAS subdivision-index and water-level requirements.

This module implements four small, rule-ready numerical pieces identified in
the GenISys hydrostatic audit:

* HDF-014 and HDF-015: cargo-ship required subdivision index ``R`` under
  SOLAS II-1/6; and
* HDF-033 and HDF-034: alarm-level requirements under SOLAS XII/12.

The functions require formal regulatory dimensions and explicit applicability
facts.  The HDF-033 alarm-height arithmetic is connected to the V1.1 adapter
as a status-aware partial metric; HDF-014/HDF-015 and HDF-034 remain standalone.
The module does not infer SOLAS ``L`` or ``Ls`` from LOA, select physical
detector coordinates, or represent approval/testing of a water-level detection
system.
"""

from __future__ import annotations

import math

import numpy as np


def _positive_finite(value: float, name: str) -> float:
    """Return one positive physical dimension used by a source formula."""

    numeric_value = float(value)

    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")

    return numeric_value


def cargo_ship_required_subdivision_index_over_100_m(
    subdivision_length_ls_m: float,
) -> float:
    """Return HDF-014 ``R`` for a cargo ship with ``Ls > 100 m``.

    SOLAS II-1/6.2.1:

    ``R = 1 - 128/(Ls + 152)``.
    """

    subdivision_length_ls_m = _positive_finite(
        subdivision_length_ls_m,
        "SOLAS subdivision length Ls",
    )

    if subdivision_length_ls_m <= 100.0:
        raise ValueError(
            "The SOLAS II-1/6.2.1 branch requires Ls greater than 100 m."
        )

    return float(
        1.0 - 128.0/(subdivision_length_ls_m + 152.0)
    )


def cargo_ship_required_subdivision_index_not_over_100_m(
    subdivision_length_ls_m: float,
) -> float:
    """Return HDF-015 ``R`` for the applicable ``Ls <= 100 m`` branch.

    SOLAS II-1/6.2.2 defines ``R0`` from the long-ship expression and then:

    ``R = 1 - 1/[1 + (Ls/100)*(R0/(1-R0))]``.

    The current applicability chapeau separately uses formal SOLAS length
    ``L >= 80 m``.  This arithmetic helper therefore takes only ``Ls``; the
    aggregate evaluator below keeps the distinct ``L`` gate explicit.
    """

    subdivision_length_ls_m = _positive_finite(
        subdivision_length_ls_m,
        "SOLAS subdivision length Ls",
    )

    if subdivision_length_ls_m > 100.0:
        raise ValueError(
            "The SOLAS II-1/6.2.2 branch requires Ls not greater than 100 m."
        )

    r_zero = (
        1.0 - 128.0/(subdivision_length_ls_m + 152.0)
    )
    length_factor = subdivision_length_ls_m/100.0

    return float(
        1.0
        - 1.0/(
            1.0 + length_factor*r_zero/(1.0 - r_zero)
        )
    )


def evaluate_cargo_ship_required_subdivision_index(
    *,
    formal_solas_length_l_m: float | None = None,
    subdivision_length_ls_m: float | None = None,
    otherwise_subject_to_solas_ii_1_6_to_7_3: bool | None = None,
) -> dict:
    """Return the candidate formula and applicable cargo-ship requirement.

    ``otherwise_subject_to_solas_ii_1_6_to_7_3`` must resolve the non-length
    applicability and alternative-instrument exclusions in SOLAS II-1/4.
    This function applies the current length gate separately: formal SOLAS
    ``L`` must be at least 80 m.  It never substitutes LOA for ``L`` or ``Ls``.

    A calculated ``R`` is only the required-index side of the rule.  Attained
    index ``A``, the partial-index minima, and ``A >= R`` are not evaluated.
    """

    missing_inputs = []

    if formal_solas_length_l_m is None:
        formal_length = None
        missing_inputs.append("Formal_SOLAS_Length_L_m")
    else:
        formal_length = _positive_finite(
            formal_solas_length_l_m,
            "Formal SOLAS length L",
        )

    if subdivision_length_ls_m is None:
        subdivision_length = None
        formula_r = None
        formula_branch = None
        hdf_rule_id = None
        missing_inputs.append("Subdivision_Length_Ls_m")
    else:
        subdivision_length = _positive_finite(
            subdivision_length_ls_m,
            "SOLAS subdivision length Ls",
        )

        if subdivision_length > 100.0:
            formula_r = (
                cargo_ship_required_subdivision_index_over_100_m(
                    subdivision_length
                )
            )
            formula_branch = "SOLAS_II-1_6.2.1_Ls_greater_than_100_m"
            hdf_rule_id = "HDF-014"
        else:
            formula_r = (
                cargo_ship_required_subdivision_index_not_over_100_m(
                    subdivision_length
                )
            )
            formula_branch = "SOLAS_II-1_6.2.2_Ls_not_over_100_m"
            hdf_rule_id = "HDF-015"

    # Determine applicability without guessing whether an alternative SOLAS
    # instrument displaces regulations 6 through 7-3 for the actual vessel.
    if otherwise_subject_to_solas_ii_1_6_to_7_3 is False:
        rule_applicable = False
        applicability_status = "not_applicable_other_solas_basis_selected"

    elif formal_length is not None and formal_length < 80.0:
        rule_applicable = False
        applicability_status = "not_applicable_formal_solas_L_below_80_m"

    elif (
        otherwise_subject_to_solas_ii_1_6_to_7_3 is True
        and formal_length is not None
        and formal_length >= 80.0
    ):
        rule_applicable = True
        applicability_status = "applicable"

    else:
        rule_applicable = None
        applicability_status = "applicability_unresolved"

        if otherwise_subject_to_solas_ii_1_6_to_7_3 is None:
            missing_inputs.append(
                "Otherwise_Subject_To_SOLAS_II-1_6_To_7-3"
            )

    required_r = formula_r if rule_applicable is True else None
    requirement_complete = (
        rule_applicable is True and formula_r is not None
    )

    if rule_applicable is False:
        calculation_status = applicability_status
        compliance_status = "not_applicable"
        missing_inputs = []

    elif formula_r is None:
        calculation_status = "incomplete_missing_regulatory_length"
        compliance_status = "not_determined_incomplete_inputs"

    elif rule_applicable is None:
        calculation_status = (
            "formula_calculated_applicability_unresolved"
        )
        compliance_status = "not_determined_applicability_unresolved"

    else:
        calculation_status = (
            "required_index_calculated_attained_index_not_evaluated"
        )
        compliance_status = "not_determined_attained_A_not_evaluated"

    return {
        "Formal_SOLAS_Length_L_m": formal_length,
        "Subdivision_Length_Ls_m": subdivision_length,
        "Otherwise_Subject_To_SOLAS_II-1_6_To_7-3": (
            otherwise_subject_to_solas_ii_1_6_to_7_3
        ),
        "Formula_Branch": formula_branch,
        "HDF_Rule_ID": hdf_rule_id,
        "Calculated_Formula_R": formula_r,
        "Required_Subdivision_Index_R": required_r,
        "Rule_Applicable": rule_applicable,
        "Applicability_Status": applicability_status,
        "Numeric_Formula_Complete": formula_r is not None,
        "Rule_Requirement_Complete": requirement_complete,
        "Missing_Inputs": missing_inputs,
        "Calculation_Status": calculation_status,
        "Subdivision_Compliance_Status": compliance_status,
        "Source_Citations": (
            "46 CFR 174.360; SOLAS II-1/4 and II-1/6; "
            "MSC.216(82); MSC.421(98)"
        ),
    }


def cargo_hold_water_level_alarm_requirements(
    *,
    inner_bottom_elevation_m: float | None = None,
    cargo_hold_depth_m: float | None = None,
    solas_xii_12_applies: bool | None = None,
    solas_xii_9_2_single_alarm_exception_applies: bool | None = None,
) -> dict:
    """Return HDF-033 cargo-hold alarm-level requirements.

    SOLAS XII/12.1 requires one alarm at 0.5 m above the inner bottom and a
    second at a height not less than 15% of hold depth and not more than 2 m.
    The function preserves those two second-alarm bounds independently.  It
    does not choose a physical sensor/setpoint or silently reinterpret the
    source wording when the calculated bounds do not overlap.
    """

    missing_inputs = []

    if inner_bottom_elevation_m is None:
        inner_bottom_elevation = None
        first_alarm_elevation = None
        main_alarm_minimum_elevation = None
        main_alarm_maximum_elevation = None
        missing_inputs.append("Inner_Bottom_Elevation_m")
    else:
        inner_bottom_elevation = float(inner_bottom_elevation_m)

        if not math.isfinite(inner_bottom_elevation):
            raise ValueError("Inner-bottom elevation must be finite.")

        first_alarm_elevation = inner_bottom_elevation + 0.5
        main_alarm_minimum_elevation = None
        main_alarm_maximum_elevation = inner_bottom_elevation + 2.0

    if cargo_hold_depth_m is None:
        cargo_hold_depth = None
        main_alarm_minimum_height = None
        source_bounds_overlap = None
        missing_inputs.append("Cargo_Hold_Depth_m")
    else:
        cargo_hold_depth = _positive_finite(
            cargo_hold_depth_m,
            "Cargo-hold depth",
        )
        main_alarm_minimum_height = 0.15*cargo_hold_depth
        source_bounds_overlap = main_alarm_minimum_height <= 2.0

        if inner_bottom_elevation is not None:
            main_alarm_minimum_elevation = (
                inner_bottom_elevation + main_alarm_minimum_height
            )

    if solas_xii_12_applies is False:
        rule_applicable = False
        first_alarm_required = False
        main_alarm_required = False
        applicability_status = "not_applicable"

    elif solas_xii_12_applies is True:
        rule_applicable = True
        main_alarm_required = True
        applicability_status = "applicable"

        if solas_xii_9_2_single_alarm_exception_applies is None:
            first_alarm_required = None
            missing_inputs.append(
                "SOLAS_XII_9_2_Single_Alarm_Exception_Applies"
            )
        else:
            first_alarm_required = not bool(
                solas_xii_9_2_single_alarm_exception_applies
            )

    else:
        rule_applicable = None
        first_alarm_required = None
        main_alarm_required = None
        applicability_status = "applicability_unresolved"
        missing_inputs.append("SOLAS_XII_12_Applies")

    numeric_requirements_complete = (
        inner_bottom_elevation is not None
        and cargo_hold_depth is not None
    )

    if rule_applicable is False:
        calculation_status = "not_applicable"
        compliance_status = "not_applicable"
        missing_inputs = []

    elif not numeric_requirements_complete:
        calculation_status = "incomplete_missing_hold_geometry"
        compliance_status = "not_determined_incomplete_inputs"

    elif source_bounds_overlap is False:
        calculation_status = (
            "source_second_alarm_bounds_do_not_overlap_"
            "setpoint_interpretation_required"
        )
        compliance_status = (
            "not_determined_requires_regulatory_interpretation"
        )

    elif rule_applicable is None:
        calculation_status = (
            "numeric_alarm_levels_calculated_applicability_unresolved"
        )
        compliance_status = "not_determined_applicability_unresolved"

    elif first_alarm_required is None:
        calculation_status = (
            "numeric_alarm_levels_calculated_first_alarm_exception_"
            "unresolved"
        )
        compliance_status = "not_determined_exception_unresolved"

    else:
        calculation_status = (
            "numeric_alarm_bounds_calculated_detector_not_evaluated"
        )
        compliance_status = (
            "not_determined_requires_detector_design_and_approval"
        )

    return {
        "HDF_Rule_ID": "HDF-033",
        "Inner_Bottom_Elevation_m": inner_bottom_elevation,
        "Cargo_Hold_Depth_m": cargo_hold_depth,
        "First_Alarm_Height_Above_Inner_Bottom_m": 0.5,
        "First_Alarm_Elevation_m": first_alarm_elevation,
        "Main_Alarm_Minimum_Height_Above_Inner_Bottom_m": (
            main_alarm_minimum_height
        ),
        "Main_Alarm_Maximum_Height_Above_Inner_Bottom_m": 2.0,
        "Main_Alarm_Minimum_Elevation_m": (
            main_alarm_minimum_elevation
        ),
        "Main_Alarm_Maximum_Elevation_m": (
            main_alarm_maximum_elevation
        ),
        "Main_Alarm_Source_Bounds_Overlap": source_bounds_overlap,
        "Selected_Main_Alarm_Setpoint_m": None,
        "First_Alarm_Required": first_alarm_required,
        "Main_Alarm_Required": main_alarm_required,
        "Rule_Applicable": rule_applicable,
        "Applicability_Status": applicability_status,
        "Numeric_Requirements_Complete": numeric_requirements_complete,
        "Physical_Detector_Coordinates_Determined": False,
        "Missing_Inputs": missing_inputs,
        "Calculation_Status": calculation_status,
        "Detector_Compliance_Status": compliance_status,
        "Source_Citations": (
            "SOLAS XII/12.1; MSC.170(79); "
            "MSC.188(79)/Rev.2; MSC.1/Circ.1572/Rev.2"
        ),
    }


def forward_ballast_tank_water_level_alarm_requirement(
    *,
    tank_capacity_m3: float | None = None,
    capacity_curve_level_elevations_m=None,
    capacity_curve_cumulative_volumes_m3=None,
    solas_xii_12_applies: bool | None = None,
    tank_is_forward_of_collision_bulkhead: bool | None = None,
) -> dict:
    """Return HDF-034's forward-ballast alarm capacity/level limit.

    SOLAS XII/12.1.2 requires alarm before the tank exceeds 10% capacity.  A
    capacity curve is optional because the fraction can be calculated from
    total volume alone.  When supplied, it is inverted to give the maximum
    corresponding liquid-level elevation; no sensor coordinate is selected.
    """

    missing_inputs = []

    if tank_capacity_m3 is None:
        tank_capacity = None
        maximum_alarm_volume = None
        missing_inputs.append("Tank_Capacity_m3")
    else:
        tank_capacity = _positive_finite(
            tank_capacity_m3,
            "Forward ballast-tank capacity",
        )
        maximum_alarm_volume = 0.10*tank_capacity

    has_level_curve = capacity_curve_level_elevations_m is not None
    has_volume_curve = capacity_curve_cumulative_volumes_m3 is not None

    if has_level_curve != has_volume_curve:
        raise ValueError(
            "Capacity-curve elevations and cumulative volumes must be "
            "supplied together."
        )

    maximum_alarm_level_elevation = None
    capacity_curve_complete = False

    if has_level_curve and maximum_alarm_volume is not None:
        level_elevations = np.asarray(
            capacity_curve_level_elevations_m,
            dtype=float,
        )
        cumulative_volumes = np.asarray(
            capacity_curve_cumulative_volumes_m3,
            dtype=float,
        )

        if (
            level_elevations.ndim != 1
            or cumulative_volumes.ndim != 1
            or len(level_elevations) != len(cumulative_volumes)
            or len(level_elevations) < 2
        ):
            raise ValueError(
                "A tank capacity curve requires matching one-dimensional "
                "level and cumulative-volume arrays."
            )

        if (
            np.any(np.diff(level_elevations) <= 0.0)
            or np.any(np.diff(cumulative_volumes) <= 0.0)
        ):
            raise ValueError(
                "Tank capacity-curve levels and cumulative volumes must be "
                "strictly increasing."
            )

        if (
            maximum_alarm_volume < cumulative_volumes[0]
            or maximum_alarm_volume > cumulative_volumes[-1]
        ):
            raise ValueError(
                "The supplied tank capacity curve does not bracket 10% of "
                "the stated tank capacity."
            )

        maximum_alarm_level_elevation = float(
            np.interp(
                maximum_alarm_volume,
                cumulative_volumes,
                level_elevations,
            )
        )
        capacity_curve_complete = True

    if solas_xii_12_applies is False:
        rule_applicable = False
        applicability_status = "not_applicable_solas_xii_12"

    elif tank_is_forward_of_collision_bulkhead is False:
        rule_applicable = False
        applicability_status = "not_applicable_tank_not_forward_of_collision_bulkhead"

    elif (
        solas_xii_12_applies is True
        and tank_is_forward_of_collision_bulkhead is True
    ):
        rule_applicable = True
        applicability_status = "applicable"

    else:
        rule_applicable = None
        applicability_status = "applicability_unresolved"

        if solas_xii_12_applies is None:
            missing_inputs.append("SOLAS_XII_12_Applies")

        if tank_is_forward_of_collision_bulkhead is None:
            missing_inputs.append(
                "Tank_Is_Forward_Of_Collision_Bulkhead"
            )

    if rule_applicable is False:
        calculation_status = applicability_status
        compliance_status = "not_applicable"
        missing_inputs = []

    elif maximum_alarm_volume is None:
        calculation_status = "incomplete_missing_tank_capacity"
        compliance_status = "not_determined_incomplete_inputs"

    elif rule_applicable is None:
        calculation_status = (
            "capacity_limit_calculated_applicability_unresolved"
        )
        compliance_status = "not_determined_applicability_unresolved"

    elif not capacity_curve_complete:
        calculation_status = (
            "capacity_limit_calculated_level_requires_capacity_curve"
        )
        compliance_status = (
            "not_determined_requires_tank_curve_detector_and_approval"
        )

    else:
        calculation_status = (
            "maximum_alarm_level_calculated_detector_not_evaluated"
        )
        compliance_status = (
            "not_determined_requires_detector_design_and_approval"
        )

    return {
        "HDF_Rule_ID": "HDF-034",
        "Tank_Capacity_m3": tank_capacity,
        "Maximum_Alarm_Activation_Capacity_Fraction": 0.10,
        "Maximum_Alarm_Activation_Volume_m3": maximum_alarm_volume,
        "Maximum_Alarm_Activation_Level_Elevation_m": (
            maximum_alarm_level_elevation
        ),
        "Tank_Capacity_Curve_Complete": capacity_curve_complete,
        "Selected_Alarm_Setpoint_m": None,
        "Rule_Applicable": rule_applicable,
        "Applicability_Status": applicability_status,
        "Physical_Detector_Coordinates_Determined": False,
        "Missing_Inputs": missing_inputs,
        "Calculation_Status": calculation_status,
        "Detector_Compliance_Status": compliance_status,
        "Source_Citations": (
            "SOLAS XII/12.1.2; MSC.170(79); "
            "MSC.188(79)/Rev.2; MSC.1/Circ.1572/Rev.2"
        ),
    }


def forward_dry_space_water_level_alarm_requirement(
    *,
    deck_elevation_m: float | None = None,
    enclosed_space_volume_m3: float | None = None,
    ship_maximum_displacement_volume_m3: float | None = None,
    solas_xii_12_applies: bool | None = None,
    space_is_dry_or_void: bool | None = None,
    any_part_extends_forward_of_foremost_cargo_hold: bool | None = None,
    space_is_chain_cable_locker: bool | None = None,
) -> dict:
    """Return HDF-034's forward dry/void-space alarm requirement.

    SOLAS XII/12.1.3 sets the alarm at 0.1 m above the deck.  Chain cable
    lockers are excluded.  An enclosed space not exceeding 0.1% of maximum
    displacement volume is eligible to omit the alarm.
    """

    missing_inputs = []

    if deck_elevation_m is None:
        deck_elevation = None
        alarm_level_elevation = None
        missing_inputs.append("Deck_Elevation_m")
    else:
        deck_elevation = float(deck_elevation_m)

        if not math.isfinite(deck_elevation):
            raise ValueError("Dry-space deck elevation must be finite.")

        alarm_level_elevation = deck_elevation + 0.1

    if ship_maximum_displacement_volume_m3 is None:
        maximum_exempt_space_volume = None
        missing_inputs.append("Ship_Maximum_Displacement_Volume_m3")
    else:
        ship_maximum_displacement_volume = _positive_finite(
            ship_maximum_displacement_volume_m3,
            "Ship maximum displacement volume",
        )
        maximum_exempt_space_volume = (
            0.001*ship_maximum_displacement_volume
        )

    if enclosed_space_volume_m3 is None:
        enclosed_space_volume = None
        small_space_exemption_eligible = None
        missing_inputs.append("Enclosed_Space_Volume_m3")
    else:
        enclosed_space_volume = _positive_finite(
            enclosed_space_volume_m3,
            "Enclosed dry/void-space volume",
        )

        if maximum_exempt_space_volume is None:
            small_space_exemption_eligible = None
        else:
            small_space_exemption_eligible = (
                enclosed_space_volume <= maximum_exempt_space_volume
            )

    # Resolve the space scope separately from the optional small-volume
    # exemption so the output explains why an alarm is or is not mandatory.
    scope_facts = (
        solas_xii_12_applies,
        space_is_dry_or_void,
        any_part_extends_forward_of_foremost_cargo_hold,
        space_is_chain_cable_locker,
    )

    if (
        solas_xii_12_applies is False
        or space_is_dry_or_void is False
        or any_part_extends_forward_of_foremost_cargo_hold is False
        or space_is_chain_cable_locker is True
    ):
        space_within_rule_scope = False
        alarm_required = False
        applicability_status = "not_applicable_space_outside_rule_scope"

    elif all(value is not None for value in scope_facts):
        space_within_rule_scope = True
        applicability_status = "space_within_rule_scope"

        if small_space_exemption_eligible is None:
            alarm_required = None
        else:
            alarm_required = not small_space_exemption_eligible

    else:
        space_within_rule_scope = None
        alarm_required = None
        applicability_status = "space_scope_unresolved"

        if solas_xii_12_applies is None:
            missing_inputs.append("SOLAS_XII_12_Applies")

        if space_is_dry_or_void is None:
            missing_inputs.append("Space_Is_Dry_Or_Void")

        if any_part_extends_forward_of_foremost_cargo_hold is None:
            missing_inputs.append(
                "Any_Part_Extends_Forward_Of_Foremost_Cargo_Hold"
            )

        if space_is_chain_cable_locker is None:
            missing_inputs.append("Space_Is_Chain_Cable_Locker")

    if space_within_rule_scope is False:
        calculation_status = applicability_status
        compliance_status = "not_applicable"
        missing_inputs = []

    elif alarm_level_elevation is None:
        calculation_status = "incomplete_missing_deck_elevation"
        compliance_status = "not_determined_incomplete_inputs"

    elif space_within_rule_scope is None:
        calculation_status = (
            "numeric_alarm_level_calculated_space_scope_unresolved"
        )
        compliance_status = "not_determined_applicability_unresolved"

    elif small_space_exemption_eligible is None:
        calculation_status = (
            "numeric_alarm_level_calculated_exemption_unresolved"
        )
        compliance_status = "not_determined_exemption_unresolved"

    elif small_space_exemption_eligible:
        calculation_status = "small_space_exemption_eligible"
        compliance_status = "alarm_not_required_by_solas_xii_12_1_3"

    else:
        calculation_status = (
            "required_alarm_level_calculated_detector_not_evaluated"
        )
        compliance_status = (
            "not_determined_requires_detector_design_and_approval"
        )

    return {
        "HDF_Rule_ID": "HDF-034",
        "Deck_Elevation_m": deck_elevation,
        "Required_Alarm_Height_Above_Deck_m": 0.1,
        "Required_Alarm_Level_Elevation_m": alarm_level_elevation,
        "Enclosed_Space_Volume_m3": enclosed_space_volume,
        "Maximum_Exempt_Space_Fraction_Of_Max_Displacement": 0.001,
        "Maximum_Exempt_Space_Volume_m3": maximum_exempt_space_volume,
        "Small_Space_Exemption_Eligible": (
            small_space_exemption_eligible
        ),
        "Space_Within_Rule_Scope": space_within_rule_scope,
        "Alarm_Required": alarm_required,
        "Applicability_Status": applicability_status,
        "Physical_Detector_Coordinates_Determined": False,
        "Missing_Inputs": missing_inputs,
        "Calculation_Status": calculation_status,
        "Detector_Compliance_Status": compliance_status,
        "Source_Citations": (
            "SOLAS XII/12.1.3; MSC.170(79); "
            "MSC.188(79)/Rev.2; MSC.1/Circ.1572/Rev.2"
        ),
    }
