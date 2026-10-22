"""Standalone preliminary EEDI arithmetic for a conventional bulk carrier.

This module implements only the unambiguous arithmetic that can be evaluated
after a caller supplies the regulation-defined inputs.  Its reference-line,
phase-schedule, and required-EEDI functions are connected to V1.1 as explicitly
labelled evaluated-draft proxy metrics; the attained-EEDI calculation remains
standalone.  In particular, it does not reinterpret the current operating-
condition speed as ``Vref`` and it never reports statutory EEDI compliance.

The calculation basis is:

* MARPOL Annex VI, regulation 24 (MEPC.328(76)): bulk-carrier reference line
  and reduction-factor table;
* MEPC.364(79), paragraphs 2.1 and 2.2: attained-EEDI equation and terms; and
* MEPC.410(84): 2026 amendment to the calculation guidelines.  Its dual-fuel
  changes do not alter the conventional single-fuel diesel branch here.

The numerical branch excludes shaft generators (PTO), propulsion motors
(PTI), hybrid/electric propulsion, dual-fuel engines, steam turbines, and
innovative-energy-efficiency deductions.  Those configurations need the
additional terms in the full MEPC.364(79) equation.
"""

from __future__ import annotations

import math


# MARPOL Annex VI regulation 24, table 2: reference-line parameters for a
# bulk carrier.  The regulation defines b as the lesser of DWT and 279,000 t.
BULK_CARRIER_REFERENCE_A = 961.79
BULK_CARRIER_REFERENCE_C = 0.477
BULK_CARRIER_REFERENCE_DWT_CAP_T = 279_000.0


# MEPC.364(79), paragraph 2.2.5.1: for a conventional main engine, PME is
# 75 percent of the engine's rated installed power MCR.
MAIN_ENGINE_EEDI_LOAD_FACTOR = 0.75


# MARPOL Annex VI regulation 24, table 1.  Phase 4 has not been adopted in
# the regulation as of 2026, so it is deliberately absent from this mapping.
BULK_CARRIER_MAX_REDUCTION_PERCENT_BY_PHASE = {
    0: 0.0,
    1: 10.0,
    2: 20.0,
    3: 30.0,
}


def _require_positive(value: float, name: str) -> float:
    """Return a positive finite numeric input used in a denominator."""

    numeric_value = float(value)

    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")

    return numeric_value


def _require_nonnegative(value: float, name: str) -> float:
    """Return a non-negative finite numeric input."""

    numeric_value = float(value)

    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")

    return numeric_value


def bulk_carrier_reference_eedi(deadweight_tonnes: float) -> float:
    """Return the bulk-carrier EEDI reference line in g CO2/(t nm).

    MARPOL Annex VI regulation 24, table 2 defines the reference value as
    ``a * b**(-c)`` with ``a = 961.79``, ``c = 0.477``, and
    ``b = min(DWT, 279000)`` for a bulk carrier.
    """

    deadweight_tonnes = _require_positive(
        deadweight_tonnes,
        "Deadweight",
    )
    reference_deadweight_tonnes = min(
        deadweight_tonnes,
        BULK_CARRIER_REFERENCE_DWT_CAP_T,
    )

    return float(
        BULK_CARRIER_REFERENCE_A
        * reference_deadweight_tonnes**(-BULK_CARRIER_REFERENCE_C)
    )


def bulk_carrier_reduction_factor_percent(
    deadweight_tonnes: float,
    phase: int,
) -> float | None:
    """Return the regulation-24 bulk-carrier reduction factor ``X``.

    ``None`` means that the selected phase/size cell in regulation 24,
    table 1 is not applicable.  For 10,000 to 20,000 DWT, the regulation
    requires linear interpolation from zero at 10,000 DWT to the phase's
    maximum reduction at 20,000 DWT.
    """

    deadweight_tonnes = _require_positive(
        deadweight_tonnes,
        "Deadweight",
    )

    if phase not in BULK_CARRIER_MAX_REDUCTION_PERCENT_BY_PHASE:
        raise ValueError(
            "EEDI phase must be one of 0, 1, 2, or 3 under the "
            "current MARPOL Annex VI regulation-24 table."
        )

    # MARPOL Annex VI regulation 24 applies the bulk-carrier requirement at
    # 10,000 DWT and above.  Smaller ships have no table-1 reduction value.
    if deadweight_tonnes < 10_000.0:
        return None

    # Phase 0 has a value only for bulk carriers of at least 20,000 DWT.
    if phase == 0 and deadweight_tonnes < 20_000.0:
        return None

    maximum_reduction_percent = (
        BULK_CARRIER_MAX_REDUCTION_PERCENT_BY_PHASE[phase]
    )

    if deadweight_tonnes >= 20_000.0:
        return maximum_reduction_percent

    interpolation_fraction = (
        (deadweight_tonnes - 10_000.0) / 10_000.0
    )

    return float(maximum_reduction_percent * interpolation_fraction)


def required_bulk_carrier_eedi(
    deadweight_tonnes: float,
    reduction_factor_percent: float,
) -> float:
    """Return required EEDI from regulation 24, paragraph 3.

    The mandatory relation is ``required = (1 - X/100) * reference``.
    Applicability and the correct value of ``X`` remain the caller's
    responsibility when this low-level function is used directly.
    """

    reduction_factor_percent = _require_nonnegative(
        reduction_factor_percent,
        "EEDI reduction factor",
    )

    if reduction_factor_percent >= 100.0:
        raise ValueError("EEDI reduction factor must be less than 100 percent.")

    reference_eedi = bulk_carrier_reference_eedi(deadweight_tonnes)

    return float(
        (1.0 - reduction_factor_percent/100.0) * reference_eedi
    )


def conventional_main_engine_eedi_power_kw(
    total_main_engine_mcr_kw: float,
) -> float:
    """Return aggregate ``PME`` under MEPC.364(79), paragraph 2.2.5.1."""

    total_main_engine_mcr_kw = _require_positive(
        total_main_engine_mcr_kw,
        "Total main-engine MCR",
    )

    return float(
        MAIN_ENGINE_EEDI_LOAD_FACTOR * total_main_engine_mcr_kw
    )


def generic_auxiliary_eedi_power_kw(
    total_main_engine_mcr_kw: float,
) -> float:
    """Return generic ``PAE`` from MEPC.364(79), paragraph 2.2.5.6.

    This is the simple no-PTI branch.  A ship for which the guideline requires
    an electric power table (EPT) must use the verified EPT value instead and
    is outside this preliminary module's scope.
    """

    total_main_engine_mcr_kw = _require_positive(
        total_main_engine_mcr_kw,
        "Total main-engine MCR",
    )

    if total_main_engine_mcr_kw >= 10_000.0:
        auxiliary_power_kw = (
            0.025 * total_main_engine_mcr_kw + 250.0
        )
    else:
        auxiliary_power_kw = 0.05 * total_main_engine_mcr_kw

    return float(auxiliary_power_kw)


def calculate_attained_eedi_conventional_diesel(
    *,
    total_main_engine_mcr_kw: float,
    reference_speed_knots: float,
    deadweight_tonnes: float,
    main_engine_sfc_g_per_kwh: float,
    auxiliary_engine_sfc_g_per_kwh: float,
    main_engine_cf_t_co2_per_t_fuel: float,
    auxiliary_engine_cf_t_co2_per_t_fuel: float,
    ship_specific_power_factor_fj: float,
    capacity_factor_fi: float,
    cubic_capacity_factor_fc: float,
    cargo_gear_factor_fl: float,
    weather_factor_fw: float,
    ice_class_factor_fm: float,
) -> dict:
    """Calculate a preliminary attained EEDI for the bounded diesel branch.

    Every fuel and correction factor is explicit.  This prevents a unit value
    from being silently awarded when the factor's regulatory applicability has
    not yet been established for a GenISys design.

    The result is arithmetic only.  MEPC.411(84) requires an EEDI Technical
    File, model-test/power-curve evidence, preliminary verification, and final
    sea-trial verification before statutory compliance can be established.
    """

    total_main_engine_mcr_kw = _require_positive(
        total_main_engine_mcr_kw,
        "Total main-engine MCR",
    )
    reference_speed_knots = _require_positive(
        reference_speed_knots,
        "Reference speed Vref",
    )
    deadweight_tonnes = _require_positive(
        deadweight_tonnes,
        "Deadweight",
    )
    main_engine_sfc_g_per_kwh = _require_positive(
        main_engine_sfc_g_per_kwh,
        "Main-engine SFC",
    )
    auxiliary_engine_sfc_g_per_kwh = _require_positive(
        auxiliary_engine_sfc_g_per_kwh,
        "Auxiliary-engine SFC",
    )
    main_engine_cf_t_co2_per_t_fuel = _require_positive(
        main_engine_cf_t_co2_per_t_fuel,
        "Main-engine carbon factor CF",
    )
    auxiliary_engine_cf_t_co2_per_t_fuel = _require_positive(
        auxiliary_engine_cf_t_co2_per_t_fuel,
        "Auxiliary-engine carbon factor CF",
    )

    # MEPC.364(79), paragraph 2.1 places fj on the main-engine emissions
    # contribution.  fi, fc, fl, fw, and fm multiply the transport-work
    # denominator.  Combined/product values must be supplied where more than
    # one instance of a factor applies to the actual ship.
    ship_specific_power_factor_fj = _require_positive(
        ship_specific_power_factor_fj,
        "Ship-specific power factor fj",
    )
    capacity_factor_fi = _require_positive(
        capacity_factor_fi,
        "Capacity factor fi",
    )
    cubic_capacity_factor_fc = _require_positive(
        cubic_capacity_factor_fc,
        "Cubic-capacity factor fc",
    )
    cargo_gear_factor_fl = _require_positive(
        cargo_gear_factor_fl,
        "Cargo-gear factor fl",
    )
    weather_factor_fw = _require_positive(
        weather_factor_fw,
        "Weather factor fw",
    )
    ice_class_factor_fm = _require_positive(
        ice_class_factor_fm,
        "Ice-class factor fm",
    )

    main_engine_power_kw = conventional_main_engine_eedi_power_kw(
        total_main_engine_mcr_kw,
    )
    auxiliary_engine_power_kw = generic_auxiliary_eedi_power_kw(
        total_main_engine_mcr_kw,
    )

    # PME [kW] * SFC [g fuel/kWh] * CF [g CO2/g fuel] gives an hourly
    # CO2-mass rate [g CO2/h].  CF is numerically dimensionless when supplied
    # in t CO2/t fuel, as it is in the IMO table.
    main_engine_co2_rate_g_per_h = (
        ship_specific_power_factor_fj
        * main_engine_power_kw
        * main_engine_cf_t_co2_per_t_fuel
        * main_engine_sfc_g_per_kwh
    )
    auxiliary_engine_co2_rate_g_per_h = (
        auxiliary_engine_power_kw
        * auxiliary_engine_cf_t_co2_per_t_fuel
        * auxiliary_engine_sfc_g_per_kwh
    )
    total_co2_rate_g_per_h = (
        main_engine_co2_rate_g_per_h
        + auxiliary_engine_co2_rate_g_per_h
    )

    # DWT [t] * Vref [nm/h] is transport work [t nm/h].  Division therefore
    # leaves the EEDI unit g CO2/(t nm), as required by MEPC.364(79).
    corrected_transport_work_t_nm_per_h = (
        capacity_factor_fi
        * cubic_capacity_factor_fc
        * cargo_gear_factor_fl
        * deadweight_tonnes
        * weather_factor_fw
        * reference_speed_knots
        * ice_class_factor_fm
    )
    attained_eedi = (
        total_co2_rate_g_per_h / corrected_transport_work_t_nm_per_h
    )

    return {
        "Total_Main_Engine_MCR_kW": total_main_engine_mcr_kw,
        "Main_Engine_EEDI_Power_PME_kW": main_engine_power_kw,
        "Auxiliary_Engine_EEDI_Power_PAE_kW": (
            auxiliary_engine_power_kw
        ),
        "Reference_Speed_Vref_knots": reference_speed_knots,
        "Capacity_DWT_tonnes": deadweight_tonnes,
        "Main_Engine_SFC_g_per_kWh": main_engine_sfc_g_per_kwh,
        "Auxiliary_Engine_SFC_g_per_kWh": (
            auxiliary_engine_sfc_g_per_kwh
        ),
        "Main_Engine_CF_tCO2_per_tFuel": (
            main_engine_cf_t_co2_per_t_fuel
        ),
        "Auxiliary_Engine_CF_tCO2_per_tFuel": (
            auxiliary_engine_cf_t_co2_per_t_fuel
        ),
        "Ship_Specific_Power_Factor_fj": ship_specific_power_factor_fj,
        "Capacity_Factor_fi": capacity_factor_fi,
        "Cubic_Capacity_Factor_fc": cubic_capacity_factor_fc,
        "Cargo_Gear_Factor_fl": cargo_gear_factor_fl,
        "Weather_Factor_fw": weather_factor_fw,
        "Ice_Class_Factor_fm": ice_class_factor_fm,
        "Main_Engine_CO2_Rate_g_per_h": main_engine_co2_rate_g_per_h,
        "Auxiliary_Engine_CO2_Rate_g_per_h": (
            auxiliary_engine_co2_rate_g_per_h
        ),
        "Total_CO2_Rate_g_per_h": total_co2_rate_g_per_h,
        "Corrected_Transport_Work_t_nm_per_h": (
            corrected_transport_work_t_nm_per_h
        ),
        "Attained_EEDI_gCO2_per_t_nm": float(attained_eedi),
        "Attained_EEDI_Numeric_Complete": True,
        "Auxiliary_Power_Basis": (
            "MEPC.364(79)_2.2.5.6_generic_no_PTI_formula"
        ),
        "EEDI_Compliance_Status": (
            "not_determined_requires_imo_verification"
        ),
        "Source_Citations": (
            "MARPOL Annex VI regulations 22 and 24; "
            "MEPC.364(79) 2.1 and 2.2; MEPC.410(84); MEPC.411(84)"
        ),
    }


def evaluate_preliminary_bulk_carrier_eedi(
    *,
    total_main_engine_mcr_kw: float | None = None,
    reference_speed_knots: float | None = None,
    deadweight_tonnes: float | None = None,
    main_engine_sfc_g_per_kwh: float | None = None,
    auxiliary_engine_sfc_g_per_kwh: float | None = None,
    main_engine_cf_t_co2_per_t_fuel: float | None = None,
    auxiliary_engine_cf_t_co2_per_t_fuel: float | None = None,
    ship_specific_power_factor_fj: float | None = None,
    capacity_factor_fi: float | None = None,
    cubic_capacity_factor_fc: float | None = None,
    cargo_gear_factor_fl: float | None = None,
    weather_factor_fw: float | None = None,
    ice_class_factor_fm: float | None = None,
    eedi_phase: int | None = None,
    reduction_factor_percent: float | None = None,
) -> dict:
    """Report preliminary EEDI completeness without implying compliance.

    The wrapper preserves partial results.  It may calculate the reference
    line from DWT even when the attained-EEDI inputs are incomplete.  A caller
    may provide either a regulation-24 phase or an independently established
    reduction factor.  If both are supplied, the two must agree.
    """

    attained_input_values = {
        "Total_Main_Engine_MCR_kW": total_main_engine_mcr_kw,
        "Reference_Speed_Vref_knots": reference_speed_knots,
        "Deadweight_tonnes": deadweight_tonnes,
        "Main_Engine_SFC_g_per_kWh": main_engine_sfc_g_per_kwh,
        "Auxiliary_Engine_SFC_g_per_kWh": (
            auxiliary_engine_sfc_g_per_kwh
        ),
        "Main_Engine_CF_tCO2_per_tFuel": (
            main_engine_cf_t_co2_per_t_fuel
        ),
        "Auxiliary_Engine_CF_tCO2_per_tFuel": (
            auxiliary_engine_cf_t_co2_per_t_fuel
        ),
        "Ship_Specific_Power_Factor_fj": ship_specific_power_factor_fj,
        "Capacity_Factor_fi": capacity_factor_fi,
        "Cubic_Capacity_Factor_fc": cubic_capacity_factor_fc,
        "Cargo_Gear_Factor_fl": cargo_gear_factor_fl,
        "Weather_Factor_fw": weather_factor_fw,
        "Ice_Class_Factor_fm": ice_class_factor_fm,
    }
    missing_attained_inputs = [
        name
        for name, value in attained_input_values.items()
        if value is None
    ]

    result = {
        "Ship_Type": "bulk_carrier",
        "EEDI_Phase": eedi_phase,
        "Supplied_Reduction_Factor_percent": reduction_factor_percent,
        "Scheduled_Reduction_Factor_percent": None,
        "Applied_Reduction_Factor_percent": None,
        "Bulk_Carrier_Reference_EEDI_gCO2_per_t_nm": None,
        "Required_EEDI_gCO2_per_t_nm": None,
        "Attained_EEDI_gCO2_per_t_nm": None,
        "Required_EEDI_Applicable": None,
        "Attained_EEDI_Inputs_Complete": not missing_attained_inputs,
        "Required_EEDI_Inputs_Complete": False,
        "Preliminary_EEDI_Comparison_Complete": False,
        "Preliminary_Numeric_Requirement_Met": None,
        "Missing_Inputs": list(missing_attained_inputs),
        "Calculation_Status": "incomplete_missing_inputs",
        "EEDI_Compliance_Status": (
            "not_determined_incomplete_or_unverified_inputs"
        ),
        "Source_Citations": (
            "MARPOL Annex VI regulations 22 and 24; "
            "MEPC.364(79); MEPC.410(84); MEPC.411(84)"
        ),
    }

    scheduled_reduction_factor = None
    phase_and_reduction_conflict = False

    if deadweight_tonnes is not None:
        reference_eedi = bulk_carrier_reference_eedi(deadweight_tonnes)
        result["Bulk_Carrier_Reference_EEDI_gCO2_per_t_nm"] = (
            reference_eedi
        )

        # Regulation 24 table 1 sets 10,000 DWT as the lower applicability
        # boundary for bulk carriers.  No phase selection can make a smaller
        # ship subject to this required-EEDI row.
        if float(deadweight_tonnes) < 10_000.0:
            result["Required_EEDI_Applicable"] = False
        else:
            result["Required_EEDI_Applicable"] = True

    if eedi_phase is not None and deadweight_tonnes is not None:
        scheduled_reduction_factor = (
            bulk_carrier_reduction_factor_percent(
                deadweight_tonnes,
                eedi_phase,
            )
        )
        result["Scheduled_Reduction_Factor_percent"] = (
            scheduled_reduction_factor
        )

        # A None schedule means the particular phase/size table cell is N/A.
        if scheduled_reduction_factor is None:
            result["Required_EEDI_Applicable"] = False

    if (
        scheduled_reduction_factor is not None
        and reduction_factor_percent is not None
    ):
        supplied_reduction = float(reduction_factor_percent)

        # This tolerance only absorbs floating-point representation in a
        # duplicated input.  It is not an engineering or regulatory margin.
        phase_and_reduction_conflict = not math.isclose(
            scheduled_reduction_factor,
            supplied_reduction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )

    if phase_and_reduction_conflict:
        result["Calculation_Status"] = (
            "inconsistent_phase_and_reduction_factor"
        )

        return result

    if reduction_factor_percent is not None:
        applied_reduction_factor = _require_nonnegative(
            reduction_factor_percent,
            "EEDI reduction factor",
        )

        if applied_reduction_factor >= 100.0:
            raise ValueError(
                "EEDI reduction factor must be less than 100 percent."
            )
    else:
        applied_reduction_factor = scheduled_reduction_factor

    result["Applied_Reduction_Factor_percent"] = (
        applied_reduction_factor
    )

    if (
        result["Required_EEDI_Applicable"] is True
        and applied_reduction_factor is None
    ):
        result["Missing_Inputs"].append(
            "EEDI_Phase_or_Reduction_Factor_percent"
        )

    elif (
        result["Required_EEDI_Applicable"] is None
        and eedi_phase is None
        and reduction_factor_percent is None
    ):
        # Until DWT is supplied, applicability cannot be resolved.  Retain the
        # phase/reduction selection in the missing-input inventory so an empty
        # readiness call lists every input needed to complete the comparison.
        result["Missing_Inputs"].append(
            "EEDI_Phase_or_Reduction_Factor_percent"
        )

    if (
        result["Required_EEDI_Applicable"] is True
        and applied_reduction_factor is not None
        and deadweight_tonnes is not None
    ):
        required_eedi = required_bulk_carrier_eedi(
            deadweight_tonnes,
            applied_reduction_factor,
        )
        result["Required_EEDI_gCO2_per_t_nm"] = required_eedi
        result["Required_EEDI_Inputs_Complete"] = True

    if not missing_attained_inputs:
        attained_result = calculate_attained_eedi_conventional_diesel(
            total_main_engine_mcr_kw=total_main_engine_mcr_kw,
            reference_speed_knots=reference_speed_knots,
            deadweight_tonnes=deadweight_tonnes,
            main_engine_sfc_g_per_kwh=main_engine_sfc_g_per_kwh,
            auxiliary_engine_sfc_g_per_kwh=(
                auxiliary_engine_sfc_g_per_kwh
            ),
            main_engine_cf_t_co2_per_t_fuel=(
                main_engine_cf_t_co2_per_t_fuel
            ),
            auxiliary_engine_cf_t_co2_per_t_fuel=(
                auxiliary_engine_cf_t_co2_per_t_fuel
            ),
            ship_specific_power_factor_fj=ship_specific_power_factor_fj,
            capacity_factor_fi=capacity_factor_fi,
            cubic_capacity_factor_fc=cubic_capacity_factor_fc,
            cargo_gear_factor_fl=cargo_gear_factor_fl,
            weather_factor_fw=weather_factor_fw,
            ice_class_factor_fm=ice_class_factor_fm,
        )
        result.update(attained_result)

        # Restore the wrapper's compliance status after copying the atomic
        # result.  Neither level is allowed to call this a compliance result.
        result["EEDI_Compliance_Status"] = (
            "not_determined_requires_imo_verification"
        )

    if result["Required_EEDI_Applicable"] is False:
        result["Calculation_Status"] = (
            "required_eedi_not_applicable_for_selected_size_and_phase"
        )
        result["EEDI_Compliance_Status"] = "not_applicable_under_table_1"

    elif (
        result["Attained_EEDI_Inputs_Complete"]
        and result["Required_EEDI_Inputs_Complete"]
    ):
        result["Preliminary_EEDI_Comparison_Complete"] = True

        # Regulation 24 requires attained EEDI to be less than or equal to
        # required EEDI.  No extra policy tolerance is inserted here.
        result["Preliminary_Numeric_Requirement_Met"] = (
            result["Attained_EEDI_gCO2_per_t_nm"]
            <= result["Required_EEDI_gCO2_per_t_nm"]
        )
        result["Calculation_Status"] = (
            "preliminary_numeric_comparison_complete"
        )

    elif result["Attained_EEDI_Inputs_Complete"]:
        result["Calculation_Status"] = (
            "attained_complete_required_eedi_incomplete"
        )

    elif result["Required_EEDI_Inputs_Complete"]:
        result["Calculation_Status"] = (
            "required_complete_attained_eedi_incomplete"
        )

    return result
