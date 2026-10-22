"""Explicit scenario inputs for the current preliminary economic model.

The V1.1 required-freight-rate calculation is a comparative early-design
model, not a contracted ship price or a life-cycle-cost estimate.  Every
coefficient below already appeared as a literal in
``ShipSystemDesign.calc_Ship_System_Performance``.  Centralizing those values
therefore adds scenario provenance without changing the legacy calculation.

The mapping deliberately remains outside the 189-value ship design vector.
Economic assumptions can then be held fixed, swept, or replaced for a dataset
without changing the geometric/design-variable schema learned by GenISys.
"""

from __future__ import annotations


# These are the exact values used by the pre-parameterization V1.1 equations.
# They are retained as a backwards-compatible baseline until the project
# selects and documents a dated economic scenario.
DEFAULT_ECONOMIC_INPUTS = {
    "Acquisition_Cost_Multiplier": 1.3,
    "Steel_Cost_Factor": 2000.0,
    "Steel_Cost_Exponent": 0.85,
    "Outfit_Cost_Factor": 3500.0,
    "Engine_Cost_Factor": 2400.0,
    "Engine_Cost_Exponent": 0.8,
    "Port_Call_Count_Per_Voyage": 2.0,
    "Cargo_Handling_Rate_t_per_day": 8000.0,
    "Fixed_Port_Allowance_days_per_call": 0.5,
    "Annual_Operating_Days": 350.0,
    "Fuel_Price_pounds_per_tonne": 100.0,
    "Fuel_Cost_Factor": 1.05,
    "Port_Cost_Factor": 6.3,
    "Port_Cost_Deadweight_Exponent": 0.8,
    "Capital_Annual_Factor": 0.2,
    "Running_Cost_Factor": 40000.0,
    "Running_Cost_Deadweight_Exponent": 0.3,
}


def resolve_economic_inputs(overrides=None) -> dict:
    """Return a complete economic scenario and its field-level provenance.

    ``overrides`` must use the public names in ``DEFAULT_ECONOMIC_INPUTS``.
    Unknown fields are rejected because a misspelled price or utilization
    input must not silently fall back to the legacy value.  Equation-specific
    domain checks remain beside the performance equation that consumes the
    scenario.
    """

    supplied = {} if overrides is None else dict(overrides)
    unknown = sorted(set(supplied) - set(DEFAULT_ECONOMIC_INPUTS))

    if unknown:
        raise ValueError(
            "Unknown preliminary economic input(s): " + ", ".join(unknown)
        )

    values = DEFAULT_ECONOMIC_INPUTS.copy()
    values.update(supplied)

    if not supplied:
        basis = "legacy_global_assumptions"
    elif len(supplied) == len(DEFAULT_ECONOMIC_INPUTS):
        basis = "explicit_scenario_overrides"
    else:
        basis = "mixed_legacy_assumptions_and_explicit_scenario"

    values["Economic_Input_Basis"] = basis
    values["Explicit_Economic_Input_Fields"] = ";".join(sorted(supplied))

    return values
