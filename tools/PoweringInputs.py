"""Explicit scenario inputs for the current preliminary powering equations.

GenISys V1.1 still uses a resistance-to-installed-power correlation rather
than a selected propeller and engine.  The seven values below already existed
as hard-coded assumptions in :mod:`tools.HullDesign`.  This module does not
add new naval-architecture assumptions; it gives callers a reproducible way
to override the existing values for a named dataset scenario.

Propeller diameter, RPM, blade count/series, cavitation limits, gear details,
certified MCR/SFC, and EEDI verification inputs are deliberately absent.  They
require a propulsion design decision and must not be inferred here.
"""

from __future__ import annotations


# Keep the existing V1.1 values as the backwards-compatible base scenario.
# Their source remains ``legacy_global_assumptions`` until the user supplies a
# project scenario file with explicit replacements.
DEFAULT_POWERING_INPUTS = {
    "Appendage_Area_m2": 1000.0,
    "Appendage_Resistance_Factor": 2.0,
    "Maximum_Continuous_Rating_Fraction": 0.85,
    "Aggregate_Propulsive_Efficiency": 0.637,
    "Main_Engine_SFOC_kg_per_kWh": 0.19,
    "Auxiliary_Fuel_t_per_day": 0.2,
    "Fuel_Reserve_Days": 5.0,
}


def resolve_powering_inputs(overrides=None) -> dict:
    """Return the complete current powering-input set and its provenance.

    ``overrides`` uses the exact public field names in
    ``DEFAULT_POWERING_INPUTS``.  Rejecting unknown names is important here:
    a misspelled scenario key must not silently fall back to a global value.
    Numeric/domain checks remain beside the equation that consumes each value
    in ``HullDesign``.
    """

    supplied = {} if overrides is None else dict(overrides)
    unknown = sorted(set(supplied) - set(DEFAULT_POWERING_INPUTS))

    if unknown:
        raise ValueError(
            "Unknown preliminary powering input(s): " + ", ".join(unknown)
        )

    values = DEFAULT_POWERING_INPUTS.copy()
    values.update(supplied)

    if not supplied:
        basis = "legacy_global_assumptions"
    elif len(supplied) == len(DEFAULT_POWERING_INPUTS):
        basis = "explicit_scenario_overrides"
    else:
        basis = "mixed_legacy_assumptions_and_explicit_scenario"

    values["Powering_Input_Basis"] = basis
    values["Explicit_Powering_Input_Fields"] = ";".join(sorted(supplied))

    return values
