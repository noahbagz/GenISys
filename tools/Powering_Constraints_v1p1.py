"""Powering-method domain checks and assumption metrics for GenISys V1.1.

The Boolean rows are early-design method-validity checks, not statutory or
class rules.  Blank-bit metrics also preserve the preliminary MARPOL Annex VI
EEDI reference/required arithmetic that can be calculated without certified
inputs.  They prevent a numerically completed resistance or EEDI proxy from
being silently treated as regulatory compliance and make the remaining fixed
propulsion assumptions explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.EEDI import (
    bulk_carrier_reduction_factor_percent,
    bulk_carrier_reference_eedi,
    required_bulk_carrier_eedi,
)


CONSTRAINT_COLUMNS = [
    "Class",
    "Name",
    "Requirement",
    "Threshold Value",
    "Constraint Value",
    "Units",
    "Constraint Eval",
    "Constraint Bit",
]


# HDF-ASM-036: the fractional-power form-factor term contains
# ``(LWL/Lr)**0.121563``.  A merely positive run length is mathematically
# defined but becomes numerically ill-conditioned as Lr approaches zero.
# The approved early-design method-domain clearance therefore requires the
# modeled run length to be at least one percent of the waterline length.
HOLTROP_MINIMUM_RUN_LENGTH_RATIO = 0.01


def holtrop_run_length_ratio(prismatic_coefficient, lcb_percent_lwl):
    """Return the normalized Holtrop run length ``Lr/LWL``.

    ``lcb_percent_lwl`` follows Holtrop's signed convention: percent of LWL
    forward of the waterline midpoint.  Keeping this algebra in one helper
    prevents the resistance calculation and its exposed method-domain row
    from silently using different definitions.
    """

    prismatic_coefficient = float(prismatic_coefficient)
    lcb_percent_lwl = float(lcb_percent_lwl)
    denominator = 4.0*prismatic_coefficient - 1.0

    if abs(denominator) <= np.finfo(float).eps:
        return np.nan

    return (
        1.0
        - prismatic_coefficient
        + (
            0.06
            * prismatic_coefficient
            * lcb_percent_lwl
            / denominator
        )
    )


def evaluate_holtrop_run_length_constraint(run_length_ratio):
    """Return the approved explicit ``Lr/LWL >= 0.01`` domain row."""

    return pd.DataFrame(
        [
            _minimum(
                "Holtrop run-length to waterline-length equation domain",
                (
                    "Holtrop and Mennen (1982), implemented form-factor "
                    "equation; HDF-ASM-036 numerical clearance"
                ),
                HOLTROP_MINIMUM_RUN_LENGTH_RATIO,
                run_length_ratio,
                "dimensionless",
            )
        ],
        columns=CONSTRAINT_COLUMNS,
    )


def _maximum(name, requirement, maximum, value, units, strict=False):
    value = float(value)
    maximum = float(maximum)

    return {
        "Class": "Powering Method Domain",
        "Name": name,
        "Requirement": requirement,
        "Threshold Value": maximum,
        "Constraint Value": value,
        "Units": units,
        "Constraint Eval": value - maximum,
        "Constraint Bit": bool(value < maximum if strict else value <= maximum),
    }


def _minimum(name, requirement, minimum, value, units, strict=False):
    value = float(value)
    minimum = float(minimum)
    passes = value > minimum if strict else value >= minimum

    return {
        "Class": "Powering Method Domain",
        "Name": name,
        "Requirement": requirement,
        "Threshold Value": minimum,
        "Constraint Value": value,
        "Units": units,
        "Constraint Eval": minimum - value,
        "Constraint Bit": bool(passes),
    }


def _metric(
    name,
    requirement,
    value,
    units,
    result_class="Powering Assumption Metric",
):
    return {
        "Class": result_class,
        "Name": name,
        "Requirement": requirement,
        "Threshold Value": np.nan,
        "Constraint Value": float(value),
        "Units": units,
        "Constraint Eval": np.nan,
        "Constraint Bit": np.nan,
    }


def evaluate_powering_constraints_v1p1(hull_dict, deadweight_kg=None):
    """Return method-domain checks and transparent fixed-input metrics.

    Holtrop's 1984 re-analysis states that predictions were often incorrect for
    high-speed craft above Froude number 0.5.  The implemented low/intermediate
    speed GenISys dataset is therefore screened at that published boundary.
    The other four checks are exact mathematical domains of the equations
    coded in ``HullDesign.calc_Hull_Drag_Holtrop``.  When DWT is supplied, the
    function also returns explicitly preliminary regulation-24 metrics with
    blank pass/fail bits.
    """

    waterline_beam_ratio = (
        float(hull_dict["Waterline_Length"])/float(hull_dict["B"])
    )
    prismatic_coefficient = float(hull_dict["Cp"])

    run_length_ratio = float(hull_dict["Holtrop_Run_Length_Ratio"])

    rows = [
        _maximum(
            "Holtrop low/intermediate-speed Froude boundary",
            "Holtrop (1984), high-speed caveat above Fn 0.5",
            0.5,
            hull_dict["Fn"],
            "dimensionless",
        ),
        _maximum(
            "Holtrop prismatic-coefficient equation domain",
            "Holtrop and Mennen (1982), implemented form-factor equation",
            1.0,
            prismatic_coefficient,
            "dimensionless",
            strict=True,
        ),
        _minimum(
            "Holtrop waterline-length to breadth equation domain",
            "Holtrop (1984), implemented c17 equation",
            2.0,
            waterline_beam_ratio,
            "dimensionless",
            strict=True,
        ),
        _minimum(
            "Holtrop run-length denominator magnitude",
            "Holtrop and Mennen (1982), |4 Cp - 1| must be nonzero",
            0.0,
            abs(4.0*prismatic_coefficient - 1.0),
            "dimensionless",
            strict=True,
        ),
        _minimum(
            "Holtrop run-length to waterline-length equation domain",
            (
                "Holtrop and Mennen (1982), implemented form-factor "
                "equation; HDF-ASM-036 numerical clearance"
            ),
            HOLTROP_MINIMUM_RUN_LENGTH_RATIO,
            run_length_ratio,
            "dimensionless",
        ),
        _metric(
            "Assumed appendage wetted area",
            "Current GenISys powering assumption pending parameterization",
            hull_dict["Appendage_Area_m2"],
            "m^2",
        ),
        _metric(
            "Assumed appendage resistance factor",
            "Current GenISys powering assumption pending parameterization",
            hull_dict["Appendage_Resistance_Factor"],
            "dimensionless",
        ),
        _metric(
            "Maximum continuous rating fraction",
            "Current GenISys powering assumption pending engine definition",
            hull_dict["Maximum_Continuous_Rating_Fraction"],
            "dimensionless",
        ),
        _metric(
            "Aggregate propulsive efficiency",
            "Current GenISys powering assumption pending propulsor definition",
            hull_dict["Aggregate_Propulsive_Efficiency"],
            "dimensionless",
        ),
        _metric(
            "Main-engine specific fuel consumption",
            "Current GenISys fuel assumption pending engine definition",
            hull_dict["Main_Engine_SFOC_kg_per_kWh"],
            "kg/kWh",
        ),
        _metric(
            "Auxiliary fuel consumption",
            "Current GenISys fuel assumption pending auxiliary-load model",
            hull_dict["Auxiliary_Fuel_t_per_day"],
            "t/day",
        ),
        _metric(
            "Fuel reserve duration",
            "Current GenISys voyage-fuel assumption",
            hull_dict["Fuel_Reserve_Days"],
            "days",
        ),
    ]

    if deadweight_kg is not None:
        deadweight_tonnes = float(deadweight_kg)/1000.0

        # MARPOL Annex VI regulation 24, table 2.  The arithmetic reference
        # line depends only on ship type and DWT, so GenISys can calculate an
        # early-design value from the displacement-derived deadweight already
        # used by the systems model.  Keep both rows as metrics: the current
        # deadweight is evaluated at the sampled design draft and is not a
        # certified summer-load-line DWT, so this is not an attained-EEDI or
        # required-EEDI compliance result.
        rows.extend(
            [
                _metric(
                    "Preliminary EEDI deadweight proxy at evaluated draft",
                    (
                        "MARPOL Annex VI regulation 24; early-design input "
                        "lineage only"
                    ),
                    deadweight_tonnes,
                    "tonnes",
                    result_class="EEDI Preliminary Metric",
                ),
                _metric(
                    "Preliminary bulk-carrier EEDI reference line",
                    (
                        "MARPOL Annex VI regulation 24 table 2; reference "
                        "line only, not attained or required EEDI"
                    ),
                    bulk_carrier_reference_eedi(deadweight_tonnes),
                    "g CO2/(t nm)",
                    result_class="EEDI Preliminary Metric",
                ),
            ]
        )

        # HDF-ASM-004 defines GenISys as a current, not-yet-planned new
        # cargo-vessel design.  Under the current MARPOL regulation-24
        # schedule this selects the phase-3 bulk-carrier row without inventing
        # historical contract, keel, or delivery dates.  Because the active
        # DWT remains an evaluated-draft proxy rather than certified summer-
        # load DWT, the phase result and required value remain metrics only.
        eedi_phase = 3
        reduction_factor_percent = (
            bulk_carrier_reduction_factor_percent(
                deadweight_tonnes,
                eedi_phase,
            )
        )

        # ``None`` is the source table's below-10,000-DWT N/A cell.  Retain
        # that state as NaN instead of manufacturing a zero-percent factor.
        if reduction_factor_percent is None:
            reduction_factor_metric = np.nan
            required_eedi_proxy = np.nan
        else:
            reduction_factor_metric = reduction_factor_percent
            required_eedi_proxy = required_bulk_carrier_eedi(
                deadweight_tonnes,
                reduction_factor_percent,
            )

        rows.extend(
            [
                _metric(
                    "Preliminary bulk-carrier EEDI phase",
                    (
                        "MARPOL Annex VI regulation 24 table 1; current-new-"
                        "build basis HDF-ASM-004"
                    ),
                    eedi_phase,
                    "phase index",
                    result_class="EEDI Preliminary Metric",
                ),
                _metric(
                    "Preliminary bulk-carrier EEDI reduction factor",
                    (
                        "MARPOL Annex VI regulation 24 table 1; phase-3 and "
                        "evaluated-draft DWT proxy"
                    ),
                    reduction_factor_metric,
                    "%",
                    result_class="EEDI Preliminary Metric",
                ),
                _metric(
                    "Preliminary required bulk-carrier EEDI proxy",
                    (
                        "MARPOL Annex VI regulation 24; phase-3 reduction "
                        "applied to evaluated-draft DWT proxy"
                    ),
                    required_eedi_proxy,
                    "g CO2/(t nm)",
                    result_class="EEDI Preliminary Metric",
                ),
            ]
        )

    return pd.DataFrame(rows, columns=CONSTRAINT_COLUMNS)
