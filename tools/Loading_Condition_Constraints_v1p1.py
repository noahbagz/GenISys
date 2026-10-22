"""Legacy-format predicates for named V1.1 loading-condition results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.LoadingConditionLoads import (
    LOAD_CLOSURE_RELATIVE_TOLERANCE,
    MASS_CLOSURE_RELATIVE_TOLERANCE,
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


def evaluate_loading_condition_constraints_v1p1(
    upright_evaluations: pd.DataFrame,
    longitudinal_load_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate equilibrium, closure, departure trim, and corrected GM.

    HDF-ASM-033 applies the one-percent-draft trim intent only to full-cargo
    departure, because that is the condition for which the immutable cargo
    allocation is optimized.  Arrival remains a named hydrostatic/load state,
    but HDF-ASM-038 specifies compensating ballast rather than a second trim
    constraint.  When ``longitudinal_load_summary`` is supplied, the departure
    trim row uses endpoint drafts from the simultaneous displacement-and-
    first-moment load solution.  The upright MCT estimate remains an explicit
    fallback for isolated departure callers without that load solution.

    HDF-ASM-035 treats mass, force, first-moment, endpoint-shear, and
    endpoint-bending closure as engineering numerical-quality gates.  A failed
    row means that the calculation is invalid; it is not a regulatory finding
    that the ship itself is infeasible.

    HDF-007 corrected GM is emitted only when a real condition-specific
    free-surface correction was supplied.  Missing free-surface geometry
    therefore remains missing data, rather than becoming a zero correction or
    an artificial pass.
    """

    rows = []
    longitudinal_by_condition = {}

    if longitudinal_load_summary is not None:
        longitudinal_by_condition = {
            str(row["Loading_Condition_ID"]): row
            for _, row in longitudinal_load_summary.iterrows()
        }

    for _, condition in upright_evaluations.iterrows():
        condition_id = str(condition["Loading_Condition_ID"])
        condition_name = str(condition["Loading_Condition_Name"])
        equilibrium_solved = bool(condition["Equilibrium_Solved"])

        rows.append(
            {
                "Class": "Loading Condition Numerical Quality",
                "Name": f"{condition_name} hydrostatic equilibrium solved",
                "Requirement": (
                    "GenISys loading-condition foundation; "
                    f"condition={condition_id}"
                ),
                "Threshold Value": True,
                "Constraint Value": equilibrium_solved,
                "Units": "Boolean",
                "Constraint Eval": int(not equilibrium_solved),
                "Constraint Bit": equilibrium_solved,
            }
        )

        # The condition-specific still-water model is a separate numerical
        # solve from the upright hydrostatic interpolation above.  Preserve a
        # separate Boolean row so a missing load equilibrium cannot be hidden
        # behind a successful upright displacement solution.
        longitudinal_state = longitudinal_by_condition.get(condition_id)

        if longitudinal_state is not None:
            longitudinal_equilibrium_solved = bool(
                longitudinal_state["Equilibrium_Solved"]
            )
            rows.append(
                {
                    "Class": "Loading Condition Numerical Quality",
                    "Name": (
                        f"{condition_name} longitudinal load equilibrium "
                        "solved"
                    ),
                    "Requirement": (
                        "GenISys simultaneous displacement and longitudinal-"
                        f"moment equilibrium; condition={condition_id}"
                    ),
                    "Threshold Value": True,
                    "Constraint Value": longitudinal_equilibrium_solved,
                    "Units": "Boolean",
                    "Constraint Eval": int(
                        not longitudinal_equilibrium_solved
                    ),
                    "Constraint Bit": longitudinal_equilibrium_solved,
                }
            )

            # Evaluate every approved signed closure residual by magnitude.
            # The source table retains signs for diagnosis, while an absolute
            # value makes the legacy maximum-allowed predicate unambiguous.
            if longitudinal_equilibrium_solved:
                closure_definitions = (
                    (
                        "distributed-mass closure",
                        "Mass_Residual_Fraction",
                        MASS_CLOSURE_RELATIVE_TOLERANCE,
                    ),
                    (
                        "net vertical-force closure",
                        "Net_Vertical_Force_Residual_Fraction",
                        LOAD_CLOSURE_RELATIVE_TOLERANCE,
                    ),
                    (
                        "longitudinal first-moment closure",
                        "Net_Vertical_First_Moment_Residual_Fraction",
                        LOAD_CLOSURE_RELATIVE_TOLERANCE,
                    ),
                    (
                        "aft endpoint shear closure",
                        "Aft_Shear_Force_Residual_Fraction",
                        LOAD_CLOSURE_RELATIVE_TOLERANCE,
                    ),
                    (
                        "aft endpoint bending closure",
                        "Aft_Bending_Moment_Residual_Fraction",
                        LOAD_CLOSURE_RELATIVE_TOLERANCE,
                    ),
                )

                for closure_name, field, tolerance in closure_definitions:
                    absolute_residual = abs(float(longitudinal_state[field]))
                    rows.append(
                        {
                            "Class": "Loading Condition Numerical Quality",
                            "Name": f"{condition_name} {closure_name}",
                            "Requirement": (
                                "HDF-ASM-035 engineering numerical-quality "
                                f"gate; condition={condition_id}"
                            ),
                            "Threshold Value": tolerance,
                            "Constraint Value": absolute_residual,
                            "Units": "dimensionless",
                            "Constraint Eval": (
                                absolute_residual - tolerance
                            ),
                            "Constraint Bit": bool(
                                absolute_residual <= tolerance
                            ),
                        }
                    )
        else:
            longitudinal_equilibrium_solved = False

        if longitudinal_state is not None and longitudinal_equilibrium_solved:
            equilibrium_draft_m = float(
                longitudinal_state["Equilibrium_Draft_m"]
            )
            signed_trim_m = float(
                longitudinal_state[
                    "Signed_Trim_Aft_Minus_Forward_m"
                ]
            )
            trim_requirement = (
                "HDF-ASM-033: |T_aft - T_forward| <= "
                "0.01*T_equilibrium from simultaneous load equilibrium; "
                f"condition={condition_id}"
            )
        elif longitudinal_state is None and equilibrium_solved:
            equilibrium_draft_m = float(condition["Equilibrium_Draft_m"])
            signed_trim_m = (
                float(condition["Draft_Aft_m"])
                - float(condition["Draft_Forward_m"])
            )
            trim_requirement = (
                "HDF-ASM-033: |T_aft - T_forward| <= "
                "0.01*T_equilibrium from upright MCT fallback; "
                f"condition={condition_id}"
            )
        else:
            equilibrium_draft_m = None
            signed_trim_m = None

        # HDF-ASM-038: arrival does not carry an independent trim constraint.
        # Its ballast policy is intended to reproduce the departure
        # displacement and trim, so counting another tolerance row would be a
        # duplicate operational requirement.  Continue to emit arrival
        # equilibrium and numerical-closure evidence above.
        trim_constraint_applicable = (
            condition_id == "full_cargo_departure"
        )

        if equilibrium_draft_m is not None and trim_constraint_applicable:
            absolute_trim_m = abs(signed_trim_m)
            trim_limit_m = 0.01*equilibrium_draft_m
            numerical_equality_allowance_m = (
                64.0
                * np.finfo(float).eps
                * max(1.0, absolute_trim_m, abs(trim_limit_m))
            )
            rows.append(
                {
                    "Class": "Loading Condition Trim",
                    "Name": f"{condition_name} one-percent-draft trim",
                    "Requirement": trim_requirement,
                    "Threshold Value": trim_limit_m,
                    "Constraint Value": absolute_trim_m,
                    "Units": "m",
                    "Constraint Eval": absolute_trim_m - trim_limit_m,
                    "Constraint Bit": bool(
                        absolute_trim_m
                        <= trim_limit_m + numerical_equality_allowance_m
                    ),
                }
            )

        corrected_gm_m = condition.get("GM0_Corrected_m", np.nan)

        if pd.notna(corrected_gm_m):
            corrected_gm_m = float(corrected_gm_m)
            rows.append(
                {
                    "Class": "Intact Stability",
                    "Name": f"{condition_name} minimum corrected initial GM",
                    "Requirement": (
                        "46 CFR 170.165; 2008 IS Code Part A 2.2.1.5; "
                        f"condition={condition_id}"
                    ),
                    "Threshold Value": 0.15,
                    "Constraint Value": corrected_gm_m,
                    "Units": "m",
                    "Constraint Eval": 0.15 - corrected_gm_m,
                    "Constraint Bit": bool(corrected_gm_m >= 0.15),
                }
            )

    return pd.DataFrame(rows, columns=CONSTRAINT_COLUMNS)
