"""Project-level physical feasibility constraints for GenISys V1.1.

These predicates are not classification-society or statutory requirements.
They state the minimum physical conditions required for the integrated ship
calculation to continue.  Keeping them in the same eight-column format as the
rule constraints makes both feasible and rejected samples useful to later
design-space learning.
"""

from __future__ import annotations

import pandas as pd

from tools.Structure_Constraints_v1p1 import CONSTRAINT_COLUMNS


def _constraint_table(row: dict) -> pd.DataFrame:
    """Return one constraint row with the shared legacy column ordering."""

    return pd.DataFrame([row], columns=CONSTRAINT_COLUMNS)


def evaluate_lightship_displacement_constraint(
    displacement_mass_kg: float,
    modeled_lightship_mass_kg: float,
) -> pd.DataFrame:
    """Require lightship to remain strictly below sampled displacement."""

    displacement_mass_kg = float(displacement_mass_kg)
    modeled_lightship_mass_kg = float(modeled_lightship_mass_kg)

    return _constraint_table(
        {
            "Class": "System Mass Feasibility",
            "Name": "Modeled lightship fits selected displacement",
            "Requirement": (
                "GenISys physical feasibility: modeled lightship mass shall "
                "be less than displacement mass at the selected operating "
                "draft"
            ),
            "Threshold Value": displacement_mass_kg,
            "Constraint Value": modeled_lightship_mass_kg,
            "Units": "kg",
            # Positive residuals consistently denote violations throughout
            # the V1.1 constraint tables.
            "Constraint Eval": (
                modeled_lightship_mass_kg - displacement_mass_kg
            ),
            "Constraint Bit": bool(
                modeled_lightship_mass_kg < displacement_mass_kg
            ),
        }
    )


def evaluate_positive_cargo_payload_constraint(
    available_cargo_mass_kg: float,
) -> pd.DataFrame:
    """Require positive payload after lightship, fuel, and miscellaneous DWT."""

    available_cargo_mass_kg = float(available_cargo_mass_kg)

    return _constraint_table(
        {
            "Class": "System Mass Feasibility",
            "Name": "Positive available cargo payload",
            "Requirement": (
                "GenISys physical feasibility: available cargo mass after "
                "lightship, voyage fuel, and miscellaneous deadweight shall "
                "be positive"
            ),
            "Threshold Value": 0.0,
            "Constraint Value": available_cargo_mass_kg,
            "Units": "kg",
            "Constraint Eval": -available_cargo_mass_kg,
            "Constraint Bit": bool(available_cargo_mass_kg > 0.0),
        }
    )


def evaluate_total_cargo_capacity_constraint(
    required_cargo_mass_kg: float,
    available_cargo_mass_kg: float,
) -> pd.DataFrame:
    """Require the generated holds to contain the selected full-load cargo."""

    required_cargo_mass_kg = float(required_cargo_mass_kg)
    available_cargo_mass_kg = float(available_cargo_mass_kg)

    return _constraint_table(
        {
            "Class": "System Capacity",
            "Name": "Total cargo-hold mass capacity",
            "Requirement": (
                "GenISys physical feasibility: generated cargo-hold capacity "
                "shall be at least the selected full-load cargo mass"
            ),
            "Threshold Value": required_cargo_mass_kg,
            "Constraint Value": available_cargo_mass_kg,
            "Units": "kg",
            "Constraint Eval": (
                required_cargo_mass_kg - available_cargo_mass_kg
            ),
            "Constraint Bit": bool(
                available_cargo_mass_kg >= required_cargo_mass_kg
            ),
        }
    )


def evaluate_cargo_equilibrium_constraint(
    valid_equilibrium_candidate_count: int,
) -> pd.DataFrame:
    """Require at least one cargo allocation with a solved equilibrium.

    The value is a candidate count instead of a Boolean so failed and passing
    samples retain more information while the pass criterion remains exactly
    the approved "at least one" condition.
    """

    valid_equilibrium_candidate_count = int(valid_equilibrium_candidate_count)

    return _constraint_table(
        {
            "Class": "Loading Condition Feasibility",
            "Name": (
                "Capacity-feasible cargo allocation reaches longitudinal "
                "equilibrium"
            ),
            "Requirement": (
                "GenISys physical feasibility: at least one capacity-feasible "
                "full-load cargo allocation shall reach displacement and "
                "longitudinal-moment equilibrium within the generated hull "
                "draft grid"
            ),
            "Threshold Value": 1.0,
            "Constraint Value": float(valid_equilibrium_candidate_count),
            "Units": "valid candidates",
            "Constraint Eval": float(
                1 - valid_equilibrium_candidate_count
            ),
            "Constraint Bit": bool(valid_equilibrium_candidate_count >= 1),
        }
    )
