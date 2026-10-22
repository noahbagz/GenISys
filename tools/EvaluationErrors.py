"""Structured exceptions for incomplete GenISys system evaluations.

The batch runner still records the full traceback for debugging, but known
physical infeasibilities carry enough rule metadata to produce an ordinary
legacy-format failed constraint row.  Unknown programming or numerical errors
remain clearly separate from a design that simply cannot fit the requested
load.
"""

from __future__ import annotations

import pandas as pd


class PhysicalConstraintFailure(RuntimeError):
    """Stop a dependent calculation after one physical predicate fails."""

    failure_category = "physical_infeasibility"

    def __init__(
        self,
        message: str,
        *,
        name: str,
        requirement: str,
        threshold_value: float,
        constraint_value: float,
        units: str,
        comparison: str = "minimum",
        result_class: str = "System Capacity",
        constraint_rows: pd.DataFrame | None = None,
        diagnostics: dict | None = None,
    ):
        super().__init__(message)
        self.name = name
        self.requirement = requirement
        self.threshold_value = float(threshold_value)
        self.constraint_value = float(constraint_value)
        self.units = units
        self.comparison = comparison
        self.result_class = result_class
        # A stopped integrated calculation may have evaluated several valid
        # predicates before reaching the failing prerequisite.  Carry those
        # rows with the exception so the batch writer does not discard them.
        self.constraint_rows = constraint_rows
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)


class MethodDomainConstraintFailure(PhysicalConstraintFailure):
    """Stop a calculation at an exposed empirical-method domain boundary.

    This deliberately retains the structured constraint-row contract of
    :class:`PhysicalConstraintFailure`, allowing the batch and repair paths to
    preserve the failed method-domain predicate.  Its distinct category keeps
    a correlation-domain rejection separate from physical ship infeasibility.
    """

    failure_category = "method_domain_rejection"


def classify_evaluation_error(error: Exception, stage: str) -> str:
    """Classify one stopped evaluation for dataset filtering and restart."""

    explicit_category = getattr(error, "failure_category", None)

    if explicit_category is not None:
        return str(explicit_category)

    message = str(error).casefold()

    if "constraint" in message or "capacity" in message:
        return "physical_infeasibility"

    if any(
        marker in message
        for marker in (
            "applicability",
            "outside the",
            "do not extrapolate",
            "method domain",
        )
    ):
        return "method_domain_rejection"

    if stage in {
        "load_structural_elements",
        "prepare_system_parameters",
        "write_individual_constraint_file",
    }:
        return "data_or_io_failure"

    return "numerical_or_implementation_failure"
