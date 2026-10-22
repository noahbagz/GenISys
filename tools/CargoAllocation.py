"""Deterministic, capacity-aware cargo allocation helpers.

The allocation problem has a particularly useful structure: total cargo mass
is fixed, each hold has a simple upper bound, and every convex combination of
two feasible allocations remains feasible.  GenISys therefore does not need a
general nonlinear optimizer merely to find a longitudinal cargo moment.  The
helpers below construct the two exact moment extremes.  The active V1.1 path
then calculates the cargo LCG required for even keel and projects that target
directly onto the feasible moment interval.
"""

from __future__ import annotations

import numpy as np


class CargoAllocationSearchError(RuntimeError):
    """Report that no candidate on the feasible segment was evaluable."""

    def __init__(
        self,
        message,
        *,
        evaluation_count,
        invalid_evaluation_count,
        closest_diagnostics=None,
    ):
        super().__init__(message)
        self.evaluation_count = int(evaluation_count)
        self.invalid_evaluation_count = int(invalid_evaluation_count)
        self.valid_evaluation_count = (
            self.evaluation_count - self.invalid_evaluation_count
        )
        self.closest_diagnostics = (
            {} if closest_diagnostics is None
            else dict(closest_diagnostics)
        )


def capacity_extreme_allocation(
    capacity_fractions,
    longitudinal_coordinates_m,
    maximize_longitudinal_moment=False,
):
    """Return the feasible bounded-simplex allocation at one moment extreme.

    Cargo is greedily placed into holds ordered by longitudinal coordinate.
    This is the exact solution of a linear moment objective subject to
    ``0 <= fraction[i] <= capacity[i]`` and ``sum(fraction) = 1``.
    """

    capacities = np.asarray(capacity_fractions, dtype=float)
    coordinates = np.asarray(longitudinal_coordinates_m, dtype=float)

    if capacities.shape != coordinates.shape:
        raise ValueError(
            "Cargo-hold capacities and longitudinal coordinates must have "
            "the same shape."
        )

    if capacities.ndim != 1 or len(capacities) == 0:
        raise ValueError("At least one cargo hold is required for allocation.")

    if not np.all(np.isfinite(capacities)) or np.any(capacities < 0.0):
        raise ValueError("Cargo-hold capacity fractions must be finite and nonnegative.")

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Cargo-hold longitudinal coordinates must be finite.")

    # Use a scale-aware floating-point allowance solely for arithmetic at the
    # simplex boundary.  It is not an engineering capacity margin.
    arithmetic_tolerance = (
        64.0*np.finfo(float).eps*max(1.0, float(np.sum(capacities)))
    )

    if float(np.sum(capacities)) < 1.0 - arithmetic_tolerance:
        raise ValueError(
            "Combined cargo-hold capacity is smaller than the required "
            "full-load cargo mass."
        )

    order = np.argsort(coordinates, kind="stable")

    if maximize_longitudinal_moment:
        order = order[::-1]

    allocation = np.zeros_like(capacities)
    remaining = 1.0

    for index in order:
        assigned = min(float(capacities[index]), remaining)
        allocation[index] = assigned
        remaining -= assigned

        if remaining <= arithmetic_tolerance:
            break

    # Remove only roundoff from the final active hold.  Because this correction
    # is no larger than the arithmetic tolerance established above, it cannot
    # conceal a physical capacity deficit.
    if abs(remaining) <= arithmetic_tolerance:
        active = np.flatnonzero(allocation > 0.0)

        if len(active):
            allocation[active[-1]] += remaining
            remaining = 0.0

    if remaining > 0.0:
        raise ValueError(
            "Combined cargo-hold capacity is smaller than the required "
            "full-load cargo mass."
        )

    return allocation


def direct_required_lcg_allocation(
    forwardmost_allocation,
    aftmost_allocation,
    longitudinal_coordinates_m,
    required_cargo_lcg_m,
):
    """Allocate cargo directly to the required longitudinal cargo center.

    The forwardmost and aftmost allocations are exact extrema of the bounded
    cargo-allocation simplex.  Their convex segment therefore spans every
    attainable cargo moment between those limits.  Interpolating on that
    segment satisfies total cargo mass, every hold-capacity bound, and the
    requested geometric cargo LCG without a sampled optimization loop.

    If the requested LCG lies outside the attainable interval, return the
    nearest moment extreme and retain the signed deficit.  The complete ship
    evaluator will then report the resulting physical trim failure, while the
    deficit identifies which direction the ship arrangement must move.
    """

    forward = np.asarray(forwardmost_allocation, dtype=float)
    aft = np.asarray(aftmost_allocation, dtype=float)
    coordinates = np.asarray(longitudinal_coordinates_m, dtype=float)
    required = float(required_cargo_lcg_m)

    if forward.shape != aft.shape or forward.shape != coordinates.shape:
        raise ValueError(
            "Cargo-allocation endpoints and coordinates must have the same "
            "shape."
        )

    if not np.isfinite(required):
        raise ValueError("Required cargo LCG must be finite.")

    forward_lcg = float(np.dot(forward, coordinates))
    aft_lcg = float(np.dot(aft, coordinates))
    interval = aft_lcg-forward_lcg
    arithmetic_tolerance = 64.0*np.finfo(float).eps*max(
        1.0,
        abs(required),
        abs(forward_lcg),
        abs(aft_lcg),
    )

    if abs(interval) <= arithmetic_tolerance:
        alpha_unbounded = 0.0
    else:
        alpha_unbounded = (required-forward_lcg)/interval

    alpha = float(np.clip(alpha_unbounded, 0.0, 1.0))
    allocation = (1.0-alpha)*forward + alpha*aft
    achieved_lcg = float(np.dot(allocation, coordinates))
    deficit = required-achieved_lcg
    target_reachable = bool(abs(deficit) <= arithmetic_tolerance)

    return {
        "allocation": allocation,
        "alpha": alpha,
        "alpha_unbounded": float(alpha_unbounded),
        "required_cargo_lcg_m": required,
        "forwardmost_cargo_lcg_m": forward_lcg,
        "aftmost_cargo_lcg_m": aft_lcg,
        "achieved_geometric_cargo_lcg_m": achieved_lcg,
        "cargo_lcg_deficit_m": float(deficit),
        "target_reachable": target_reachable,
    }


def deterministic_trim_search(
    low_moment_allocation,
    high_moment_allocation,
    evaluate_allocation,
    sample_count=33,
    bisection_iterations=48,
    recoverable_exceptions=(),
):
    """Find the least-absolute-trim point on a feasible allocation segment.

    A coarse deterministic scan first captures nonlinearity caused by changing
    cargo VCG and hydrostatic properties.  Every adjacent sign change is then
    refined by bisection.  The returned allocation is always one that was
    explicitly evaluated; no optimizer success flag is trusted as a proxy for
    physical feasibility.
    """

    low = np.asarray(low_moment_allocation, dtype=float)
    high = np.asarray(high_moment_allocation, dtype=float)

    if low.shape != high.shape:
        raise ValueError("Cargo-allocation endpoints must have the same shape.")

    if sample_count < 2:
        raise ValueError("Deterministic trim search requires at least two samples.")

    evaluation_count = 0
    invalid_evaluation_count = 0
    invalid_diagnostics = []
    sampled = []

    def evaluate_alpha(alpha):
        nonlocal evaluation_count, invalid_evaluation_count

        allocation = (1.0 - alpha)*low + alpha*high
        evaluation_count += 1

        try:
            state = dict(evaluate_allocation(allocation))
        except recoverable_exceptions as error:
            # A longitudinal-moment extreme may lie outside the hull's
            # attainable buoyancy-centroid range.  This invalidates that
            # allocation, not every interior allocation on the segment.
            invalid_evaluation_count += 1
            diagnostics = getattr(error, "diagnostics", None)

            if diagnostics:
                diagnostics = dict(diagnostics)

                # Preserve which point on the capacity-feasible allocation
                # segment produced each failed equilibrium solve.  The repair
                # pipeline may compare successive closest states, and this
                # coordinate prevents those residuals from being mistaken for
                # fixed-allocation finite differences.
                diagnostics["Cargo_Allocation_Alpha"] = float(alpha)
                invalid_diagnostics.append(diagnostics)

            return None

        trim_m = float(state["trim_m"])

        if not np.isfinite(trim_m):
            raise ValueError("Cargo trim evaluation produced a nonfinite result.")

        return {
            "alpha": float(alpha),
            "allocation": allocation,
            "state": state,
            "trim_m": trim_m,
        }

    for alpha in np.linspace(0.0, 1.0, int(sample_count)):
        sampled.append(evaluate_alpha(float(alpha)))

    candidates = [
        candidate
        for candidate in sampled
        if candidate is not None
    ]

    if not candidates:
        # Retain the sampled candidate whose normalized force/moment closure
        # came closest to a solution.  Some failures occur before a nonlinear
        # solve and therefore have no residual pair; the remaining range
        # diagnostics are still kept on the search error.
        closest_diagnostics = {}
        residual_diagnostics = [
            diagnostics
            for diagnostics in invalid_diagnostics
            if (
                diagnostics.get("Normalized_Displacement_Volume_Residual")
                is not None
                and diagnostics.get(
                    "Normalized_Displacement_First_Moment_Residual"
                ) is not None
            )
        ]

        if residual_diagnostics:
            closest_diagnostics = min(
                residual_diagnostics,
                key=lambda diagnostics: max(
                    abs(
                        diagnostics[
                            "Normalized_Displacement_Volume_Residual"
                        ]
                    ),
                    abs(
                        diagnostics[
                            "Normalized_Displacement_First_Moment_Residual"
                        ]
                    ),
                ),
            )
        elif invalid_diagnostics:
            closest_diagnostics = invalid_diagnostics[0]

        raise CargoAllocationSearchError(
            "No capacity-feasible cargo allocation on the deterministic "
            "moment segment reached displacement and longitudinal-moment "
            "equilibrium within the generated hull draft grid.",
            evaluation_count=evaluation_count,
            invalid_evaluation_count=invalid_evaluation_count,
            closest_diagnostics=closest_diagnostics,
        )

    bracket_count = 0

    for left, right in zip(sampled[:-1], sampled[1:]):
        # Do not form a sign-change bracket across an allocation that has no
        # physical equilibrium.  Valid neighboring samples still bracket and
        # refine normally.
        if left is None or right is None:
            continue

        if left["trim_m"] == 0.0 or right["trim_m"] == 0.0:
            continue

        if np.signbit(left["trim_m"]) == np.signbit(right["trim_m"]):
            continue

        bracket_count += 1
        lower = left
        upper = right

        for _ in range(int(bisection_iterations)):
            midpoint = evaluate_alpha(
                0.5*(lower["alpha"] + upper["alpha"])
            )

            if midpoint is None:
                # The feasible-equilibrium set ended inside this numerical
                # interval.  Retain the valid endpoints as candidates instead
                # of aborting the complete system evaluation.
                break

            candidates.append(midpoint)

            if midpoint["trim_m"] == 0.0:
                lower = midpoint
                upper = midpoint
                break

            if np.signbit(midpoint["trim_m"]) == np.signbit(lower["trim_m"]):
                lower = midpoint
            else:
                upper = midpoint

    best = min(candidates, key=lambda item: abs(item["trim_m"]))

    return {
        "allocation": best["allocation"],
        "state": best["state"],
        "alpha": best["alpha"],
        "evaluation_count": evaluation_count,
        "invalid_evaluation_count": invalid_evaluation_count,
        "valid_evaluation_count": (
            evaluation_count - invalid_evaluation_count
        ),
        "sign_change_bracket_count": bracket_count,
    }
