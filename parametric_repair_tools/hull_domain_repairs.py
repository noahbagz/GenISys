"""Bounded repairs for the active hull and Holtrop domain predicates.

The 49 hull rows are generated, in a fixed order, by
``Hull_Parameterization.input_Constraints``.  Their mathematical convention
is ``g(x) <= 0``.  This module evaluates those *existing* equations without
changing them and searches only explicitly listed independent controls.

The five Holtrop rows are evaluated later in the integrated system model.
Where their reported value admits an analytical, physically directed update,
the inverse is applied here and the ordinary exact-evaluation loop verifies
the result.  No repair-specific surrogate is allowed to declare success.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .parameters import ParameterRegistry


# Keep this tuple in the exact concatenation order used by
# Hull_Parameterization.input_Constraints(): general, cross-section, bow,
# stern, and bulb predicates.  The names are the stable public identities in
# tools/Hull_constraints_list.csv.
HULL_CONSTRAINT_NAMES = (
    "Bow and Stern Taper",
    "WL Height ",
    "Positive Gunwale-chine Intersection",
    "Positive Chine Radius",
    "Positive Beam at Chine",
    "Positive Chine Height",
    "Chine - Bottom Fillet",
    "Keel - Bottom Fillet ",
    "Keel Radius is not exactly 0",
    "Keel Rise Fwd of Bow Taper End",
    "Drift Angle at Baseline < 90deg",
    "Drift Angle at Baseline > 0",
    "Drift Angle at Deck < 90deg",
    "Drift Angle at Deck > 0deg",
    "Drift Angle atDrift  Vertex < 90deg",
    "Drift Angle at Drift Vertex > 0deg",
    "Bow Rake and Keel Rise in LOA",
    "Bow Rake and Keel Rise Intersection fwd of Keel Rise start",
    "Bow Rake and Keel Rise above baseline",
    "Bow Rake and Keel Rise below deck",
    "Bow taper length at deck is positive",
    "Bow taper length at Bow rake-Keel intersect is positive",
    "Bow taper length is positive in Bow Rake (1)",
    "Bow taper length is positive in Bow Rake (2)",
    "Stern rise starts aft of stern taper start",
    "stern taper length is positive",
    "stern-transom intersect is aft of sternrise start",
    "stern taper is positive at deck",
    "stern taper is positive at transom-stern rise interect",
    "Bc_trans < B@Dc_trans",
    "Positive transom Gunwale-chine Intersection",
    "Positive transom chine radius",
    "positive transom chine beam",
    "positive transom chine height",
    "transom chine fillet is inboard of chine beam",
    "transom keel fillet is inboard of chine fillet",
    "Lower vertical bulbous bow radius is less than Rk",
    "longitudinal bulb radius less than Rk",
    "bow bulb beam is less than B@Z = bulb's widest point",
    "bulb is strictly forward of the end of the bow taper (1)",
    "bulb is strictly forward of the end of the bow taper (2)",
    "bulb is strictly forward of the end of the bow taper (3)",
    "Stern Bulb lower vertical radius is less than Rk",
    "Stern bulb longitudinal radius is less than Rk",
    "Stern bulb beam is less than B@Z = bulb's widest point",
    "Stern bulb height is less than stern-transom intersect",
    "Stern bulb is aft of the start of the stern taper (1)",
    "Stern bulb is aft of the start of the stern taper (2)",
    "Stern bulb is aft of the start of the stern taper (3)",
)

HULL_CONSTRAINT_INDEX = {
    name: index for index, name in enumerate(HULL_CONSTRAINT_NAMES)
}


# Every mapping below follows the actual algebra in HullParameterization.py.
# Coupled aliases Bc and bit_EP_T are intentionally absent because they are
# owned by ParameterRegistry.canonicalize, not independent repair controls.
_GENERAL_CONTROLS = {
    "Bow and Stern Taper": ("Lb", "Ls"),
    "WL Height ": ("WL",),
}

_MIDSHIP_CONTROLS = {
    "Positive Gunwale-chine Intersection": ("Rc", "Rk", "Bd", "Dd"),
    "Positive Chine Radius": ("Rc",),
    "Positive Beam at Chine": ("Bd",),
    "Positive Chine Height": ("Rc", "Rk", "Bd", "Dd"),
    "Chine - Bottom Fillet": ("Rc", "Rk", "Bd", "Dd"),
    "Keel - Bottom Fillet ": ("Rc", "Rk", "Bd", "Dd"),
    "Keel Radius is not exactly 0": ("Rk",),
}

_DRIFT_CONTROLS = {
    name: ("Drift_A", "Drift_B", "Drift_C")
    for name in HULL_CONSTRAINT_NAMES[10:16]
}

_BOW_CONTROLS = {
    name: (
        "Lb",
        "Bow_A",
        "Bow_B",
        "Bk",
        "Kappa_Bow",
        "Delta_Bow_A",
        "Delta_Bow_B",
    )
    for name in HULL_CONSTRAINT_NAMES[9:10]
    + HULL_CONSTRAINT_NAMES[16:24]
}

_STERN_CONTROLS = {
    name: (
        "Ls",
        "Bs",
        "Trans_A",
        "SK",
        "Kappa_Stern",
        "Delta_Stern_A",
        "Delta_Stern_B",
        "Beta_Trans",
        "Bc_Trans",
        "Rc_Trans",
        "Rk_Trans",
        "bit_EP_S",
    )
    for name in HULL_CONSTRAINT_NAMES[24:36]
}

_BOW_BULB_CONTROLS = {
    name: (
        "bit_BB",
        "Lbb",
        "Hbb",
        "Bbb",
        "Lbbm",
        "Rbb",
        "Rk",
        "Lb",
        "Delta_Bow_A",
        "Delta_Bow_B",
    )
    for name in HULL_CONSTRAINT_NAMES[36:42]
}

_STERN_BULB_CONTROLS = {
    name: (
        "bit_SB",
        "Kappa_SB",
        "Lsb",
        "HSBOA",
        "Hsb",
        "Bsb",
        "Lsbm",
        "Rsb",
        "Rk",
        "Ls",
        "SK",
        "Delta_Stern_A",
        "Delta_Stern_B",
    )
    for name in HULL_CONSTRAINT_NAMES[42:49]
}

HULL_REPAIR_CONTROLS: dict[str, tuple[str, ...]] = {
    **_GENERAL_CONTROLS,
    **_MIDSHIP_CONTROLS,
    **_DRIFT_CONTROLS,
    **_BOW_CONTROLS,
    **_STERN_CONTROLS,
    **_BOW_BULB_CONTROLS,
    **_STERN_BULB_CONTROLS,
}


HOLTROP_REPAIR_CONTROLS = {
    "Holtrop low/intermediate-speed Froude boundary": ("Fn_ship",),
    "Holtrop prismatic-coefficient equation domain": (
        "Lb",
        "Ls",
        "Draft_frac",
    ),
    "Holtrop waterline-length to breadth equation domain": (
        "Bd",
        "Lb",
        "Ls",
        "Draft_frac",
    ),
    "Holtrop run-length denominator magnitude": (
        "Lb",
        "Ls",
        "Draft_frac",
    ),
    "Holtrop run-length to waterline-length equation domain": (
        "Ls",
        "Lb",
        "Draft_frac",
    ),
}


@dataclass(frozen=True)
class HullDomainChange:
    """One independent hull control selected by the bounded search."""

    parameter: str
    old_value: float
    new_value: float
    equation: str
    constraint_names: tuple[str, ...]


@dataclass(frozen=True)
class HullDomainResult:
    """Hull-domain proposal plus any identities exhausted by the search."""

    vector: np.ndarray
    changes: tuple[HullDomainChange, ...]
    unresolved_names: tuple[str, ...]


def evaluate_hull_constraints(
    vector: Iterable[float],
    parameters: ParameterRegistry,
) -> np.ndarray:
    """Evaluate the unchanged 49-row hull equation vector.

    Candidate evaluation applies only the two established hull aliases.  It
    deliberately avoids full system canonicalization because changing a hull
    coordinate during a local search must not silently change structure or
    arrangement coordinates before a candidate is selected.
    """

    values = np.asarray(vector, dtype=float).copy()
    values[parameters.index["Bc"]] = values[parameters.index["Bd"]]
    values[parameters.index["bit_EP_T"]] = values[
        parameters.index["bit_EP_S"]
    ]

    try:
        # HullParameterization imports optional plotting support.  Import it
        # only when a failed hull row actually triggers local geometry work;
        # registry inspection and non-hull repairs should remain lightweight.
        from tools.HullParameterization import Hull_Parameterization

        # Invalid trial coordinates are expected during a bounded domain
        # search.  Convert their non-finite algebra to an infinite residual
        # below instead of emitting thousands of divide/sqrt warnings.
        with np.errstate(all="ignore"):
            hull = Hull_Parameterization(values[:45])
            result = np.asarray(hull.input_Constraints(), dtype=float)
    except (ArithmeticError, FloatingPointError, ValueError, np.linalg.LinAlgError):
        return np.full(len(HULL_CONSTRAINT_NAMES), np.inf, dtype=float)

    if result.shape != (len(HULL_CONSTRAINT_NAMES),):
        return np.full(len(HULL_CONSTRAINT_NAMES), np.inf, dtype=float)

    result[~np.isfinite(result)] = np.inf
    return result


class HullDomainRepairSearch:
    """Project failed hull predicates into a bounded admissible region."""

    # Six inexpensive coordinate sweeps are enough for these algebraic
    # predicates.  The integrated exact evaluator remains responsible for all
    # volume, resistance, structure, and system consequences.
    MAX_COORDINATE_SWEEPS = 6

    def __init__(self, parameters: ParameterRegistry):
        self.parameters = parameters

    def _characteristic_scale(self, index: int, vector: np.ndarray) -> float:
        """Return a small dimensional scale for passing-side clearance."""

        if 10 <= index <= 15:
            # The active implementation reports radians despite the CSV's
            # historical ``degrees`` label.  One degree is therefore the
            # natural numerical clearance scale.
            return math.radians(1.0)

        loa = abs(float(vector[self.parameters.index["LOA"]]))
        return max(0.01, 0.001*loa)

    def _score(
        self,
        values: np.ndarray,
        target_indices: tuple[int, ...],
        target_margins: Mapping[int, float],
        reference: np.ndarray,
    ) -> tuple[float, ...]:
        """Prefer globally valid candidates, then stochastic target slack."""

        hard_failures = int(np.count_nonzero(values > 0.0))
        margin_failures = 0
        margin_residual = 0.0

        for index in target_indices:
            scale = self._characteristic_scale(index, reference)
            requested_slack = float(target_margins[index])*scale
            residual = float(values[index]) + requested_slack

            if residual > 0.0:
                margin_failures += 1
                margin_residual += residual/scale

        finite = values[np.isfinite(values)]
        global_residual = float(
            np.sum(np.maximum(finite, 0.0))
            + np.count_nonzero(~np.isfinite(values))*1.0e6
        )

        return (
            float(hard_failures),
            float(margin_failures),
            float(margin_residual),
            global_residual,
        )

    def _candidate_values(
        self,
        parameter: str,
        current: float,
        margin: float,
    ) -> tuple[float, ...]:
        """Build a stable, bounded grid with a stochastic step location."""

        lower, upper = self.parameters.repair_bounds(parameter)

        if parameter in self.parameters.boolean_names:
            return (0.0, 1.0)

        if upper <= lower + 1.0e-15:
            return (float(lower),)

        span = upper - lower
        stochastic_fraction = min(0.45, max(0.05, float(margin)))
        raw = (
            current - stochastic_fraction*span,
            current + stochastic_fraction*span,
            lower + stochastic_fraction*span,
            upper - stochastic_fraction*span,
            lower + 0.10*span,
            lower + 0.25*span,
            lower + 0.50*span,
            lower + 0.75*span,
            lower + 0.90*span,
            lower,
            upper,
        )
        clipped = [
            self.parameters.clip_to_repair_domain(parameter, value)
            for value in raw
        ]
        return tuple(dict.fromkeys(float(value) for value in clipped))

    def _set_control(
        self,
        vector: np.ndarray,
        parameter: str,
        value: float,
    ) -> None:
        """Write one independent control to a local search vector."""

        vector[self.parameters.index[parameter]] = (
            self.parameters.clip_to_repair_domain(parameter, value)
        )

    def _apply_direct_projections(
        self,
        vector: np.ndarray,
        target_names: set[str],
        margin_by_name: Mapping[str, float],
        equations: dict[str, str],
    ) -> None:
        """Apply exact/simple projections before nonlinear coordinate search."""

        # g = LOA*(Lb + Ls - 1) <= 0.  Preserve the current bow/stern
        # proportion while moving the sum onto the passing side.
        if "Bow and Stern Taper" in target_names:
            margin = margin_by_name["Bow and Stern Taper"]
            lb = float(vector[self.parameters.index["Lb"]])
            ls = float(vector[self.parameters.index["Ls"]])
            current_sum = max(lb + ls, 1.0e-12)
            target_sum = 1.0/(1.0 + margin)

            for parameter, value in (
                ("Lb", lb*target_sum/current_sum),
                ("Ls", ls*target_sum/current_sum),
            ):
                self._set_control(vector, parameter, value)
                equations[parameter] = (
                    "Lb_new+Ls_new = 1/(1+margin), preserving Lb:Ls"
                )

        # g = Dd*(WL - 1) <= 0 because WL is stored as a depth fraction.
        if "WL Height " in target_names:
            margin = margin_by_name["WL Height "]
            self._set_control(vector, "WL", 1.0/(1.0 + margin))
            equations["WL"] = "WL_new = 1/(1+margin)"

        # Project an invalid quadratic drift law to a constant, positive
        # entrance angle.  The sampled margin selects the interior angle and
        # therefore retains diversity without inventing a new relationship.
        drift_targets = target_names.intersection(_DRIFT_CONTROLS)

        if drift_targets:
            margin = float(np.mean([margin_by_name[name] for name in drift_targets]))
            lower_margin, upper_margin = 0.10, 0.35
            unit = np.clip(
                (margin - lower_margin)/(upper_margin - lower_margin),
                0.0,
                1.0,
            )
            angle_degrees = 20.0 + 25.0*float(unit)

            for parameter, value in (
                ("Drift_A", 0.0),
                ("Drift_B", 0.0),
                ("Drift_C", angle_degrees),
            ):
                self._set_control(vector, parameter, value)
                equations[parameter] = (
                    f"project drift(z) to constant {angle_degrees:.9g} deg"
                )

        # These inequalities are strict physical positivity requirements.
        # Move a zero/negative radius or beam to an interior point selected by
        # the same saved margin used for the failed identity.
        positive_controls = {
            "Positive Chine Radius": "Rc",
            "Positive Beam at Chine": "Bd",
            "Keel Radius is not exactly 0": "Rk",
            "Positive transom chine radius": "Rc_Trans",
            "positive transom chine beam": "Bc_Trans",
        }

        for name, parameter in positive_controls.items():
            if name not in target_names:
                continue

            margin = margin_by_name[name]
            lower, upper = self.parameters.repair_bounds(parameter)
            candidate = lower + max(0.05, margin)*(upper - lower)
            self._set_control(vector, parameter, candidate)
            equations[parameter] = (
                "p_new = lower + max(0.05, margin)*(upper-lower)"
            )

        # Removing an optional bulb is an exact topology enumeration already
        # permitted by the 189-parameter interface.  It deactivates only the
        # corresponding conditional bulb rows; no out-of-scope system is
        # introduced.  Shape-preserving bulb optimization would require a
        # resistance objective and belongs in the outer design optimization.
        if target_names.intersection(_BOW_BULB_CONTROLS):
            self._set_control(vector, "bit_BB", 0.0)
            equations["bit_BB"] = (
                "enumerate optional bow-bulb topology: bit_BB_new = 0"
            )

        if target_names.intersection(_STERN_BULB_CONTROLS):
            self._set_control(vector, "bit_SB", 0.0)
            equations["bit_SB"] = (
                "enumerate optional stern-bulb topology: bit_SB_new = 0"
            )

    def repair_hull_rows(
        self,
        vector: Iterable[float],
        rows: pd.DataFrame,
        margin_by_name: Mapping[str, float],
    ) -> HullDomainResult:
        """Repair known failed hull rows using direct and bounded steps."""

        original = np.asarray(vector, dtype=float).copy()
        candidate = original.copy()
        target_names = {
            str(name) for name in rows["Name"] if str(name) in HULL_REPAIR_CONTROLS
        }

        if not target_names:
            return HullDomainResult(candidate, (), ())

        target_indices = tuple(
            sorted(HULL_CONSTRAINT_INDEX[name] for name in target_names)
        )
        margin_by_index = {
            HULL_CONSTRAINT_INDEX[name]: float(margin_by_name[name])
            for name in target_names
        }
        equations: dict[str, str] = {}
        self._apply_direct_projections(
            candidate,
            target_names,
            margin_by_name,
            equations,
        )

        current_values = evaluate_hull_constraints(candidate, self.parameters)
        current_score = self._score(
            current_values,
            target_indices,
            margin_by_index,
            original,
        )

        # Search only controls that occur in the remaining failed/margin-short
        # identities.  Each accepted coordinate must improve the unchanged
        # hull equations; an arbitrary random mutation is never accepted.
        for _ in range(self.MAX_COORDINATE_SWEEPS):
            active_names = {
                name
                for name in target_names
                if (
                    current_values[HULL_CONSTRAINT_INDEX[name]]
                    + margin_by_name[name]
                    * self._characteristic_scale(
                        HULL_CONSTRAINT_INDEX[name], original
                    )
                    > 0.0
                )
            }

            if not active_names:
                break

            controls = tuple(
                dict.fromkeys(
                    parameter
                    for name in sorted(active_names)
                    for parameter in HULL_REPAIR_CONTROLS[name]
                    if self.parameters.role(parameter).startswith("control_")
                )
            )
            improved = False

            for parameter in controls:
                position = self.parameters.index[parameter]
                current = float(candidate[position])
                related = [
                    name
                    for name in active_names
                    if parameter in HULL_REPAIR_CONTROLS[name]
                ]
                margin = float(
                    np.mean([margin_by_name[name] for name in related])
                )
                best_vector = candidate
                best_values = current_values
                best_score = current_score

                for value in self._candidate_values(
                    parameter,
                    current,
                    margin,
                ):
                    trial = candidate.copy()
                    self._set_control(trial, parameter, value)
                    trial_values = evaluate_hull_constraints(
                        trial,
                        self.parameters,
                    )
                    trial_score = self._score(
                        trial_values,
                        target_indices,
                        margin_by_index,
                        original,
                    )

                    if trial_score < best_score:
                        best_vector = trial
                        best_values = trial_values
                        best_score = trial_score

                if best_score < current_score:
                    candidate = best_vector.copy()
                    current_values = best_values
                    current_score = best_score
                    equations[parameter] = (
                        "bounded coordinate search minimizing the unchanged "
                        "49-row hull-domain residual vector"
                    )
                    improved = True

            if not improved:
                break

        unresolved = tuple(
            name
            for name in sorted(target_names)
            if current_values[HULL_CONSTRAINT_INDEX[name]] > 0.0
        )
        changes = []

        for parameter in sorted(equations, key=self.parameters.index.get):
            old = float(original[self.parameters.index[parameter]])
            new = float(candidate[self.parameters.index[parameter]])

            if np.isclose(old, new, rtol=0.0, atol=1.0e-12):
                continue

            causal_names = tuple(
                name
                for name in sorted(target_names)
                if parameter in HULL_REPAIR_CONTROLS[name]
                or (
                    parameter == "bit_BB"
                    and name in _BOW_BULB_CONTROLS
                )
                or (
                    parameter == "bit_SB"
                    and name in _STERN_BULB_CONTROLS
                )
            )
            changes.append(
                HullDomainChange(
                    parameter=parameter,
                    old_value=old,
                    new_value=new,
                    equation=equations[parameter],
                    constraint_names=causal_names,
                )
            )

        return HullDomainResult(candidate, tuple(changes), unresolved)

    def repair_holtrop_row(
        self,
        vector: Iterable[float],
        row: pd.Series,
        margin: float,
    ) -> HullDomainResult:
        """Apply a reported-value inverse for one Holtrop domain row."""

        original = np.asarray(vector, dtype=float).copy()
        candidate = original.copy()
        name = str(row.get("Name", ""))
        threshold = pd.to_numeric(
            pd.Series([row.get("Threshold Value")]), errors="coerce"
        ).iloc[0]
        offered = pd.to_numeric(
            pd.Series([row.get("Constraint Value")]), errors="coerce"
        ).iloc[0]
        equations: dict[str, str] = {}

        if name == "Holtrop low/intermediate-speed Froude boundary":
            if pd.notna(threshold) and float(threshold) > 0.0:
                value = float(threshold)/(1.0 + float(margin))
                self._set_control(candidate, "Fn_ship", value)
                equations["Fn_ship"] = "Fn_new = Fn_max/(1+margin)"

        elif name == "Holtrop waterline-length to breadth equation domain":
            if (
                pd.notna(threshold)
                and pd.notna(offered)
                and float(threshold) > 0.0
                and float(offered) > 0.0
            ):
                current_bd = float(candidate[self.parameters.index["Bd"]])
                target_ratio = float(threshold)*(1.0 + float(margin))
                value = current_bd*float(offered)/target_ratio
                self._set_control(candidate, "Bd", value)
                equations["Bd"] = (
                    "Bd_new = Bd_old*(reported LWL/B)/(minimum*(1+margin))"
                )

        elif name == "Holtrop prismatic-coefficient equation domain":
            if (
                pd.notna(threshold)
                and pd.notna(offered)
                and float(threshold) > 0.0
                and float(offered) > 0.0
            ):
                # Cp = displaced volume/(LOA*immersed midship area).  At
                # fixed midship section, lengthening both end tapers removes
                # volume from the prismatic reference and is the directed
                # shape change for Cp >= 1.
                target_cp = float(threshold)/(1.0 + float(margin))
                scale = max(1.0, float(offered)/target_cp)

                for parameter in ("Lb", "Ls"):
                    current = float(candidate[self.parameters.index[parameter]])
                    self._set_control(candidate, parameter, current*scale)
                    equations[parameter] = (
                        "increase bow/stern taper fractions by reported "
                        "Cp/(Cp_max/(1+margin)); exact evaluator verifies Cp"
                    )

        elif name == "Holtrop run-length denominator magnitude":
            # At Cp=0.25 the reported |4Cp-1| is exactly zero.  Increasing
            # end-taper length reduces prismatic fullness and moves Cp away
            # from the singular point.  The saved stochastic margin selects
            # the bounded perturbation; exact reevaluation supplies the new Cp.
            current = float(candidate[self.parameters.index["Lb"]])
            lower, upper = self.parameters.repair_bounds("Lb")
            step = max(0.05, float(margin))*(upper - lower)
            value = current + step if current < upper - 1.0e-12 else current - step
            self._set_control(candidate, "Lb", value)
            equations["Lb"] = (
                "move Cp away from 0.25 by a bounded end-taper perturbation"
            )

        elif name == (
            "Holtrop run-length to waterline-length equation domain"
        ):
            # Lr/LWL = 1-Cp + 0.06*Cp*lcb/(4*Cp-1).  Increasing the stern
            # taper removes aft immersed volume: it generally reduces Cp and
            # moves the buoyancy center forward in Holtrop's signed LCB
            # convention.  Both effects move the normalized run length in the
            # passing direction for the observed negative-run cases.  Ls is
            # therefore the first legitimate upstream control.  A smaller
            # symmetric Lb change also reduces Cp without relying only on an
            # LCB shift.  The exact integrated evaluator recomputes Cp, LCB,
            # Lr, and every coupled ship result after this proposal.
            if pd.notna(threshold) and pd.notna(offered):
                target_ratio = float(threshold)*(1.0 + float(margin))
                deficit_ratio = max(0.0, target_ratio - float(offered))
                normalized_deficit = min(
                    1.0,
                    deficit_ratio/max(target_ratio, 1.0e-12),
                )
                step_fraction = max(0.05, float(margin))*max(
                    normalized_deficit,
                    0.25,
                )

                for parameter, share in (("Ls", 1.0), ("Lb", 0.35)):
                    current = float(
                        candidate[self.parameters.index[parameter]]
                    )
                    lower, upper = self.parameters.repair_bounds(parameter)
                    value = current + share*step_fraction*(upper - lower)
                    self._set_control(candidate, parameter, value)
                    equations[parameter] = (
                        "increase end-taper fraction to reduce Cp and, for "
                        "Ls, shift LCB forward; exact evaluator verifies "
                        "Lr/LWL >= 0.01*(1+margin)"
                    )

        changes = tuple(
            HullDomainChange(
                parameter=parameter,
                old_value=float(original[self.parameters.index[parameter]]),
                new_value=float(candidate[self.parameters.index[parameter]]),
                equation=equation,
                constraint_names=(name,),
            )
            for parameter, equation in equations.items()
            if not np.isclose(
                original[self.parameters.index[parameter]],
                candidate[self.parameters.index[parameter]],
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        unresolved = () if changes else (name,)
        return HullDomainResult(candidate, changes, unresolved)


__all__ = [
    "HOLTROP_REPAIR_CONTROLS",
    "HULL_CONSTRAINT_INDEX",
    "HULL_CONSTRAINT_NAMES",
    "HULL_REPAIR_CONTROLS",
    "HullDomainChange",
    "HullDomainRepairSearch",
    "HullDomainResult",
    "evaluate_hull_constraints",
]
