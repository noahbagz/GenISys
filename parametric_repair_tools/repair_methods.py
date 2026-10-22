"""Generate one equation-informed repaired GenISys parameter vector.

This module follows the maintained Ship_Structures ordering: direct
dimensions and spacings first, then section-modulus reconciliation. Every
failure comes from the saved source evaluation, every mapped parameter is
modified at most once in one simultaneous pass, and no post-repair result is
fed back into this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from cad_query_tools.parameters import ParametricStructureDefinition
from tools.ABS_Part3_Constraints import ABS_Part3_Constraints

from .configuration import RepairConfiguration
from .evaluation import failed_constraint_rows
from .hull_domain_repairs import HullDomainRepairSearch
from .parameters import ParameterRegistry
from .registry import ConstraintRepairRegistry, RepairRule


def constraint_key(row: pd.Series | dict) -> str:
    """Return the stable composite identity used by V1.1 artifacts."""

    return f"{row.get('Requirement', '')}::{row.get('Name', '')}"


class MarginLedger:
    """Record the seeded random decisions used by one repair pass."""

    def __init__(
        self,
        seed: int,
        configuration: RepairConfiguration,
    ):
        self.seed = int(seed)
        self.configuration = configuration
        self.rng = np.random.default_rng(self.seed)
        self.draws: dict[str, dict[str, float | str]] = {}

    def _range_for(self, method: str) -> tuple[float, float]:
        if method in {
            "section_modulus",
            "global_hull_girder",
            "uniform_bulkhead_capacity",
        }:
            return self.configuration.section_margin_range

        if method in {"maximum_spacing_count", "subdivision"}:
            return self.configuration.spacing_margin_range

        if method in {
            "trim",
            "tank_capacity",
            "system_capacity",
            "system_displacement",
            "cargo_equilibrium",
        }:
            return self.configuration.system_step_range

        return self.configuration.direct_margin_range

    def margin_for(self, key: str, method: str, family: str) -> float:
        """Return one reproducible passing-side margin for a source failure."""

        if key not in self.draws:
            lower, upper = self._range_for(method)
            value = float(self.rng.uniform(lower, upper))
            self.draws[key] = {
                "Constraint_Key": key,
                "Repair_Method": method,
                "Repair_Family": family,
                "Range_Lower": float(lower),
                "Range_Upper": float(upper),
                "Sampled_Margin": value,
                "Design_Repair_Seed": self.seed,
            }

        return float(self.draws[key]["Sampled_Margin"])

    def choice_for(
        self,
        key: str,
        method: str,
        family: str,
        options: tuple[str, ...],
    ) -> str:
        """Return one reproducible equal-probability discrete repair choice."""

        if not options:
            raise ValueError("A stochastic repair choice needs options.")

        # Create the ordinary margin record first so the complete stochastic
        # decision for this failed constraint is kept in one provenance row.
        self.margin_for(key, method, family)
        record = self.draws[key]

        if "Stochastic_Choice" not in record:
            selected = int(self.rng.integers(0, len(options)))
            record["Stochastic_Choice"] = str(options[selected])
            record["Choice_Probability"] = 1.0/len(options)

        return str(record["Stochastic_Choice"])

    def table(self) -> pd.DataFrame:
        """Return every actual random draw, not merely its reconstructing seed."""

        return pd.DataFrame(list(self.draws.values()))


@dataclass
class ProposedChange:
    """One parameter update proposed by one or more mapped failures."""

    parameter: str
    old_value: float
    proposed_value: float
    repair_method: str
    repair_family: str
    constraint_keys: tuple[str, ...]
    sampled_margins: tuple[float, ...]
    equation: str
    repair_lower_limit: float
    repair_upper_limit: float
    at_repair_domain_limit: bool
    outside_sampling_bounds: bool


@dataclass
class RepairProposal:
    """Complete candidate vector and the evidence used to construct it."""

    vector: np.ndarray
    changes: list[ProposedChange]
    unmapped: pd.DataFrame


def _numeric(row: pd.Series, column: str) -> float | None:
    """Read one finite scalar or return ``None`` for a nonnumeric row."""

    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]

    if pd.isna(value) or not np.isfinite(float(value)):
        return None

    return float(value)


class EquationRepairMethods:
    """Generate one simultaneous Ship_Structures-style repair pass."""

    COUNT_OFFSET = {
        "num bottom girders": -1.0,
        "num intermediate bottom stiffeners": 1.0,
        "web frames per hold": 1.0,
        "num intermediate trans frames": 1.0,
        "num top girders": -1.0,
        "num intermediate deck stiffeners": 1.0,
        "num side shell stiffeners": 2.0,
        "num vertical bulkhead stiffeners": 0.0,
        "num transverse bulkhead stiffeners": 1.0,
        "num bulk car bottom hopper stiff": 1.0,
        "num bulk car top hopper stiff": 1.0,
    }

    SECTION_PRIMARY = {
        "bottom_stringer": (
            "intermediate bottom stiffener h",
            "intermediate bottom stiffener t",
            "intermediate bottom stiffener w",
        ),
        "bottom_frame": (
            "intermediate bottom frame h",
            "intermediate bottom frame t",
            "intermediate bottom frame w",
        ),
        "bottom_girder": ("Db", "bottom girder thickness", None),
        "bottom_floor": ("Db", "floor t", None),
        "deck_stringer": ("deck stiff h", "deck stiff t", "deck stiff w"),
        "deck_frame": (
            "intermediate deck frame h",
            "intermediate deck frame t",
            "intermediate deck frame w",
        ),
        "deck_beam": (
            "trans deck beam h",
            "trans deck beam t",
            "trans deck beam w",
        ),
        "deck_girder": ("deck girder h", "deck girder t", "deck girder w"),
        "side_stringer": (
            "side shell Long stiff h",
            "side shell long stiff t",
            "side shell long stiff w",
        ),
        "side_frame": (
            "intermediate side frame h",
            "intermediate side frame t",
            "intermediate side frame w",
        ),
        "web_frame": ("web frame h", "web frame t", "web frame w"),
        "bulkhead_horizontal": (
            "blkhd trans stiff h",
            "blkhd trans stiff t",
            "blkhd trans stiff w",
        ),
        "bulkhead_vertical": (
            "blkhd vert stiff h",
            "blkhd vert stiff t",
            "blkhd vert stiff w",
        ),
        "hopper_lower": (
            "bulk car bottom hopper stiff h",
            "bulk car bottom hopper stiff t",
            "bulk car bottom hopper stiff w",
        ),
        "hopper_upper": (
            "bulk car top hopper stiff h",
            "bulk car top hopper stiff t",
            "bulk car top hopper stiff w",
        ),
        "shear_strake": ("Shear Strake Height", "Shear Strake thickness", None),
    }

    SECTION_COUNT = {
        "bottom_stringer": "num intermediate bottom stiffeners",
        "bottom_frame": "num intermediate trans frames",
        "bottom_girder": "num bottom girders",
        "bottom_floor": "web frames per hold",
        "deck_stringer": "num intermediate bottom stiffeners",
        "deck_frame": "num intermediate trans frames",
        "deck_beam": "web frames per hold",
        "deck_girder": "num bottom girders",
        "side_stringer": "num side shell stiffeners",
        "side_frame": "num intermediate trans frames",
        "web_frame": "web frames per hold",
        "bulkhead_horizontal": "num transverse bulkhead stiffeners",
        "bulkhead_vertical": "num vertical bulkhead stiffeners",
        "hopper_lower": "num bulk car bottom hopper stiff",
        "hopper_upper": "num bulk car top hopper stiff",
    }

    SECTION_PLATE = {
        "bottom_stringer": "bottom shell thickness",
        "bottom_frame": "bottom shell thickness",
        "deck_stringer": "Top Deck Thickness",
        "deck_frame": "Top Deck Thickness",
        "deck_beam": "Top Deck Thickness",
        "deck_girder": "Top Deck Thickness",
        "side_stringer": "side shell thickness",
        "side_frame": "side shell thickness",
        "web_frame": "side shell thickness",
        "bulkhead_horizontal": "Trans Bulkhead Thickness",
        "bulkhead_vertical": "Trans Bulkhead Thickness",
        "hopper_lower": "bulk car bottom hopper t",
        "hopper_upper": "bulk car top hopper t",
    }

    def __init__(
        self,
        parameters: ParameterRegistry,
        constraints: ConstraintRepairRegistry,
        configuration: RepairConfiguration,
    ):
        self.parameters = parameters
        self.constraints = constraints
        self.configuration = configuration
        self.hull_domain_search = HullDomainRepairSearch(parameters)

    def _pass_start_value(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        parameter: str,
    ) -> float:
        """Return a control's value at the beginning of this repair pass.

        Several independently reported rules can own the same design
        coordinate.  The candidate for each rule must be calculated from the
        same pass-start value; calculating from an earlier rule's proposal
        compounds independent safety margins and over-repairs the design.
        ``base_old`` is captured by the first setter call and remains the
        authoritative anchor for every later owner of that coordinate.
        """

        existing = updates.get(parameter)

        if existing is not None:
            return float(existing["base_old"])

        return float(proposal[self.parameters.index[parameter]])

    @staticmethod
    def _append_cause(
        update: dict,
        key: str,
        margin: float,
        equation: str | None = None,
    ) -> None:
        """Record each causal key once with its corresponding margin.

        Keeping the two lists aligned is important: downstream provenance can
        reconstruct which stochastic satisfaction margin belonged to which
        failed rule, including non-governing rules that shared the repaired
        parameter.
        """

        if key not in update["keys"]:
            update["keys"].append(key)
            update["margins"].append(float(margin))

        if equation is not None and equation not in update["equations"]:
            update["equations"].append(equation)

    def _set_larger(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        parameter: str,
        candidate: float,
        row: pd.Series,
        rule: RepairRule,
        margin: float,
        equation: str,
    ) -> None:
        """Merge increasing repairs using their governing pass-start target."""

        position = self.parameters.index[parameter]
        old = self._pass_start_value(proposal, updates, parameter)
        candidate = self.parameters.clip_to_repair_domain(parameter, candidate)
        candidate = max(old, candidate)
        key = constraint_key(row)
        existing = updates.get(parameter)

        if existing is None:
            if candidate <= old + 1.0e-12:
                return

            updates[parameter] = {
                "base_old": old,
                "value": candidate,
                "method": rule.method,
                "family": rule.family,
                "keys": [key],
                "margins": [margin],
                "equations": [equation],
            }
        else:
            # Retain the most demanding passing-side candidate, while still
            # retaining every failed rule and random margin that participated
            # in this shared-coordinate repair.
            if candidate > float(existing["value"]):
                existing["value"] = candidate
                existing["method"] = rule.method
                existing["family"] = rule.family

            self._append_cause(existing, key, margin, equation)

        proposal[position] = float(updates[parameter]["value"])

    def _set_smaller(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        parameter: str,
        candidate: float,
        row: pd.Series,
        rule: RepairRule,
        margin: float,
        equation: str,
    ) -> None:
        """Merge decreasing repairs using their governing pass-start target."""

        position = self.parameters.index[parameter]
        old = self._pass_start_value(proposal, updates, parameter)
        candidate = self.parameters.clip_to_repair_domain(parameter, candidate)
        candidate = min(old, candidate)
        key = constraint_key(row)
        existing = updates.get(parameter)

        if existing is None:
            if candidate >= old - 1.0e-12:
                return

            updates[parameter] = {
                "base_old": old,
                "value": candidate,
                "method": rule.method,
                "family": rule.family,
                "keys": [key],
                "margins": [margin],
                "equations": [equation],
            }
        else:
            if candidate < float(existing["value"]):
                existing["value"] = candidate
                existing["method"] = rule.method
                existing["family"] = rule.family

            self._append_cause(existing, key, margin, equation)

        proposal[position] = float(updates[parameter]["value"])

    def _set_value(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        parameter: str,
        candidate: float,
        row: pd.Series,
        rule: RepairRule,
        margin: float,
        equation: str,
    ) -> None:
        """Merge a bounded non-monotonic update such as bracket angle."""

        position = self.parameters.index[parameter]
        old = self._pass_start_value(proposal, updates, parameter)
        candidate = self.parameters.clip_to_repair_domain(parameter, candidate)
        key = constraint_key(row)
        existing = updates.get(parameter)

        # A simultaneous pass can touch the same control from more than one
        # loading condition.  Preserve the value at the beginning of the pass
        # and every causal constraint instead of replacing earlier evidence.
        if existing is None:
            updates[parameter] = {
                "base_old": old,
                "value": candidate,
                "method": rule.method,
                "family": rule.family,
                "keys": [key],
                "margins": [margin],
                "equations": [equation],
            }
        else:
            existing["value"] = candidate
            existing["method"] = rule.method
            existing["family"] = rule.family
            self._append_cause(existing, key, margin, equation)

        proposal[position] = candidate

    def _direct_minimum(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        row: pd.Series,
        rule: RepairRule,
        margin: float,
    ) -> None:
        threshold = _numeric(row, "Threshold Value")
        offered = _numeric(row, "Constraint Value")

        if threshold is None or offered is None:
            return

        parameter = rule.parameters[0]
        target = threshold*(1.0 + margin)

        # Every registry entry using ``direct_minimum`` maps to the dimension
        # reported in Constraint Value.  Solve that relationship directly
        # instead of adding the residual to a potentially stale source
        # parameter.  ``offered_scale`` handles the approved center-girder
        # convention, where the reported plate is twice the ordinary-girder
        # thickness control.
        candidate = target/float(rule.offered_scale)

        if parameter in self.parameters.integer_names:
            candidate = math.ceil(candidate - 1.0e-12)

        self._set_larger(
            proposal,
            updates,
            parameter,
            candidate,
            row,
            rule,
            margin,
            (
                "p_new = threshold*(1+margin)/"
                f"{rule.offered_scale:g}"
            ),
        )

    def _maximum_spacing_count(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        row: pd.Series,
        rule: RepairRule,
        margin: float,
    ) -> None:
        maximum = _numeric(row, "Threshold Value")
        spacing = _numeric(row, "Constraint Value")

        if maximum is None or spacing is None or maximum <= 0.0:
            return

        parameter = rule.parameters[0]
        current = self._pass_start_value(proposal, updates, parameter)
        offset = self.COUNT_OFFSET.get(parameter, 1.0)
        target_spacing = maximum/(1.0 + margin)
        candidate = math.ceil(
            (current + offset)*(spacing/target_spacing) - offset
        )
        self._set_larger(
            proposal,
            updates,
            parameter,
            candidate,
            row,
            rule,
            margin,
            (
                "count_new = ceil((count_old+offset)*"
                "spacing/(maximum/(1+margin))-offset)"
            ),
        )

    def _maximum_aspect_ratio(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        row: pd.Series,
        rule: RepairRule,
        margin: float,
    ) -> None:
        maximum = _numeric(row, "Threshold Value")
        aspect = _numeric(row, "Constraint Value")

        if maximum is None or aspect is None or maximum <= 0.0:
            return

        thickness_parameter, height_parameter = rule.parameters
        thickness = self._pass_start_value(
            proposal,
            updates,
            thickness_parameter,
        )
        height = self._pass_start_value(
            proposal,
            updates,
            height_parameter,
        )
        target_aspect = maximum/(1.0 + margin)

        # Use both the exact reported ratio and the current parameter height.
        # The maximum catches small CAD/rounding differences safely.
        candidate = max(
            thickness*aspect/target_aspect,
            height/target_aspect,
        )

        if thickness_parameter in self.parameters.integer_names:
            candidate = math.ceil(candidate - 1.0e-12)

        self._set_larger(
            proposal,
            updates,
            thickness_parameter,
            candidate,
            row,
            rule,
            margin,
            "t_new >= h/(maximum/(1+margin))",
        )

    def _structure_definition_for_repair(
        self,
        vector: np.ndarray,
    ) -> ParametricStructureDefinition:
        """Build the same snapped structural definition used by evaluation.

        A simultaneous repair can change ``LOA``, ``Bd``, ``Dd``, or draft
        after the incoming evaluated vector was created.  Its stored ``B``,
        ``D``, ``T``, and ``L_3h`` coordinates are consequently stale until
        canonicalization.  Recompute those owned coordinates before predicting
        CadQuery's stiffener grids and bracket endpoint snapping.
        """

        # Bilge/double-bottom compatibility belongs to the exact evaluator and
        # its own repair family.  This preview needs only refreshed owned hull
        # dimensions and must not turn an unrelated bilge-domain issue into a
        # bracket-repair exception.
        canonical, _ = self.parameters.canonicalize(
            vector,
            ensure_bilge_fit=False,
        )
        structural_names = self.parameters.catalog.loc[
            self.parameters.catalog["Class"].eq("Structure"),
            "Name",
        ].astype(str).tolist()
        structural_values = np.asarray(
            [
                canonical[self.parameters.index[name]]
                for name in structural_names
            ],
            dtype=float,
        )

        # The constructor intentionally snaps a mutable copy in place.  The
        # caller's proposal must remain untouched until a documented setter
        # records the corresponding repair action.
        return ParametricStructureDefinition(structural_values.copy())

    def _bracket_angle(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        row: pd.Series,
        rule: RepairRule,
        margin: float,
    ) -> None:
        """Select bracket endpoints that remain valid after generator snapping.

        CadQuery connects both ends of each hopper bracket to generated
        stiffener grids.  Setting ``L2 = L1*tan(angle)`` alone is therefore not
        an inverse of the evaluated geometry: the requested endpoint can snap
        to the next side stiffener and move back outside the 40--50 degree
        feasibility band.  Enumerate the *actual* endpoint pairs available on
        the current generated grids and write one pair whose snapped angle is
        closest to the design's persistent stochastic target.

        The complete CadQuery/system evaluation remains authoritative.  This
        inexpensive parameter-definition calculation only predicts the same
        grid snapping used by that evaluator.
        """

        l1_name, l2_name = rule.parameters

        # Start from L1=L2, the 45-degree center of the approved band.  The
        # discrete-grid search below then makes the smallest correction needed
        # to land both endpoints on generated supporting members.
        target_degrees = 45.0

        # ParametricStructureDefinition performs the same count-grid creation
        # and endpoint snapping as the full CadQuery generator without building
        # any solids.  Pass a copy because the legacy-compatible constructor
        # intentionally updates its input row in place.
        definition = self._structure_definition_for_repair(proposal)
        vertex_y_mm = 1_000.0*(
            definition.Hull_Dict["B"]/2.0
            - definition.Web_Frame_Dict["Side_Dict"]["h"]/1_000.0
        )
        depth_mm = 1_000.0*definition.Hull_Dict["D"]
        double_bottom_mm = float(definition.Struct_Params["Db"])
        side_grid_mm = (
            np.asarray(definition.z_side_stiffeners, dtype=float)*depth_mm
        )

        if rule.family == "lower_hopper_bracket_angle":
            transverse_grid_mm = np.sort(
                np.concatenate(
                    (
                        definition.y_bottom_stiffeners,
                        definition.y_bottom_girders,
                    )
                )
                * 500.0
                * definition.Hull_Dict["B"]
            )
            transverse_grid_mm = transverse_grid_mm[
                transverse_grid_mm <= vertex_y_mm+1.0e-9
            ]
            l1_options = vertex_y_mm-transverse_grid_mm
            l2_options = side_grid_mm[
                side_grid_mm >= double_bottom_mm-1.0e-9
            ]-double_bottom_mm
        else:
            transverse_grid_mm = np.sort(
                np.concatenate(
                    (
                        definition.y_deck_stiffeners,
                        definition.y_deck_girders,
                    )
                )
                * 500.0
                * definition.Hull_Dict["B"]
            )

            # Match the generator's hatch-opening exclusion before choosing a
            # deck endpoint for the upper bracket.
            transverse_grid_mm = transverse_grid_mm[
                transverse_grid_mm >= definition.Deck_Hatch_Half_width-1.0e-9
            ]
            transverse_grid_mm = transverse_grid_mm[
                transverse_grid_mm <= vertex_y_mm+1.0e-9
            ]
            l1_options = vertex_y_mm-transverse_grid_mm
            l2_options = depth_mm-side_grid_mm[
                side_grid_mm <= depth_mm+1.0e-9
            ]

        l1_lower, l1_upper = self.parameters.repair_bounds(l1_name)
        l2_lower, l2_upper = self.parameters.repair_bounds(l2_name)
        l1_options = sorted(
            {
                float(value)
                for value in l1_options
                if l1_lower <= value <= l1_upper and value > 0.0
            }
        )
        l2_options = sorted(
            {
                float(value)
                for value in l2_options
                if l2_lower <= value <= l2_upper and value > 0.0
            }
        )

        current_l1 = self._pass_start_value(proposal, updates, l1_name)
        current_l2 = self._pass_start_value(proposal, updates, l2_name)
        mean_length = 0.5*(current_l1 + current_l2)
        candidates = []

        for candidate_l1 in l1_options:
            for candidate_l2 in l2_options:
                angle = math.degrees(math.atan2(candidate_l2, candidate_l1))

                if not 40.0 <= angle <= 50.0:
                    continue

                relative_movement = math.hypot(
                    (candidate_l1-mean_length)/max(abs(mean_length), 1.0),
                    (candidate_l2-mean_length)/max(abs(mean_length), 1.0),
                )
                candidates.append(
                    (
                        relative_movement,
                        abs(angle-target_degrees),
                        candidate_l1,
                        candidate_l2,
                        angle,
                    )
                )

        if candidates:
            _, _, snapped_l1, snapped_l2, snapped_angle = min(candidates)

            # The generator selects grid points with inclusive floating-point
            # comparisons.  Request a point infinitesimally inside the desired
            # cell so roundoff cannot place an exactly coincident endpoint in
            # the neighboring cell.  CadQuery then writes the exact snapped
            # lengths back into the evaluated vector.
            snap_clearance_mm = max(
                1.0e-6,
                1.0e-9*max(snapped_l1, snapped_l2),
            )
            candidate_l1 = max(l1_lower, snapped_l1-snap_clearance_mm)
            candidate_l2 = max(l2_lower, snapped_l2-snap_clearance_mm)
            equation = (
                "select generated stiffener-grid endpoints minimizing "
                f"|angle-{target_degrees:.9g}|; snapped angle="
                f"{snapped_angle:.9g} deg"
            )
        else:
            candidate_l1 = mean_length
            candidate_l2 = mean_length
            equation = (
                "set L1=L2=mean(L1,L2); no in-band generated endpoint pair "
                "existed on the current grids"
            )

        self._set_value(
            proposal,
            updates,
            l1_name,
            candidate_l1,
            row,
            rule,
            margin,
            equation,
        )
        self._set_value(
            proposal,
            updates,
            l2_name,
            candidate_l2,
            row,
            rule,
            margin,
            equation,
        )

    def _refresh_geometry_dependent_plate_minima(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        constraints: pd.DataFrame,
        ledger: MarginLedger,
    ) -> None:
        """Recalculate all direct plate minima after upstream repair stages.

        Hull dimensions, tank-envelope dimensions, and generated member counts
        are repaired before local scantlings.  Those changes alter the Rule
        length, pressure head, and panel spacing used by several plate rules.
        Reusing only the source-design thresholds can therefore create a new
        violation in an otherwise correctly directed one-pass repair.

        This block mirrors only the explicit, monotone plate equations already
        active in ``Structure_Constraints_v1p1``.  It does not approximate
        cargo depth or invent a new load case.  The cargo-pressure row keeps
        its source threshold, which was already handled by ``_direct_minimum``;
        the complete evaluator remains authoritative for the repaired design.
        """

        canonical, _ = self.parameters.canonicalize(
            proposal,
            ensure_bilge_fit=False,
        )

        frame_spacing_mm = float(
            self._section_spacing_mm(canonical, "deck_frame")
        )
        bottom_stringer_spacing_mm = float(
            self._section_spacing_mm(canonical, "bottom_stringer")
        )
        deck_stringer_spacing_mm = float(
            self._section_spacing_mm(canonical, "deck_stringer")
        )
        side_stringer_spacing_mm = float(
            self._section_spacing_mm(canonical, "side_stringer")
        )
        webframe_spacing_mm = float(
            self._section_spacing_mm(canonical, "web_frame")
        )
        bottom_girder_spacing_mm = float(
            self._section_spacing_mm(canonical, "deck_girder")
        )
        vertical_bulkhead_spacing_mm = float(
            self._section_spacing_mm(canonical, "bulkhead_vertical")
        )
        horizontal_bulkhead_spacing_mm = float(
            self._section_spacing_mm(canonical, "bulkhead_horizontal")
        )

        bottom_spacing_mm = max(
            frame_spacing_mm,
            bottom_stringer_spacing_mm,
        )
        deck_spacing_mm = max(
            frame_spacing_mm,
            deck_stringer_spacing_mm,
        )
        side_spacing_mm = max(
            frame_spacing_mm,
            side_stringer_spacing_mm,
        )

        loa_m = self._value(canonical, "LOA")
        depth_m = self._value(canonical, "D")
        draft_m = self._value(canonical, "T")
        abs_rules = ABS_Part3_Constraints(
            {
                "L": loa_m,
                "B": self._value(canonical, "B"),
                "D": depth_m,
                "d": draft_m,
            }
        )

        side_required_mm, _ = (
            abs_rules.calc_Shell_Plating_Constraints(
                frame_spacing_mm,
                side_spacing_mm,
                bottom_spacing_mm,
            )
        )

        lower_hopper_spacing_mm = (
            math.hypot(
                self._value(canonical, "floor web bracket L1"),
                self._value(canonical, "floor web bracket L2"),
            )
            / (self._value(canonical, "num bulk car bottom hopper stiff") + 1.0)
        )

        def wheel_requirement(panel_ratio: float) -> float:
            """Mirror the approved 13.8 t excavator wheel equation."""

            k1 = 8.05
            k2 = 0.20
            c1 = 1.5
            weight_kN = 13_800.0*9.81/1_000.0
            n = min(1.0, 0.85 + 0.15*(panel_ratio - 1.0))
            return abs_rules.round_rule_thickness(
                1.1*k1*k2*n*np.sqrt(c1*weight_kN)
            )

        governing_bulkhead_spacing_mm = max(
            vertical_bulkhead_spacing_mm,
            horizontal_bulkhead_spacing_mm,
        )
        bulkhead_aspect_ratio = (
            governing_bulkhead_spacing_mm
            / min(
                vertical_bulkhead_spacing_mm,
                horizontal_bulkhead_spacing_mm,
            )
        )
        bulkhead_panel_factor = (
            (
                3.075*np.sqrt(bulkhead_aspect_ratio)-2.077
            )/(bulkhead_aspect_ratio+0.272)
            if bulkhead_aspect_ratio <= 2.0
            else 1.0
        )
        bulkhead_pressure_mm = abs_rules.round_rule_thickness(
            governing_bulkhead_spacing_mm
            * bulkhead_panel_factor
            * np.sqrt(depth_m)
            / 290.0
            + 1.5
        )
        bulkhead_spacing_minimum_mm = abs_rules.round_rule_thickness(
            governing_bulkhead_spacing_mm/200.0 + 2.5
        )

        inner_grab_mm = abs_rules.round_rule_thickness(
            0.037*loa_m + 0.009*bottom_spacing_mm + 5.5
        )
        lower_hopper_grab_mm = abs_rules.round_rule_thickness(
            0.037*loa_m + 0.009*lower_hopper_spacing_mm + 5.5
        )
        deck_grab_mm = abs_rules.round_rule_thickness(
            0.037*loa_m + 0.009*deck_spacing_mm + 5.5
        )
        inner_general_mm = abs_rules.round_rule_thickness(
            0.037*loa_m + 0.009*bottom_spacing_mm - 1.5
        )
        deck_pressure_mm = abs_rules.round_rule_thickness(
            deck_spacing_mm*np.sqrt(3.66)/254.0 + 1.5
        )
        inner_wheel_mm = wheel_requirement(
            max(webframe_spacing_mm, bottom_girder_spacing_mm)
            / min(webframe_spacing_mm, bottom_girder_spacing_mm)
        )
        deck_wheel_mm = wheel_requirement(
            max(webframe_spacing_mm, bottom_girder_spacing_mm)
            / min(webframe_spacing_mm, bottom_girder_spacing_mm)
        )

        thresholds_by_name = {
            "Min Inner Side Shell Thickness": side_required_mm,
            "MVR side-shell pressure thickness": (
                abs_rules.calc_side_shell_pressure_thickness(side_spacing_mm)
            ),
            "MVR side-shell amidship end-rule minimum": (
                abs_rules.calc_shell_end_thickness(610.0)
            ),
            "Side-shell end thickness at actual spacing": (
                abs_rules.calc_shell_end_thickness(side_spacing_mm)
            ),
            "MVR bottom-shell pressure thickness": (
                abs_rules.calc_bottom_shell_pressure_thickness(
                    bottom_spacing_mm
                )
            ),
            "Bottom-shell absolute minimum thickness": 8.5,
            "Inner bottom plating - general Rule thickness": inner_general_mm,
            "Min Inner Bottom Grab-Service Thickness": inner_grab_mm,
            "Inner bottom plating - grab loading": inner_grab_mm,
            "Inner bottom plating - excavator wheel loading": inner_wheel_mm,
            "Min Lower Hopper Grab-Service Thickness": lower_hopper_grab_mm,
            "Lower hopper plating - grab loading": lower_hopper_grab_mm,
            # These project continuity constraints depend on plating that may
            # have increased during hull-girder or local plate repair. Refresh
            # them from the final proposal so the hopper thickness preserves
            # the required 25-50 percent passing-side margin.
            "Min Upper Hopper Plate Thickness": self._value(
                proposal,
                "Top Deck Thickness",
            ),
            "Min Lower Hopper Plate Thickness": self._value(
                proposal,
                "Inner Bottom Thickness",
            ),
            "Strength deck plating - unloaded exposed-deck pressure": (
                deck_pressure_mm
            ),
            "Strength deck plating - absolute minimum": 5.0,
            "Strength deck plating - grab loading": deck_grab_mm,
            "Strength deck plating - excavator wheel loading": deck_wheel_mm,
            "Watertight bulkhead plating - pressure equation": (
                bulkhead_pressure_mm
            ),
            "Watertight bulkhead plating - absolute minimum": 6.0,
            "Watertight bulkhead plating - spacing minimum": (
                bulkhead_spacing_minimum_mm
            ),
        }

        for name, required_mm in thresholds_by_name.items():
            matching = constraints.loc[
                constraints["Name"].astype(str).eq(name)
            ]

            for _, source_row in matching.iterrows():
                rule = self.constraints.classify(source_row)

                if rule.method != "direct_minimum" or not rule.parameters:
                    continue

                source_bit = str(source_row.get("Constraint Bit", "")).strip().lower()
                source_failed = source_bit in {"false", "0", "0.0"}
                parameter = rule.parameters[0]
                offered_mm = (
                    self._value(proposal, parameter)
                    * float(rule.offered_scale)
                )

                # A source pass needs repair only when an upstream geometry
                # change has made the recomputed bare Rule threshold fail.
                if not source_failed and offered_mm >= required_mm-1.0e-12:
                    continue

                key = constraint_key(source_row)
                margin = ledger.margin_for(key, rule.method, rule.family)
                target_parameter = (
                    required_mm*(1.0 + margin)
                    / float(rule.offered_scale)
                )

                if parameter in self.parameters.integer_names:
                    target_parameter = math.ceil(target_parameter-1.0e-12)

                refreshed_row = source_row.copy()
                refreshed_row["Threshold Value"] = required_mm
                refreshed_row["Constraint Value"] = offered_mm
                refreshed_row["Constraint Eval"] = required_mm-offered_mm
                self._set_larger(
                    proposal,
                    updates,
                    parameter,
                    target_parameter,
                    refreshed_row,
                    rule,
                    margin,
                    (
                        "final-geometry governing plate minimum: p_new = "
                        "recomputed_threshold*(1+margin)/offered_scale"
                    ),
                )

    def _hatch_area(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        row: pd.Series,
        rule: RepairRule,
        margin: float,
    ) -> None:
        threshold = _numeric(row, "Threshold Value")
        offered = _numeric(row, "Constraint Value")

        if threshold is None or offered is None or offered <= 0.0:
            return

        scale = math.sqrt(threshold*(1.0 + margin)/offered)

        for parameter in ("Hatch Opening L", "Hatch Opening W"):
            current = self._pass_start_value(proposal, updates, parameter)
            self._set_larger(
                proposal,
                updates,
                parameter,
                current*scale,
                row,
                rule,
                margin,
                "L_new,W_new = old*sqrt(target_area/offered_area)",
            )

    def _engine_room_length(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        row: pd.Series,
        rule: RepairRule,
        margin: float,
    ) -> None:
        required = _numeric(row, "Threshold Value")

        if required is None:
            return

        loa = float(proposal[self.parameters.index["LOA"]])
        candidate = required*(1.0 + margin)/loa
        self._set_larger(
            proposal,
            updates,
            "frac_engineRoom_L",
            candidate,
            row,
            rule,
            margin,
            "engine-room fraction = required dimensional length*(1+margin)/LOA",
        )

    def _section_modulus_groups(
        self,
        failed: pd.DataFrame,
        ledger: MarginLedger,
    ) -> dict[str, list[tuple[pd.Series, RepairRule, float]]]:
        groups: dict[str, list[tuple[pd.Series, RepairRule, float]]] = {}

        for _, row in failed.iterrows():
            rule = self.constraints.classify(row)

            if rule.method != "section_modulus" or rule.family not in self.SECTION_PRIMARY:
                continue

            key = constraint_key(row)
            margin = ledger.margin_for(key, rule.method, rule.family)
            groups.setdefault(rule.family, []).append((row, rule, margin))

        return groups

    def _value(self, vector: np.ndarray, name: str) -> float:
        """Read one named coordinate while keeping equations legible."""

        return float(vector[self.parameters.index[name]])

    def _section_spacing_mm(
        self,
        vector: np.ndarray,
        family: str,
    ) -> float | None:
        """Reproduce uniform spacings that CadQuery generates from controls.

        ``StructureCalcs`` now measures the structural-elements table directly
        and remains the evaluation authority.  This repair-side equation is
        only a directed inverse for proposing the next member-count/section
        change.  It mirrors the generator's half-breadth and boundary-interval
        layout so the proposal is close to the generated result; CadQuery
        snapping and the complete evaluator still decide whether it passes.
        """

        length = self._value(vector, "L_3h")
        overhang = self._value(vector, "l_overhang")
        webframe_count = self._value(vector, "web frames per hold")
        frame_count = self._value(vector, "num intermediate trans frames")
        beam = self._value(vector, "B")
        depth = self._value(vector, "D")
        web_depth_m = self._value(vector, "web frame h")/1_000.0
        bottom_girders = self._value(vector, "num bottom girders")
        bottom_stiffeners = self._value(
            vector, "num intermediate bottom stiffeners"
        )

        webframe_spacing_m = (
            length*(1.0 - 2.0*overhang)
            / (3.0*(webframe_count + 1.0))
        )
        frame_spacing_m = webframe_spacing_m/(frame_count + 1.0)
        bottom_girder_spacing_m = (
            0.5*beam - web_depth_m
        )/max(bottom_girders - 1.0, 1.0)
        bottom_stringer_spacing_m = bottom_girder_spacing_m/(
            bottom_stiffeners + 1.0
        )

        if family == "bottom_stringer":
            return 1_000.0*bottom_stringer_spacing_m

        if family in {"bottom_frame", "deck_frame", "deck_beam", "side_frame"}:
            return 1_000.0*frame_spacing_m

        if family == "web_frame":
            return 1_000.0*webframe_spacing_m

        if family == "deck_girder":
            # Deck and bottom girders share the same generated half-breadth
            # stations.  The former full-breadth reconstruction was exactly
            # twice the actual spacing.
            return 1_000.0*bottom_girder_spacing_m

        if family == "deck_stringer":
            return 1_000.0*bottom_girder_spacing_m/(
                bottom_stiffeners + 1.0
            )

        if family == "side_stringer":
            count = self._value(vector, "num side shell stiffeners")
            return 1_000.0*depth/(count + 1.0)

        if family == "bulkhead_horizontal":
            count = self._value(
                vector, "num transverse bulkhead stiffeners"
            )
            return 1_000.0*depth/(count + 1.0)

        if family == "bulkhead_vertical":
            count = self._value(vector, "num vertical bulkhead stiffeners")
            clear_half_breadth_m = max(0.0, 0.5*beam - web_depth_m)
            return 1_000.0*clear_half_breadth_m/max(count - 1.0, 1.0)

        # Hopper spacing is presently assembled from sloped panel geometry in
        # StructureCalcs.  Its repair remains source-anchored and is reconciled
        # by exact reevaluation rather than duplicating that geometry here.
        return None

    @staticmethod
    def _stiffener_section_modulus(
        plate_t: float,
        spacing: float,
        height: float,
        thickness: float,
        width: float,
    ) -> float:
        """Reproduce ``StructureCalcs.calc_stiffener_SM`` in mm and cm^3."""

        area = (
            thickness*width
            + height*thickness
            + plate_t*spacing
        )

        if area <= 0.0:
            return 0.0

        centroid = (
            thickness*width*height
            + height*thickness*(height/2.0)
        )/area
        inertia = (
            spacing*plate_t**3.0/12.0
            + spacing*plate_t*centroid**2.0
            + width*thickness**3.0/12.0
            + width*thickness*(centroid - height)**2.0
            + thickness*height**3.0/12.0
            + height*thickness*(height/2.0 - centroid)**2.0
        )
        extreme = max(centroid, height - centroid)

        if extreme <= 0.0:
            return 0.0

        return float(inertia/extreme/1_000.0)

    def _solve_source_anchored_section(
        self,
        source: np.ndarray,
        proposal: np.ndarray,
        family: str,
        offered: float,
        target: float,
    ) -> tuple[float, float, float] | None:
        """Solve h/w with the maintained 61-point and bisection pattern.

        The supplied exact source value calibrates the local analytical
        section model.  This is the important Ship_Structures behavior: the
        repair equation guides a candidate, while the real evaluated value is
        never replaced by the analytical model.
        """

        height_name, thickness_name, width_name = self.SECTION_PRIMARY[family]

        if width_name is None or family not in self.SECTION_PLATE:
            return None

        source_spacing = self._section_spacing_mm(source, family)
        current_spacing = self._section_spacing_mm(proposal, family)

        if (
            source_spacing is None
            or current_spacing is None
            or source_spacing <= 0.0
            or current_spacing <= 0.0
        ):
            return None

        plate_name = self.SECTION_PLATE[family]
        source_h = self._value(source, height_name)
        source_t = self._value(source, thickness_name)
        source_w = min(
            self._value(source, width_name),
            source_h,
            source_spacing,
        )
        source_model = self._stiffener_section_modulus(
            self._value(source, plate_name),
            source_spacing,
            source_h,
            source_t,
            source_w,
        )

        if source_model <= 0.0 or offered <= 0.0 or target <= 0.0:
            return None

        calibration = offered/source_model
        base_h = self._value(proposal, height_name)
        base_t = self._value(proposal, thickness_name)
        base_w = min(
            self._value(proposal, width_name),
            base_h,
            current_spacing,
        )
        plate_t = self._value(proposal, plate_name)

        def utilization(h_scale: float, w_scale: float, t_value: float) -> float:
            height = min(
                base_h*h_scale,
                self.parameters.repair_bounds(height_name)[1],
            )
            width = min(
                base_w*w_scale,
                height,
                current_spacing,
                self.parameters.repair_bounds(width_name)[1],
            )
            value = calibration*self._stiffener_section_modulus(
                plate_t,
                current_spacing,
                height,
                t_value,
                width,
            )
            return value/target

        best: tuple[float, float, float, float] | None = None

        # Match Ship_Structures: try the current integer web thickness and a
        # sparse series of small increments before expanding the h/w domain.
        thickness_upper = self.parameters.repair_bounds(thickness_name)[1]

        for thickness_increment in range(13):
            trial_t = min(
                thickness_upper,
                math.ceil(base_t) + thickness_increment,
            )

            for height_scale in np.linspace(1.0, 4.0, 61):
                if utilization(height_scale, 4.0, trial_t) < 1.0:
                    continue

                if utilization(height_scale, 1.0, trial_t) >= 1.0:
                    width_scale = 1.0
                else:
                    lower_width = 1.0
                    upper_width = 4.0

                    for _ in range(50):
                        middle = 0.5*(lower_width + upper_width)

                        if utilization(height_scale, middle, trial_t) >= 1.0:
                            upper_width = middle
                        else:
                            lower_width = middle

                    width_scale = upper_width

                movement = math.hypot(
                    height_scale - 1.0,
                    width_scale - 1.0,
                ) + 0.05*thickness_increment

                if best is None or movement < best[0]:
                    best = (movement, height_scale, width_scale, trial_t)

            if best is not None:
                break

        # Retain the maintained extended h/w fallback for unusually large
        # deficits.  It is deliberately sparse because every result still
        # receives a complete exact GenISys evaluation.
        if best is None:
            trial_t = base_t

            for height_scale in np.geomspace(4.0, 64.0, 61):
                if utilization(height_scale, 64.0, trial_t) < 1.0:
                    continue

                lower_width = 1.0
                upper_width = 64.0

                for _ in range(50):
                    middle = 0.5*(lower_width + upper_width)

                    if utilization(height_scale, middle, trial_t) >= 1.0:
                        upper_width = middle
                    else:
                        lower_width = middle

                movement = math.hypot(
                    height_scale - 1.0,
                    upper_width - 1.0,
                )
                best = (movement, height_scale, upper_width, trial_t)
                break

        if best is None:
            return None

        _, height_scale, width_scale, trial_t = best
        return (
            min(
                base_h*height_scale,
                self.parameters.repair_bounds(height_name)[1],
            ),
            min(
                base_w*width_scale,
                self.parameters.repair_bounds(width_name)[1],
            ),
            trial_t,
        )

    def _repair_section_family(
        self,
        source: np.ndarray,
        proposal: np.ndarray,
        updates: dict[str, dict],
        family: str,
        items: list[tuple[pd.Series, RepairRule, float]],
    ) -> None:
        ratios = []

        for row, _, margin in items:
            threshold = _numeric(row, "Threshold Value")
            offered = _numeric(row, "Constraint Value")

            if threshold is None or offered is None or threshold <= 0.0:
                continue

            ratios.append(threshold*(1.0 + margin)/max(offered, 1.0e-9))

        if not ratios:
            return

        utilization_ratio = max(1.0, max(ratios))
        height_name, thickness_name, width_name = self.SECTION_PRIMARY[family]
        representative_row, representative_rule, representative_margin = max(
            items,
            key=lambda item: (
                (_numeric(item[0], "Threshold Value") or 0.0)
                * (1.0 + item[2])
                / max((_numeric(item[0], "Constraint Value") or 0.0), 1.0e-9)
            ),
        )

        # Apply the count/spacing movement before the local h/w calculation,
        # following the maintained Ship_Structures order.
        count_name = self.SECTION_COUNT.get(family)

        if count_name is not None:
            current_count = self._pass_start_value(
                proposal,
                updates,
                count_name,
            )

            if family == "bulkhead_vertical":
                candidate_count = max(current_count, 36.0)
            elif family == "bulkhead_horizontal":
                candidate_count = max(current_count, 30.0)
            elif family in {"hopper_lower", "hopper_upper"}:
                candidate_count = max(current_count, 6.0)
            else:
                candidate_count = math.ceil(
                    current_count*min(2.0, utilization_ratio**0.20)
                )

            self._set_larger(
                proposal,
                updates,
                count_name,
                candidate_count,
                representative_row,
                representative_rule,
                representative_margin,
                "ordered count increase precedes local section search",
            )

        target = (
            (_numeric(representative_row, "Threshold Value") or 0.0)
            * (1.0 + representative_margin)
        )
        offered = _numeric(representative_row, "Constraint Value") or 0.0

        # Keep already proposed independent controls such as spacing, but
        # reset this family's section coordinates to their pass-start values.
        # The section rule then competes with other rules through the
        # governing max in ``_set_larger`` instead of multiplying a previous
        # rule's already margined candidate.
        section_proposal = proposal.copy()

        for parameter in (height_name, thickness_name, width_name):
            if parameter is None:
                continue

            section_proposal[self.parameters.index[parameter]] = (
                self._pass_start_value(proposal, updates, parameter)
            )

        solved = self._solve_source_anchored_section(
            source,
            section_proposal,
            family,
            offered,
            target,
        )

        if solved is not None:
            solved_height, solved_width, solved_thickness = solved
            height_name, thickness_name, width_name = self.SECTION_PRIMARY[family]
            self._set_larger(
                proposal,
                updates,
                height_name,
                solved_height,
                representative_row,
                representative_rule,
                representative_margin,
                "61-point h search plus bounded width bisection on source-anchored exact SM equation",
            )
            self._set_larger(
                proposal,
                updates,
                thickness_name,
                solved_thickness,
                representative_row,
                representative_rule,
                representative_margin,
                "sparse integer web-thickness search on source-anchored exact SM equation",
            )
            self._set_larger(
                proposal,
                updates,
                width_name,
                solved_width,
                representative_row,
                representative_rule,
                representative_margin,
                "bounded flange-width bisection on source-anchored exact SM equation",
            )

        # Closed double-bottom sections and sloped hopper panels retain a
        # source-anchored response scale because their existing evaluator uses
        # a second face plate or panel geometry not represented by the generic
        # attached-stiffener equation above.  Exact residual passes reconcile
        # these coupled families.
        if family in {"bottom_girder", "bottom_floor"}:
            dimension_scale = utilization_ratio**(1.0/3.0)
        else:
            dimension_scale = math.sqrt(utilization_ratio)

        old_height = self._pass_start_value(
            proposal,
            updates,
            height_name,
        )
        new_height = (
            old_height
            if solved is not None
            else old_height*dimension_scale
        )
        self._set_larger(
            proposal,
            updates,
            height_name,
            new_height,
            representative_row,
            representative_rule,
            representative_margin,
            (
                "source-anchored section response: h_new = h_old*"
                f"({utilization_ratio:.9g})^"
                + ("(1/3)" if family in {"bottom_girder", "bottom_floor"} else "0.5")
            ),
        )

        # The former web-depth/thickness aspect-ratio feasibility rules were
        # removed from GenISys by project decision.  Do not silently recreate
        # them inside the repair method.  Web thickness is changed only by an
        # active minimum-thickness row or when the section-modulus solver
        # itself needs additional material.
        #
        # old behavior: t_new >= repaired h/45

        if width_name is not None and solved is None:
            current_w = self._pass_start_value(
                proposal,
                updates,
                width_name,
            )
            new_w = max(
                current_w*dimension_scale,
                2.2*float(proposal[self.parameters.index[thickness_name]]),
            )
            self._set_larger(
                proposal,
                updates,
                width_name,
                new_w,
                representative_row,
                representative_rule,
                representative_margin,
                "flange width scaled with web depth and kept greater than 2t",
            )

    def _uniform_bulkhead_loading_model(
        self,
        vector: np.ndarray,
        *,
        count: int | None = None,
        height_mm: float | None = None,
        thickness_mm: float | None = None,
        width_mm: float | None = None,
    ) -> tuple[float, float]:
        """Reproduce the active ABS 3-2-8/3.3 capacity and demand pair.

        Despite their historical deck/bottom labels, both runtime rows use
        this same transverse-bulkhead composite.  ``StructureCalcs`` treats
        the bulkhead plating and every vertical stiffener as one column-like
        section, calculates its radius of gyration, and compares the resulting
        permissible vertical load ``W_a`` with ``W_calc``.  Keeping this
        equation here makes the repair source anchored without changing the
        authoritative evaluator.
        """

        beam_m = self._value(vector, "B")
        bulkhead_plate_m = (
            self._value(vector, "Trans Bulkhead Thickness")/1_000.0
        )
        stiffener_count = (
            int(round(self._value(
                vector, "num vertical bulkhead stiffeners"
            )))
            if count is None
            else int(count)
        )
        number_on_full_bulkhead = 2*stiffener_count - 1
        stiffener_height_m = (
            self._value(vector, "blkhd vert stiff h")
            if height_mm is None
            else float(height_mm)
        )/1_000.0
        stiffener_thickness_m = (
            self._value(vector, "blkhd vert stiff t")
            if thickness_mm is None
            else float(thickness_mm)
        )/1_000.0
        stiffener_width_m = (
            self._value(vector, "blkhd vert stiff w")
            if width_mm is None
            else float(width_mm)
        )/1_000.0

        area_m2 = (
            beam_m*bulkhead_plate_m
            + number_on_full_bulkhead
            * stiffener_thickness_m
            * (stiffener_height_m + stiffener_width_m)
        )
        centroid_m = (
            number_on_full_bulkhead
            * (
                stiffener_height_m*stiffener_thickness_m
                * (stiffener_height_m/2.0)
                + stiffener_width_m*stiffener_thickness_m
                * stiffener_height_m
            )
            / area_m2
        )
        inertia_m4 = (
            beam_m*bulkhead_plate_m**3.0/12.0
            + beam_m*bulkhead_plate_m*centroid_m**2.0
            + number_on_full_bulkhead
            * (
                stiffener_width_m*stiffener_thickness_m**3.0/12.0
                + stiffener_width_m*stiffener_thickness_m
                * (centroid_m-stiffener_height_m)**2.0
                + stiffener_thickness_m*stiffener_height_m**3.0/12.0
                + stiffener_height_m*stiffener_thickness_m
                * (centroid_m-stiffener_height_m/2.0)**2.0
            )
        )
        radius_of_gyration_cm = math.sqrt(inertia_m4/area_m2)*100.0
        cargo_height_m = (
            self._value(vector, "D") - self._value(vector, "Db")/1_000.0
        )
        area_cm2 = area_m2*100.0**2.0
        allowable_kn = (
            12.09 - 4.44*cargo_height_m/radius_of_gyration_cm
        )*area_cm2

        # Match StructureCalcs' rounded four-bulkhead three-hold grid rather
        # than replacing it with an algebraically simplified spacing.
        overhang = self._value(vector, "l_overhang")
        locations = (
            np.linspace(overhang, 1.0-overhang, 4).round(10)
            * self._value(vector, "L_3h")
        )
        bulkhead_spacing_m = float(locations[1]-locations[0])
        calculated_kn = 7.01*beam_m*bulkhead_spacing_m*2.9

        return float(allowable_kn), float(calculated_kn)

    def _bulkhead_vertical_width_limit_mm(
        self,
        vector: np.ndarray,
        height_mm: float,
        *,
        count: int | None = None,
    ) -> float:
        """Return the flange-width limit applied by CadQuery generation."""

        beam_m = self._value(vector, "B")
        web_frame_depth_m = self._value(vector, "web frame h")/1_000.0
        stiffener_count = (
            int(round(self._value(
                vector, "num vertical bulkhead stiffeners"
            )))
            if count is None
            else int(count)
        )
        clear_half_breadth_m = max(0.0, 0.5*beam_m-web_frame_depth_m)
        cad_spacing_mm = (
            clear_half_breadth_m*1_000.0/max(stiffener_count-1, 1)
        )
        repair_upper = self.parameters.repair_bounds(
            "blkhd vert stiff w"
        )[1]

        return min(float(height_mm), cad_spacing_mm, repair_upper)

    def _valid_bulkhead_vertical_counts(
        self,
        vector: np.ndarray,
    ) -> list[int]:
        """Return count values which CadQuery will preserve unchanged.

        CadQuery aligns vertical bulkhead stiffeners with the derived deck
        longitudinal grid.  It snaps a requested count to ``divisor + 1``,
        where ``divisor`` divides the number of deck-grid intervals.  Searching
        this exact finite set prevents a proposal from relying on a count that
        geometry generation will immediately replace.
        """

        girder_count = int(round(
            self._value(vector, "num bottom girders")
        ))
        intermediate_count = int(round(
            self._value(vector, "num intermediate bottom stiffeners")
        ))
        grid_intervals = max(
            1,
            (girder_count-1)*(intermediate_count+1),
        )
        lower, upper = self.parameters.repair_bounds(
            "num vertical bulkhead stiffeners"
        )
        options = {
            divisor+1
            for divisor in range(2, grid_intervals+1)
            if grid_intervals % divisor == 0
        }

        return sorted(
            count
            for count in options
            if int(math.ceil(lower)) <= count <= int(math.floor(upper))
        )

    def _solve_uniform_bulkhead_section(
        self,
        source: np.ndarray,
        proposal: np.ndarray,
        offered_kn: float,
        threshold_kn: float,
        margin: float,
    ) -> tuple[int, float, float, float] | None:
        """Find a passing vertical-stiffener section with the maintained grid.

        The exact source output supplies additive capacity/demand anchors.
        Candidate depth uses the maintained 61-point search, flange breadth
        uses a sampled bracket followed by 50 bisection steps, and integer web
        thickness uses the maintained sparse 0--12 mm increments.  A complete
        GenISys evaluation remains authoritative after this proposal.
        """

        source_allowable, source_calculated = (
            self._uniform_bulkhead_loading_model(source)
        )
        proposal_allowable, proposal_calculated = (
            self._uniform_bulkhead_loading_model(proposal)
        )
        allowable_anchor = offered_kn-source_allowable
        calculated_anchor = threshold_kn-source_calculated
        target_kn = (
            proposal_calculated+calculated_anchor
        )*(1.0+margin)
        base_height = self._value(proposal, "blkhd vert stiff h")
        base_thickness = self._value(proposal, "blkhd vert stiff t")
        base_width = min(
            self._value(proposal, "blkhd vert stiff w"),
            self._bulkhead_vertical_width_limit_mm(proposal, base_height),
        )
        base_count = int(round(
            self._value(proposal, "num vertical bulkhead stiffeners")
        ))

        if proposal_allowable+allowable_anchor >= target_kn:
            return base_count, base_height, base_width, base_thickness

        # Match the maintained Ship_Structures ordering by trying a valid
        # member-count increase before enlarging individual sections.  Search
        # only exact CadQuery divisor-grid options and take the first passing
        # count.  If count alone cannot pass, retain the largest valid count as
        # the base for the subsequent h/t/w search.
        valid_counts = [
            count
            for count in self._valid_bulkhead_vertical_counts(proposal)
            if count >= base_count
        ]

        if not valid_counts:
            valid_counts = [base_count]

        working_count = valid_counts[-1]

        for candidate_count in valid_counts:
            candidate_width = min(
                base_width,
                self._bulkhead_vertical_width_limit_mm(
                    proposal,
                    base_height,
                    count=candidate_count,
                ),
            )
            allowable, _ = self._uniform_bulkhead_loading_model(
                proposal,
                count=candidate_count,
                height_mm=base_height,
                thickness_mm=base_thickness,
                width_mm=candidate_width,
            )

            if allowable+allowable_anchor >= target_kn:
                return (
                    candidate_count,
                    base_height,
                    candidate_width,
                    base_thickness,
                )

        height_upper = self.parameters.repair_bounds(
            "blkhd vert stiff h"
        )[1]
        thickness_upper = self.parameters.repair_bounds(
            "blkhd vert stiff t"
        )[1]

        def anchored_capacity(
            height_value: float,
            thickness_value: float,
            width_value: float,
        ) -> float:
            allowable, _ = self._uniform_bulkhead_loading_model(
                proposal,
                count=working_count,
                height_mm=height_value,
                thickness_mm=thickness_value,
                width_mm=width_value,
            )
            return allowable+allowable_anchor

        best: tuple[float, float, float, float] | None = None

        # Search the ordinary 1--4x section neighborhood first.  At each
        # height, sample width brackets before bisection so no global
        # monotonic-width assumption is needed.
        for thickness_increment in range(13):
            nominal_thickness = min(
                thickness_upper,
                math.ceil(base_thickness)+thickness_increment,
            )

            for height_scale in np.linspace(1.0, 4.0, 61):
                height_value = min(base_height*height_scale, height_upper)
                thickness_value = min(
                    thickness_upper,
                    max(
                        nominal_thickness,
                        math.ceil(height_value/45.0-1.0e-12),
                    ),
                )
                width_limit = self._bulkhead_vertical_width_limit_mm(
                    proposal,
                    height_value,
                    count=working_count,
                )
                width_upper = min(base_width*4.0, width_limit)

                if width_upper < base_width-1.0e-12:
                    continue

                width_scales = np.linspace(
                    1.0,
                    width_upper/max(base_width, 1.0e-9),
                    61,
                )
                previous_scale = float(width_scales[0])
                passing_scale = None

                for width_scale in width_scales:
                    width_value = base_width*float(width_scale)

                    if anchored_capacity(
                        height_value,
                        thickness_value,
                        width_value,
                    ) >= target_kn:
                        passing_scale = float(width_scale)
                        break

                    previous_scale = float(width_scale)

                if passing_scale is None:
                    continue

                lower_width_scale = previous_scale
                upper_width_scale = passing_scale

                for _ in range(50):
                    middle = 0.5*(
                        lower_width_scale+upper_width_scale
                    )
                    width_value = base_width*middle

                    if anchored_capacity(
                        height_value,
                        thickness_value,
                        width_value,
                    ) >= target_kn:
                        upper_width_scale = middle
                    else:
                        lower_width_scale = middle

                movement = (
                    math.hypot(height_scale-1.0, upper_width_scale-1.0)
                    + 0.05*thickness_increment
                )

                if best is None or movement < best[0]:
                    best = (
                        movement,
                        height_value,
                        base_width*upper_width_scale,
                        thickness_value,
                    )

            if best is not None:
                break

        # Retain the same extended geometric fallback used by the other
        # source-anchored section families for unusually large deficits.
        if best is None:
            thickness_value = min(
                thickness_upper,
                max(
                    base_thickness,
                    math.ceil(base_height/45.0-1.0e-12),
                ),
            )

            for height_scale in np.geomspace(4.0, 64.0, 61):
                height_value = min(base_height*height_scale, height_upper)
                thickness_value = min(
                    thickness_upper,
                    max(
                        thickness_value,
                        math.ceil(height_value/45.0-1.0e-12),
                    ),
                )
                width_limit = self._bulkhead_vertical_width_limit_mm(
                    proposal,
                    height_value,
                    count=working_count,
                )
                width_upper = min(base_width*64.0, width_limit)

                if anchored_capacity(
                    height_value,
                    thickness_value,
                    width_upper,
                ) < target_kn:
                    continue

                lower_width = base_width
                upper_width = width_upper

                for _ in range(50):
                    middle = 0.5*(lower_width+upper_width)

                    if anchored_capacity(
                        height_value,
                        thickness_value,
                        middle,
                    ) >= target_kn:
                        upper_width = middle
                    else:
                        lower_width = middle

                best = (
                    height_scale-1.0,
                    height_value,
                    upper_width,
                    thickness_value,
                )
                break

        if best is None:
            return None

        _, height_value, width_value, thickness_value = best
        return working_count, height_value, width_value, thickness_value

    def _repair_uniform_bulkhead_loading(
        self,
        source: np.ndarray,
        proposal: np.ndarray,
        updates: dict[str, dict],
        items: list[tuple[pd.Series, RepairRule, float]],
    ) -> None:
        """Repair the duplicated deck/bottom rows as one causal block."""

        valid_items = []

        for row, rule, margin in items:
            threshold = _numeric(row, "Threshold Value")
            offered = _numeric(row, "Constraint Value")

            if threshold is None or offered is None or threshold <= 0.0:
                continue

            valid_items.append((row, rule, margin, threshold, offered))

        if not valid_items:
            return

        representative = max(
            valid_items,
            key=lambda item: item[3]*(1.0+item[2])-item[4],
        )
        row, rule, margin, threshold, offered = representative

        # Other structural rules can own these same four coordinates.  Solve
        # this loading equation from the common pass-start section, then let
        # the setters retain the governing coordinate-wise increase.
        uniform_proposal = proposal.copy()

        for parameter in (
            "num vertical bulkhead stiffeners",
            "blkhd vert stiff h",
            "blkhd vert stiff t",
            "blkhd vert stiff w",
        ):
            uniform_proposal[self.parameters.index[parameter]] = (
                self._pass_start_value(proposal, updates, parameter)
            )

        solved = self._solve_uniform_bulkhead_section(
            source,
            uniform_proposal,
            offered,
            threshold,
            margin,
        )

        if solved is None:
            return

        count_value, height_value, width_value, thickness_value = solved
        values = {
            "num vertical bulkhead stiffeners": count_value,
            "blkhd vert stiff h": height_value,
            "blkhd vert stiff t": thickness_value,
            "blkhd vert stiff w": width_value,
        }
        equations = {
            "num vertical bulkhead stiffeners": (
                "ordered search over exact CadQuery divisor-grid bulkhead "
                "stiffener counts before local section enlargement"
            ),
            "blkhd vert stiff h": (
                "61-point vertical-stiffener depth search on the "
                "source-anchored ABS 3-2-8/3.3 W_a equation"
            ),
            "blkhd vert stiff t": (
                "sparse integer web-thickness search with h/t <= 45 on "
                "the source-anchored ABS 3-2-8/3.3 W_a equation"
            ),
            "blkhd vert stiff w": (
                "sampled width bracket plus 50-step bisection within the "
                "CadQuery flange-width limit"
            ),
        }

        for parameter, value in values.items():
            self._set_larger(
                proposal,
                updates,
                parameter,
                value,
                row,
                rule,
                margin,
                equations[parameter],
            )

        # Both labels invoke the identical active capacity/demand equation.
        # Preserve both identities on every actual parameter movement so the
        # provenance states that one coordinated section repair served both.
        for parameter in values:
            if parameter not in updates:
                continue

            for causal_row, _, causal_margin, _, _ in valid_items:
                key = constraint_key(causal_row)

                self._append_cause(
                    updates[parameter],
                    key,
                    causal_margin,
                )

    def _global_hull_girder(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        rows: list[tuple[pd.Series, RepairRule, float]],
    ) -> None:
        ratios = []

        for row, _, margin in rows:
            threshold = _numeric(row, "Threshold Value")
            offered = _numeric(row, "Constraint Value")

            if threshold is None or offered is None or threshold <= 0.0:
                continue

            if "shear-stress limit" in str(row["Name"]):
                ratios.append(offered*(1.0 + margin)/threshold)
            else:
                ratios.append(threshold*(1.0 + margin)/max(offered, 1.0e-9))

        if not ratios:
            return

        ratio = max(1.0, max(ratios))
        representative_row, representative_rule, representative_margin = rows[0]

        # The effective section includes continuous deck, bottom, side, inner
        # side, inner bottom, shear strake, and their longitudinal members.
        # Plate-area contribution is approximately linear in thickness, so a
        # one-pass repair must use at least the governing capacity ratio.  The
        # earlier 0.75 exponent deliberately under-corrected the failure and
        # depended on later repair iterations, which are not part of the
        # approved GenISys method.
        zone_scale = min(4.0, ratio)

        for parameter in representative_rule.parameters:
            current = self._pass_start_value(proposal, updates, parameter)
            candidate = current*zone_scale

            if parameter in self.parameters.integer_names:
                candidate = math.ceil(candidate - 1.0e-12)

            self._set_larger(
                proposal,
                updates,
                parameter,
                candidate,
                representative_row,
                representative_rule,
                representative_margin,
                "coordinated effective-section zone scale from governing utilization",
            )

    def _tank_or_system_repair(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        row: pd.Series,
        rule: RepairRule,
        margin: float,
        failure_diagnostics: dict | None = None,
    ) -> None:
        required = _numeric(row, "Threshold Value")
        offered = _numeric(row, "Constraint Value")

        if (
            rule.family == "fuel_tank_capacity"
            and required is not None
            and offered is not None
            and offered > required
        ):
            # Tank rows use ``Threshold Value`` for generated capacity and
            # ``Constraint Value`` for demanded fill volume.  The target fuel
            # demand is therefore capacity/(1 + margin), which puts the fixed
            # tank 25-50 percent above the repaired voyage requirement.
            fuel_scale = required/max(
                offered*(1.0 + margin),
                1.0e-9,
            )

            current_fn = self._pass_start_value(
                proposal,
                updates,
                "Fn_ship",
            )
            current_range = self._pass_start_value(
                proposal,
                updates,
                "Voyage_Range",
            )
            minimum_fn = self.parameters.repair_bounds("Fn_ship")[0]

            # For a fixed hull, resistance is approximately proportional to
            # V^2, installed propulsion power to V^3, and voyage duration to
            # Range/V.  Voyage propulsion fuel therefore scales as
            # Range*V^2, and V scales directly with Froude number.  Reduce
            # speed first because that also reduces the required engine size.
            unconstrained_fn = current_fn*math.sqrt(
                max(fuel_scale, 0.0)
            )
            repaired_fn = max(minimum_fn, unconstrained_fn)

            self._set_smaller(
                proposal,
                updates,
                "Fn_ship",
                repaired_fn,
                row,
                rule,
                margin,
                (
                    "fuel capacity repair: hold tank geometry fixed and "
                    "reduce Fn_ship using Fuel ~ Voyage_Range*Fn_ship^2"
                ),
            )

            # If the required Froude number lies below the declared repair
            # domain, use the residual fuel ratio to shorten the mission range.
            # No fuel-tank or lazarette dimension is changed by this repair.
            if unconstrained_fn < minimum_fn:
                speed_scale_squared = (
                    repaired_fn/max(current_fn, 1.0e-9)
                )**2
                repaired_range = (
                    current_range
                    * fuel_scale
                    / max(speed_scale_squared, 1.0e-12)
                )

                self._set_smaller(
                    proposal,
                    updates,
                    "Voyage_Range",
                    repaired_range,
                    row,
                    rule,
                    margin,
                    (
                        "fuel capacity repair: after Fn_ship reaches its "
                        "repair floor, reduce Voyage_Range using Fuel ~ "
                        "Voyage_Range*Fn_ship^2"
                    ),
                )

        elif rule.family == "ballast_tank_capacity":
            depth_m = (
                float(proposal[self.parameters.index["LOA"]])
                * float(proposal[self.parameters.index["Dd"]])
            )
            beam_m = (
                float(proposal[self.parameters.index["LOA"]])
                * float(proposal[self.parameters.index["Bd"]])
            )
            ratio = (
                offered/max(required, 1.0e-9)
                if required is not None and offered is not None
                else 1.5
            )
            scale = min(2.5, math.sqrt(max(1.0, ratio))*(1.0 + margin))
            current_db = self._pass_start_value(
                proposal,
                updates,
                "Db",
            )
            current_web = self._pass_start_value(
                proposal,
                updates,
                "web frame h",
            )
            self._set_larger(
                proposal,
                updates,
                "Db",
                max(current_db*scale, 0.18*depth_m*1_000.0),
                row,
                rule,
                margin,
                "increase double-bottom ballast envelope from volume deficit",
            )
            self._set_larger(
                proposal,
                updates,
                "web frame h",
                max(current_web*scale, 0.12*beam_m*1_000.0),
                row,
                rule,
                margin,
                "increase side/hopper ballast envelope from volume deficit",
            )

        elif (
            rule.family == "cargo_hold_capacity"
            and rule.method == "tank_capacity"
        ):
            ratio = (
                offered/max(required, 1.0e-9)
                if required is not None and offered is not None
                else 1.25
            )
            dimension_scale = max(1.0, ratio*(1.0 + margin))**0.5

            for parameter in ("Bd", "Dd"):
                current = self._pass_start_value(
                    proposal,
                    updates,
                    parameter,
                )
                self._set_larger(
                    proposal,
                    updates,
                    parameter,
                    current*dimension_scale,
                    row,
                    rule,
                    margin,
                    "increase cargo-envelope beam/depth ratios from volume deficit",
                )

        elif rule.method in {"system_displacement", "system_capacity"}:
            # A structured lightship rejection supplies both sides of the
            # exact mass balance: ``Constraint Value`` is modeled lightship
            # and ``Threshold Value`` is displacement at the selected draft.
            # Use that measured ratio instead of taking a blind fixed step.
            # Draft is consumed first, then beam/depth ratios share the
            # remaining approximate volume factor.  LOA is the last resort
            # because changing it also scales nearly every structural rule.
            if (
                row.get("Name")
                == "Modeled lightship fits selected displacement"
                and required is not None
                and offered is not None
                and required > 0.0
                and offered >= required
            ):
                remaining_volume_ratio = (
                    offered/required
                )*(1.0 + margin)

                for parameter in ("Draft_frac",):
                    current = self._pass_start_value(
                        proposal,
                        updates,
                        parameter,
                    )
                    upper = self.parameters.repair_bounds(parameter)[1]
                    factor = min(
                        remaining_volume_ratio,
                        upper/max(current, 1.0e-12),
                    )
                    self._set_larger(
                        proposal,
                        updates,
                        parameter,
                        current*factor,
                        row,
                        rule,
                        margin,
                        (
                            "consume exact lightship/displacement deficit "
                            "through authoritative draft fraction"
                        ),
                    )
                    remaining_volume_ratio /= max(factor, 1.0)

                if remaining_volume_ratio > 1.0+1.0e-12:
                    # B and D enter the early-design displacement estimate
                    # approximately linearly.  Equal logarithmic movement
                    # avoids forcing the entire residual into either an
                    # unusually broad or unusually deep hull.
                    shared_factor = math.sqrt(remaining_volume_ratio)
                    achieved_factor = 1.0

                    for parameter in ("Bd", "Dd"):
                        current = self._pass_start_value(
                            proposal,
                            updates,
                            parameter,
                        )
                        upper = self.parameters.repair_bounds(parameter)[1]
                        factor = min(
                            shared_factor,
                            upper/max(current, 1.0e-12),
                        )
                        self._set_larger(
                            proposal,
                            updates,
                            parameter,
                            current*factor,
                            row,
                            rule,
                            margin,
                            (
                                "share exact lightship/displacement deficit "
                                "between beam and depth ratios"
                            ),
                        )
                        achieved_factor *= max(factor, 1.0)

                    remaining_volume_ratio /= achieved_factor

                if remaining_volume_ratio > 1.0+1.0e-12:
                    current_loa = self._pass_start_value(
                        proposal,
                        updates,
                        "LOA",
                    )
                    self._set_larger(
                        proposal,
                        updates,
                        "LOA",
                        current_loa*remaining_volume_ratio**(1.0/3.0),
                        row,
                        rule,
                        margin,
                        (
                            "consume residual approximate displacement-volume "
                            "factor through LOA cubed"
                        ),
                    )
                return

            current_draft = self._pass_start_value(
                proposal,
                updates,
                "Draft_frac",
            )
            current_loa = self._pass_start_value(
                proposal,
                updates,
                "LOA",
            )
            self._set_larger(
                proposal,
                updates,
                "Draft_frac",
                max(current_draft*(1.0 + margin), 0.72),
                row,
                rule,
                margin,
                "increase submerged displacement through authoritative draft fraction",
            )
            self._set_larger(
                proposal,
                updates,
                "LOA",
                current_loa*(1.0 + margin),
                row,
                rule,
                margin,
                "increase hull volume faster than surface-dominated lightship mass",
            )

        elif rule.method == "cargo_equilibrium":
            diagnostics = failure_diagnostics or {}
            volume_residual = pd.to_numeric(
                pd.Series(
                    [
                        diagnostics.get(
                            "Normalized_Displacement_Volume_Residual"
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]
            moment_residual = pd.to_numeric(
                pd.Series(
                    [
                        diagnostics.get(
                            "Normalized_Displacement_First_Moment_Residual"
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if pd.notna(moment_residual) and abs(moment_residual) > 1.0e-12:
                moment_residual = float(moment_residual)

                # r_M is normalized by displacement times ship length.  Use it
                # only to size a bounded longitudinal movement; exact GenISys
                # reevaluation, not this proportional step, decides whether
                # the new geometry is better.  The persistent stochastic
                # system margin preserves diversity between repaired parents.
                requested_shift = min(
                    0.20,
                    2.0*abs(moment_residual)*(1.0+margin),
                )
                primary_movement_available = False

                if moment_residual > 0.0:
                    # r_M > 0 means the attainable buoyancy first moment is
                    # aft of the selected weight first moment.  Translate the
                    # cargo region aft by moving its forward boundary aft and
                    # its aft boundary aft.  Reducing the lazarette fraction
                    # supplies additional aft movement while also moving the
                    # modeled machinery/fuel regions aft.
                    for parameter in (
                        "frac_CrashBlkhd",
                    ):
                        current = self._pass_start_value(
                            proposal,
                            updates,
                            parameter,
                        )
                        upper = self.parameters.repair_bounds(parameter)[1]
                        primary_movement_available |= current < upper-1.0e-12
                        self._set_larger(
                            proposal,
                            updates,
                            parameter,
                            current+requested_shift,
                            row,
                            rule,
                            margin,
                            (
                                "positive normalized first-moment residual: "
                                "move the forward cargo boundary aft"
                            ),
                        )

                    for parameter in (
                        "frac_AftPeakBlkhd",
                        "frac_lazarette_L",
                    ):
                        current = self._pass_start_value(
                            proposal,
                            updates,
                            parameter,
                        )
                        lower = self.parameters.repair_bounds(parameter)[0]
                        primary_movement_available |= current > lower+1.0e-12
                        self._set_smaller(
                            proposal,
                            updates,
                            parameter,
                            current-requested_shift,
                            row,
                            rule,
                            margin,
                            (
                                "positive normalized first-moment residual: "
                                "move the aft cargo boundary and aft systems aft"
                            ),
                        )

                    # A shorter bow taper adds forward hull volume and was
                    # verified to move the attainable buoyancy/weight moment
                    # pair in the required direction.  This is a hull-shape
                    # control, not a beam/depth volume surrogate.
                    current_lb = self._pass_start_value(
                        proposal,
                        updates,
                        "Lb",
                    )
                    lb_lower = self.parameters.repair_bounds("Lb")[0]
                    primary_movement_available |= current_lb > lb_lower+1.0e-12
                    self._set_smaller(
                        proposal,
                        updates,
                        "Lb",
                        current_lb-2.0*requested_shift,
                        row,
                        rule,
                        margin,
                        (
                            "positive normalized first-moment residual: shorten "
                            "the bow taper to move the attainable balance point"
                        ),
                    )

                    # Once every longitudinal control is at its repair guard,
                    # use the signed volume residual to restore endpoint-draft
                    # headroom.  The target full-load displacement is rebuilt
                    # at every Draft_frac; therefore this is a bounded design-
                    # draft search, not the invalid fixed-target Bd/Dd scaling
                    # used by the earlier repair.
                    if not primary_movement_available:
                        current_draft = self._pass_start_value(
                            proposal,
                            updates,
                            "Draft_frac",
                        )
                        draft_lower, draft_upper = (
                            self.parameters.repair_bounds("Draft_frac")
                        )
                        draft_movement_available = False

                        if pd.notna(volume_residual):
                            if float(volume_residual) < 0.0:
                                draft_movement_available = (
                                    current_draft > draft_lower+1.0e-12
                                )
                                self._set_smaller(
                                    proposal,
                                    updates,
                                    "Draft_frac",
                                    current_draft*(1.0-margin),
                                    row,
                                    rule,
                                    margin,
                                    (
                                        "negative volume residual after "
                                        "longitudinal repair: lower design draft"
                                    ),
                                )
                            elif float(volume_residual) > 0.0:
                                draft_movement_available = (
                                    current_draft < draft_upper-1.0e-12
                                )
                                self._set_larger(
                                    proposal,
                                    updates,
                                    "Draft_frac",
                                    current_draft*(1.0+margin),
                                    row,
                                    rule,
                                    margin,
                                    (
                                        "positive volume residual after "
                                        "longitudinal repair: raise design draft"
                                    ),
                                )

                        # Hold count is the final discrete exploration because
                        # it also changes three-hold spacing and structural
                        # mass.  Use it only when no signed draft movement is
                        # available, one division per exact evaluation.
                        if not draft_movement_available:
                            current_holds = self._pass_start_value(
                                proposal,
                                updates,
                                "Num_Cargo_Holds",
                            )
                            self._set_smaller(
                                proposal,
                                updates,
                                "Num_Cargo_Holds",
                                current_holds-1.0,
                                row,
                                rule,
                                margin,
                                (
                                    "positive normalized first-moment residual: "
                                    "explore one fewer cargo-hold division"
                                ),
                            )
                else:
                    # Reverse the same controls for r_M < 0: translate the
                    # cargo region forward, lengthen the bow taper, and only
                    # then add one hold division if all continuous controls
                    # have reached their repair guards.
                    current_crash = self._pass_start_value(
                        proposal,
                        updates,
                        "frac_CrashBlkhd",
                    )
                    crash_lower = self.parameters.repair_bounds(
                        "frac_CrashBlkhd"
                    )[0]
                    primary_movement_available |= (
                        current_crash > crash_lower+1.0e-12
                    )
                    self._set_smaller(
                        proposal,
                        updates,
                        "frac_CrashBlkhd",
                        current_crash-requested_shift,
                        row,
                        rule,
                        margin,
                        (
                            "negative normalized first-moment residual: move "
                            "the forward cargo boundary forward"
                        ),
                    )

                    for parameter in (
                        "frac_AftPeakBlkhd",
                        "frac_lazarette_L",
                    ):
                        current = self._pass_start_value(
                            proposal,
                            updates,
                            parameter,
                        )
                        upper = self.parameters.repair_bounds(parameter)[1]
                        primary_movement_available |= current < upper-1.0e-12
                        self._set_larger(
                            proposal,
                            updates,
                            parameter,
                            current+requested_shift,
                            row,
                            rule,
                            margin,
                            (
                                "negative normalized first-moment residual: "
                                "move the aft cargo boundary and systems forward"
                            ),
                        )

                    current_lb = self._pass_start_value(
                        proposal,
                        updates,
                        "Lb",
                    )
                    lb_upper = self.parameters.repair_bounds("Lb")[1]
                    primary_movement_available |= current_lb < lb_upper-1.0e-12
                    self._set_larger(
                        proposal,
                        updates,
                        "Lb",
                        current_lb+2.0*requested_shift,
                        row,
                        rule,
                        margin,
                        (
                            "negative normalized first-moment residual: lengthen "
                            "the bow taper to move the attainable balance point"
                        ),
                    )

                    if not primary_movement_available:
                        current_draft = self._pass_start_value(
                            proposal,
                            updates,
                            "Draft_frac",
                        )
                        draft_lower, draft_upper = (
                            self.parameters.repair_bounds("Draft_frac")
                        )
                        draft_movement_available = False

                        if pd.notna(volume_residual):
                            if float(volume_residual) < 0.0:
                                draft_movement_available = (
                                    current_draft > draft_lower+1.0e-12
                                )
                                self._set_smaller(
                                    proposal,
                                    updates,
                                    "Draft_frac",
                                    current_draft*(1.0-margin),
                                    row,
                                    rule,
                                    margin,
                                    (
                                        "negative volume residual after "
                                        "longitudinal repair: lower design draft"
                                    ),
                                )
                            elif float(volume_residual) > 0.0:
                                draft_movement_available = (
                                    current_draft < draft_upper-1.0e-12
                                )
                                self._set_larger(
                                    proposal,
                                    updates,
                                    "Draft_frac",
                                    current_draft*(1.0+margin),
                                    row,
                                    rule,
                                    margin,
                                    (
                                        "positive volume residual after "
                                        "longitudinal repair: raise design draft"
                                    ),
                                )

                        if not draft_movement_available:
                            current_holds = self._pass_start_value(
                                proposal,
                                updates,
                                "Num_Cargo_Holds",
                            )
                            self._set_larger(
                                proposal,
                                updates,
                                "Num_Cargo_Holds",
                                current_holds+1.0,
                                row,
                                rule,
                                margin,
                                (
                                    "negative normalized first-moment residual: "
                                    "explore one additional cargo-hold division"
                                ),
                            )

            elif pd.notna(volume_residual) and abs(volume_residual) > 1.0e-12:
                # Only use draft after the moment equation is already closed.
                # Full-load target displacement is recalculated from each hull,
                # so Bd/Dd scaling by 1/(1+r_V) is not a valid residual repair.
                current_draft = self._pass_start_value(
                    proposal,
                    updates,
                    "Draft_frac",
                )
                draft_step = min(
                    0.10,
                    abs(float(volume_residual))*(1.0+margin),
                )

                if float(volume_residual) < 0.0:
                    candidate_draft = current_draft*(1.0-draft_step)
                    setter = self._set_smaller
                    direction = "reduce"
                else:
                    candidate_draft = current_draft*(1.0+draft_step)
                    setter = self._set_larger
                    direction = "increase"

                setter(
                    proposal,
                    updates,
                    "Draft_frac",
                    candidate_draft,
                    row,
                    rule,
                    margin,
                    (
                        f"isolated signed volume residual: {direction} the "
                        "selected design draft before exact reevaluation"
                    ),
                )

            # A legacy or synthetic caller may lack structured residuals.  Its
            # direction is genuinely unknown, so use one small reversible
            # hold-count exploration rather than claiming a volume equation.
            if pd.isna(volume_residual) and pd.isna(moment_residual):
                current_holds = self._pass_start_value(
                    proposal,
                    updates,
                    "Num_Cargo_Holds",
                )
                self._set_smaller(
                    proposal,
                    updates,
                    "Num_Cargo_Holds",
                    current_holds-1.0,
                    row,
                    rule,
                    margin,
                    (
                        "fallback discrete cargo-allocation exploration when "
                        "the evaluator supplied no signed residual pair"
                    ),
                )

    def _trim_repair(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        row: pd.Series,
        rule: RepairRule,
        margin: float,
    ) -> None:
        limit_m = _numeric(row, "Threshold Value")
        trim_m = _numeric(row, "Constraint Value")

        if limit_m is None or trim_m is None or limit_m <= 0.0:
            return

        ratio = max(1.0, trim_m/limit_m)
        current_range = self._pass_start_value(
            proposal,
            updates,
            "Voyage_Range",
        )
        current_fn = self._pass_start_value(
            proposal,
            updates,
            "Fn_ship",
        )
        current_draft = self._pass_start_value(
            proposal,
            updates,
            "Draft_frac",
        )

        # The active departure trim row is evaluated after optimizing one
        # immutable cargo allocation.  When that bounded allocation still
        # cannot reach the tolerance, reduce the aft fuel demand through range
        # and speed; do not edit evaluator-owned solved cargo shares directly.
        self._set_smaller(
            proposal,
            updates,
            "Voyage_Range",
            current_range/(ratio**0.55*(1.0 + margin)),
            row,
            rule,
            margin,
            "reduce aft fuel mass: range_new = range/trim_ratio^0.55/(1+step)",
        )
        self._set_smaller(
            proposal,
            updates,
            "Fn_ship",
            current_fn/(ratio**0.12*(1.0 + 0.25*margin)),
            row,
            rule,
            margin,
            "reduce resistance/fuel demand through the authoritative Froude number",
        )
        self._set_larger(
            proposal,
            updates,
            "Draft_frac",
            max(current_draft, 0.70),
            row,
            rule,
            margin,
            "increase trim tolerance and waterplane restoring moment",
        )

    def _subdivision_repair(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        row: pd.Series,
        rule: RepairRule,
        margin: float,
    ) -> None:
        current = int(round(self._pass_start_value(
            proposal,
            updates,
            "Num_Cargo_Holds",
        )))
        maximum_holds = int(
            self.parameters.repair_bounds("Num_Cargo_Holds")[1]
        )

        # L_3h is derived from LOA, end-region fractions, and hold count. Adding
        # a hold shortens the generated bulkhead spacing without changing LOA,
        # so it is always the first and least-disruptive flooding repair.
        if current < maximum_holds:
            candidate = min(
                maximum_holds,
                max(current+1, math.ceil(current*(1.0+margin))),
            )
            self._set_larger(
                proposal,
                updates,
                "Num_Cargo_Holds",
                candidate,
                row,
                rule,
                margin,
                (
                    "increase cargo-hold count to reduce derived L_3h with "
                    "zero change to LOA"
                ),
            )
            return

        # At the representational fifteen-hold limit, reduce LOA only enough
        # to reproduce the spacing reduction that one additional hold would
        # have supplied. No second repair is attempted after final evaluation.
        current_loa = self._pass_start_value(proposal, updates, "LOA")
        self._set_smaller(
            proposal,
            updates,
            "LOA",
            current_loa*current/(current+1.0),
            row,
            rule,
            margin,
            (
                "cargo-hold count is at its maximum; reduce LOA by the minimum "
                "one-additional-hold equivalent to shorten derived L_3h"
            ),
        )

    def _principal_dimension_repair(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        row: pd.Series,
        rule: RepairRule,
        margin: float,
    ) -> None:
        """Invert the three explicit ABS MVR principal-dimension limits."""

        name = str(row.get("Name", ""))
        maximum = _numeric(row, "Threshold Value")

        if maximum is None or maximum <= 0.0:
            return

        if name == "MVR maximum Rule length":
            self._set_smaller(
                proposal,
                updates,
                "LOA",
                maximum/(1.0 + margin),
                row,
                rule,
                margin,
                "LOA_new = maximum_rule_length/(1+margin)",
            )
            return

        loa = float(proposal[self.parameters.index["LOA"]])

        if name == "MVR maximum breadth from length":
            target_bd = (maximum/loa)/(1.0 + margin)
            self._set_smaller(
                proposal,
                updates,
                "Bd",
                target_bd,
                row,
                rule,
                margin,
                "Bd_new = maximum_breadth/LOA/(1+margin)",
            )
            return

        if name == "MVR maximum breadth from depth":
            # B and D are owned by LOA*Bd and LOA*Dd. Use the current controls
            # directly because an earlier hull-domain repair can make the saved
            # derived coordinates stale until final canonicalization.
            breadth = loa*float(proposal[self.parameters.index["Bd"]])
            depth = loa*float(proposal[self.parameters.index["Dd"]])
            factor = maximum/max(depth, 1.0e-9)
            required_depth = breadth*(1.0 + margin)/factor
            self._set_larger(
                proposal,
                updates,
                "Dd",
                required_depth/loa,
                row,
                rule,
                margin,
                "Dd_new = breadth*(1+margin)/(breadth_depth_factor*LOA)",
            )

    def _repair_hull_domains(
        self,
        proposal: np.ndarray,
        updates: dict[str, dict],
        failed: pd.DataFrame,
        ledger: MarginLedger,
        unresolved_rows: list[dict],
    ) -> None:
        """Apply the reviewed 49-row hull and five-row Holtrop repairs.

        Hull geometry is evaluated locally with the unchanged
        ``HullParameterization.input_Constraints`` equations.  Holtrop's
        reported-value inverses remain proposals until the ordinary complete
        evaluator recalculates volume, LCB, resistance, structure, and every
        system constraint.
        """

        domain_rows = []

        for _, row in failed.iterrows():
            rule = row["_Rule"]

            if rule.method != "hull_domain":
                continue

            key = row["Constraint_Key"]
            margin = ledger.margin_for(key, rule.method, rule.family)
            domain_rows.append((row, rule, margin))

        if not domain_rows:
            return

        hull_items = [
            item for item in domain_rows if item[1].family == "hull_geometry_domain"
        ]

        if hull_items:
            hull_frame = pd.DataFrame([row for row, _, _ in hull_items])
            margin_by_name = {
                str(row["Name"]): float(margin)
                for row, _, margin in hull_items
            }
            result = self.hull_domain_search.repair_hull_rows(
                proposal,
                hull_frame,
                margin_by_name,
            )

            rows_by_name = {
                str(row["Name"]): (row, rule, margin)
                for row, rule, margin in hull_items
            }

            for change in result.changes:
                causal = [
                    rows_by_name[name]
                    for name in change.constraint_names
                    if name in rows_by_name
                ]

                if not causal:
                    continue

                representative_row, representative_rule, representative_margin = causal[0]
                self._set_value(
                    proposal,
                    updates,
                    change.parameter,
                    change.new_value,
                    representative_row,
                    representative_rule,
                    representative_margin,
                    change.equation,
                )

                # Preserve every equation identity that caused a shared hull
                # coordinate to move, not just the representative row passed
                # to the generic provenance writer.
                for row, _, margin in causal[1:]:
                    self._append_cause(
                        updates[change.parameter],
                        constraint_key(row),
                        margin,
                    )

            for name in result.unresolved_names:
                row, rule, _ = rows_by_name[name]
                unresolved_rows.append(
                    {
                        **row.drop(labels=["_Rule"]).to_dict(),
                        "Repair_Status": "bounded_hull_search_exhausted",
                        "Repair_Family": rule.family,
                        "Repair_Note": (
                            "No candidate using the equation-traced controls "
                            "satisfied this hull predicate inside the declared "
                            "repair domain."
                        ),
                    }
                )

        # Each Holtrop identity has a separate reported-value inverse.  Apply
        # them in stable identity order so repeated runs with the same ledger
        # produce byte-identical vectors and action evidence.
        holtrop_items = sorted(
            (
                item
                for item in domain_rows
                if item[1].family == "holtrop_domain"
            ),
            key=lambda item: constraint_key(item[0]),
        )

        for row, rule, margin in holtrop_items:
            result = self.hull_domain_search.repair_holtrop_row(
                proposal,
                row,
                margin,
            )

            for change in result.changes:
                self._set_value(
                    proposal,
                    updates,
                    change.parameter,
                    change.new_value,
                    row,
                    rule,
                    margin,
                    change.equation,
                )

            if result.unresolved_names:
                unresolved_rows.append(
                    {
                        **row.drop(labels=["_Rule"]).to_dict(),
                        "Repair_Status": "holtrop_repair_domain_exhausted",
                        "Repair_Family": rule.family,
                        "Repair_Note": (
                            "The directed Holtrop inverse produced no "
                            "independent movement inside the declared domain."
                        ),
                    }
                )

    def propose(
        self,
        vector: Iterable[float],
        constraints: pd.DataFrame,
        ledger: MarginLedger,
        failure_diagnostics: dict | None = None,
    ) -> RepairProposal:
        """Apply one direct pass to every currently failed, mapped row."""

        proposal = np.asarray(vector, dtype=float).copy()
        source_vector = proposal.copy()
        failed = failed_constraint_rows(constraints).copy()

        if failed.empty:
            return RepairProposal(proposal, [], pd.DataFrame())

        failed["Constraint_Key"] = failed.apply(constraint_key, axis=1)
        failed["_Rule"] = [
            self.constraints.classify(row) for _, row in failed.iterrows()
        ]
        failed["_Priority"] = failed["_Rule"].map(lambda rule: rule.priority)
        failed = failed.sort_values(
            ["_Priority", "Constraint_Key"],
            kind="stable",
        )
        updates: dict[str, dict] = {}
        unresolved_rows = []

        # Stage 1: repair the hull domain before any dependent structure.  The
        # combined hull search owns all failed hull identities and may change
        # principal dimensions or hull-form coordinates used downstream.
        self._repair_hull_domains(
            proposal,
            updates,
            failed,
            ledger,
            unresolved_rows,
        )

        # Stage 2: apply direct and discrete changes in registry-priority order.
        # Dimension-changing system and subdivision rules have priorities ahead
        # of structural rules. Bracket geometry is deliberately delayed until
        # all member depths and stiffener grids have reached their final state.
        for _, row in failed.iterrows():
            rule = row["_Rule"]
            key = row["Constraint_Key"]

            if rule.status not in {"repairable", "repairable_approximate_coupled"}:
                unresolved_rows.append(
                    {
                        **row.drop(labels=["_Rule"]).to_dict(),
                        "Repair_Status": rule.status,
                        "Repair_Family": rule.family,
                        "Repair_Note": rule.note,
                    }
                )
                continue

            margin = ledger.margin_for(key, rule.method, rule.family)

            if rule.method == "direct_minimum":
                self._direct_minimum(proposal, updates, row, rule, margin)
            elif rule.method == "maximum_spacing_count":
                self._maximum_spacing_count(
                    proposal, updates, row, rule, margin
                )
            elif rule.method == "maximum_aspect_ratio":
                self._maximum_aspect_ratio(
                    proposal, updates, row, rule, margin
                )
            elif rule.method == "bounded_bracket_angle":
                pass
            elif rule.method == "hatch_area":
                self._hatch_area(proposal, updates, row, rule, margin)
            elif rule.method == "engine_room_length":
                self._engine_room_length(
                    proposal, updates, row, rule, margin
                )
            elif rule.method in {
                "tank_capacity",
                "system_capacity",
                "system_displacement",
                "cargo_equilibrium",
            }:
                self._tank_or_system_repair(
                    proposal,
                    updates,
                    row,
                    rule,
                    margin,
                    failure_diagnostics,
                )
            elif rule.method == "trim":
                self._trim_repair(proposal, updates, row, rule, margin)
            elif rule.method == "subdivision":
                self._subdivision_repair(
                    proposal, updates, row, rule, margin
                )
            elif rule.method == "principal_dimensions":
                self._principal_dimension_repair(
                    proposal, updates, row, rule, margin
                )
            elif rule.method == "hull_domain":
                # All simultaneously failed hull rows are combined below so
                # one coupled coordinate search sees the complete 49-row
                # geometry state.  Holtrop identities are also dispatched
                # there in stable identity order.
                pass
            elif rule.method == "uniform_bulkhead_capacity":
                # The deck and bottom labels call one identical active
                # equation.  Combine them below before changing the shared
                # transverse-bulkhead composite.
                pass
            elif (
                rule.method == "section_modulus"
                and rule.family not in self.SECTION_PRIMARY
            ):
                unresolved_rows.append(
                    {
                        **row.drop(labels=["_Rule"]).to_dict(),
                        "Repair_Status": "repair_method_not_implemented",
                        "Repair_Family": rule.family,
                        "Repair_Note": (
                            "This section family does not yet have an approved "
                            "source-anchored section block."
                        ),
                    }
                )
            elif rule.method in {"section_modulus", "global_hull_girder"}:
                # These families are combined below so one governing repair
                # can satisfy multiple rows without order-dependent updates.
                pass
            else:
                unresolved_rows.append(
                    {
                        **row.drop(labels=["_Rule"]).to_dict(),
                        "Repair_Status": "repair_method_not_implemented",
                        "Repair_Family": rule.family,
                        "Repair_Note": (
                            rule.note
                            or f"No approved implementation for method {rule.method}."
                        ),
                    }
                )

        # Stage 3: repair the whole longitudinal effective section before
        # sizing local supporting members.  Its plate changes improve local
        # attached-section properties and must therefore cascade into the
        # subsequent member calculations rather than being applied afterward.
        global_rows = []

        for _, row in failed.iterrows():
            rule = row["_Rule"]

            if rule.method == "global_hull_girder":
                margin = ledger.margin_for(
                    row["Constraint_Key"], rule.method, rule.family
                )
                global_rows.append((row, rule, margin))

        if global_rows:
            self._global_hull_girder(proposal, updates, global_rows)

            for parameter in global_rows[0][1].parameters:
                if parameter not in updates:
                    continue

                for row, _, margin in global_rows:
                    self._append_cause(
                        updates[parameter],
                        constraint_key(row),
                        margin,
                    )

        # Stage 4: size every local member family once, using the already
        # repaired hull, arrangement, support spacing, and effective-section
        # plate thicknesses.  Multiple rules for one member share one
        # governing section proposal.
        section_groups = self._section_modulus_groups(failed, ledger)

        for family, items in section_groups.items():
            self._repair_section_family(
                source_vector,
                proposal,
                updates,
                family,
                items,
            )

            # Every failed row in the governing family caused the combined
            # section search, even though one representative supplies the
            # tightest target used by each individual parameter setter.
            for parameter in self.SECTION_PRIMARY[family]:
                if parameter is None or parameter not in updates:
                    continue

                for row, _, margin in items:
                    self._append_cause(
                        updates[parameter],
                        constraint_key(row),
                        margin,
                    )

            count_parameter = self.SECTION_COUNT.get(family)

            if count_parameter in updates:
                for row, _, margin in items:
                    self._append_cause(
                        updates[count_parameter],
                        constraint_key(row),
                        margin,
                    )

        # Run the shared bulkhead-support repair after every other structural
        # section update.  Hull-girder repair can increase transverse-bulkhead
        # plating enough to reduce the active radius-of-gyration capacity, so
        # running this block earlier can create a passing intermediate state
        # which the later global update invalidates.
        uniform_loading_rows = []

        for _, row in failed.iterrows():
            rule = row["_Rule"]

            if rule.method == "uniform_bulkhead_capacity":
                margin = ledger.margin_for(
                    row["Constraint_Key"], rule.method, rule.family
                )
                uniform_loading_rows.append((row, rule, margin))

        if uniform_loading_rows:
            self._repair_uniform_bulkhead_loading(
                source_vector,
                proposal,
                updates,
                uniform_loading_rows,
            )

        # Stage 5: refresh direct plate rules from the final repaired
        # dimensions and support grids.  Inner-bottom, hopper, strength-deck,
        # shell, and watertight-bulkhead plates all use the same governing-max
        # process here.
        self._refresh_geometry_dependent_plate_minima(
            proposal,
            updates,
            constraints,
            ledger,
        )

        bracket_rows = {}

        for _, row in failed.iterrows():
            rule = row["_Rule"]

            if rule.method == "bounded_bracket_angle":
                bracket_rows.setdefault(rule.family, (row, rule))

        for row, rule in bracket_rows.values():
            margin = ledger.margin_for(
                row["Constraint_Key"],
                rule.method,
                rule.family,
            )
            self._bracket_angle(proposal, updates, row, rule, margin)

        # A repairable row that produced no movement is not silently dropped.
        # It usually means that every mapped control is already at a declared
        # repair-domain limit or that the current equation cannot invert the
        # supplied values.
        targeted_keys = {
            key
            for update in updates.values()
            for key in update["keys"]
        }
        explicitly_unresolved_keys = {
            str(item.get("Constraint_Key", ""))
            for item in unresolved_rows
        }

        for _, row in failed.iterrows():
            rule = row["_Rule"]
            key = row["Constraint_Key"]

            if (
                rule.status in {"repairable", "repairable_approximate_coupled"}
                and key not in targeted_keys
                and key not in explicitly_unresolved_keys
            ):
                unresolved_rows.append(
                    {
                        **row.drop(labels=["_Rule"]).to_dict(),
                        "Repair_Status": "repair_domain_exhausted_or_no_movement",
                        "Repair_Family": rule.family,
                        "Repair_Note": (
                            "The mapped equation produced no independent "
                            "parameter movement inside the declared domain."
                        ),
                    }
                )

        changes = []

        for parameter, update in updates.items():
            lower, upper = self.parameters.repair_bounds(parameter)
            sampling_lower, sampling_upper = self.parameters.sampling_bounds(
                parameter
            )
            value = float(proposal[self.parameters.index[parameter]])
            changes.append(
                ProposedChange(
                    parameter=parameter,
                    old_value=float(update["base_old"]),
                    proposed_value=float(proposal[self.parameters.index[parameter]]),
                    repair_method=str(update["method"]),
                    repair_family=str(update["family"]),
                    constraint_keys=tuple(dict.fromkeys(update["keys"])),
                    sampled_margins=tuple(float(x) for x in update["margins"]),
                    equation=" | ".join(dict.fromkeys(update["equations"])),
                    repair_lower_limit=float(lower),
                    repair_upper_limit=float(upper),
                    at_repair_domain_limit=bool(
                        np.isclose(value, lower, rtol=0.0, atol=1.0e-12)
                        or np.isclose(value, upper, rtol=0.0, atol=1.0e-12)
                    ),
                    outside_sampling_bounds=bool(
                        value < sampling_lower - 1.0e-12
                        or value > sampling_upper + 1.0e-12
                    ),
                )
            )

        unmapped = (
            pd.DataFrame(unresolved_rows)
            if unresolved_rows
            else pd.DataFrame()
        )
        return RepairProposal(proposal, changes, unmapped)


__all__ = [
    "EquationRepairMethods",
    "MarginLedger",
    "ProposedChange",
    "RepairProposal",
    "constraint_key",
]
