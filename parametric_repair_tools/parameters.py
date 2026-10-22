"""Ownership, coupling, and repair-domain rules for the 189-value vector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from tools.V1p1Dataset import (
    derive_constraint_draft_m,
    derive_structural_bilge_radius_m,
    load_parameter_catalog,
)

from .configuration import RepairConfiguration


DERIVED_NAMES = frozenset(
    {
        "L_3h",
        "B",
        "T",
        "D",
        "R_b",
        # CadQuery intentionally aligns the deck member grids to the bottom
        # grids and writes these corrected aliases back into the vector.
        "num top girders",
        "num intermediate deck stiffeners",
    }
)
COUPLED_ALIAS_NAMES = frozenset(
    {
        # These coordinates remain in the public 189-value interface, but
        # canonicalization always derives them from their authoritative peer.
        # They are therefore neither independent controls nor general module
        # outputs such as B, D, T, R_b, and L_3h.
        "Bc",
        "bit_EP_T",
    }
)
CARGO_FRACTION_NAMES = tuple(
    f"frac_Cargo_Hold_{index}" for index in range(15)
)

# These coordinates encode approved project scope, not a repair choice.
FIXED_VALUES = {
    "cat tanker": 0.0,
    "cat container": 0.0,
    "cat bulkcarrier": 1.0,
    "longitudinal bulkhead bit": 0.0,
    "num longitudinal bulkheads": 0.0,
    "Hatch Openings Bit": 1.0,
    "l_overhang": 0.03,
}

INACTIVE_NAMES = frozenset(
    {
        "cont step height",
        "cont step width",
        "cont step deck plate thickness",
        "con step side plate thickness",
        "num cont step deck stiff",
        "cont step deck stiff h",
        "cont step deck stiff t",
        "cont step deck stiff w",
        "cont step deck stiff torc",
        "num cont step side stiff",
        "cont step side stiff h",
        "cont step side stiff t",
        "cont step side stiff w",
        "cont step side stiff torc",
        "long blkhd plate t",
        "long blkhd long stiff h",
        "long blkhd long stiff t",
        "long blkhd long stiff w",
        "long blkhd long stiff torc",
        "long blkhd vert stiff h",
        "long blkhd vert stiff t",
        "long blkhd vert stiff w",
        "long blkhd vert stiff torc",
    }
)


@dataclass(frozen=True)
class CanonicalizationChange:
    """Describe one dependency or representation correction."""

    parameter: str
    old_value: float
    new_value: float
    reason: str


class ParameterRegistry:
    """Address coordinates by authoritative names and enforce ownership.

    The catalog's LL/UL values describe the random-sampling population.  They
    are retained for reporting, while :class:`RepairConfiguration` supplies
    wider repair-domain guards for selected independent controls.
    """

    END_REGION_NAMES = (
        "frac_CrashBlkhd",
        "frac_AftPeakBlkhd",
        "frac_lazarette_L",
        "frac_engineRoom_L",
        "frac_fuel_tank_L",
    )

    def __init__(self, configuration: RepairConfiguration | None = None):
        self.configuration = configuration or RepairConfiguration()
        self.catalog = load_parameter_catalog().reset_index(drop=True)
        self.names = self.catalog["Name"].astype(str).tolist()
        self.index = dict(zip(self.names, range(len(self.names)), strict=True))

        if len(self.names) != 189:
            raise ValueError(
                "The repair pipeline requires the complete 189-coordinate "
                "GenISys V1.1 schema."
            )

        self.integer_names = frozenset(
            self.catalog.loc[
                self.catalog["data type"].fillna("").str.lower()
                == "integer",
                "Name",
            ].astype(str)
        )
        self.boolean_names = frozenset(
            self.catalog.loc[
                self.catalog["data type"].fillna("").str.lower() == "bool",
                "Name",
            ].astype(str)
        )

    def vector_from_row(self, row: pd.Series | dict) -> np.ndarray:
        """Extract one ordered, independent copy of all 189 coordinates."""

        return np.asarray([row[name] for name in self.names], dtype=float)

    def row_from_vector(self, vector: Iterable[float]) -> dict[str, float]:
        """Return a name-keyed row after verifying vector length."""

        values = np.asarray(vector, dtype=float)

        if values.shape != (189,):
            raise ValueError("A GenISys repair vector must contain 189 values.")

        return dict(zip(self.names, values.tolist(), strict=True))

    def role(self, name: str) -> str:
        """Return the one authoritative repair role for a coordinate."""

        if name in COUPLED_ALIAS_NAMES:
            return "coupled_alias"

        if name in DERIVED_NAMES:
            return "derived"

        if name in CARGO_FRACTION_NAMES:
            return "solver_state"

        if name in INACTIVE_NAMES:
            return "inactive"

        if name in FIXED_VALUES:
            return "fixed"

        if name in self.boolean_names:
            return "control_boolean"

        if name in self.integer_names:
            return "control_integer"

        return "control_continuous"

    def sampling_bounds(self, name: str) -> tuple[float, float]:
        """Return the published random-sampling interval for one coordinate."""

        row = self.catalog.loc[self.catalog["Name"] == name].iloc[0]
        return float(row["LL"]), float(row["UL"])

    def repair_bounds(self, name: str) -> tuple[float, float]:
        """Return the explicit repair domain, or the sampling interval."""

        if name in self.configuration.repair_limits:
            return tuple(self.configuration.repair_limits[name])

        lower, upper = self.sampling_bounds(name)

        # Structural member sizes may move beyond the sampling population in
        # the same spirit as the maintained Ship_Structures repair.  Explicit
        # caps prevent unbounded CAD or numerical searches.
        if self.catalog.loc[self.index[name], "Class"] == "Structure":
            lowered = name.lower()

            if "thickness" in lowered or lowered.endswith(" t"):
                # The active ABS web-aspect rule permits at most 50:1 when a
                # flange is present and 15:1 otherwise.  The structural-depth
                # repair guard below is 10,000 mm, so a 100 mm thickness cap
                # made some otherwise repairable sections mathematically
                # impossible (for example, an 8,000 mm floor needs at least
                # 160 mm at 50:1).  A 1,000 mm numerical guard covers even the
                # stricter 15:1 branch across the complete declared height
                # domain; it is a repair-search guard, not a sampling bound or
                # a recommendation for a practical ship scantling.
                return max(0.0, lower), max(1_000.0, upper)

            if lowered.endswith(" h") or "height" in lowered:
                return max(0.0, lower), max(10_000.0, upper)

            if lowered.endswith(" w") or "width" in lowered:
                return max(0.0, lower), max(10_000.0, upper)

            if name in {
                "web frames per hold",
                "num bottom girders",
                "num intermediate bottom stiffeners",
                "num top girders",
                "num intermediate deck stiffeners",
                "num side shell stiffeners",
                "num vertical bulkhead stiffeners",
                "num transverse bulkhead stiffeners",
                "num bulk car bottom hopper stiff",
                "num bulk car top hopper stiff",
                "num intermediate trans frames",
            }:
                return max(1.0, lower), max(60.0, upper)

            if "bracket l" in lowered:
                return 250.0, 12_000.0

        return lower, upper

    def clip_to_repair_domain(self, name: str, value: float) -> float:
        """Clip an approved independent proposal to its declared domain."""

        if self.role(name) not in {
            "control_continuous",
            "control_integer",
            "control_boolean",
        }:
            raise ValueError(f"Parameter {name!r} is not an independent control.")

        lower, upper = self.repair_bounds(name)
        clipped = float(np.clip(float(value), lower, upper))

        if name in self.integer_names:
            # The catalog may declare a fractional engineering lower bound for
            # a coordinate stored as an integer (for example, an 8.5 mm plate
            # thickness).  Rounding a value already clipped to 8.5 with
            # ``rint`` produces 8, which violates the original bound.  Project
            # into the closed set of admissible integers instead.
            integer_lower = int(np.ceil(lower))
            integer_upper = int(np.floor(upper))

            if integer_lower > integer_upper:
                raise ValueError(
                    f"Integer parameter {name!r} has no admissible value "
                    f"inside [{lower}, {upper}]."
                )

            clipped = float(
                np.clip(
                    np.rint(float(value)),
                    integer_lower,
                    integer_upper,
                )
            )

        if name in self.boolean_names:
            clipped = float(clipped >= 0.5)

        return clipped

    def canonicalize(
        self,
        vector: Iterable[float],
        *,
        ensure_bilge_fit: bool = True,
    ) -> tuple[np.ndarray, list[CanonicalizationChange]]:
        """Recompute owned values and restore approved project scope.

        ``L_3h`` is the generated three-hold module including its two
        structural overhangs.  The active cargo region supplies the clear
        three-hold length, hence division by ``1 - 2*l_overhang``.
        """

        values = np.asarray(vector, dtype=float).copy()

        if values.shape != (189,):
            raise ValueError("A GenISys repair vector must contain 189 values.")

        if not np.all(np.isfinite(values)):
            raise ValueError("Every GenISys repair coordinate must be finite.")

        changes: list[CanonicalizationChange] = []

        def set_value(name: str, new_value: float, reason: str) -> None:
            position = self.index[name]
            old_value = float(values[position])
            new_value = float(new_value)

            if not np.isclose(old_value, new_value, rtol=0.0, atol=1.0e-12):
                changes.append(
                    CanonicalizationChange(
                        parameter=name,
                        old_value=old_value,
                        new_value=new_value,
                        reason=reason,
                    )
                )

            values[position] = new_value

        # Round discrete independent variables before relationships use them.
        for name in sorted(self.integer_names | self.boolean_names):
            if self.role(name).startswith("control_"):
                set_value(
                    name,
                    self.clip_to_repair_domain(name, values[self.index[name]]),
                    "canonical discrete representation",
                )

        for name, fixed_value in FIXED_VALUES.items():
            set_value(name, fixed_value, "approved GenISys project scope")

        for name in INACTIVE_NAMES:
            set_value(name, 0.0, "out-of-scope parameter block")

        # The current hull generator and structure coupling both treat Bc as
        # the full-beam ratio.  Preserve the established Bc=Bd convention.
        set_value(
            "Bc",
            values[self.index["Bd"]],
            "current hull/structure beam-ratio coupling",
        )
        set_value(
            "bit_EP_T",
            values[self.index["bit_EP_S"]],
            "current paired endpoint-topology convention",
        )

        # The top-member count coordinates are generated aliases of the
        # authoritative bottom grids, not independent repair controls.
        set_value(
            "num top girders",
            values[self.index["num bottom girders"]],
            "CadQuery deck/bottom girder-grid alignment",
        )
        set_value(
            "num intermediate deck stiffeners",
            values[self.index["num intermediate bottom stiffeners"]],
            "CadQuery deck/bottom longitudinal-grid alignment",
        )

        for name in CARGO_FRACTION_NAMES:
            set_value(name, 0.0, "cargo allocation is evaluator solver state")

        number_of_holds = int(values[self.index["Num_Cargo_Holds"]])

        if not 1 <= number_of_holds <= len(CARGO_FRACTION_NAMES):
            raise ValueError(
                "Num_Cargo_Holds must fit the fifteen cargo-fraction fields."
            )

        end_fraction = float(
            sum(values[self.index[name]] for name in self.END_REGION_NAMES)
        )
        cargo_region_fraction = 1.0 - end_fraction
        overhang = float(values[self.index["l_overhang"]])

        if cargo_region_fraction <= 0.0:
            raise ValueError("End regions leave no positive cargo region.")

        if 1.0 - 2.0*overhang <= 0.0:
            raise ValueError("Structural overhangs leave no clear hold length.")

        loa_m = float(values[self.index["LOA"]])
        beam_m = loa_m*float(values[self.index["Bd"]])
        depth_m = loa_m*float(values[self.index["Dd"]])
        radius_m = derive_structural_bilge_radius_m(
            beam_m,
            values[self.index["Rc"]],
        )
        three_hold_length_m = (
            3.0*loa_m*cargo_region_fraction
            / (number_of_holds*(1.0 - 2.0*overhang))
        )

        set_value("B", beam_m, "B = LOA*Bd")
        set_value("D", depth_m, "D = LOA*Dd")
        set_value("R_b", radius_m, "R_b = 0.5*B*Rc")
        set_value("L_3h", three_hold_length_m, "three-hold cargo-region module")
        set_value("T", derive_constraint_draft_m(values), "T = Draft_frac*D")

        if ensure_bilge_fit:
            required_db_mm = 1_000.0*radius_m
            current_db_mm = float(values[self.index["Db"]])

            if current_db_mm < required_db_mm:
                db_upper = self.repair_bounds("Db")[1]

                if required_db_mm > db_upper:
                    raise ValueError(
                        "Hull-derived R_b cannot fit inside the declared "
                        "double-bottom repair domain."
                    )

                set_value(
                    "Db",
                    required_db_mm,
                    "geometric compatibility Db >= 1000*R_b",
                )

        for name in ("L_3h", "B", "T", "D", "R_b", "Db"):
            if values[self.index[name]] <= 0.0:
                raise ValueError(f"Derived geometry {name} must be positive.")

        return values, changes

    def sampling_bound_exceedances(
        self,
        vector: Iterable[float],
    ) -> pd.DataFrame:
        """Report repair-domain values outside the original sampling box."""

        values = np.asarray(vector, dtype=float)
        rows = []
        columns = [
            "Parameter",
            "Value",
            "Sampling_Lower_Bound",
            "Sampling_Upper_Bound",
            "Direction",
            "Parameter_Role",
        ]

        for position, name in enumerate(self.names):
            value = float(values[position])
            lower, upper = self.sampling_bounds(name)

            if value < lower - 1.0e-12 or value > upper + 1.0e-12:
                rows.append(
                    {
                        "Parameter": name,
                        "Value": value,
                        "Sampling_Lower_Bound": lower,
                        "Sampling_Upper_Bound": upper,
                        "Direction": "below" if value < lower else "above",
                        "Parameter_Role": self.role(name),
                    }
                )

        # Preserve a readable schema even when no coordinate exceeds its
        # sampling interval; downstream validators must not see a headerless
        # CSV as an evaluation error.
        return pd.DataFrame(rows, columns=columns)

    def role_table(self) -> pd.DataFrame:
        """Export the complete machine-readable coordinate registry."""

        result = self.catalog.copy()
        result["Repair_Role"] = result["Name"].map(self.role)
        result[["Repair_Lower_Limit", "Repair_Upper_Limit"]] = pd.DataFrame(
            [self.repair_bounds(name) for name in self.names],
            index=result.index,
        )
        return result


__all__ = [
    "CARGO_FRACTION_NAMES",
    "CanonicalizationChange",
    "COUPLED_ALIAS_NAMES",
    "DERIVED_NAMES",
    "FIXED_VALUES",
    "INACTIVE_NAMES",
    "ParameterRegistry",
]
