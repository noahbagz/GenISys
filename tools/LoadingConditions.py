"""Named full-load conditions for GenISys V1.1.

The two-condition registry implements the project scope recorded by
HDF-ASM-034.  It evaluates the departure-optimized full-cargo design at
departure and arrival fuel states.  A separate ballast-service loading case,
partial/alternate-hold cases, in-port conditions, intermediate conditions, and
loading/unloading sequences are explicitly outside the present GenISys scope.

HDF-ASM-038 approves compensating ballast within the arrival condition so it
reproduces departure displacement and trim.  Its spatial allocation is not yet
active because the existing generated ballast components cover only the cargo
region and cannot generally replace the aft fuel moment.  Until an aft trim
tank is explicitly designated, the table retains zero arrival ballast and the
constraint adapter correctly emits no separate arrival trim requirement.
These two structural load cases remain independent of the 5,000 x 4 x 5
design-population arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LOADING_CONDITION_DEFINITIONS = (
    {
        "Loading_Condition_ID": "full_cargo_departure",
        "Loading_Condition_Name": "Full-cargo departure",
        "Cargo_Fraction": 1.0,
        "Ballast_Fraction": 0.0,
        "Fuel_Fraction": 1.0,
    },
    {
        "Loading_Condition_ID": "full_cargo_arrival",
        "Loading_Condition_Name": "Full-cargo arrival",
        "Cargo_Fraction": 1.0,
        "Ballast_Fraction": 0.0,
        "Fuel_Fraction": 0.10,
    },
)


def loading_condition_registry() -> pd.DataFrame:
    """Return the approved full-load departure/arrival definition."""

    registry = pd.DataFrame(LOADING_CONDITION_DEFINITIONS)
    registry["Assumption_IDs"] = (
        "HDF-ASM-010; HDF-ASM-018; HDF-ASM-034; HDF-ASM-038"
    )

    return registry


def build_condition_specific_fuel_tank_states(cargo_holds) -> pd.DataFrame:
    """Re-evaluate the generated fuel tank at each approved fuel fraction.

    The current design contains one longitudinal fuel-tank region.  This
    function preserves that model while calculating a distinct fill height,
    mass distribution, LCG, and VCG for departure and arrival.  It does not
    infer port/starboard subdivision or free-surface connectivity.
    """

    state_tables = []

    for _, condition in loading_condition_registry().iterrows():
        fuel_state = cargo_holds.calculate_FuelTank_State(
            cargo_holds.fuel_weight*float(condition["Fuel_Fraction"])
        ).copy()
        fuel_state.insert(
            0,
            "Loading_Condition_Name",
            condition["Loading_Condition_Name"],
        )
        fuel_state.insert(
            0,
            "Loading_Condition_ID",
            condition["Loading_Condition_ID"],
        )
        fuel_state["Applied_Fuel_Fraction"] = float(
            condition["Fuel_Fraction"]
        )
        fuel_state["Tank_Geometry_Status"] = (
            "longitudinal_vertical_geometry_only_missing_transverse_groups"
        )
        state_tables.append(fuel_state)

    return pd.concat(state_tables, ignore_index=True)


def _component_scale(component_name: str, condition: pd.Series) -> float:
    """Return the condition-specific mass multiplier for one component."""

    if component_name == "Cargo_Hold":
        return float(condition["Cargo_Fraction"])

    if component_name == "Ballast_Tank":
        return float(condition["Ballast_Fraction"])

    if component_name == "Fuel_Tank":
        return float(condition["Fuel_Fraction"])

    # Structure, outfit, machinery, miscellaneous deadweight, hatch covers,
    # and any later lightship components remain aboard in every condition.
    return 1.0


def build_loading_condition_component_states(
    weight_components: pd.DataFrame,
    condition_specific_fuel_tanks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand base components into full-load departure/arrival states."""

    component_columns = [
        "Component_Index",
        "Component_Name",
        "Base_Weight_kg",
        "Applied_Fraction",
        "Condition_Weight_kg",
        "LCG_m",
        "VCG_m",
    ]
    state_tables = []
    base = weight_components.reset_index(drop=True)

    for _, condition in loading_condition_registry().iterrows():
        scales = np.asarray(
            [
                _component_scale(str(name), condition)
                for name in base["Name"]
            ],
            dtype=float,
        )
        state = pd.DataFrame(
            {
                "Component_Index": np.arange(len(base)),
                "Component_Name": base["Name"].astype(str),
                "Base_Weight_kg": base["Total_Weight_kg"].astype(float),
                "Applied_Fraction": scales,
                "Condition_Weight_kg": (
                    base["Total_Weight_kg"].to_numpy(dtype=float)*scales
                ),
                "LCG_m": base["LCG"].astype(float),
                "VCG_m": base["VCG"].astype(float),
            }
        )

        # HDF-ASM-010: the ten-percent arrival fuel condition must be a real
        # geometric tank fill, not only a scalar multiple of the departure
        # weight at the departure CG.  Replace the aggregate fuel component's
        # mass and centers when the condition-specific state is supplied.
        if condition_specific_fuel_tanks is not None:
            fuel_matches = condition_specific_fuel_tanks.loc[
                condition_specific_fuel_tanks[
                    "Loading_Condition_ID"
                ].astype(str)
                == str(condition["Loading_Condition_ID"])
            ]
            fuel_component_indices = state.index[
                state["Component_Name"] == "Fuel_Tank"
            ]

            for fuel_component_index, (_, fuel_state) in zip(
                fuel_component_indices,
                fuel_matches.iterrows(),
            ):
                state.at[
                    fuel_component_index,
                    "Condition_Weight_kg",
                ] = float(fuel_state["Total_Weight_kg"])
                state.at[fuel_component_index, "LCG_m"] = float(
                    fuel_state["LCG"]
                )
                state.at[fuel_component_index, "VCG_m"] = float(
                    fuel_state["VCG"]
                )

        for column in reversed(condition.index.tolist()):
            state.insert(0, column, condition[column])

        state_tables.append(state)

    return pd.concat(state_tables, ignore_index=True)[
        list(loading_condition_registry().columns) + component_columns
    ]


def summarize_loading_conditions(
    component_states: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate mass and center of gravity for every named condition."""

    rows = []

    for condition_id, state in component_states.groupby(
        "Loading_Condition_ID",
        sort=False,
    ):
        mass_kg = float(state["Condition_Weight_kg"].sum())
        longitudinal_moment_kg_m = float(
            (state["Condition_Weight_kg"]*state["LCG_m"]).sum()
        )
        vertical_moment_kg_m = float(
            (state["Condition_Weight_kg"]*state["VCG_m"]).sum()
        )
        first = state.iloc[0]
        rows.append(
            {
                "Loading_Condition_ID": condition_id,
                "Loading_Condition_Name": first["Loading_Condition_Name"],
                "Cargo_Fraction": float(first["Cargo_Fraction"]),
                "Ballast_Fraction": float(first["Ballast_Fraction"]),
                "Fuel_Fraction": float(first["Fuel_Fraction"]),
                "Total_Weight_kg": mass_kg,
                "Displacement_t": mass_kg/1000.0,
                "LCG_m": longitudinal_moment_kg_m/mass_kg,
                "TCG_m": 0.0,
                "VCG_m": vertical_moment_kg_m/mass_kg,
                "Assumption_IDs": first["Assumption_IDs"],
            }
        )

    return pd.DataFrame(rows)
