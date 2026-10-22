"""Evaluate one V1.1 system-design shard with the existing GenISys modules.

Scalar results, audit-qualified constraints, optimized parameters, numerical
curve arrays, and failures are written to separate restartable artifacts.  The
script never embeds NumPy arrays as quoted text inside a CSV field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

from tools.ConstraintIdentity import canonical_constraint_bits
from tools.ShipSystemDesign import ShipSystemDesign
from tools.Structure_Constraints_v1p1 import CONSTRAINT_COLUMNS
from tools.V1p1Dataset import derive_constraint_draft_m, load_parameter_catalog
from tools.EvaluationErrors import (
    PhysicalConstraintFailure,
    classify_evaluation_error,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "Ship_Systems_Designs_v1p1" / "Dataset_100k"
)

IDENTIFIER_COLUMNS = [
    "Index",
    "Structure_Index",
    "Hull_Variant_Index",
    "Operating_Condition_Index",
    "Hull_Candidate_Index",
    "Structure_Sample_Attempt",
]


def _write_csv_atomic(table: pd.DataFrame, output_path: Path) -> None:
    """Replace one complete table without exposing a partial CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    compression = (
        {"method": "gzip", "compresslevel": 6, "mtime": 0}
        if output_path.suffix == ".gz"
        else None
    )
    table.to_csv(temporary_path, index=False, compression=compression)
    temporary_path.replace(output_path)


def _csv_shard_suffix(save_mode: str) -> str:
    """Use compressed canonical shards only in reduced-storage mode."""

    if save_mode not in {"all", "reduced"}:
        raise ValueError("save_mode must be either 'all' or 'reduced'.")

    return ".csv.gz" if save_mode == "reduced" else ".csv"


def _system_shard_range_label(input_csv: Path) -> str:
    """Extract the range from either a .csv or .csv.gz system shard."""

    name = input_csv.name

    for suffix in (".csv.gz", ".csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    return name.removeprefix("System_Designs_")


def _write_json_atomic(payload: dict, output_path: Path) -> None:
    """Replace one complete provenance JSON without exposing a partial file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _file_provenance(path: str | Path) -> dict:
    """Return path, size, and SHA-256 for one evaluation input/source file."""

    path = Path(path).resolve()
    digest = hashlib.sha256()
    size_bytes = 0

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size_bytes += len(block)

    try:
        displayed_path = str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        displayed_path = str(path)

    return {
        "path": displayed_path,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _captured_file_provenance(path: str | Path, content: bytes) -> dict:
    """Bind already-read bytes to their path without reopening the file."""

    path = Path(path).resolve()

    try:
        displayed_path = str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        displayed_path = str(path)

    return {
        "path": displayed_path,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _canonical_mapping_provenance(mapping, source_kind: str) -> dict:
    """Describe one optional scenario by its exact resolved override content."""

    resolved_overrides = {} if mapping is None else mapping
    canonical_bytes = json.dumps(
        resolved_overrides,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: (
            value.tolist()
            if isinstance(value, np.ndarray)
            else value.item()
            if isinstance(value, np.generic)
            else str(value)
        ),
    ).encode("utf-8")

    return {
        "provided": mapping is not None,
        "source_kind": source_kind,
        "resolved_overrides": resolved_overrides,
        "canonical_json_size_bytes": len(canonical_bytes),
        "canonical_json_sha256": hashlib.sha256(
            canonical_bytes
        ).hexdigest(),
    }


def _load_optional_json_mapping_with_provenance(
    path: str | Path | None,
) -> tuple[object, dict]:
    """Load one scenario mapping and bind it to the exact source JSON bytes."""

    if path is None:
        return None, _canonical_mapping_provenance(
            None,
            "built_in_defaults_or_inactive_optional_group",
        )

    path = Path(path)
    source_bytes = path.read_bytes()
    mapping = json.loads(source_bytes.decode("utf-8"))
    provenance = _canonical_mapping_provenance(mapping, "json_file")
    provenance["source_file"] = _captured_file_provenance(
        path,
        source_bytes,
    )

    return mapping, provenance


def _write_npz_atomic(output_path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Replace one complete compressed numerical-array shard."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)

    temporary_path.replace(output_path)


def _per_design_table(table: pd.DataFrame) -> pd.DataFrame:
    """Remove shard identifiers from a table stored in one design folder."""

    return table.drop(
        columns=[
            column
            for column in IDENTIFIER_COLUMNS
            if column in table.columns
        ]
    )


def _scalar_hydrostatic_summary(ship_system) -> pd.DataFrame:
    """Build the compact hydrostatic summary used by review folders."""

    hull = ship_system.Hull_Dict
    cg = ship_system.Weights.Ship_CG
    full_cargo = cg.loc[cg["Condition"] == "Full Cargo"].iloc[0]
    waterplane_area, lcf, ixx, iyy = ship_system.HS.calc_WaterPlane(
        hull["d"]
    )

    return pd.DataFrame(
        [
            {
                "LOA_m": hull["L"],
                "Beam_m": hull["B"],
                "Depth_m": hull["D"],
                "Design_Draft_m": hull["d"],
                "Displacement_Volume_m3": hull["Vol_Disp"],
                "Block_Coefficient_Full_Beam": ship_system.Weights.Cb,
                "Full_Cargo_Displacement_t": full_cargo[
                    "Displacement_t"
                ],
                "Full_Cargo_LCG_m": full_cargo["LCG"],
                "Full_Cargo_VCG_m": full_cargo["VCG"],
                "Waterplane_Area_m2": waterplane_area,
                "LCF_m": lcf,
                "Waterplane_Ixx_m4": ixx,
                "Waterplane_Iyy_m4": iyy,
                "Forward_Perpendicular_X_m": hull[
                    "Forward_Perpendicular_X_m"
                ],
                "Aft_Perpendicular_X_m": hull["Aft_Perpendicular_X_m"],
                "Length_Between_Perpendiculars_m": hull[
                    "Length_Between_Perpendiculars_m"
                ],
                "Integration_Method": "Composite trapezoidal rule",
            }
        ]
    )


def _save_per_design_numerical_arrays(
    ship_system,
    output_path: Path,
) -> None:
    """Save one system's load and flooding curves as numeric arrays."""

    arrays = {
        "x_m": np.asarray(ship_system.Weights.X, dtype=float),
        "distributed_load_n_per_m": np.asarray(
            ship_system.Weights.Distributed_Load,
            dtype=float,
        ),
        "buoyancy_load_n_per_m": np.asarray(
            ship_system.Weights.buoyancy_dist,
            dtype=float,
        ),
        "shear_force_n": np.asarray(
            ship_system.Weights.Shear_Force,
            dtype=float,
        ),
        "bending_moment_nm": np.asarray(
            ship_system.Weights.Bending_Moment,
            dtype=float,
        ),
    }

    if not ship_system.FloodableLengthCurve.empty:
        flood_curve = ship_system.FloodableLengthCurve
        arrays.update(
            {
                "flood_x_position_m": flood_curve[
                    "X Position"
                ].to_numpy(dtype=float),
                "flood_x_index": flood_curve[
                    "X Index"
                ].to_numpy(dtype=int),
                "flood_permeability": flood_curve[
                    "Permeability"
                ].to_numpy(dtype=float),
                "floodable_length_m": flood_curve[
                    "Floodable Length"
                ].to_numpy(dtype=float),
                "single_compartment_margin_m": flood_curve[
                    "Single-Compartment Margin"
                ].to_numpy(dtype=float),
                "two_compartment_margin_m": flood_curve[
                    "Two-Compartment Margin"
                ].to_numpy(dtype=float),
            }
        )

    _write_npz_atomic(output_path, arrays)


def _load_optional_json_mapping(path: str | Path | None):
    """Load one optional, explicitly supplied dataset-scenario mapping.

    The same mapping object is passed to every system in a shard.  It remains
    outside the fixed 189-value design vector so scenario studies can change
    preliminary assumptions without changing the learned parameter schema.
    """

    mapping, _ = _load_optional_json_mapping_with_provenance(path)

    return mapping


def _evaluation_invocation_payload(
    input_csv: str | Path,
    output_root: str | Path,
    run_flooding_diagnostic: bool,
    flooding_step: int,
    flooding_permeabilities,
    include_legacy_constraints: bool,
    write_per_design_review_files: bool,
    scenario_provenance: dict,
    save_mode: str = "all",
) -> dict:
    """Describe the exact resolved evaluator configuration for one shard."""

    return {
        "schema_version": 1,
        "stage": "evaluation",
        "entry_point": _file_provenance(Path(__file__)),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
        },
        "input_system_shard": _file_provenance(input_csv),
        "configuration": {
            "output_root": str(Path(output_root).resolve()),
            "run_flooding_diagnostic": bool(run_flooding_diagnostic),
            "flooding_step": int(flooding_step),
            "flooding_permeabilities": [
                float(value)
                for value in flooding_permeabilities
            ],
            "include_legacy_constraints": bool(
                include_legacy_constraints
            ),
            "write_per_design_review_files": bool(
                write_per_design_review_files
            ),
            "save_mode": save_mode,
        },
        "optional_scenarios": scenario_provenance,
    }


def _individual_constraint_path(
    output_root: Path,
    structure_index: int,
    system_index: int,
) -> Path:
    """Return the stable one-file-per-system constraint path.

    Files are grouped by structure so the 100,000-design production dataset
    does not place every directory entry in one filesystem directory.
    """

    return (
        output_root
        / "evaluations"
        / "constraints_v1p1"
        / "individual"
        / f"Structure_{structure_index:06d}"
        / f"Ship_{system_index:06d}_Constraints_Evaluations.csv"
    )


def _failure_constraint_table(
    identifiers: dict,
    stage: str,
    error: Exception,
) -> pd.DataFrame:
    """Represent a failed design using the same eight-column CSV contract.

    The filename and row name both retain the system index.  The complete
    traceback remains in the separate failure shard, while this small file
    records the stage and immediate cause where every expected design's
    constraint file would normally be found.
    """

    if isinstance(error, PhysicalConstraintFailure):
        # New V1.1 physical failures carry every normal constraint row that
        # was evaluated before the dependent calculation stopped.  Return the
        # complete partial table so passing prerequisites remain available to
        # the dataset instead of saving only the final failed predicate.
        if error.constraint_rows is not None:
            return error.constraint_rows[CONSTRAINT_COLUMNS].copy()

        if error.comparison == "minimum":
            residual = error.threshold_value - error.constraint_value
        elif error.comparison == "maximum":
            residual = error.constraint_value - error.threshold_value
        else:
            raise ValueError(
                "Structured physical failures require a minimum or maximum "
                "comparison."
            )

        row = {
            "Class": error.result_class,
            "Name": error.name,
            "Requirement": error.requirement,
            "Threshold Value": error.threshold_value,
            "Constraint Value": error.constraint_value,
            "Units": error.units,
            "Constraint Eval": residual,
            "Constraint Bit": False,
        }
    else:
        row = {
            "Class": "Evaluation Failure",
            "Name": (
                f"System {identifiers['Index']} evaluation failure: "
                f"{type(error).__name__}"
            ),
            "Requirement": f"Complete GenISys evaluation; failed at {stage}",
            "Threshold Value": np.nan,
            "Constraint Value": str(error),
            "Units": "",
            "Constraint Eval": 1.0,
            "Constraint Bit": False,
        }

    return pd.DataFrame([row], columns=CONSTRAINT_COLUMNS)


def _weight_value(weights: pd.DataFrame, name: str) -> float:
    """Return the total mass for one named weight row, or NaN if absent."""

    matches = weights.loc[weights["Name"] == name, "Total_Weight_kg"]

    if matches.empty:
        return np.nan

    return float(matches.sum())


def _audited_value(audited: pd.DataFrame, name: str) -> float:
    """Return one audit-qualified scalar value, or NaN if not emitted."""

    matches = audited.loc[audited["Name"] == name, "Constraint Value"]

    if matches.empty:
        return np.nan

    return float(matches.iloc[0])


def _prepend_identifiers(table: pd.DataFrame, identifiers: dict) -> pd.DataFrame:
    """Return a copy of one detail table with stable dataset identifiers."""

    result = table.copy()

    for column in reversed(IDENTIFIER_COLUMNS):
        result.insert(0, column, identifiers[column])

    return result


def _build_structure_property_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Preserve the legacy structural-evaluation table in long form."""

    properties = ship_system.SC.Struct_Dict[
        ["Parameter", "Value", "Units"]
    ].copy()
    properties.insert(0, "Property_Index", np.arange(len(properties)))

    return _prepend_identifiers(properties, identifiers)


def _build_weight_component_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Preserve weight/CG detail and prove each distribution's mass closure.

    The legacy weight CSV embedded NumPy arrays as quoted text.  V1.1 keeps the
    useful scalar component table and adds independently calculated integral
    residuals; the common ship load arrays remain in the compressed NPZ shard.
    """

    weights = ship_system.Weights.Weights.copy().reset_index(drop=True)
    x_m = np.asarray(ship_system.Weights.X, dtype=float)
    integrated_masses = []

    # Stable component types preserve semantic identity without depending on
    # row order in an individual CSV.  The mapping describes existing weight
    # rows only; it does not create any new component or mass assumption.
    def component_type(name):
        name = str(name)

        if name.startswith("Transverse_Bulkhead"):
            return "transverse_bulkhead_structure"

        return {
            "Structure": "longitudinal_structure",
            "Outfit": "outfit",
            "Machinery": "machinery",
            "Hatch_Cover": "hatch_cover",
            "Simple Deckhouse": "simple_deckhouse",
            "Lightship Growth Margin": "lightship_growth_margin",
            "Misc_DWT": "miscellaneous_deadweight",
            "Cargo_Hold": "cargo_hold_load",
            "Ballast_Tank": "ballast_tank_load",
            "Fuel_Tank": "fuel_tank_load",
        }.get(name, "other_existing_component")

    weights["Component_Type"] = weights["Name"].map(component_type)
    type_sequence = weights.groupby("Component_Type").cumcount()
    weights["Component_ID"] = [
        f"{component_type_name}_{int(sequence):03d}"
        for component_type_name, sequence in zip(
            weights["Component_Type"],
            type_sequence,
        )
    ]
    weights["LCG_m"] = weights["LCG"].astype(float)
    weights["VCG_m"] = weights["VCG"].astype(float)
    weights["TCG_m"] = 0.0
    weights["Transverse_Center_Basis"] = (
        "HDF-ASM-013 symmetric intact condition"
    )

    for distribution in weights["Dist_Weight"]:
        integrated_masses.append(
            float(np.trapezoid(np.asarray(distribution, dtype=float), x=x_m))
        )

    weights.insert(0, "Component_Index", np.arange(len(weights)))
    weights["Distributed_Mass_Integral_kg"] = integrated_masses
    weights["Distributed_Mass_Residual_kg"] = (
        weights["Distributed_Mass_Integral_kg"]
        - weights["Total_Weight_kg"].astype(float)
    )
    denominator = np.maximum(
        np.abs(weights["Total_Weight_kg"].to_numpy(dtype=float)),
        1.0,
    )
    weights["Distributed_Mass_Residual_Fraction"] = (
        weights["Distributed_Mass_Residual_kg"].to_numpy(dtype=float)
        / denominator
    )
    weights["Longitudinal_Distribution_Status"] = (
        "trapezoidal_mass_and_first_moment_checked"
    )
    weights["Input_Status"] = "active_existing_genisys_weight_model"
    weights.loc[
        weights["Component_Type"] == "hatch_cover",
        "Input_Status",
    ] = "active_approved_hatch_cover_assumptions"

    # Optional items remain in the same component ledger rather than creating
    # a parallel mass table.  When they are active, retain every explicit
    # scenario input required to reproduce the calculation.  Inactive status
    # is reported in the scalar table because no component row exists.
    optional_numeric_columns = [
        "Geometry_Length_m",
        "Geometry_Breadth_m",
        "Geometry_Height_m",
        "Geometry_Base_X_m",
        "Geometry_Base_Z_m",
        "Equivalent_Areal_Mass_kg_m2",
        "Lightship_Margin_Fraction",
    ]

    for column in optional_numeric_columns:
        weights[column] = np.nan

    deckhouse = getattr(ship_system, "DeckhouseComponents", pd.DataFrame())

    if not deckhouse.empty:
        source = deckhouse.iloc[0]
        target = weights["Component_Type"] == "simple_deckhouse"
        weights.loc[target, "Geometry_Length_m"] = float(source["Length_m"])
        weights.loc[target, "Geometry_Breadth_m"] = float(source["Breadth_m"])
        weights.loc[target, "Geometry_Height_m"] = float(source["Height_m"])
        weights.loc[target, "Geometry_Base_X_m"] = float(source["Base_X_m"])
        weights.loc[target, "Geometry_Base_Z_m"] = float(source["Base_Z_m"])
        weights.loc[target, "Equivalent_Areal_Mass_kg_m2"] = float(
            source["Equivalent_Areal_Mass_kg_m2"]
        )
        weights.loc[target, "Input_Status"] = (
            "active_explicit_input_not_spatially_validated"
        )

    margin = getattr(
        ship_system,
        "LightshipMarginComponent",
        pd.DataFrame(),
    )

    if not margin.empty:
        target = weights["Component_Type"] == "lightship_growth_margin"
        weights.loc[target, "Lightship_Margin_Fraction"] = float(
            margin.iloc[0]["Margin_Fraction"]
        )
        weights.loc[target, "Input_Status"] = (
            "active_explicit_fraction_and_mass_center"
        )

    # Array text is deliberately excluded.  It is neither numerically robust
    # nor storage-efficient in a 100,000-system CSV dataset.
    weights = weights.drop(columns=["Dist_Weight"])

    return _prepend_identifiers(weights, identifiers)


def _build_compartment_load_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Normalize cargo, ballast, and fuel state into one compartment table."""

    tables = []
    cargo_system = ship_system.CargoHolds
    x_bulkheads = np.asarray(cargo_system.X_bulkheads, dtype=float)

    cargo = ship_system.CargoHolds.CargoHolds.copy().reset_index(drop=True)
    if not cargo.empty:
        cargo_table = pd.DataFrame(
            {
                "Compartment_ID": [
                    f"cargo_hold_{int(number):03d}"
                    for number in cargo["Hold_Number"]
                ],
                "Compartment_Type": "cargo_hold",
                "Compartment_Number": cargo["Hold_Number"].astype(int),
                "X_Start_m": [
                    x_bulkheads[int(number)]
                    for number in cargo["Hold_Number"]
                ],
                "X_End_m": [
                    x_bulkheads[int(number) + 1]
                    for number in cargo["Hold_Number"]
                ],
                "Total_Capacity_m3": cargo["Total_Hold_Volume_m3"],
                "Loaded_Volume_m3": cargo["Cargo_Volume_Filled_m3"],
                "Fill_Height_m": cargo["Cargo_Fill_Height_m"],
                "Liquid_or_Cargo_Density_kg_per_m3": (
                    ship_system.CargoHolds.rho_cargo
                ),
                "Total_Weight_kg": cargo["Total_Weight_kg"],
                "VCG_m": cargo["VCG"],
                "LCG_m": cargo["LCG"],
            }
        )
        tables.append(cargo_table)

    ballast = ship_system.CargoHolds.BallastTanks.copy().reset_index(drop=True)
    if not ballast.empty:
        ballast_table = pd.DataFrame(
            {
                "Compartment_ID": [
                    f"ballast_region_{int(number):03d}"
                    for number in ballast["Tank_Number"]
                ],
                "Compartment_Type": "ballast_tank",
                "Compartment_Number": ballast["Tank_Number"].astype(int),
                "X_Start_m": [
                    x_bulkheads[int(number)]
                    for number in ballast["Tank_Number"]
                ],
                "X_End_m": [
                    x_bulkheads[int(number) + 1]
                    for number in ballast["Tank_Number"]
                ],
                "Total_Capacity_m3": ballast["Total_Tank_Volume_m3"],
                "Loaded_Volume_m3": ballast[
                    "Ballast_Volume_toReplaceCargo_m3"
                ],
                "Fill_Height_m": ballast["Ballast_Fill_Height_m"],
                "Liquid_or_Cargo_Density_kg_per_m3": (
                    ship_system.CargoHolds.rho
                ),
                "Total_Weight_kg": ballast["Total_Weight_kg"],
                "VCG_m": ballast["VCG"],
                "LCG_m": ballast["LCG"],
            }
        )
        tables.append(ballast_table)

    fuel = ship_system.CargoHolds.Fuel_Tank.copy().reset_index(drop=True)
    if not fuel.empty:
        fuel_table = pd.DataFrame(
            {
                "Compartment_ID": ["fuel_tank_000"],
                "Compartment_Type": "fuel_tank",
                "Compartment_Number": fuel["Tank_Number"].astype(int),
                "X_Start_m": [
                    x_bulkheads[int(cargo_system.FuelTank_Assignment[0])]
                ],
                "X_End_m": [
                    x_bulkheads[int(cargo_system.FuelTank_Assignment[1])]
                ],
                "Total_Capacity_m3": fuel["Total_Tank_Volume_m3"],
                "Loaded_Volume_m3": fuel["Fuel_Volume_Filled_m3"],
                "Fill_Height_m": fuel["Fuel_Fill_Height_m"],
                "Liquid_or_Cargo_Density_kg_per_m3": (
                    ship_system.CargoHolds.rho_fuel
                ),
                "Total_Weight_kg": fuel["Total_Weight_kg"],
                "VCG_m": fuel["VCG"],
                "LCG_m": fuel["LCG"],
            }
        )
        tables.append(fuel_table)

    result = pd.concat(tables, ignore_index=True)
    result.insert(0, "Compartment_State_Index", np.arange(len(result)))
    result["Capacity_Margin_m3"] = (
        result["Total_Capacity_m3"].astype(float)
        - result["Loaded_Volume_m3"].astype(float)
    )
    result["Loaded_Fraction"] = (
        result["Loaded_Volume_m3"].astype(float)
        / result["Total_Capacity_m3"].astype(float)
    )
    usable_volume_factor_by_type = {
        "cargo_hold": float(cargo_system.cargo_hold_permeability),
        "ballast_tank": float(cargo_system.ballast_tank_permeability),
        # The current single fuel-region geometry has no additional usable-
        # volume deduction.  This is a model factor, not a flooding value.
        "fuel_tank": 1.0,
    }
    result["Usable_Volume_Factor"] = result["Compartment_Type"].map(
        usable_volume_factor_by_type
    )
    result["Usable_Volume_Factor_Basis"] = (
        "HDF-ASM-014 preliminary capacity factor; not flooding permeability"
    )
    result["TCG_m"] = 0.0
    result["Transverse_Center_Basis"] = (
        "HDF-ASM-013 symmetric intact condition"
    )
    capacity_scale_m3 = np.maximum(
        1.0,
        np.abs(result["Total_Capacity_m3"].to_numpy(dtype=float)),
    )
    result["Capacity_Constraint_Passed"] = (
        result["Capacity_Margin_m3"].to_numpy(dtype=float)
        >= -64.0*np.finfo(float).eps*capacity_scale_m3
    )
    result["Regulatory_Flooding_Permeability"] = np.nan
    result["Regulatory_Flooding_Permeability_Status"] = (
        "not_assigned_requires_rule_space_and_condition"
    )
    result["Free_Surface_Status"] = np.where(
        result["Compartment_Type"].isin(["ballast_tank", "fuel_tank"]),
        "not_evaluated_missing_transverse_tank_groups",
        "not_applicable_to_solid_cargo_model",
    )
    result["Geometry_Completeness_Status"] = (
        "longitudinal_bounds_and_aggregate_volume_only"
    )
    result["Missing_Regulatory_Geometry"] = (
        "transverse/vertical watertight bounds; connectivity; openings"
    )

    return _prepend_identifiers(result, identifiers)


def _build_bulkhead_location_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Publish the generated longitudinal subdivision with stable labels."""

    cargo_holds = ship_system.CargoHolds
    number_of_holds = len(cargo_holds.CargoHold_Assignments) - 1
    grid_indices = np.asarray(cargo_holds.Bulkhead_Locations, dtype=int)
    x_locations = np.asarray(cargo_holds.X_bulkheads, dtype=float)
    boundary_types = []

    for bulkhead_number in range(len(grid_indices)):
        if bulkhead_number == 0:
            boundary_types.append("collision_bulkhead")
        elif bulkhead_number < number_of_holds:
            boundary_types.append("cargo_hold_bulkhead")
        elif bulkhead_number == number_of_holds:
            boundary_types.append("fuel_tank_forward_bulkhead")
        elif bulkhead_number == number_of_holds + 1:
            boundary_types.append("engine_room_forward_bulkhead")
        elif bulkhead_number == number_of_holds + 2:
            boundary_types.append("engine_room_aft_bulkhead")
        else:
            boundary_types.append("aft_peak_bulkhead")

    table = pd.DataFrame(
        {
            "Bulkhead_Number": np.arange(len(grid_indices)),
            "Boundary_Type": boundary_types,
            "Hydrostatic_Grid_Index": grid_indices,
            "X_m": x_locations,
        }
    )

    return _prepend_identifiers(table, identifiers)


def _build_compartment_condition_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Expand aggregate compartment loads into the two full-load conditions.

    This table does not invent transverse tank geometry.  It simply applies
    the already approved cargo, ballast, and fuel fractions to each generated
    longitudinal region, preserving the remaining geometry blocker alongside
    every state.
    """

    base = _build_compartment_load_table(
        identifiers,
        ship_system,
    )
    state_tables = []

    for condition_index, condition in (
        ship_system.LoadingConditions.reset_index(drop=True).iterrows()
    ):
        fractions = {
            "cargo_hold": float(condition["Cargo_Fraction"]),
            "ballast_tank": float(condition["Ballast_Fraction"]),
            "fuel_tank": float(condition["Fuel_Fraction"]),
        }
        state = base.copy()
        applied = state["Compartment_Type"].map(fractions).astype(float)
        state.insert(6, "Loading_Condition_Index", int(condition_index))
        state.insert(
            7,
            "Loading_Condition_ID",
            str(condition["Loading_Condition_ID"]),
        )
        state.insert(
            8,
            "Loading_Condition_Name",
            str(condition["Loading_Condition_Name"]),
        )
        state["Base_Loaded_Volume_m3"] = state["Loaded_Volume_m3"]
        state["Base_Total_Weight_kg"] = state["Total_Weight_kg"]
        state["Applied_Fraction"] = applied
        state["Loaded_Volume_m3"] = (
            state["Base_Loaded_Volume_m3"].astype(float)*applied
        )
        state["Total_Weight_kg"] = (
            state["Base_Total_Weight_kg"].astype(float)*applied
        )

        # The fuel region is recalculated geometrically at 100-percent and
        # 10-percent fill.  Apply its condition-specific fill surface and
        # centers instead of retaining the departure geometry after scaling
        # only the scalar mass.
        condition_fuel = ship_system.LoadingConditionFuelTankStates.loc[
            ship_system.LoadingConditionFuelTankStates[
                "Loading_Condition_ID"
            ].astype(str)
            == str(condition["Loading_Condition_ID"])
        ]
        fuel_rows = state.index[
            state["Compartment_Type"] == "fuel_tank"
        ]

        for fuel_row, (_, fuel_state) in zip(
            fuel_rows,
            condition_fuel.iterrows(),
        ):
            state.at[fuel_row, "Loaded_Volume_m3"] = float(
                fuel_state["Fuel_Volume_Filled_m3"]
            )
            state.at[fuel_row, "Fill_Height_m"] = float(
                fuel_state["Fuel_Fill_Height_m"]
            )
            state.at[fuel_row, "Total_Weight_kg"] = float(
                fuel_state["Total_Weight_kg"]
            )
            state.at[fuel_row, "VCG_m"] = float(fuel_state["VCG"])
            state.at[fuel_row, "LCG_m"] = float(fuel_state["LCG"])

        state["Loaded_Fraction"] = (
            state["Loaded_Volume_m3"].astype(float)
            / state["Total_Capacity_m3"].astype(float)
        )
        state["Capacity_Margin_m3"] = (
            state["Total_Capacity_m3"].astype(float)
            - state["Loaded_Volume_m3"].astype(float)
        )
        state_tables.append(state)

    return pd.concat(state_tables, ignore_index=True)


def _build_hatch_cover_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Publish the preliminary hatch-cover mass and geometry assumptions."""

    covers = ship_system.HatchCoverComponents.copy().reset_index(drop=True)

    if covers.empty:
        return _prepend_identifiers(covers, identifiers)

    x_m = np.asarray(ship_system.Weights.X, dtype=float)
    distributed_mass_kg = [
        float(np.trapezoid(np.asarray(row, dtype=float), x=x_m))
        for row in covers["Dist_Weight"]
    ]
    covers.insert(0, "Hatch_Cover_Index", np.arange(len(covers)))
    covers["Distributed_Mass_Integral_kg"] = distributed_mass_kg
    covers["Distributed_Mass_Residual_kg"] = (
        covers["Distributed_Mass_Integral_kg"]
        - covers["Total_Weight_kg"].astype(float)
    )
    covers = covers.drop(columns=["Dist_Weight"])

    return _prepend_identifiers(covers, identifiers)


def _build_loading_condition_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Publish the mass and CG summary for every named loading condition."""

    conditions = ship_system.LoadingConditions.copy().reset_index(drop=True)
    conditions.insert(0, "Loading_Condition_Index", np.arange(len(conditions)))

    return _prepend_identifiers(conditions, identifiers)


def _build_loading_condition_component_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Publish each component's condition-specific applied mass fraction."""

    return _prepend_identifiers(
        ship_system.LoadingConditionComponentStates,
        identifiers,
    )


def _build_upright_stability_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Publish per-condition equilibrium and upright stability foundations."""

    evaluations = (
        ship_system.UprightLoadingConditionEvaluations.copy()
        .reset_index(drop=True)
    )
    evaluations.insert(
        0,
        "Loading_Condition_Index",
        np.arange(len(evaluations)),
    )

    return _prepend_identifiers(evaluations, identifiers)


def _build_named_condition_load_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Publish still-water load extrema and closure for both conditions."""

    loads = (
        ship_system.NamedConditionLongitudinalLoadSummary.copy()
        .reset_index(drop=True)
    )
    loads.insert(
        0,
        "Loading_Condition_Index",
        np.arange(len(loads)),
    )

    return _prepend_identifiers(loads, identifiers)


def _build_load_closure_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Publish independent mass, force, moment, shear, and BM residuals."""

    diagnostics = pd.DataFrame([ship_system.Weights.LoadClosureDiagnostics])

    return _prepend_identifiers(diagnostics, identifiers)


def _build_economic_performance_table(
    identifiers: dict,
    ship_system: ShipSystemDesign,
) -> pd.DataFrame:
    """Publish the resolved scenario and complete annual-cost breakdown."""

    return _prepend_identifiers(
        ship_system.EconomicPerformanceSummary,
        identifiers,
    )


def _build_scalar_result(
    identifiers: dict,
    ship_system: ShipSystemDesign,
    ship_cost: float,
    annual_cargo: float,
    required_freight_rate: float,
    evaluation_seconds: float,
) -> dict:
    """Build the scalar-only performance record for one completed system."""

    hull = ship_system.Hull_Dict
    weights = ship_system.Weights
    audited = ship_system.AuditedConstraints
    expanded_constraints = ship_system.ConstraintsV1p1
    cargo_optimization = ship_system.CargoOptimizationSummary.iloc[0]
    load_closure = ship_system.Weights.LoadClosureDiagnostics
    full_cargo = weights.Ship_CG.loc[
        weights.Ship_CG["Condition"] == "Full Cargo"
    ].iloc[0]
    waterplane_area, lcf, ixx, iyy = ship_system.HS.calc_WaterPlane(hull["d"])

    # Status-aware audit tables may contain calculated requirement metrics
    # whose regulatory comparison is not yet implementable.  Only rows with
    # an actual Boolean result can enter feasibility counts.  In particular,
    # HDF-033 alarm-height metrics must not become implicit True values merely
    # because ``bool(np.nan)`` is true in Python.  This covers both HDF-030
    # applicability geometry and HDF-033 alarm requirements.
    # A not-applicable row is an explicit pass under the current dataset
    # contract.  Partial calculations and diagnostics remain blank and are
    # still excluded because they have no Boolean result.
    applicable = audited["Constraint Bit"].notna()
    # Preserve every long-form compatibility alias, but use one reviewed
    # canonical identity per physical equation/member pair in scalar totals.
    # Logical-all aggregation makes an aliased constraint fail if any retained
    # implementation row fails.
    applicable_bits = canonical_constraint_bits(
        audited.loc[applicable],
        audited=True,
    )
    expanded_bits = canonical_constraint_bits(expanded_constraints)

    row = {
        **identifiers,
        "Evaluation_Status": "completed",
        "Evaluation_Seconds": evaluation_seconds,
        "LOA_m": float(hull["L"]),
        "Beam_m": float(hull["B"]),
        "Depth_m": float(hull["D"]),
        "Draft_m": float(hull["d"]),
        "Froude_Number": float(hull["Fn"]),
        "Speed_m_per_s": float(hull["U_ship"]),
        "Speed_knots": float(hull["U_knots"]),
        "Voyage_Range_nmi": float(hull["Voyange_Range"]),
        "Days_at_Sea": float(hull["Days_at_Sea"]),
        "Waterline_Length_m": float(hull["Waterline_Length"]),
        "Wetted_Surface_Area_m2": float(hull["Wet_Surface_Area"]),
        "Displacement_Volume_m3": float(hull["Vol_Disp"]),
        "LCB_m": float(hull["LCB"]),
        "VCB_m": float(hull["VCB"]),
        "Block_Coefficient": float(hull["Cb"]),
        "Midship_Coefficient": float(hull["Cm"]),
        "Prismatic_Coefficient": float(hull["Cp"]),
        "Reynolds_Number": float(hull["Re"]),
        "Total_Resistance_N": float(hull["Total_Resistance"]),
        "Appendage_Area_m2": float(hull["Appendage_Area_m2"]),
        "Appendage_Resistance_Factor": float(
            hull["Appendage_Resistance_Factor"]
        ),
        "Waterline_Entrance_Half_Angle_deg": float(
            hull["Waterline_Entrance_Half_Angle_deg"]
        ),
        "Bulb_Transverse_Area_m2": float(
            hull["Bulb_Transverse_Area_m2"]
        ),
        "Engine_Power_kW": float(hull["Engine_Power"]),
        "Maximum_Continuous_Rating_Fraction": float(
            hull["Maximum_Continuous_Rating_Fraction"]
        ),
        "Aggregate_Propulsive_Efficiency": float(
            hull["Aggregate_Propulsive_Efficiency"]
        ),
        "Powering_Input_Basis": hull["Powering_Input_Basis"],
        "Explicit_Powering_Input_Fields": hull[
            "Explicit_Powering_Input_Fields"
        ],
        "Propulsion_Design_Status": (
            "preliminary_aggregate_correlation_no_selected_propeller_or_engine"
        ),
        "Main_Engine_SFOC_kg_per_kWh": float(
            hull["Main_Engine_SFOC_kg_per_kWh"]
        ),
        "Auxiliary_Fuel_t_per_day": float(
            hull["Auxiliary_Fuel_t_per_day"]
        ),
        "Fuel_Reserve_Days": float(hull["Fuel_Reserve_Days"]),
        # MARPOL Annex VI regulation 24 tables 1 and 2.  These are deliberately
        # labelled preliminary metrics: GenISys deadweight is derived at the
        # sampled design draft, not copied from a verified EEDI Technical File
        # or an assigned summer load line.  HDF-ASM-004 selects the current
        # phase-3 new-build row, but no attained-EEDI pass is inferred.
        "Preliminary_EEDI_Deadweight_Proxy_t": _audited_value(
            ship_system.PoweringConstraintsV1p1,
            "Preliminary EEDI deadweight proxy at evaluated draft",
        ),
        "Preliminary_Bulk_Carrier_Reference_EEDI_gCO2_per_t_nm": (
            _audited_value(
                ship_system.PoweringConstraintsV1p1,
                "Preliminary bulk-carrier EEDI reference line",
            )
        ),
        "Preliminary_Bulk_Carrier_EEDI_Phase": _audited_value(
            ship_system.PoweringConstraintsV1p1,
            "Preliminary bulk-carrier EEDI phase",
        ),
        "Preliminary_Bulk_Carrier_EEDI_Reduction_Factor_percent": (
            _audited_value(
                ship_system.PoweringConstraintsV1p1,
                "Preliminary bulk-carrier EEDI reduction factor",
            )
        ),
        "Preliminary_Required_Bulk_Carrier_EEDI_Proxy_gCO2_per_t_nm": (
            _audited_value(
                ship_system.PoweringConstraintsV1p1,
                "Preliminary required bulk-carrier EEDI proxy",
            )
        ),
        "Minimum_Engine_Room_Length_m": float(
            hull["Min_Engine_Room_Length"]
        ),
        "Required_Fuel_Weight_kg": float(hull["Fuel_Weight_kg"]),
        "Structure_Weight_kg": float(weights.structure_weight),
        "Outfit_Weight_kg": float(weights.outfit_weight),
        "Machinery_Weight_kg": _weight_value(weights.Weights, "Machinery"),
        "Miscellaneous_Deadweight_kg": _weight_value(
            weights.Weights,
            "Misc_DWT",
        ),
        "Deadweight_kg": float(weights.deadweight),
        "Cargo_Weight_kg": float(weights.cargo_weight),
        "Hatch_Cover_Count": int(len(ship_system.HatchCoverComponents)),
        "Hatch_Cover_Weight_kg": float(
            ship_system.HatchCoverComponents["Total_Weight_kg"].sum()
        ),
        "Hatch_Cover_Projected_Lateral_Area_m2": float(
            ship_system.HatchCoverComponents[
                "Projected_Lateral_Area_m2"
            ].sum()
        ),
        "Hatch_Cover_CAD_Proxy_Status": (
            "approved_geometry_available_for_on_demand_serialization"
        ),
        "Deckhouse_Model_Status": (
            "active_explicit_inputs_not_spatially_validated"
            if not ship_system.DeckhouseComponents.empty
            else "inactive_missing_dimensions_placement_and_areal_mass"
        ),
        "Deckhouse_Weight_kg": _weight_value(
            weights.Weights,
            "Simple Deckhouse",
        ),
        "Lightship_Growth_Margin_Status": (
            "active_explicit_fraction_and_mass_center"
            if not ship_system.LightshipMarginComponent.empty
            else "inactive_missing_fraction_and_mass_center"
        ),
        "Lightship_Growth_Margin_kg": _weight_value(
            weights.Weights,
            "Lightship Growth Margin",
        ),
        "Modeled_Lightship_Weight_kg": float(
            weights.modeled_lightship_weight
        ),
        "Displacement_Mass_kg": float(weights.rho*weights.V_disp),
        "Lightship_Displacement_Margin_kg": float(weights.deadweight),
        "Available_Cargo_Displacement_Fraction": float(
            weights.cargo_weight/(weights.rho*weights.V_disp)
        ),
        "Full_Cargo_Displacement_t": float(full_cargo["Displacement_t"]),
        "Full_Cargo_LCG_m": float(full_cargo["LCG"]),
        "Full_Cargo_VCG_m": float(full_cargo["VCG"]),
        "Full_Cargo_YCG_m": 0.0,
        "Cargo_Optimization_Success": bool(
            cargo_optimization["Solver_Success"]
        ),
        "Cargo_Optimization_Iterations": int(
            cargo_optimization["Iterations"]
        ),
        "Cargo_Optimization_Solver_Phase": cargo_optimization[
            "Solver_Phase"
        ],
        "Cargo_Multivariable_Optimizer_Used": bool(
            cargo_optimization["Multivariable_Optimizer_Used"]
        ),
        "Cargo_Multivariable_Optimizer_Success": bool(
            cargo_optimization["Multivariable_Optimizer_Success"]
        ),
        "Cargo_Multivariable_Optimizer_Iterations": int(
            cargo_optimization["Multivariable_Optimizer_Iterations"]
        ),
        "Cargo_Multivariable_Physical_Evaluation_Count": int(
            cargo_optimization[
                "Multivariable_Physical_Evaluation_Count"
            ]
        ),
        "Cargo_Allocation_Valid_Equilibrium_Evaluation_Count": int(
            cargo_optimization["Valid_Equilibrium_Evaluation_Count"]
        ),
        "Cargo_Allocation_Invalid_Equilibrium_Evaluation_Count": int(
            cargo_optimization["Invalid_Equilibrium_Evaluation_Count"]
        ),
        "Cargo_Allocation_Interpolation_Alpha": float(
            cargo_optimization["Interpolation_Alpha"]
        ),
        "Required_Cargo_LCG_m": float(
            cargo_optimization["Required_Cargo_LCG_m"]
        ),
        "Forwardmost_Cargo_LCG_m": float(
            cargo_optimization["Forwardmost_Cargo_LCG_m"]
        ),
        "Aftmost_Cargo_LCG_m": float(
            cargo_optimization["Aftmost_Cargo_LCG_m"]
        ),
        "Achieved_Geometric_Cargo_LCG_m": float(
            cargo_optimization["Achieved_Geometric_Cargo_LCG_m"]
        ),
        "Cargo_LCG_Deficit_m": float(
            cargo_optimization["Cargo_LCG_Deficit_m"]
        ),
        "Cargo_LCG_Target_Reachable": bool(
            cargo_optimization["Cargo_LCG_Target_Reachable"]
        ),
        "Cargo_Allocation_Alpha_Unbounded": float(
            cargo_optimization["Cargo_Allocation_Alpha_Unbounded"]
        ),
        "Cargo_Trim_Sign_Change_Bracket_Count": int(
            cargo_optimization["Trim_Sign_Change_Bracket_Count"]
        ),
        "Equilibrium_Draft_m": float(
            cargo_optimization["Equilibrium_Draft_m"]
        ),
        "Draft_Forward_m": float(cargo_optimization["Draft_Forward_m"]),
        "Draft_Aft_m": float(cargo_optimization["Draft_Aft_m"]),
        "Signed_Trim_Aft_Minus_Forward_m": float(
            cargo_optimization["Signed_Trim_Aft_Minus_Forward_m"]
        ),
        "Absolute_Trim_m": float(cargo_optimization["Absolute_Trim_m"]),
        "Trim_Limit_One_Percent_Draft_m": float(
            cargo_optimization["Trim_Limit_One_Percent_Draft_m"]
        ),
        "Trim_Margin_m": float(cargo_optimization["Trim_Margin_m"]),
        "Trim_Constraint_Passed": bool(
            cargo_optimization["Trim_Constraint_Passed"]
        ),
        "Minimum_Hold_Capacity_Margin_Fraction": float(
            cargo_optimization["Minimum_Hold_Capacity_Margin_Fraction"]
        ),
        "Available_Cargo_Hold_Capacity_kg": float(
            cargo_optimization["Available_Cargo_Hold_Capacity_kg"]
        ),
        "Total_Cargo_Hold_Capacity_Margin_kg": float(
            cargo_optimization["Total_Cargo_Hold_Capacity_Margin_kg"]
        ),
        "Hold_Capacity_Constraints_Passed": bool(
            cargo_optimization["Hold_Capacity_Constraints_Passed"]
        ),
        "Waterplane_Area_m2": float(waterplane_area),
        "LCF_m": float(lcf),
        "Waterplane_Ixx_m4": float(ixx),
        "Waterplane_Iyy_m4": float(iyy),
        "Forward_Perpendicular_X_m": float(
            hull["Forward_Perpendicular_X_m"]
        ),
        "Aft_Perpendicular_X_m": float(hull["Aft_Perpendicular_X_m"]),
        "Length_Between_Perpendiculars_m": float(
            hull["Length_Between_Perpendiculars_m"]
        ),
        "Convention_GT_Lower_Bound": _audited_value(
            audited,
            "46 CFR Part 174 dry-cargo tonnage gate",
        ),
        "Maximum_Positive_SWBM_kNm": float(
            ship_system.distributed_loading_dict["Mpsw"]
        ),
        "Maximum_Negative_SWBM_kNm": float(
            ship_system.distributed_loading_dict["Mnsw"]
        ),
        "Maximum_Positive_Shear_kN": float(
            ship_system.distributed_loading_dict["Fpsw"]
        ),
        "Maximum_Negative_Shear_kN": float(
            ship_system.distributed_loading_dict["Fnsw"]
        ),
        "Maximum_Positive_SWBM_Governing_Condition_ID": (
            ship_system.StructuralLoadEnvelope["Mpsw_Condition_ID"]
        ),
        "Maximum_Negative_SWBM_Governing_Condition_ID": (
            ship_system.StructuralLoadEnvelope["Mnsw_Condition_ID"]
        ),
        "Maximum_Positive_Shear_Governing_Condition_ID": (
            ship_system.StructuralLoadEnvelope["Fpsw_Condition_ID"]
        ),
        "Maximum_Negative_Shear_Governing_Condition_ID": (
            ship_system.StructuralLoadEnvelope["Fnsw_Condition_ID"]
        ),
        "Structural_Still_Water_Load_Case_Count": int(
            ship_system.StructuralLoadEnvelope["Load_Case_Count"]
        ),
        "Structural_Still_Water_Load_Basis": (
            ship_system.StructuralLoadEnvelope["Load_Basis"]
        ),
        "Mass_Displacement_Residual_kg": float(
            full_cargo["Total_Weight_kg"] - ship_system.HS.rho*hull["Vol_Disp"]
        ),
        "Load_Equilibrium_Draft_m": float(
            load_closure["Load_Equilibrium_Draft_m"]
        ),
        "Load_Equilibrium_Forward_Draft_m": float(
            load_closure["Load_Equilibrium_Forward_Draft_m"]
        ),
        "Load_Equilibrium_Aft_Draft_m": float(
            load_closure["Load_Equilibrium_Aft_Draft_m"]
        ),
        "Load_Equilibrium_Trim_m": float(
            load_closure["Load_Equilibrium_Trim_m"]
        ),
        "Load_Equilibrium_Sinkage_m": float(
            load_closure["Load_Equilibrium_Sinkage_m"]
        ),
        "Longitudinal_Equilibrium_Solved": bool(
            load_closure["Longitudinal_Equilibrium_Solved"]
        ),
        "Normalized_Displacement_Volume_Residual": float(
            load_closure["Normalized_Displacement_Volume_Residual"]
        ),
        "Normalized_Displacement_First_Moment_Residual": float(
            load_closure[
                "Normalized_Displacement_First_Moment_Residual"
            ]
        ),
        "Distributed_Mass_Residual_kg": float(
            load_closure["Mass_Distribution_Residual_kg"]
        ),
        "Distributed_First_Moment_Residual_kg_m": float(
            load_closure["Distribution_First_Moment_Residual_kg_m"]
        ),
        "Net_Vertical_Force_Residual_N": float(
            load_closure["Net_Vertical_Force_Residual_N"]
        ),
        "Net_Vertical_Force_Residual_Fraction": float(
            load_closure["Net_Vertical_Force_Residual_Fraction"]
        ),
        "Net_Vertical_First_Moment_Residual_Nm": float(
            load_closure["Net_Vertical_First_Moment_Residual_Nm"]
        ),
        "Aft_Shear_Force_Residual_N": float(
            load_closure["Aft_Shear_Force_Residual_N"]
        ),
        "Aft_Bending_Moment_Residual_Nm": float(
            load_closure["Aft_Bending_Moment_Residual_Nm"]
        ),
        "Ship_Cost_pounds": float(ship_cost),
        "Annual_Cargo_t": float(annual_cargo),
        "Required_Freight_Rate_pounds_per_t": float(required_freight_rate),
        "Audited_Constraint_Row_Count": int(len(audited)),
        "Applicable_Audited_Constraint_Count": int(len(applicable_bits)),
        "Audited_Constraint_Pass_Count": int(applicable_bits.sum()),
        "Audited_Constraint_Fail_Count": int((~applicable_bits).sum()),
        "Global_Audited_Feasible": bool(applicable_bits.all()),
        "V1p1_Constraint_Row_Count": int(len(expanded_constraints)),
        "V1p1_Evaluated_Constraint_Count": int(len(expanded_bits)),
        "V1p1_Constraint_Pass_Count": int(expanded_bits.sum()),
        "V1p1_Constraint_Fail_Count": int((~expanded_bits).sum()),
        "Global_V1p1_Feasible": bool(expanded_bits.all()),
        "Integration_Method": "Composite trapezoidal rule",
    }

    # The one-row economic summary contains the complete legacy/default or
    # explicitly overridden scenario plus each intermediate cost term.  Keep
    # those values in the scalar table as well as the long-form detail shard
    # so the primary ML/performance table is self-describing.
    row.update(ship_system.EconomicPerformanceSummary.iloc[0].to_dict())

    if not ship_system.FloodingDiagnosticSummary.empty:
        flooding = ship_system.FloodingDiagnosticSummary
        row.update(
            {
                "Flooding_Diagnostic_Executed": True,
                "Flooding_Sample_Count": int(flooding["Sample Count"].sum()),
                "Flooding_Monotonic_Boundary_Observed": bool(
                    flooding["Monotonic Boundary Observed"].all()
                ),
                "Minimum_Single_Compartment_Margin_m": float(
                    flooding["Minimum Single-Compartment Margin"].min()
                ),
                "Minimum_Two_Compartment_Margin_m": float(
                    flooding["Minimum Two-Compartment Margin"].min()
                ),
                "Flooding_Diagnostic_Single_Pass": bool(
                    flooding["Diagnostic Single Pass"].all()
                ),
                "Flooding_Diagnostic_Two_Pass": bool(
                    flooding["Diagnostic Two Pass"].all()
                ),
            }
        )
    else:
        row.update(
            {
                "Flooding_Diagnostic_Executed": False,
                "Flooding_Sample_Count": 0,
                "Flooding_Monotonic_Boundary_Observed": np.nan,
                "Minimum_Single_Compartment_Margin_m": np.nan,
                "Minimum_Two_Compartment_Margin_m": np.nan,
                "Flooding_Diagnostic_Single_Pass": np.nan,
                "Flooding_Diagnostic_Two_Pass": np.nan,
            }
        )

    return row


def _write_per_design_review_outputs(
    output_root: Path,
    identifiers: dict,
    source_row: pd.Series,
    parameter_names: list[str],
    ship_system: ShipSystemDesign,
    structural_elements_path: Path,
    scalar_result: dict,
    expanded_constraints: pd.DataFrame,
    weight_components: pd.DataFrame,
    compartment_loads: pd.DataFrame,
    compartment_conditions: pd.DataFrame,
    bulkhead_locations: pd.DataFrame,
    hatch_covers: pd.DataFrame,
    loading_conditions: pd.DataFrame,
    loading_components: pd.DataFrame,
    upright_stability: pd.DataFrame,
    named_condition_loads: pd.DataFrame,
    load_closure: pd.DataFrame,
    economic_performance: pd.DataFrame,
) -> Path:
    """Write one human-readable folder matching the selected-design layout.

    The normal V1.1 dataset remains shard-oriented for efficient parallel
    computation and compilation.  This optional representation is intended
    for the 200-system laptop trial, where opening one self-contained design
    folder is more useful than minimizing the number of files.

    ``Audited_Constraints.csv`` is deliberately omitted.  Its rule-status
    metadata remains available in the aggregate audit shard, while the
    authoritative per-design evaluation is the eight-column
    ``Ship_<index>_Constraints_Evaluations.csv`` file.
    """

    system_index = identifiers["Index"]
    design_directory = output_root / f"Design_{system_index}"
    design_directory.mkdir(parents=True, exist_ok=True)

    # Retain the unmodified source vector and the exact structural-element
    # input so an individual laptop result can be reproduced on its own.
    _write_csv_atomic(
        source_row.to_frame().T,
        design_directory / "Source_System_Design_Parameters.csv",
    )
    structural_elements_copy = design_directory / structural_elements_path.name
    temporary_structure_copy = structural_elements_copy.with_suffix(
        structural_elements_copy.suffix + ".tmp"
    )
    shutil.copy2(structural_elements_path, temporary_structure_copy)
    temporary_structure_copy.replace(structural_elements_copy)

    # Preserve the exact legacy-style eight-column constraint contract while
    # retaining every expanded V1.1 constraint as its own independent row.
    _write_csv_atomic(
        expanded_constraints[CONSTRAINT_COLUMNS],
        design_directory
        / f"Ship_{system_index}_Constraints_Evaluations.csv",
    )

    # These direct module tables retain the selected-design filenames.  They
    # make the laptop result easy to compare to designs 99, 170, and 178.
    direct_tables = {
        "Flooding_Diagnostic_Summary.csv": (
            ship_system.FloodingDiagnosticSummary
        ),
        "Floodable_Length_Curve.csv": ship_system.FloodableLengthCurve,
        "Cargo_Optimization_Summary.csv": (
            ship_system.CargoOptimizationSummary
        ),
        f"Ship_{system_index}_Weights_Evaluations.csv": (
            ship_system.Weights.Weights
        ),
        "Loading_Condition_Centers.csv": ship_system.Weights.Ship_CG,
        "Cargo_Holds.csv": ship_system.CargoHolds.CargoHolds,
        "Ballast_Tanks.csv": ship_system.CargoHolds.BallastTanks,
        "Fuel_Tank.csv": ship_system.CargoHolds.Fuel_Tank,
        f"Ship_{system_index}_Structure_Evaluations.csv": (
            ship_system.SC.Struct_Dict
        ),
        "Hydrostatic_Summary.csv": _scalar_hydrostatic_summary(ship_system),
        "Evaluated_System_Parameters.csv": pd.DataFrame(
            [ship_system.system_params],
            columns=parameter_names,
        ),
    }

    for filename, table in direct_tables.items():
        _write_csv_atomic(table, design_directory / filename)

    # The normalized V1.1 tables avoid arrays embedded in CSV fields and
    # expose the intermediate calculations used by the constraints.
    detail_tables = {
        "Weight_Components.csv": weight_components,
        "Compartment_Loads.csv": compartment_loads,
        "Compartment_Condition_States.csv": compartment_conditions,
        "Bulkhead_Locations.csv": bulkhead_locations,
        "Hatch_Covers.csv": hatch_covers,
        "Loading_Conditions.csv": loading_conditions,
        "Loading_Condition_Components.csv": loading_components,
        "Upright_Stability.csv": upright_stability,
        "Named_Condition_Loads.csv": named_condition_loads,
        "Load_Closure.csv": load_closure,
        "Economic_Performance.csv": economic_performance,
    }

    for filename, table in detail_tables.items():
        _write_csv_atomic(
            _per_design_table(table),
            design_directory / filename,
        )

    _write_csv_atomic(
        pd.DataFrame([scalar_result]).drop(columns=IDENTIFIER_COLUMNS),
        design_directory / "System_Performance.csv",
    )
    _save_per_design_numerical_arrays(
        ship_system,
        design_directory / "Numerical_Arrays.npz",
    )

    return design_directory


def _write_failed_per_design_review_outputs(
    output_root: Path,
    identifiers: dict,
    source_row: pd.Series,
    failure_constraint: pd.DataFrame,
    failure_record: dict,
) -> Path:
    """Write the available evidence when a laptop design cannot complete."""

    system_index = identifiers["Index"]
    design_directory = output_root / f"Design_{system_index}"
    design_directory.mkdir(parents=True, exist_ok=True)

    _write_csv_atomic(
        source_row.to_frame().T,
        design_directory / "Source_System_Design_Parameters.csv",
    )
    _write_csv_atomic(
        failure_constraint[CONSTRAINT_COLUMNS],
        design_directory
        / f"Ship_{system_index}_Constraints_Evaluations.csv",
    )
    _write_csv_atomic(
        pd.DataFrame([failure_record]),
        design_directory / "Evaluation_Failure.csv",
    )

    return design_directory


def evaluate_system_shard(
    input_csv: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_flooding_diagnostic: bool = False,
    flooding_step: int = 15,
    include_legacy_constraints: bool = False,
    powering_inputs=None,
    early_design_mass_inputs=None,
    economic_inputs=None,
    scenario_provenance: dict | None = None,
    flooding_permeabilities=(0.95,),
    write_per_design_review_files: bool = False,
    save_mode: str = "all",
) -> dict[str, Path]:
    """Evaluate every row in one generated V1.1 system shard."""

    input_csv = Path(input_csv)
    output_root = Path(output_root)
    range_label = _system_shard_range_label(input_csv)
    shard_suffix = _csv_shard_suffix(save_mode)

    # Review folders intentionally repeat most canonical tables once per
    # design.  Reduced mode suppresses them while retaining each design's
    # required legacy-format constraint CSV in the individual hierarchy.
    write_per_design_review_files = bool(
        write_per_design_review_files and save_mode == "all"
    )

    # Direct Python callers may provide dictionaries without source files.
    # Preserve those exact overrides using canonical-JSON hashes.  The CLI
    # supplies richer records that additionally bind each original JSON file.
    if scenario_provenance is None:
        scenario_provenance = {
            "powering_inputs": _canonical_mapping_provenance(
                powering_inputs,
                (
                    "in_memory_mapping"
                    if powering_inputs is not None
                    else "built_in_defaults_or_inactive_optional_group"
                ),
            ),
            "early_design_mass_inputs": _canonical_mapping_provenance(
                early_design_mass_inputs,
                (
                    "in_memory_mapping"
                    if early_design_mass_inputs is not None
                    else "built_in_defaults_or_inactive_optional_group"
                ),
            ),
            "economic_inputs": _canonical_mapping_provenance(
                economic_inputs,
                (
                    "in_memory_mapping"
                    if economic_inputs is not None
                    else "built_in_defaults_or_inactive_optional_group"
                ),
            ),
        }

    scalar_path = (
        output_root
        / "evaluations"
        / "scalar_results"
        / f"System_Results_{range_label}{shard_suffix}"
    )
    updated_path = (
        output_root
        / "evaluations"
        / "updated_parameters"
        / f"Updated_System_Parameters_{range_label}{shard_suffix}"
    )
    constraint_path = (
        output_root
        / "evaluations"
        / "constraints"
        / f"Audited_Constraints_{range_label}{shard_suffix}"
    )
    expanded_constraint_path = (
        output_root
        / "evaluations"
        / "constraints_v1p1"
        / "shards"
        / f"Constraints_v1p1_{range_label}{shard_suffix}"
    )
    legacy_path = (
        output_root
        / "evaluations"
        / "legacy_constraints"
        / f"Legacy_Constraints_{range_label}{shard_suffix}"
    )
    failure_path = (
        output_root
        / "evaluations"
        / "failures"
        / f"Evaluation_Failures_{range_label}{shard_suffix}"
    )
    array_path = (
        output_root
        / "evaluations"
        / "arrays"
        / f"Numerical_Arrays_{range_label}.npz"
    )
    structure_property_path = (
        output_root
        / "evaluations"
        / "structure_properties"
        / f"Structure_Evaluations_{range_label}{shard_suffix}"
    )
    weight_component_path = (
        output_root
        / "evaluations"
        / "weight_components"
        / f"Weight_Components_{range_label}{shard_suffix}"
    )
    compartment_load_path = (
        output_root
        / "evaluations"
        / "compartment_loads"
        / f"Compartment_Loads_{range_label}{shard_suffix}"
    )
    compartment_condition_path = (
        output_root
        / "evaluations"
        / "compartment_condition_states"
        / f"Compartment_Condition_States_{range_label}{shard_suffix}"
    )
    bulkhead_location_path = (
        output_root
        / "evaluations"
        / "bulkhead_locations"
        / f"Bulkhead_Locations_{range_label}{shard_suffix}"
    )
    hatch_cover_path = (
        output_root
        / "evaluations"
        / "hatch_covers"
        / f"Hatch_Covers_{range_label}{shard_suffix}"
    )
    loading_condition_path = (
        output_root
        / "evaluations"
        / "loading_conditions"
        / f"Loading_Conditions_{range_label}{shard_suffix}"
    )
    loading_component_path = (
        output_root
        / "evaluations"
        / "loading_condition_components"
        / f"Loading_Condition_Components_{range_label}{shard_suffix}"
    )
    upright_stability_path = (
        output_root
        / "evaluations"
        / "upright_stability"
        / f"Upright_Stability_{range_label}{shard_suffix}"
    )
    named_condition_load_path = (
        output_root
        / "evaluations"
        / "named_condition_loads"
        / f"Named_Condition_Loads_{range_label}{shard_suffix}"
    )
    load_closure_path = (
        output_root
        / "evaluations"
        / "load_closure"
        / f"Load_Closure_{range_label}{shard_suffix}"
    )
    economic_performance_path = (
        output_root
        / "evaluations"
        / "economic_performance"
        / f"Economic_Performance_{range_label}{shard_suffix}"
    )

    systems = pd.read_csv(input_csv)
    catalog = load_parameter_catalog()
    names = catalog["Name"].tolist()
    parameter_index = dict(zip(names, catalog["idx"], strict=True))

    scalar_rows = []
    updated_rows = []
    constraint_tables = []
    expanded_constraint_tables = []
    legacy_tables = []
    failure_rows = []
    load_array_rows = []
    flooding_curve_tables = []
    structure_property_tables = []
    weight_component_tables = []
    compartment_load_tables = []
    compartment_condition_tables = []
    bulkhead_location_tables = []
    hatch_cover_tables = []
    loading_condition_tables = []
    loading_component_tables = []
    upright_stability_tables = []
    named_condition_load_tables = []
    load_closure_tables = []
    economic_performance_tables = []

    for _, source_row in systems.iterrows():
        identifiers = {
            name: int(source_row[name])
            for name in IDENTIFIER_COLUMNS
        }
        stage = "load_structural_elements"
        start_time = time.perf_counter()

        try:
            structure_index = identifiers["Structure_Index"]
            structural_elements_path = (
                output_root
                / "structures"
                / "elements"
                / f"Structure_{structure_index:06d}_Structural_Elements.csv"
            )
            structural_elements = pd.read_csv(structural_elements_path)

            stage = "prepare_system_parameters"
            system_parameters = source_row[names].to_numpy(
                dtype=float,
                copy=True,
            )

            # Re-derive T at the evaluation boundary.  This makes the origin of
            # the regulatory draft explicit even if a hand-edited input shard
            # contains a stale structural T value.
            system_parameters[parameter_index["T"]] = (
                derive_constraint_draft_m(system_parameters)
            )

            stage = "evaluate_ship_system"
            ship_system = ShipSystemDesign(
                system_params=system_parameters,
                structural_elements=structural_elements,
                file_id=f"System_{identifiers['Index']:06d}",
                structure_id=str(structure_index),
                save_path=str(output_root),
                run_flooding_diagnostic=run_flooding_diagnostic,
                flooding_step=flooding_step,
                flooding_permeabilities=flooding_permeabilities,
                powering_inputs=powering_inputs,
                early_design_mass_inputs=early_design_mass_inputs,
                economic_inputs=economic_inputs,
            )

            if ship_system.HD.Hull_Dict is None:
                raise RuntimeError(
                    "Hull design stopped after its geometric constraint checks."
                )

            stage = "calculate_system_performance"
            ship_cost, annual_cargo, required_freight_rate = (
                ship_system.calc_Ship_System_Performance()
            )
            evaluation_seconds = time.perf_counter() - start_time
            scalar_result = _build_scalar_result(
                identifiers,
                ship_system,
                ship_cost,
                annual_cargo,
                required_freight_rate,
                evaluation_seconds,
            )

            # Preserve the optimized cargo fractions without overwriting the
            # source system-design shard.
            updated_row = identifiers.copy()
            updated_row.update(
                dict(zip(names, ship_system.system_params, strict=True))
            )

            audited = ship_system.AuditedConstraints.copy()

            for column in reversed(IDENTIFIER_COLUMNS):
                audited.insert(0, column, identifiers[column])

            # Save one legacy-formatted file for every successfully evaluated
            # system.  No identifiers or audit metadata are added to this
            # file: it has exactly the eight columns used by the old dataset.
            expanded = ship_system.ConstraintsV1p1.copy()
            individual_path = _individual_constraint_path(
                output_root,
                identifiers["Structure_Index"],
                identifiers["Index"],
            )
            stage = "write_individual_constraint_file"
            _write_csv_atomic(expanded[CONSTRAINT_COLUMNS], individual_path)

            # A companion shard retains identifiers for efficient compilation
            # and model training without opening 100,000 small CSV files.
            for column in reversed(IDENTIFIER_COLUMNS):
                expanded.insert(0, column, identifiers[column])

            legacy = None
            if include_legacy_constraints:
                legacy = ship_system.Constraints.copy()

                for column in reversed(IDENTIFIER_COLUMNS):
                    legacy.insert(0, column, identifiers[column])

            # The hydrostatic grid has a fixed point count.  One matrix row per
            # successful system keeps these numerical arrays aligned without
            # converting them to fragile CSV strings.
            load_array_row = {
                "Index": identifiers["Index"],
                "X_m": np.asarray(ship_system.Weights.X, dtype=float),
                "Distributed_Load_N_per_m": np.asarray(
                    ship_system.Weights.Distributed_Load,
                    dtype=float,
                ),
                "Buoyancy_Load_N_per_m": np.asarray(
                    ship_system.Weights.buoyancy_dist,
                    dtype=float,
                ),
                "Shear_Force_N": np.asarray(
                    ship_system.Weights.Shear_Force,
                    dtype=float,
                ),
                "Bending_Moment_Nm": np.asarray(
                    ship_system.Weights.Bending_Moment,
                    dtype=float,
                ),
            }

            flood_curve = None
            if not ship_system.FloodableLengthCurve.empty:
                flood_curve = ship_system.FloodableLengthCurve.copy()
                flood_curve.insert(0, "Index", identifiers["Index"])

            # Restore the useful scalar detail from the legacy per-design
            # files as efficient long-form shards.  These are prepared before
            # the in-memory commit so a partial design cannot enter only some
            # compiled table families.
            structure_properties = _build_structure_property_table(
                identifiers,
                ship_system,
            )
            weight_components = _build_weight_component_table(
                identifiers,
                ship_system,
            )
            compartment_loads = _build_compartment_load_table(
                identifiers,
                ship_system,
            )
            compartment_conditions = _build_compartment_condition_table(
                identifiers,
                ship_system,
            )
            bulkhead_locations = _build_bulkhead_location_table(
                identifiers,
                ship_system,
            )
            hatch_covers = _build_hatch_cover_table(
                identifiers,
                ship_system,
            )
            loading_conditions = _build_loading_condition_table(
                identifiers,
                ship_system,
            )
            loading_components = _build_loading_condition_component_table(
                identifiers,
                ship_system,
            )
            upright_stability = _build_upright_stability_table(
                identifiers,
                ship_system,
            )
            named_condition_loads = _build_named_condition_load_table(
                identifiers,
                ship_system,
            )
            load_closure = _build_load_closure_table(
                identifiers,
                ship_system,
            )
            economic_performance = _build_economic_performance_table(
                identifiers,
                ship_system,
            )

            # The laptop trial additionally writes one review folder per
            # system.  Production/cluster jobs leave this switch disabled and
            # retain the scalable shard-oriented representation only.
            if write_per_design_review_files:
                stage = "write_per_design_review_files"
                _write_per_design_review_outputs(
                    output_root=output_root,
                    identifiers=identifiers,
                    source_row=source_row,
                    parameter_names=names,
                    ship_system=ship_system,
                    structural_elements_path=structural_elements_path,
                    scalar_result=scalar_result,
                    expanded_constraints=expanded,
                    weight_components=weight_components,
                    compartment_loads=compartment_loads,
                    compartment_conditions=compartment_conditions,
                    bulkhead_locations=bulkhead_locations,
                    hatch_covers=hatch_covers,
                    loading_conditions=loading_conditions,
                    loading_components=loading_components,
                    upright_stability=upright_stability,
                    named_condition_loads=named_condition_loads,
                    load_closure=load_closure,
                    economic_performance=economic_performance,
                )

            # Commit every in-memory shard row together only after all
            # per-system artifacts above have been prepared.  If preparation
            # fails, the exception path records only the failed system.
            scalar_rows.append(scalar_result)
            updated_rows.append(updated_row)
            constraint_tables.append(audited)
            expanded_constraint_tables.append(expanded)
            load_array_rows.append(load_array_row)
            structure_property_tables.append(structure_properties)
            weight_component_tables.append(weight_components)
            compartment_load_tables.append(compartment_loads)
            compartment_condition_tables.append(compartment_conditions)
            bulkhead_location_tables.append(bulkhead_locations)
            hatch_cover_tables.append(hatch_covers)
            loading_condition_tables.append(loading_conditions)
            loading_component_tables.append(loading_components)
            upright_stability_tables.append(upright_stability)
            named_condition_load_tables.append(named_condition_loads)
            load_closure_tables.append(load_closure)
            economic_performance_tables.append(economic_performance)

            if legacy is not None:
                legacy_tables.append(legacy)

            if flood_curve is not None:
                flooding_curve_tables.append(flood_curve)

        except Exception as error:
            failure_category = classify_evaluation_error(error, stage)
            failure_constraint = _failure_constraint_table(
                identifiers,
                stage,
                error,
            )
            individual_path = _individual_constraint_path(
                output_root,
                identifiers["Structure_Index"],
                identifiers["Index"],
            )
            _write_csv_atomic(
                failure_constraint,
                individual_path,
            )

            # Preserve a structured failure row in the compiled V1.1 shard as
            # well as in the individual eight-column file.  Dependent rules
            # remain missing, because they were not mathematically evaluated;
            # the failed physical predicate itself is a normal result.
            expanded_failure = failure_constraint.copy()
            for column in reversed(IDENTIFIER_COLUMNS):
                expanded_failure.insert(0, column, identifiers[column])
            expanded_constraint_tables.append(expanded_failure)

            failure_record = {
                **identifiers,
                "Stage": stage,
                "Failure_Category": failure_category,
                "Error_Type": type(error).__name__,
                "Error_Message": str(error),
                "Traceback": traceback.format_exc(),
                "Evaluation_Seconds": time.perf_counter() - start_time,
            }

            if isinstance(error, PhysicalConstraintFailure):
                # Numeric failure diagnostics are deliberately separate from
                # the fixed eight-column constraint contract.  Keeping them
                # as ordinary tabular columns allows later design-generation
                # bounds to use the magnitude and cause of a rejection.
                failure_record.update(error.diagnostics)

            failure_rows.append(failure_record)

            if write_per_design_review_files:
                _write_failed_per_design_review_outputs(
                    output_root=output_root,
                    identifiers=identifiers,
                    source_row=source_row,
                    failure_constraint=failure_constraint,
                    failure_record=failure_record,
                )
            scalar_rows.append(
                {
                    **identifiers,
                    "Evaluation_Status": "failed",
                    "Failure_Stage": stage,
                    "Failure_Category": failure_category,
                    "Error_Type": type(error).__name__,
                    "Error_Message": str(error),
                    "Evaluation_Seconds": time.perf_counter() - start_time,
                    **(
                        error.diagnostics
                        if isinstance(error, PhysicalConstraintFailure)
                        else {}
                    ),
                }
            )

    _write_csv_atomic(pd.DataFrame(scalar_rows), scalar_path)
    _write_csv_atomic(pd.DataFrame(updated_rows), updated_path)
    _write_csv_atomic(
        (
            pd.concat(constraint_tables, ignore_index=True)
            if constraint_tables
            else pd.DataFrame()
        ),
        constraint_path,
    )
    _write_csv_atomic(
        (
            pd.concat(expanded_constraint_tables, ignore_index=True)
            if expanded_constraint_tables
            else pd.DataFrame(columns=IDENTIFIER_COLUMNS + CONSTRAINT_COLUMNS)
        ),
        expanded_constraint_path,
    )
    failure_columns = (
        IDENTIFIER_COLUMNS
        + [
            "Stage",
            "Failure_Category",
            "Error_Type",
            "Error_Message",
            "Traceback",
            "Evaluation_Seconds",
        ]
    )
    failure_table = pd.DataFrame(failure_rows)

    if failure_table.empty:
        failure_table = pd.DataFrame(columns=failure_columns)
    else:
        diagnostic_columns = sorted(
            set(failure_table.columns) - set(failure_columns)
        )
        failure_table = failure_table.reindex(
            columns=failure_columns + diagnostic_columns
        )

    _write_csv_atomic(failure_table, failure_path)

    if include_legacy_constraints:
        _write_csv_atomic(
            (
                pd.concat(legacy_tables, ignore_index=True)
                if legacy_tables
                else pd.DataFrame()
            ),
            legacy_path,
        )

    detail_outputs = (
        (
            structure_property_tables,
            structure_property_path,
        ),
        (
            weight_component_tables,
            weight_component_path,
        ),
        (
            compartment_load_tables,
            compartment_load_path,
        ),
        (
            compartment_condition_tables,
            compartment_condition_path,
        ),
        (
            bulkhead_location_tables,
            bulkhead_location_path,
        ),
        (
            hatch_cover_tables,
            hatch_cover_path,
        ),
        (
            loading_condition_tables,
            loading_condition_path,
        ),
        (
            loading_component_tables,
            loading_component_path,
        ),
        (
            upright_stability_tables,
            upright_stability_path,
        ),
        (
            named_condition_load_tables,
            named_condition_load_path,
        ),
        (
            load_closure_tables,
            load_closure_path,
        ),
        (
            economic_performance_tables,
            economic_performance_path,
        ),
    )

    for detail_tables, detail_path in detail_outputs:
        _write_csv_atomic(
            (
                pd.concat(detail_tables, ignore_index=True)
                if detail_tables
                else pd.DataFrame()
            ),
            detail_path,
        )

    if load_array_rows:
        array_data = {
            "system_index": np.asarray(
                [row["Index"] for row in load_array_rows],
                dtype=np.int64,
            ),
            "x_m": np.stack([row["X_m"] for row in load_array_rows]),
            "distributed_load_n_per_m": np.stack(
                [row["Distributed_Load_N_per_m"] for row in load_array_rows]
            ),
            "buoyancy_load_n_per_m": np.stack(
                [row["Buoyancy_Load_N_per_m"] for row in load_array_rows]
            ),
            "shear_force_n": np.stack(
                [row["Shear_Force_N"] for row in load_array_rows]
            ),
            "bending_moment_nm": np.stack(
                [row["Bending_Moment_Nm"] for row in load_array_rows]
            ),
        }
    else:
        array_data = {
            "system_index": np.empty(0, dtype=np.int64),
            "x_m": np.empty((0, 0), dtype=float),
            "distributed_load_n_per_m": np.empty((0, 0), dtype=float),
            "buoyancy_load_n_per_m": np.empty((0, 0), dtype=float),
            "shear_force_n": np.empty((0, 0), dtype=float),
            "bending_moment_nm": np.empty((0, 0), dtype=float),
        }

    if flooding_curve_tables:
        flood_curves = pd.concat(flooding_curve_tables, ignore_index=True)
        array_data.update(
            {
                "flood_system_index": flood_curves["Index"].to_numpy(
                    dtype=np.int64
                ),
                "flood_x_position_m": flood_curves["X Position"].to_numpy(
                    dtype=float
                ),
                "flood_x_index": flood_curves["X Index"].to_numpy(
                    dtype=np.int64
                ),
                "flood_permeability": flood_curves["Permeability"].to_numpy(
                    dtype=float
                ),
                "floodable_length_m": flood_curves[
                    "Floodable Length"
                ].to_numpy(dtype=float),
                "single_compartment_demand_m": flood_curves[
                    "Single-Compartment Demand"
                ].to_numpy(dtype=float),
                "two_compartment_demand_m": flood_curves[
                    "Two-Compartment Demand"
                ].to_numpy(dtype=float),
                "single_compartment_margin_m": flood_curves[
                    "Single-Compartment Margin"
                ].to_numpy(dtype=float),
                "two_compartment_margin_m": flood_curves[
                    "Two-Compartment Margin"
                ].to_numpy(dtype=float),
            }
        )

    _write_npz_atomic(array_path, array_data)

    outputs = {
        "scalar_results": scalar_path,
        "updated_parameters": updated_path,
        "audited_constraints": constraint_path,
        "expanded_constraints_v1p1": expanded_constraint_path,
        "failures": failure_path,
        "numerical_arrays": array_path,
        "structure_properties": structure_property_path,
        "weight_components": weight_component_path,
        "compartment_loads": compartment_load_path,
        "compartment_condition_states": compartment_condition_path,
        "bulkhead_locations": bulkhead_location_path,
        "hatch_covers": hatch_cover_path,
        "loading_conditions": loading_condition_path,
        "loading_condition_components": loading_component_path,
        "upright_stability": upright_stability_path,
        "named_condition_loads": named_condition_load_path,
        "load_closure": load_closure_path,
        "economic_performance": economic_performance_path,
    }

    if include_legacy_constraints:
        outputs["legacy_constraints"] = legacy_path

    # Bind this output shard to the actual switches, input system CSV, and
    # optional scenario JSON content used for the evaluation.  The compiler
    # embeds these sidecars and their hashes in Dataset_Manifest.json.
    invocation_path = (
        output_root
        / "manifests"
        / "invocations"
        / "evaluation"
        / f"Evaluation_{range_label}.json"
    )
    invocation_payload = _evaluation_invocation_payload(
        input_csv=input_csv,
        output_root=output_root,
        run_flooding_diagnostic=run_flooding_diagnostic,
        flooding_step=flooding_step,
        flooding_permeabilities=flooding_permeabilities,
        include_legacy_constraints=include_legacy_constraints,
        write_per_design_review_files=write_per_design_review_files,
        scenario_provenance=scenario_provenance,
        save_mode=save_mode,
    )
    invocation_payload["output_shards"] = {
        label: _file_provenance(path)
        for label, path in outputs.items()
    }
    _write_json_atomic(invocation_payload, invocation_path)
    outputs["invocation_provenance"] = invocation_path

    return outputs


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one generated GenISys V1.1 system shard."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-flooding-diagnostic", action="store_true")
    parser.add_argument("--flooding-step", type=int, default=15)
    parser.add_argument("--include-legacy-constraints", action="store_true")
    parser.add_argument(
        "--save-mode",
        choices=("all", "reduced"),
        default="all",
        help=(
            "Write ordinary CSV shards and optional review folders ('all') "
            "or gzip-compressed canonical shards without review folders "
            "('reduced'). Individual constraint CSVs and NPZ arrays are "
            "retained in both modes."
        ),
    )
    parser.add_argument(
        "--write-per-design-review-files",
        action="store_true",
        help=(
            "Also write one Design_<Index> review folder per system. "
            "Intended for the 200-system laptop trial, not the 100,000-design "
            "cluster dataset."
        ),
    )
    parser.add_argument(
        "--powering-inputs-json",
        help=(
            "Optional JSON mapping that overrides the seven existing "
            "preliminary powering assumptions for every row in this shard."
        ),
    )
    parser.add_argument(
        "--early-design-mass-inputs-json",
        help=(
            "Optional JSON mapping with explicit 'deckhouse' and/or "
            "'lightship_margin' inputs. No dimensions, placement, or margin "
            "values are assumed when this option is absent."
        ),
    )
    parser.add_argument(
        "--economic-inputs-json",
        help=(
            "Optional JSON mapping that overrides the existing preliminary "
            "cost, utilization, cargo-handling, and annualization inputs for "
            "every row in this shard."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()

    # Read each optional scenario exactly once.  The calculation receives the
    # parsed mappings, while the sidecar receives both canonical content hashes
    # and raw source-file hashes/sizes.
    powering_inputs, powering_provenance = (
        _load_optional_json_mapping_with_provenance(
            args.powering_inputs_json
        )
    )
    early_design_mass_inputs, early_design_mass_provenance = (
        _load_optional_json_mapping_with_provenance(
            args.early_design_mass_inputs_json
        )
    )
    economic_inputs, economic_provenance = (
        _load_optional_json_mapping_with_provenance(
            args.economic_inputs_json
        )
    )

    outputs = evaluate_system_shard(
        input_csv=args.input_csv,
        output_root=args.output_root,
        run_flooding_diagnostic=args.run_flooding_diagnostic,
        flooding_step=args.flooding_step,
        include_legacy_constraints=args.include_legacy_constraints,
        save_mode=args.save_mode,
        write_per_design_review_files=(
            args.write_per_design_review_files
        ),
        powering_inputs=powering_inputs,
        early_design_mass_inputs=early_design_mass_inputs,
        economic_inputs=economic_inputs,
        scenario_provenance={
            "powering_inputs": powering_provenance,
            "early_design_mass_inputs": early_design_mass_provenance,
            "economic_inputs": economic_provenance,
        },
    )

    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
