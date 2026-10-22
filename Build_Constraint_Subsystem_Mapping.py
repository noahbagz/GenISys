"""Build the reviewable V1.1 constraint-to-guidance-group mapping.

This is a training-data preparation script.  It reads the immutable compiled
constraint catalogs and does not import or modify the evaluator.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

CATALOG_PATHS = (
    PROJECT_ROOT
    / "Ship_Systems_Designs_v1p1"
    / "Dataset_100k"
    / "compiled"
    / "Constraint_Catalog.csv",
    PROJECT_ROOT
    / "Ship_Systems_Designs_v1p1"
    / "repair_experiments"
    / "Dataset_100k_Repaired"
    / "compiled"
    / "Constraint_Catalog.csv",
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "Ship_Systems_Designs_v1p1"
    / "Unified_Training_Dataset"
    / "Constraint_Subsystem_Mapping.csv"
)

LABEL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "Ship_Systems_Designs_v1p1"
    / "Unified_Training_Dataset"
    / "Granular_Subsystem_Labels.csv.gz"
)

SOURCE_POPULATIONS = (
    (
        "random",
        PROJECT_ROOT
        / "Ship_Systems_Designs_v1p1"
        / "Dataset_100k"
        / "compiled",
    ),
    (
        "repaired",
        PROJECT_ROOT
        / "Ship_Systems_Designs_v1p1"
        / "repair_experiments"
        / "Dataset_100k_Repaired"
        / "compiled",
    ),
)


# Each current evaluator class has one primary guidance scope.  Local classes
# contribute to the pass/fail label for one generated subsystem.  Classes that
# depend on coupled ship behavior are reserved for full-system guidance.
CLASS_TO_GUIDANCE_GROUP = {
    "Hull": "Hull",
    "Hydrostatic Numerical Foundation": "Hull",
    "Fuel Tank Volume": "Powering",
    "Powering Method Domain": "Powering",
    "Bulkhead_Spacing": "Bulkhead_Spacing",
    "Cargo Hold Volume": "Bulkhead_Spacing",
    "Structural Material": "Structure",
    "Structure": "Structure",
    "Structure Applicability Metric": "Structure",
    "Structure Rule Method": "Structure",
    "Subdivision Arrangement": "Structure",
    "Load Line Applicability": "System",
    "Load Line Input": "System",
    "Loading Condition Feasibility": "System",
    "Loading Condition Numerical Quality": "System",
    "Loading Condition Trim": "System",
    "Longitudinal Load Numerical Check": "System",
    "Regulatory Applicability": "System",
    "System Capacity": "System",
    "System Mass Feasibility": "System",
}


GROUP_RATIONALES = {
    "Hull": (
        "Constraint primarily tests hull geometry or hull hydrostatic "
        "numerical behavior."
    ),
    "Powering": (
        "Constraint primarily tests propulsion-domain inputs or voyage-fuel "
        "capacity."
    ),
    "Bulkhead_Spacing": (
        "Constraint primarily tests compartment arrangement, engine-room "
        "length, or cargo-hold capacity."
    ),
    "Structure": (
        "Constraint primarily tests structural arrangement, material, "
        "scantlings, or a structural-rule method."
    ),
    "System": (
        "Constraint depends on coupled full-ship behavior, loading, mass, "
        "capacity, longitudinal equilibrium, or regulatory applicability."
    ),
}


SYSTEM_GRANULAR_LABELS = {
    "Load Line Applicability": "System_Regulatory",
    "Load Line Input": "System_Regulatory",
    "Regulatory Applicability": "System_Regulatory",
    "Loading Condition Feasibility": "System_Loading_Condition",
    "Loading Condition Numerical Quality": "System_Loading_Condition",
    "Loading Condition Trim": "System_Loading_Condition",
    "Longitudinal Load Numerical Check": "System_Loading_Condition",
    "System Capacity": "System_Capacity",
    "System Mass Feasibility": "System_Mass",
}


def _hull_granular_label(name: str) -> str:
    """Assign a current hull constraint to its geometric region."""

    normalized = name.strip().lower()

    if normalized.startswith("stern bulb"):
        return "Hull_Stern_Bulb"

    if "bulb" in normalized:
        return "Hull_Bulbous_Bow"

    if normalized.startswith(("drift angle", "bow rake", "bow taper")):
        return "Hull_Bow"

    if (
        "stern" in normalized
        or "transom" in normalized
        or normalized.startswith("bc_trans")
    ):
        return "Hull_Stern"

    return "Hull_Midbody"


def _structure_granular_label(name: str) -> str:
    """Assign a current structural constraint to its affected structure."""

    normalized = name.strip().lower()

    if "hull-girder" in normalized:
        return "Structure_Hull_Girder"

    if "hopper" in normalized:
        return "Structure_Hopper"

    # These two named loading checks use the transverse-bulkhead vertical-
    # support equation in the current implementation.
    if (
        "bulkhead" in normalized
        or normalized.startswith("max allowable uniform bottom loading")
        or normalized.startswith("max allowable uniform deck loading")
    ):
        return "Structure_Bulkhead"

    if "deck" in normalized or "hatch" in normalized:
        return "Structure_Top_Deck"

    if (
        "bottom" in normalized
        or "floor" in normalized
        or "center-girder" in normalized
        or "center bottom girder" in normalized
    ):
        return "Structure_Bottom"

    if (
        "side" in normalized
        or "frame" in normalized
        or "web-frame" in normalized
        or "web frame" in normalized
    ):
        return "Structure_Side"

    return "Structure_General"


def granular_label(row: pd.Series) -> str:
    """Return the fine label nested under the broad model block."""

    constraint_class = row["Class"]
    name = row["Name"]

    if constraint_class == "Hull":
        return _hull_granular_label(name)

    if constraint_class == "Hydrostatic Numerical Foundation":
        return "Hull_Hydrostatics"

    if constraint_class == "Fuel Tank Volume":
        return "Powering_Fuel"

    if constraint_class == "Powering Method Domain":
        return "Powering_Hydrodynamics"

    if constraint_class == "Bulkhead_Spacing":
        return "Bulkhead_Arrangement"

    if constraint_class == "Cargo Hold Volume":
        return "Cargo_Hold"

    if constraint_class == "Structure":
        return _structure_granular_label(name)

    if constraint_class == "Subdivision Arrangement":
        return "Structure_Bottom"

    if constraint_class == "Structure Applicability Metric":
        return "Structure_Top_Deck"

    if constraint_class in {"Structural Material", "Structure Rule Method"}:
        return "Structure_General"

    if constraint_class in SYSTEM_GRANULAR_LABELS:
        return SYSTEM_GRANULAR_LABELS[constraint_class]

    raise ValueError(
        f"No granular label for class={constraint_class!r}, name={name!r}."
    )


def load_engineering_constraint_catalog() -> pd.DataFrame:
    """Load the union of current engineering constraint identities."""

    tables = []

    for path in CATALOG_PATHS:
        table = pd.read_csv(path)
        table = table.loc[
            table["Output_Type"].eq("constraint")
            & ~table["Class"].eq("Evaluation Failure"),
            ["Constraint_Key", "Class", "Name", "Requirement", "Units"],
        ].copy()
        tables.append(table)

    # The catalogs contain repeated runtime rows for some conditional cargo
    # holds.  Constraint_Key is the stable identity used by the training data.
    catalog = (
        pd.concat(tables, ignore_index=True)
        .drop_duplicates(subset=["Constraint_Key"])
        .sort_values(["Class", "Constraint_Key"], kind="stable")
        .reset_index(drop=True)
    )

    return catalog


def build_mapping(catalog: pd.DataFrame) -> pd.DataFrame:
    """Assign each engineering constraint to its training guidance group."""

    result = catalog.copy()
    result["Guidance_Group"] = result["Class"].map(CLASS_TO_GUIDANCE_GROUP)

    missing_classes = sorted(
        result.loc[result["Guidance_Group"].isna(), "Class"].unique()
    )
    if missing_classes:
        raise ValueError(
            "Add explicit subsystem mappings for constraint classes: "
            f"{missing_classes}"
        )

    result["Guidance_Scope"] = result["Guidance_Group"].map(
        lambda group: "system" if group == "System" else "subsystem"
    )
    result["Granular_Label"] = result.apply(granular_label, axis=1)
    result["Subsystem_Condition_Label"] = result["Guidance_Group"].where(
        result["Guidance_Scope"].eq("subsystem"),
        "",
    )

    # HCDDPM can use only the local label for the block it is generating.
    # SCDDPM can use those local labels and full-system guidance.
    result["Use_For_HCDDPM_Subsystem_Label"] = result[
        "Guidance_Scope"
    ].eq("subsystem")
    result["Use_For_SCDDPM_Subsystem_Label"] = result[
        "Guidance_Scope"
    ].eq("subsystem")
    result["Use_For_SCDDPM_System_Guidance"] = result[
        "Guidance_Scope"
    ].eq("system")

    result["Mapping_Rationale"] = result["Guidance_Group"].map(
        GROUP_RATIONALES
    )
    result["Mapping_Status"] = "proposed_for_review"

    return result


def build_row_labels(mapping: pd.DataFrame) -> pd.DataFrame:
    """Aggregate current constraint bits into broad and granular labels."""

    key_mapping = mapping.set_index("Constraint_Key")[
        ["Guidance_Group", "Granular_Label"]
    ]
    population_tables = []

    for population, compiled_root in SOURCE_POPULATIONS:
        status = pd.read_csv(
            compiled_root / "Ship_System_Performance.csv.gz",
            usecols=["Index", "Evaluation_Status"],
        )
        completed = (
            status["Evaluation_Status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq("completed")
        )

        result = status.loc[:, ["Index", "Evaluation_Status"]].copy()
        result.insert(0, "Population", population)
        result["Labels_Known"] = completed

        # Begin with the evaluator convention that absent/non-applicable rules
        # pass.  The long margin table is authoritative for the 197 current
        # identities; its older wide bit matrix omits some added constraints.
        broad_groups = sorted(mapping["Guidance_Group"].unique())
        granular_groups = sorted(mapping["Granular_Label"].unique())

        for broad_group in broad_groups:
            result[f"Feasible::Broad::{broad_group}"] = True

        for granular_group in granular_groups:
            result[f"Feasible::Granular::{granular_group}"] = True

        failure_parts = []
        margin_path = compiled_root / "Normalized_Constraint_Margins.csv.gz"

        for chunk in pd.read_csv(
            margin_path,
            usecols=["Index", "Constraint_Key", "Constraint_Bit"],
            chunksize=500_000,
        ):
            failed = chunk.loc[
                chunk["Constraint_Bit"].eq(False)
                & chunk["Constraint_Key"].isin(key_mapping.index),
                ["Index", "Constraint_Key"],
            ].copy()

            if failed.empty:
                continue

            failed = failed.join(key_mapping, on="Constraint_Key")
            failure_parts.append(
                failed.loc[
                    :,
                    ["Index", "Guidance_Group", "Granular_Label"],
                ].drop_duplicates()
            )

        failures = (
            pd.concat(failure_parts, ignore_index=True).drop_duplicates()
            if failure_parts
            else pd.DataFrame(
                columns=["Index", "Guidance_Group", "Granular_Label"]
            )
        )
        index_to_position = pd.Series(result.index, index=result["Index"])

        for broad_group, rows in failures.groupby("Guidance_Group"):
            positions = index_to_position.loc[rows["Index"].unique()].to_numpy()
            result.loc[positions, f"Feasible::Broad::{broad_group}"] = False

        for granular_group, rows in failures.groupby("Granular_Label"):
            positions = index_to_position.loc[rows["Index"].unique()].to_numpy()
            result.loc[
                positions,
                f"Feasible::Granular::{granular_group}",
            ] = False

        # Evaluator failures are unknown labels rather than negative evidence
        # for every subsystem.
        label_columns = [
            column for column in result.columns if column.startswith("Feasible::")
        ]
        result[label_columns] = result[label_columns].astype("boolean")
        result.loc[~completed, label_columns] = pd.NA

        population_tables.append(result)

    return pd.concat(population_tables, ignore_index=True)


def main() -> None:
    """Write the complete constraint-to-subsystem mapping."""

    catalog = load_engineering_constraint_catalog()
    mapping = build_mapping(catalog)
    row_labels = build_row_labels(mapping)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUTPUT_PATH, index=False)
    row_labels.to_csv(
        LABEL_OUTPUT_PATH,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )

    print(f"Wrote {len(mapping)} constraint mappings to {OUTPUT_PATH}")
    print(f"Wrote {len(row_labels)} row-label records to {LABEL_OUTPUT_PATH}")
    print(mapping["Guidance_Group"].value_counts().sort_index().to_string())
    print("\nGranular labels:")
    print(mapping["Granular_Label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
