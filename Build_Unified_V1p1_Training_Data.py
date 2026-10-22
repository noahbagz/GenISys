#!/usr/bin/env python3
"""Combine the random and repaired V1.1 populations into one training table.

The output keeps every source design, including failed evaluations. The
``Training_Eligible_RP`` flag reproduces the current experiment runner's data
selection: all random rows and completed repaired rows. No evaluation result is
recomputed or changed here.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Ship_Systems_Designs_v1p1"
RANDOM_COMPILED = DATA_ROOT / "Dataset_100k" / "compiled"
REPAIRED_COMPILED = (
    DATA_ROOT / "repair_experiments" / "Dataset_100k_Repaired" / "compiled"
)
OUTPUT_ROOT = DATA_ROOT / "Unified_Training_Dataset"
OUTPUT_TABLE = OUTPUT_ROOT / "Training_Rows.csv.gz"
OUTPUT_MANIFEST = OUTPUT_ROOT / "Training_Manifest.json"

DESIGN_IDENTIFIERS = (
    "Index",
    "Structure_Index",
    "Hull_Variant_Index",
    "Operating_Condition_Index",
    "Hull_Candidate_Index",
    "Structure_Sample_Attempt",
)
PERFORMANCE_COLUMNS = (
    "Index",
    "Evaluation_Status",
    "Global_V1p1_Feasible",
    "V1p1_Constraint_Fail_Count",
    "Required_Freight_Rate_pounds_per_t",
    "Engine_Power_kW",
    "Annual_Cargo_t",
)
LINEAGE_COLUMNS = (
    "Index",
    "Parent_Index",
    "Alternative_Number",
    "Selection_Order",
    "Task_ID",
    "Final_Status",
    "Fully_Feasible",
)
PROVENANCE_COLUMNS = (
    "Population",
    "Source_Design_Index",
    "Source_Structure_Index",
    "Parent_Index",
    "Alternative_Number",
    "Selection_Order",
    "Task_ID",
    "Final_Status",
    "Fully_Feasible",
    "Training_Eligible_RP",
    "Verified_Global_Feasible",
)


def sha256_file(path: Path) -> str:
    """Hash one source artifact or the finished training table."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_unique_index(table: pd.DataFrame, name: str) -> None:
    """Refuse an ambiguous one-to-one join."""

    if table["Index"].isna().any() or table["Index"].duplicated().any():
        raise ValueError(f"{name} has missing or duplicate Index values.")


def join_all_rows(
    designs: pd.DataFrame,
    other: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """Attach one indexed table without silently dropping design rows."""

    joined = designs.merge(
        other,
        on="Index",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"] != "both", "Index"].head(10)
        raise ValueError(f"{name} is missing design IDs: {missing.tolist()}")
    return joined.drop(columns="_merge")


def read_shared_schema() -> tuple[list[str], list[str], list[str]]:
    """Freeze the compact design and class-level constraint columns."""

    random_catalog = RANDOM_COMPILED / "Parameter_Catalog.csv"
    repaired_catalog = REPAIRED_COMPILED / "Parameter_Catalog.csv"
    if sha256_file(random_catalog) != sha256_file(repaired_catalog):
        raise ValueError("The two populations have different parameter catalogs.")

    catalog = pd.read_csv(random_catalog).sort_values("idx")
    parameter_names = catalog["Name"].astype(str).tolist()
    if len(parameter_names) != 189 or len(set(parameter_names)) != 189:
        raise ValueError("Expected 189 unique V1.1 design parameters.")

    design_columns = list(DESIGN_IDENTIFIERS) + parameter_names
    bit_path = RANDOM_COMPILED / "V1p1_Subsystem_Feasibility_Bits.csv.gz"
    bit_columns = pd.read_csv(bit_path, nrows=0).columns.tolist()
    if bit_columns[0] != "Index" or not all(
        name.startswith("Feasible::") for name in bit_columns[1:]
    ):
        raise ValueError("The class-level feasibility table has an unknown schema.")

    for root in (RANDOM_COMPILED, REPAIRED_COMPILED):
        observed_design = pd.read_csv(
            root / "System_Designs.csv.gz", nrows=0
        ).columns.tolist()
        observed_bits = pd.read_csv(
            root / "V1p1_Subsystem_Feasibility_Bits.csv.gz", nrows=0
        ).columns.tolist()
        if observed_design != design_columns or observed_bits != bit_columns:
            raise ValueError(f"Shared training schema differs in {root}.")
        observed_performance = pd.read_csv(
            root / "Ship_System_Performance.csv.gz", nrows=0
        ).columns
        if not set(PERFORMANCE_COLUMNS).issubset(observed_performance):
            raise ValueError(f"Required performance fields are absent in {root}.")

    return parameter_names, design_columns, bit_columns


def build_training_table() -> dict:
    """Stream source designs into one non-overwriting gzip CSV and manifest."""

    if OUTPUT_TABLE.exists() or OUTPUT_MANIFEST.exists():
        raise FileExistsError(
            f"Unified dataset already exists in {OUTPUT_ROOT}; refusing overwrite."
        )

    parameter_names, design_columns, bit_columns = read_shared_schema()

    # Random design IDs are the only authoritative route from a repair back to
    # its source structural family. Repaired Structure_Index values are new IDs.
    random_ids = pd.read_csv(
        RANDOM_COMPILED / "System_Designs.csv.gz",
        usecols=["Index", "Structure_Index"],
    )
    require_unique_index(random_ids, "random System_Designs")
    source_structure = random_ids.set_index("Index")["Structure_Index"]

    lineage_path = REPAIRED_COMPILED / "Repair_Lineage.csv.gz"
    lineage = pd.read_csv(lineage_path, usecols=list(LINEAGE_COLUMNS))
    require_unique_index(lineage, "Repair_Lineage")
    if not lineage["Parent_Index"].isin(source_structure.index).all():
        raise ValueError("At least one repaired row has no random source design.")

    output_columns = [
        "Population",
        "Index",
        *PROVENANCE_COLUMNS[1:],
        *(name for name in design_columns if name != "Index"),
        *(name for name in PERFORMANCE_COLUMNS if name != "Index"),
        *(name for name in bit_columns if name != "Index"),
    ]
    if len(output_columns) != len(set(output_columns)):
        raise ValueError("The unified training schema has duplicate column names.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_table = OUTPUT_ROOT / ".Training_Rows.csv.gz.tmp"
    if temporary_table.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_table}")

    counts = Counter()
    all_ids: set[int] = set()
    source_files: list[Path] = [
        RANDOM_COMPILED / "Parameter_Catalog.csv",
        lineage_path,
    ]

    try:
        # Write in source order. The explicit source and parent columns preserve
        # the lineage needed for leakage-safe train/validation/test splits.
        with temporary_table.open("wb") as raw_output, gzip.GzipFile(
            fileobj=raw_output, mode="wb", filename="", mtime=0
        ) as compressed_output, io.TextIOWrapper(
            compressed_output, encoding="utf-8", newline=""
        ) as output:
            first_chunk = True

            for population, root in (
                ("random", RANDOM_COMPILED),
                ("repaired", REPAIRED_COMPILED),
            ):
                design_path = root / "System_Designs.csv.gz"
                performance_path = root / "Ship_System_Performance.csv.gz"
                bit_path = root / "V1p1_Subsystem_Feasibility_Bits.csv.gz"
                source_files.extend((design_path, performance_path, bit_path))

                performance = pd.read_csv(
                    performance_path, usecols=list(PERFORMANCE_COLUMNS)
                )
                bits = pd.read_csv(bit_path, usecols=bit_columns)
                require_unique_index(performance, f"{population} performance")
                require_unique_index(bits, f"{population} feasibility bits")
                seen_in_population: set[int] = set()

                for design in pd.read_csv(design_path, chunksize=10_000):
                    require_unique_index(design, f"{population} design chunk")
                    chunk_ids = set(design["Index"].astype(int))
                    if chunk_ids & (seen_in_population | all_ids):
                        raise ValueError("Duplicate design IDs across sources or chunks.")
                    seen_in_population.update(chunk_ids)
                    all_ids.update(chunk_ids)

                    joined = join_all_rows(design, performance, "performance")
                    joined = join_all_rows(joined, bits, "feasibility bits")

                    if population == "repaired":
                        joined = join_all_rows(joined, lineage, "repair lineage")
                        joined["Source_Design_Index"] = joined["Parent_Index"]
                        joined["Source_Structure_Index"] = joined[
                            "Parent_Index"
                        ].map(source_structure)
                    else:
                        for name in LINEAGE_COLUMNS[1:]:
                            joined[name] = pd.NA
                        joined["Source_Design_Index"] = joined["Index"]
                        joined["Source_Structure_Index"] = joined["Structure_Index"]

                    completed = joined["Evaluation_Status"].eq("completed")
                    feasible = joined["Global_V1p1_Feasible"].eq(True)
                    joined["Population"] = population
                    joined["Verified_Global_Feasible"] = completed & feasible
                    joined["Training_Eligible_RP"] = (
                        True if population == "random" else completed
                    )

                    if population == "repaired":
                        lineage_feasible = joined["Fully_Feasible"].eq(True)
                        if not lineage_feasible.eq(
                            joined["Verified_Global_Feasible"]
                        ).all():
                            raise ValueError(
                                "Repair lineage and compiled global feasibility disagree."
                            )

                    counts[f"{population}_rows"] += len(joined)
                    counts[f"{population}_completed"] += int(completed.sum())
                    counts[f"{population}_training_eligible"] += int(
                        joined["Training_Eligible_RP"].sum()
                    )
                    counts[f"{population}_verified_feasible"] += int(
                        joined["Verified_Global_Feasible"].sum()
                    )

                    joined.loc[:, output_columns].to_csv(
                        output, index=False, header=first_chunk
                    )
                    first_chunk = False

                if seen_in_population != set(performance["Index"].astype(int)):
                    raise ValueError(f"{population} performance IDs do not match designs.")
                if seen_in_population != set(bits["Index"].astype(int)):
                    raise ValueError(f"{population} bit IDs do not match designs.")
                if population == "repaired" and seen_in_population != set(
                    lineage["Index"].astype(int)
                ):
                    raise ValueError("Repair lineage IDs do not match designs.")

        os.replace(temporary_table, OUTPUT_TABLE)

    except Exception:
        if temporary_table.exists():
            temporary_table.unlink()
        raise

    manifest = {
        "schema_version": 1,
        "description": "One row per random or repaired V1.1 design, including failed evaluations.",
        "table": str(OUTPUT_TABLE.relative_to(PROJECT_ROOT)),
        "row_count": int(sum(counts[f"{name}_rows"] for name in ("random", "repaired"))),
        "counts": dict(sorted(counts.items())),
        "parameter_count": len(parameter_names),
        "feasibility_class_count": len(bit_columns) - 1,
        "split_group_column": "Source_Structure_Index",
        "training_eligibility_rule": "All random rows; repaired rows only when Evaluation_Status is completed.",
        "feasibility_rule": "Evaluation_Status completed and Global_V1p1_Feasible true.",
        "note": (
            "Individual constraint margins remain in the two source compiled folders. "
            "Their long-form tables are not one-row-per-design training features. "
            "The two wide ML matrices have different columns and are not concatenated."
        ),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in source_files
        },
        "table_sha256": sha256_file(OUTPUT_TABLE),
    }
    with OUTPUT_MANIFEST.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    return manifest


if __name__ == "__main__":
    summary = build_training_table()
    print(json.dumps({"row_count": summary["row_count"], "counts": summary["counts"]}, indent=2))
