"""Load matching saved source rows for a system visualization.

The unified training table identifies designs, while the original random
population shards retain the detailed geometry and load data.  Shard names
use *structure* index ranges, even though each detail row uses a system Index.
Imported generated evaluations retain their own compiled performance and
updated-parameter records under an explicitly selected evaluation directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


_IDENTIFIER_COLUMNS = (
    "Index",
    "Structure_Index",
    "Hull_Variant_Index",
    "Operating_Condition_Index",
    "Hull_Candidate_Index",
    "Structure_Sample_Attempt",
)

_DETAIL_FAMILIES = {
    "bulkheads": ("bulkhead_locations", "Bulkhead_Locations"),
    "hatches": ("hatch_covers", "Hatch_Covers"),
    "compartments": ("compartment_loads", "Compartment_Loads"),
    "weight_components": ("weight_components", "Weight_Components"),
}


@dataclass(frozen=True)
class VisualizationDesign:
    """One unified design and its matching source geometry/detail rows."""

    population: str
    index: int
    structure_index: int
    training_row: pd.Series
    system_row: pd.Series
    structural_row: pd.Series
    structural_elements: pd.DataFrame
    structural_elements_path: Path
    bulkheads: pd.DataFrame
    hatches: pd.DataFrame
    compartments: pd.DataFrame
    weight_components: pd.DataFrame
    provenance_paths: dict[str, Path]


def _training_row(training_path: Path, population: str, index: int) -> pd.Series:
    """Locate one design without materializing the entire training table."""

    for chunk in pd.read_csv(training_path, chunksize=1_000, low_memory=False):
        matches = chunk.loc[
            (chunk["Population"] == population) & (chunk["Index"] == index)
        ]

        if not matches.empty:
            return matches.iloc[0].copy()

    raise ValueError(f"No unified training row for {population} design {index}.")


def _find_shard(directory: Path, prefix: str, structure_index: int) -> Path:
    """Resolve the unique compressed shard that contains a structure index."""

    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)_(\d+)\.csv\.gz")
    matches = []

    for path in directory.glob(f"{prefix}_*.csv.gz"):
        range_match = pattern.fullmatch(path.name)

        if range_match is None:
            continue

        lower, upper = (int(value) for value in range_match.groups())

        if lower <= structure_index <= upper:
            matches.append(path)

    if len(matches) != 1:
        raise ValueError(
            f"Expected one {prefix} shard for Structure_Index "
            f"{structure_index}; found {len(matches)} in {directory}."
        )

    return matches[0]


def _one_row(path: Path, column: str, value: int) -> pd.Series:
    """Read one system or structure row from its source shard."""

    table = pd.read_csv(path, low_memory=False)
    matches = table.loc[table[column] == value]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one {column}={value} row in {path}; "
            f"found {len(matches)}."
        )

    return matches.iloc[0].copy()


def _detail_rows(path: Path, index: int) -> pd.DataFrame:
    """Read the rows for one system from a structure-range detail shard."""

    table = pd.read_csv(path, low_memory=False)
    rows = table.loc[table["Index"] == index].copy()

    if rows.empty:
        raise ValueError(f"No Index={index} detail rows in {path}.")

    return rows.reset_index(drop=True)


def _verify_identifiers(
    training_row: pd.Series,
    system_row: pd.Series,
    structural_row: pd.Series,
    details: dict[str, pd.DataFrame],
) -> None:
    """Check each shared source identifier before combining the tables."""

    for column in _IDENTIFIER_COLUMNS:
        expected = int(training_row[column])

        if int(system_row[column]) != expected:
            raise ValueError(f"System row differs from training row at {column}.")

        for family, rows in details.items():
            if not rows[column].eq(expected).all():
                raise ValueError(
                    f"{family} rows differ from training row at {column}."
                )

    for column in ("Structure_Index", "Structure_Sample_Attempt"):
        if int(structural_row[column]) != int(training_row[column]):
            raise ValueError(
                f"Structural row differs from training row at {column}."
            )

    if int(training_row["Source_Design_Index"]) != int(system_row["Index"]):
        raise ValueError("Unified source design index differs from source system.")

    if int(training_row["Source_Structure_Index"]) != int(
        structural_row["Structure_Index"]
    ):
        raise ValueError(
            "Unified source structure index differs from source structure."
        )


def load_visualization_design(
    repo_root: Path,
    population: str,
    index: int,
) -> VisualizationDesign:
    """Load one completed random-population design for CAD visualization.

    The structural-elements path embedded in historical structure rows can be
    an obsolete cluster absolute path.  This loader uses the same named file
    from the current repository's ``Dataset_100k/structures/elements`` folder.
    """

    if population != "random":
        raise ValueError("Visualization loading currently supports random only.")

    repo_root = Path(repo_root).resolve()
    designs_root = repo_root / "Ship_Systems_Designs_v1p1"
    source_root = designs_root / "Dataset_100k"
    training_path = (
        designs_root / "Unified_Training_Dataset" / "Training_Rows.csv.gz"
    )

    # The training row provides the exact source system and structure keys.
    training_row = _training_row(training_path, population, index)

    if training_row["Evaluation_Status"] != "completed":
        raise ValueError(
            f"Unified {population} design {index} has no completed evaluation."
        )

    source_index = int(training_row["Source_Design_Index"])
    structure_index = int(training_row["Source_Structure_Index"])

    # Source shard filenames encode Structure_Index intervals, not Index.
    system_path = _find_shard(
        source_root / "systems" / "shards",
        "System_Designs",
        structure_index,
    )
    structural_path = _find_shard(
        source_root / "structures" / "parameters",
        "Structure_Parameters",
        structure_index,
    )
    system_row = _one_row(system_path, "Index", source_index)
    structural_row = _one_row(
        structural_path, "Structure_Index", structure_index
    )

    # Resolve the local structural-elements file by the accepted structure ID.
    structural_elements_path = (
        source_root
        / "structures"
        / "elements"
        / f"Structure_{structure_index:06d}_Structural_Elements.csv"
    )
    structural_elements = pd.read_csv(
        structural_elements_path, low_memory=False
    )

    # Detail families all carry the same identifiers as the source system row.
    details = {}
    provenance_paths = {
        "training": training_path,
        "system": system_path,
        "structural_parameters": structural_path,
        "structural_elements": structural_elements_path,
    }

    for family, (directory, prefix) in _DETAIL_FAMILIES.items():
        path = _find_shard(
            source_root / "evaluations" / directory,
            prefix,
            structure_index,
        )
        details[family] = _detail_rows(path, source_index)
        provenance_paths[family] = path

    _verify_identifiers(training_row, system_row, structural_row, details)

    return VisualizationDesign(
        population=population,
        index=index,
        structure_index=structure_index,
        training_row=training_row,
        system_row=system_row,
        structural_row=structural_row,
        structural_elements=structural_elements,
        structural_elements_path=structural_elements_path,
        bulkheads=details["bulkheads"],
        hatches=details["hatches"],
        compartments=details["compartments"],
        weight_components=details["weight_components"],
        provenance_paths=provenance_paths,
    )


def load_evaluated_visualization_design(
    evaluation_root: Path,
    population: str,
    index: int,
) -> VisualizationDesign:
    """Load a generated design from its saved exact-evaluation directory.

    ``population`` identifies the experiment as well as the architecture:
    generated indices are reused in separate experiments. Updated parameters
    and geometry details belong to the completed evaluation, not the proposal.
    """

    evaluation_root = Path(evaluation_root).resolve()
    performance_path = evaluation_root / "compiled" / "Performance.csv.gz"
    performance_row = _one_row(performance_path, "Index", index)
    if performance_row["Evaluation_Status"] != "completed":
        raise ValueError(f"Generated design {index} has no completed evaluation.")

    # The generated evaluation stores one structure and system per index.
    structure_index = int(performance_row["Structure_Index"])
    system_path = _find_shard(
        evaluation_root / "evaluations" / "updated_parameters",
        "Updated_System_Parameters", structure_index,
    )
    system_row = _one_row(system_path, "Index", index)
    structural_elements_path = (
        evaluation_root / "structures" / "elements"
        / f"Structure_{structure_index:06d}_Structural_Elements.csv"
    )

    # Reuse the source-identity checks used by the training-design loader.
    performance_row["Source_Design_Index"] = index
    performance_row["Source_Structure_Index"] = structure_index
    details = {}
    provenance_paths = {
        "performance": performance_path,
        "system": system_path,
        "structural_parameters": system_path,
        "structural_elements": structural_elements_path,
    }
    for family, (directory, prefix) in _DETAIL_FAMILIES.items():
        path = _find_shard(evaluation_root / "evaluations" / directory,
                           prefix, structure_index)
        details[family] = _detail_rows(path, index)
        provenance_paths[family] = path
    _verify_identifiers(performance_row, system_row, system_row, details)

    return VisualizationDesign(
        population=population,
        index=index,
        structure_index=structure_index,
        training_row=performance_row,
        system_row=system_row,
        structural_row=system_row,
        structural_elements=pd.read_csv(structural_elements_path),
        structural_elements_path=structural_elements_path,
        bulkheads=details["bulkheads"],
        hatches=details["hatches"],
        compartments=details["compartments"],
        weight_components=details["weight_components"],
        provenance_paths=provenance_paths,
    )
