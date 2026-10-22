"""Load the 0--399 pilot data and freeze reproducible repair batches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .configuration import RepairConfiguration
from .parameters import ParameterRegistry


IDENTIFIER_COLUMNS = [
    "Index",
    "Structure_Index",
    "Hull_Variant_Index",
    "Operating_Condition_Index",
    "Hull_Candidate_Index",
    "Structure_Sample_Attempt",
]

FROZEN_SUBSET_FILENAMES = (
    "selected_source_designs.csv",
    "original_189_parameter_vectors.csv.gz",
    "original_constraint_results.csv.gz",
    "original_performance_metrics.csv.gz",
    "original_evaluation_failures.csv.gz",
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one immutable experiment input."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)

    return digest.hexdigest()


def write_csv_atomic(table: pd.DataFrame, path: str | Path) -> Path:
    """Write one complete CSV (optionally gzip) through a temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compression = (
        {"method": "gzip", "compresslevel": 6, "mtime": 0}
        if path.suffix == ".gz"
        else None
    )
    table.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)
    return path


def write_json_atomic(payload: dict, path: str | Path) -> Path:
    """Write one stable, human-readable JSON artifact atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


class SourceDataset:
    """Provide indexed access to vectors, constraints, and diagnostics."""

    def __init__(
        self,
        source_root: str | Path,
        parameter_registry: ParameterRegistry | None = None,
    ):
        self.root = Path(source_root).resolve()
        self.compiled = self.root / "compiled"
        self.parameters = parameter_registry or ParameterRegistry()

        def compiled_table(filename: str) -> Path:
            """Resolve either ordinary or reduced-mode compiled CSV."""

            plain = self.compiled / filename

            if plain.exists():
                return plain

            compressed = plain.with_name(f"{plain.name}.gz")

            if compressed.exists():
                return compressed

            # Return the preferred reduced-mode path so the consolidated
            # missing-file message remains explicit below.
            return compressed

        required = {
            "source_vectors": compiled_table("System_Designs.csv"),
            "evaluated_vectors": compiled_table(
                "Updated_Ship_System_Parameters.csv"
            ),
            "performance": compiled_table("Ship_System_Performance.csv"),
            "failures": compiled_table("Evaluation_Failures.csv"),
            "constraint_catalog": compiled_table("Constraint_Catalog.csv"),
        }

        # The standalone violation-count table is produced by the plotting
        # helper after compilation, not by Compile_V1p1_Dataset.py itself.
        # Production repair therefore treats it as an optional convenience
        # and derives the same selection columns from the compiled performance
        # table when it has not been created.
        violation_counts_path = (
            self.compiled / "Constraint_Violation_Counts.csv"
        )

        if violation_counts_path.exists():
            required["violation_counts"] = violation_counts_path
        elif violation_counts_path.with_name(
            f"{violation_counts_path.name}.gz"
        ).exists():
            required["violation_counts"] = violation_counts_path.with_name(
                f"{violation_counts_path.name}.gz"
            )

        missing = [str(path) for path in required.values() if not path.exists()]

        if missing:
            raise FileNotFoundError(
                "The repair source dataset is incomplete: " + ", ".join(missing)
            )

        self.paths = required
        shard_root = self.root / "evaluations" / "constraints_v1p1" / "shards"
        self.constraint_shard_paths = tuple(
            sorted(shard_root.glob("Constraints_v1p1_*.csv*"))
        )

        if not self.constraint_shard_paths:
            raise FileNotFoundError(
                f"No V1.1 constraint shards were found under {shard_root}."
            )

        self.source_vectors = pd.read_csv(required["source_vectors"])
        self.evaluated_vectors = pd.read_csv(required["evaluated_vectors"])
        self.performance = pd.read_csv(required["performance"])
        self.failures = pd.read_csv(required["failures"])
        self.constraint_catalog = pd.read_csv(required["constraint_catalog"])

        # Integer-to-row maps keep repeated production repair lookups O(1)
        # without duplicating the wide 189-coordinate dataframes in memory.
        self._source_vector_positions = {
            int(index): position
            for position, index in enumerate(self.source_vectors["Index"])
        }
        self._evaluated_vector_positions = {
            int(index): position
            for position, index in enumerate(self.evaluated_vectors["Index"])
        }
        self._performance_positions = {
            int(index): position
            for position, index in enumerate(self.performance["Index"])
        }

        for label, table, positions in (
            (
                "compiled source vectors",
                self.source_vectors,
                self._source_vector_positions,
            ),
            (
                "compiled evaluated vectors",
                self.evaluated_vectors,
                self._evaluated_vector_positions,
            ),
            (
                "compiled performance rows",
                self.performance,
                self._performance_positions,
            ),
        ):
            if len(positions) != len(table):
                raise ValueError(f"Duplicate Index values in {label}.")

        if "violation_counts" in required:
            self.violation_counts = pd.read_csv(
                required["violation_counts"]
            )
        else:
            count_columns = [
                "Index",
                "Structure_Index",
                "Hull_Variant_Index",
                "Operating_Condition_Index",
                "Evaluation_Status",
                "V1p1_Evaluated_Constraint_Count",
                "V1p1_Constraint_Fail_Count",
                "Failure_Stage",
                "Failure_Category",
                "Error_Type",
                "Error_Message",
            ]
            missing_count_columns = [
                column
                for column in count_columns
                if column not in self.performance.columns
            ]

            if missing_count_columns:
                raise ValueError(
                    "The compiled performance table cannot supply repair "
                    "selection counts; missing columns: "
                    + ", ".join(missing_count_columns)
                )

            self.violation_counts = self.performance[count_columns].copy()

        self._constraint_cache: pd.DataFrame | None = None
        self._constraint_shard_cache_path: Path | None = None
        self._constraint_shard_cache: pd.DataFrame | None = None
        self._constraint_shard_ranges = self._parse_constraint_shard_ranges()

    def _parse_constraint_shard_ranges(
        self,
    ) -> tuple[tuple[int, int, Path], ...]:
        """Index the evaluator's design-range filenames once.

        The production dataset contains hundreds of compressed constraint
        shards.  Reading every shard in every SLURM repair task would consume
        unnecessary memory.  Their standard ``start_end`` filenames encode
        the source Structure_Index interval and therefore provide an exact
        lookup for the one source design currently being repaired.
        """

        pattern = re.compile(
            r"^Constraints_v1p1_(\d+)_(\d+)\.csv(?:\.gz)?$"
        )
        ranges = []

        for path in self.constraint_shard_paths:
            match = pattern.match(path.name)

            if match is None:
                continue

            ranges.append((int(match.group(1)), int(match.group(2)), path))

        return tuple(sorted(ranges))

    def completed_below_median(
        self,
        excluded_indices: set[int] | None = None,
    ) -> tuple[pd.DataFrame, float]:
        """Return completed below-median designs after explicit exclusions."""

        completed = self.violation_counts.loc[
            self.violation_counts["Evaluation_Status"].eq("completed")
        ].copy()

        # Candidate-selection orchestration can exclude a completed scalar row
        # when its detailed loading-condition solution is numerically invalid.
        # Existing repair workflows pass no exclusions and retain their prior
        # population definition unchanged.
        if excluded_indices:
            completed = completed.loc[
                ~completed["Index"].astype(int).isin(excluded_indices)
            ].copy()

        median_failures = float(
            completed["V1p1_Constraint_Fail_Count"].median()
        )
        eligible = completed.loc[
            completed["V1p1_Constraint_Fail_Count"] < median_failures
        ].sort_values("Index").reset_index(drop=True)
        return eligible, median_failures

    def vector(self, index: int) -> np.ndarray:
        """Return the completed evaluator's full 189-coordinate vector."""

        position = self._evaluated_vector_positions.get(int(index))

        if position is None:
            raise KeyError(
                f"Completed evaluated vector for source Index {index} "
                "was not found exactly once."
            )

        return self.parameters.vector_from_row(
            self.evaluated_vectors.iloc[position]
        )

    def identifiers(self, index: int) -> dict[str, int]:
        """Return the six stable source identifiers."""

        position = self._source_vector_positions.get(int(index))

        if position is None:
            raise KeyError(f"Source Index {index} was not found exactly once.")

        row = self.source_vectors.iloc[position]
        return {name: int(row[name]) for name in IDENTIFIER_COLUMNS}

    def all_constraints(self) -> pd.DataFrame:
        """Load all long-form V1.1 constraint shards once, preserving rows."""

        if self._constraint_cache is not None:
            return self._constraint_cache

        self._constraint_cache = pd.concat(
            [pd.read_csv(path) for path in self.constraint_shard_paths],
            ignore_index=True,
        )
        return self._constraint_cache

    def constraints(self, index: int) -> pd.DataFrame:
        """Return every applicable result row for one source design."""

        table = None

        # Prefer one structure-range shard so a production-array task does not
        # load the full 100,000-design long-form constraint table. Keep only
        # the most recently used shard because selections are normally ordered
        # by parent design and therefore also group nearby structures.
        structure_index = int(self.identifiers(index)["Structure_Index"])

        for start_index, end_index, path in self._constraint_shard_ranges:
            if start_index <= structure_index <= end_index:
                if self._constraint_shard_cache_path != path:
                    self._constraint_shard_cache = pd.read_csv(path)
                    self._constraint_shard_cache_path = path

                table = self._constraint_shard_cache
                break

        # Preserve compatibility with nonstandard test or legacy filenames.
        if table is None:
            table = self.all_constraints()

        return table.loc[table["Index"].astype(int) == int(index)].copy()

    def performance_row(self, index: int) -> pd.DataFrame:
        """Return the scalar evaluator row for one source design."""

        position = self._performance_positions.get(int(index))

        if position is None:
            return self.performance.iloc[0:0].copy()

        return self.performance.iloc[[position]].copy()

    def failure_rows(self, index: int) -> pd.DataFrame:
        """Return structured evaluator failures for one source design."""

        if self.failures.empty:
            return self.failures.copy()

        return self.failures.loc[
            self.failures["Index"].astype(int) == int(index)
        ].copy()

    def input_fingerprints(self) -> dict[str, dict[str, str | int]]:
        """Bind a repair experiment to the exact compiled source artifacts."""

        result = {}

        for label, path in self.input_paths().items():
            result[label] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        return result

    def input_paths(self) -> dict[str, Path]:
        """Return every file that can affect frozen repair inputs.

        The long-form constraint rows are read from the uncompiled shards,
        not from ``Constraint_Catalog.csv``.  Listing every shard here closes
        a provenance gap in the original selection manifest.
        """

        paths = dict(self.paths)

        for shard in self.constraint_shard_paths:
            paths[f"constraint_shard::{shard.name}"] = shard

        return paths


def frozen_subset_fingerprints(
    output_directory: str | Path,
) -> dict[str, dict[str, str | int]]:
    """Hash each saved subset whose contents define a frozen batch."""

    output_directory = Path(output_directory)
    result = {}

    for filename in FROZEN_SUBSET_FILENAMES:
        path = output_directory / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Frozen batch artifact is missing: {path}"
            )

        result[filename] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return result


def _validate_or_upgrade_saved_manifest(
    dataset: SourceDataset,
    output_directory: Path,
    manifest: dict,
) -> dict:
    """Validate committed hashes and add provenance absent from schema 1.

    Existing recorded hashes are always checked before any schema upgrade.
    This permits the already-frozen development and validation selections to
    gain the missing shard/subset records without masking later mutation.
    """

    current_sources = dataset.input_fingerprints()
    saved_sources = manifest.get("source_fingerprints", {})

    if (
        int(manifest.get("schema_version", 1)) >= 2
        and set(saved_sources) != set(current_sources)
    ):
        raise RuntimeError(
            "The committed set of repair source files changed after selection."
        )

    for label, saved in saved_sources.items():
        current = current_sources.get(label)

        if current is None or current["sha256"] != saved.get("sha256"):
            raise RuntimeError(
                f"Committed repair source changed after selection: {label}"
            )

    current_subsets = frozen_subset_fingerprints(output_directory)
    saved_subsets = manifest.get("frozen_subset_fingerprints", {})

    if (
        int(manifest.get("schema_version", 1)) >= 2
        and set(saved_subsets) != set(current_subsets)
    ):
        raise RuntimeError(
            "The committed set of frozen subset files changed after selection."
        )

    for label, saved in saved_subsets.items():
        current = current_subsets.get(label)

        if current is None or current["sha256"] != saved.get("sha256"):
            raise RuntimeError(
                f"Committed frozen subset changed after selection: {label}"
            )

    upgraded = dict(manifest)
    upgraded["schema_version"] = 2
    upgraded["source_fingerprints"] = current_sources
    upgraded["frozen_subset_fingerprints"] = current_subsets

    if upgraded != manifest:
        write_json_atomic(
            upgraded,
            output_directory / "selection_manifest.json",
        )

    return upgraded


def freeze_random_batch(
    dataset: SourceDataset,
    output_directory: str | Path,
    *,
    seed: int,
    batch_size: int = 10,
    excluded_indices: set[int] | None = None,
    batch_kind: str = "development",
) -> pd.DataFrame:
    """Randomly select and commit one immutable below-median repair batch."""

    if not 1 <= int(batch_size) <= 10:
        raise ValueError("A repair batch must contain between 1 and 10 designs.")

    output_directory = Path(output_directory)
    selection_path = output_directory / "selected_source_designs.csv"
    manifest_path = output_directory / "selection_manifest.json"

    # Never redraw a committed batch to improve its eventual repair outcome.
    if selection_path.exists() or manifest_path.exists():
        if not (selection_path.exists() and manifest_path.exists()):
            raise RuntimeError(
                "The batch-selection commit is incomplete; preserve it for "
                "audit and choose a new batch directory."
            )

        saved = pd.read_csv(selection_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if int(manifest["batch_seed"]) != int(seed):
            raise ValueError(
                "The requested seed differs from the already committed batch."
            )

        _validate_or_upgrade_saved_manifest(
            dataset,
            output_directory,
            manifest,
        )

        return saved

    eligible, median_failures = dataset.completed_below_median()
    excluded = set() if excluded_indices is None else set(excluded_indices)
    eligible = eligible.loc[
        ~eligible["Index"].astype(int).isin(excluded)
    ].copy()

    if len(eligible) < int(batch_size):
        raise ValueError("The eligible population is smaller than the batch.")

    rng = np.random.default_rng(int(seed))
    selected_indices = rng.choice(
        eligible["Index"].astype(int).to_numpy(),
        size=int(batch_size),
        replace=False,
    )

    # Draw one stable uint32 child seed for each selected design.  Each repair
    # uses its own generator, so changing one design's iteration count cannot
    # shift the random margins of another design.
    design_seeds = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=int(batch_size),
        dtype=np.uint32,
    )
    selected_lookup = eligible.set_index("Index")
    rows = []

    for draw_order, (index, design_seed) in enumerate(
        zip(selected_indices, design_seeds, strict=True)
    ):
        source = selected_lookup.loc[int(index)]
        identifiers = dataset.identifiers(int(index))
        rows.append(
            {
                "Batch_Kind": batch_kind,
                "Selection_Order": draw_order,
                "Index": int(index),
                "Structure_Index": identifiers["Structure_Index"],
                "Hull_Variant_Index": identifiers["Hull_Variant_Index"],
                "Operating_Condition_Index": identifiers[
                    "Operating_Condition_Index"
                ],
                "Original_V1p1_Constraint_Fail_Count": int(
                    source["V1p1_Constraint_Fail_Count"]
                ),
                "Completed_Population_Median_Fail_Count": median_failures,
                "Batch_Seed": int(seed),
                "Design_Repair_Seed": int(design_seed),
            }
        )

    selection = pd.DataFrame(rows)
    write_csv_atomic(selection, selection_path)

    vectors = []
    constraints = []
    performance = []
    failures = []

    for index in selection["Index"].astype(int):
        vector_row = {
            "Parent_Index": int(index),
            **dataset.parameters.row_from_vector(dataset.vector(index)),
        }
        vectors.append(vector_row)
        constraints.append(dataset.constraints(index))
        performance.append(dataset.performance_row(index))
        source_failures = dataset.failure_rows(index)

        if not source_failures.empty:
            failures.append(source_failures)

    write_csv_atomic(
        pd.DataFrame(vectors),
        output_directory / "original_189_parameter_vectors.csv.gz",
    )
    write_csv_atomic(
        pd.concat(constraints, ignore_index=True),
        output_directory / "original_constraint_results.csv.gz",
    )
    write_csv_atomic(
        pd.concat(performance, ignore_index=True),
        output_directory / "original_performance_metrics.csv.gz",
    )
    write_csv_atomic(
        (
            pd.concat(failures, ignore_index=True)
            if failures
            else dataset.failures.iloc[0:0].copy()
        ),
        output_directory / "original_evaluation_failures.csv.gz",
    )

    write_json_atomic(
        {
            "schema_version": 2,
            "batch_kind": batch_kind,
            "batch_seed": int(seed),
            "batch_size": int(batch_size),
            "selection_rule": (
                "completed source evaluation AND V1p1 fail count strictly "
                "below completed-population median; random without replacement"
            ),
            "completed_population_median_fail_count": median_failures,
            "eligible_population_count_after_exclusions": int(len(eligible)),
            "excluded_indices": sorted(int(value) for value in excluded),
            "selected_indices_in_draw_order": [
                int(value) for value in selected_indices
            ],
            "source_fingerprints": dataset.input_fingerprints(),
            "frozen_subset_fingerprints": frozen_subset_fingerprints(
                output_directory
            ),
        },
        manifest_path,
    )
    return selection


def freeze_below_median_alternatives(
    dataset: SourceDataset,
    output_directory: str | Path,
    *,
    seed: int,
    alternatives_per_source: int = 2,
    batch_kind: str = "below_median",
    save_source_snapshots: bool = True,
) -> pd.DataFrame:
    """Freeze every partially feasible below-median source for repair.

    The parent population is deterministic: all completed designs with more
    than zero failures and strictly fewer failures than the completed-
    population median are retained in ascending source-index order.  The
    seeded generator supplies a separate stochastic repair seed for each
    alternative, preserving repair diversity and reproducibility.  Small
    experiments retain full frozen source snapshots.  The production cluster
    selection can omit those duplicate tables because its manifest binds the
    immutable compiled source and constraint shards directly.
    """

    alternatives_per_source = int(alternatives_per_source)

    if alternatives_per_source < 1:
        raise ValueError("alternatives_per_source must be at least one.")

    output_directory = Path(output_directory)
    selection_path = output_directory / "selected_source_designs.csv"
    manifest_path = output_directory / "selection_manifest.json"

    # Reuse an immutable committed selection only when its defining seed,
    # alternative count, and evidence-storage contract match exactly.
    if selection_path.exists() or manifest_path.exists():
        if not (selection_path.exists() and manifest_path.exists()):
            raise RuntimeError(
                "The below-median selection commit is incomplete; preserve "
                "it for audit and choose a new batch directory."
            )

        saved = pd.read_csv(selection_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if int(manifest["batch_seed"]) != int(seed):
            raise ValueError(
                "The requested seed differs from the committed below-median "
                "batch."
            )

        if (
            int(manifest["alternatives_per_source"])
            != alternatives_per_source
        ):
            raise ValueError(
                "The requested alternatives-per-source count differs from "
                "the committed below-median batch."
            )

        saved_snapshot_mode = bool(
            manifest.get("save_source_snapshots", True)
        )

        if saved_snapshot_mode != bool(save_source_snapshots):
            raise ValueError(
                "The requested source-snapshot mode differs from the "
                "committed below-median batch."
            )

        if save_source_snapshots:
            _validate_or_upgrade_saved_manifest(
                dataset,
                output_directory,
                manifest,
            )
        else:
            current_sources = dataset.input_fingerprints()

            if manifest.get("source_fingerprints") != current_sources:
                raise RuntimeError(
                    "The production repair source files changed after "
                    "selection was frozen."
                )

        return saved

    eligible, median_failures = dataset.completed_below_median()
    eligible = eligible.loc[
        eligible["V1p1_Constraint_Fail_Count"].astype(float) > 0.0
    ].sort_values("Index").reset_index(drop=True)

    if eligible.empty:
        raise ValueError(
            "No completed, partially feasible designs are strictly below "
            "the completed-population median."
        )

    rng = np.random.default_rng(int(seed))
    number_of_repairs = len(eligible) * alternatives_per_source
    design_seeds = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=number_of_repairs,
        dtype=np.uint32,
    )
    rows = []
    selection_order = 0

    # Keep parents ordered for transparent dataset lineage.  Diversity comes
    # from the two independent stochastic margin/choice streams, not from
    # randomly omitting otherwise eligible parent designs.
    for parent_order, (_, source) in enumerate(eligible.iterrows()):
        index = int(source["Index"])
        identifiers = dataset.identifiers(index)

        for alternative_number in range(1, alternatives_per_source + 1):
            rows.append(
                {
                    "Batch_Kind": batch_kind,
                    "Selection_Order": selection_order,
                    "Parent_Order": parent_order,
                    "Alternative_Number": alternative_number,
                    "Index": index,
                    "Structure_Index": identifiers["Structure_Index"],
                    "Hull_Variant_Index": identifiers[
                        "Hull_Variant_Index"
                    ],
                    "Operating_Condition_Index": identifiers[
                        "Operating_Condition_Index"
                    ],
                    "Original_V1p1_Constraint_Fail_Count": int(
                        source["V1p1_Constraint_Fail_Count"]
                    ),
                    "Completed_Population_Median_Fail_Count": (
                        median_failures
                    ),
                    "Batch_Seed": int(seed),
                    "Design_Repair_Seed": int(
                        design_seeds[selection_order]
                    ),
                }
            )
            selection_order += 1

    selection = pd.DataFrame(rows)
    write_csv_atomic(selection, selection_path)

    if save_source_snapshots:
        vectors = []
        constraints = []
        performance = []
        failures = []

        # Save each source exactly once even though it owns several
        # alternative repair rows.  Alternative-specific vectors and
        # evaluations are written later by the repair pipeline.
        for index in eligible["Index"].astype(int):
            vectors.append(
                {
                    "Parent_Index": int(index),
                    **dataset.parameters.row_from_vector(
                        dataset.vector(index)
                    ),
                }
            )
            constraints.append(dataset.constraints(index))
            performance.append(dataset.performance_row(index))
            source_failures = dataset.failure_rows(index)

            if not source_failures.empty:
                failures.append(source_failures)

        write_csv_atomic(
            pd.DataFrame(vectors),
            output_directory / "original_189_parameter_vectors.csv.gz",
        )
        write_csv_atomic(
            pd.concat(constraints, ignore_index=True),
            output_directory / "original_constraint_results.csv.gz",
        )
        write_csv_atomic(
            pd.concat(performance, ignore_index=True),
            output_directory / "original_performance_metrics.csv.gz",
        )
        write_csv_atomic(
            (
                pd.concat(failures, ignore_index=True)
                if failures
                else dataset.failures.iloc[0:0].copy()
            ),
            output_directory / "original_evaluation_failures.csv.gz",
        )

        subset_fingerprints = frozen_subset_fingerprints(output_directory)
    else:
        subset_fingerprints = {}

    write_json_atomic(
        {
            "schema_version": 2,
            "batch_kind": batch_kind,
            "batch_seed": int(seed),
            "alternatives_per_source": alternatives_per_source,
            "source_design_count": int(len(eligible)),
            "repair_design_count": int(len(selection)),
            "save_source_snapshots": bool(save_source_snapshots),
            "selection_rule": (
                "completed source evaluation AND 0 < V1p1 fail count "
                "strictly below the completed-population median; every "
                "eligible source retained; independently seeded one-pass "
                "alternatives per source"
            ),
            "completed_population_median_fail_count": median_failures,
            "selected_parent_indices": (
                eligible["Index"].astype(int).tolist()
            ),
            "source_fingerprints": dataset.input_fingerprints(),
            "frozen_subset_fingerprints": subset_fingerprints,
        },
        manifest_path,
    )
    return selection


def read_candidate_indices(path: str | Path) -> list[int]:
    """Read one headered or headerless column of unique design indices."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Candidate-index file was not found: {path}")

    table = pd.read_csv(path, header=None, comment="#")

    if table.shape[1] != 1:
        raise ValueError(
            "Candidate_Indices.csv must contain exactly one column."
        )

    raw = table.iloc[:, 0]
    numeric = pd.to_numeric(raw, errors="coerce")

    # Accept either a descriptive first-row header or a purely numeric file.
    # Any nonnumeric value after the first row remains an input error.
    if len(numeric) > 0 and pd.isna(numeric.iloc[0]):
        numeric = numeric.iloc[1:]

    if numeric.empty or numeric.isna().any():
        raise ValueError(
            "Candidate_Indices.csv must contain only integer design indices "
            "after its optional first-row header."
        )

    values = numeric.to_numpy(dtype=float)

    if not np.all(np.equal(values, np.floor(values))):
        raise ValueError("Every candidate design index must be an integer.")

    indices = [int(value) for value in values]

    if any(index < 0 for index in indices):
        raise ValueError("Candidate design indices must be nonnegative.")

    if len(indices) != len(set(indices)):
        raise ValueError(
            "Candidate_Indices.csv contains duplicate design indices."
        )

    return indices


def freeze_candidate_index_alternatives(
    dataset: SourceDataset,
    candidate_path: str | Path,
    output_directory: str | Path,
    *,
    seed: int,
    alternatives_per_source: int = 3,
    batch_kind: str = "candidate_indices",
) -> pd.DataFrame:
    """Freeze three seeded repairs for each explicitly listed candidate.

    The candidate CSV is the complete selection authority.  No median,
    ranking, or additional feasibility filter changes its membership.  The
    validation below only establishes that every listed parent completed the
    source evaluator and therefore has the vector and constraint evidence
    required by the one-pass repair method.
    """

    candidate_path = Path(candidate_path).resolve()
    output_directory = Path(output_directory)
    alternatives_per_source = int(alternatives_per_source)

    if alternatives_per_source < 1:
        raise ValueError("alternatives_per_source must be at least one.")

    candidate_indices = read_candidate_indices(candidate_path)
    selection_path = output_directory / "selected_source_designs.csv"
    manifest_path = output_directory / "selection_manifest.json"
    candidate_fingerprint = {
        "path": str(candidate_path),
        "size_bytes": candidate_path.stat().st_size,
        "sha256": sha256_file(candidate_path),
    }

    if selection_path.exists() or manifest_path.exists():
        if not (selection_path.exists() and manifest_path.exists()):
            raise RuntimeError(
                "The candidate-index selection commit is incomplete; "
                "preserve it for audit and choose a new output directory."
            )

        saved = pd.read_csv(selection_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        expected = {
            "batch_seed": int(seed),
            "alternatives_per_source": alternatives_per_source,
            "candidate_indices_fingerprint": candidate_fingerprint,
        }

        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"Committed candidate repair selection changed: {key}."
                )

        if manifest.get("source_fingerprints") != dataset.input_fingerprints():
            raise RuntimeError(
                "The compiled source dataset changed after candidate repair "
                "selection was frozen."
            )

        return saved

    count_lookup = dataset.violation_counts.copy()
    count_lookup["Index"] = count_lookup["Index"].astype(int)
    count_lookup = count_lookup.set_index("Index", drop=False)
    rows = []
    rng = np.random.default_rng(int(seed))
    repair_count = len(candidate_indices)*alternatives_per_source
    design_seeds = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=repair_count,
        dtype=np.uint32,
    )
    selection_order = 0

    for parent_order, index in enumerate(candidate_indices):
        try:
            matches = count_lookup.loc[[int(index)]]
        except KeyError:
            matches = count_lookup.iloc[0:0]

        if len(matches) != 1:
            raise ValueError(
                f"Candidate Index {index} was not found exactly once in the "
                "compiled source population."
            )

        source = matches.iloc[0]

        if str(source["Evaluation_Status"]) != "completed":
            raise ValueError(
                f"Candidate Index {index} did not complete source "
                "evaluation and cannot supply a repair constraint vector."
            )

        # These accessors additionally prove that the evaluated 189-value
        # vector and original identifier row exist exactly once.
        dataset.vector(index)
        identifiers = dataset.identifiers(index)

        for alternative_number in range(1, alternatives_per_source + 1):
            rows.append(
                {
                    "Batch_Kind": batch_kind,
                    "Selection_Order": selection_order,
                    "Parent_Order": parent_order,
                    "Alternative_Number": alternative_number,
                    "Index": int(index),
                    "Structure_Index": identifiers["Structure_Index"],
                    "Hull_Variant_Index": identifiers[
                        "Hull_Variant_Index"
                    ],
                    "Operating_Condition_Index": identifiers[
                        "Operating_Condition_Index"
                    ],
                    "Original_V1p1_Constraint_Fail_Count": int(
                        source["V1p1_Constraint_Fail_Count"]
                    ),
                    "Batch_Seed": int(seed),
                    "Design_Repair_Seed": int(
                        design_seeds[selection_order]
                    ),
                }
            )
            selection_order += 1

    selection = pd.DataFrame(rows)
    write_csv_atomic(selection, selection_path)
    source_fingerprints = dataset.input_fingerprints()
    write_json_atomic(
        {
            "schema_version": 1,
            "batch_kind": batch_kind,
            "batch_seed": int(seed),
            "alternatives_per_source": alternatives_per_source,
            "source_design_count": len(candidate_indices),
            "repair_design_count": len(selection),
            "selection_rule": (
                "exactly the unique completed design indices listed in "
                "compiled/Candidate_Indices.csv; no median or ranking filter"
            ),
            "candidate_indices": candidate_indices,
            "candidate_indices_fingerprint": candidate_fingerprint,
            "source_fingerprints": source_fingerprints,
        },
        manifest_path,
    )
    return selection


def freeze_configured_batches(
    configuration: RepairConfiguration | None = None,
) -> dict[str, pd.DataFrame]:
    """Commit nonoverlapping development and validation batches."""

    configuration = configuration or RepairConfiguration()
    registry = ParameterRegistry(configuration)
    dataset = SourceDataset(configuration.source_root, registry)
    experiment_root = Path(configuration.experiment_root)

    development = freeze_random_batch(
        dataset,
        experiment_root / "development_batch",
        seed=configuration.batch_seed,
        batch_size=configuration.batch_size,
        batch_kind="development",
    )
    validation = freeze_random_batch(
        dataset,
        experiment_root / "validation_batch",
        seed=configuration.validation_batch_seed,
        batch_size=configuration.batch_size,
        excluded_indices=set(development["Index"].astype(int)),
        batch_kind="validation",
    )
    return {"development": development, "validation": validation}


__all__ = [
    "IDENTIFIER_COLUMNS",
    "FROZEN_SUBSET_FILENAMES",
    "SourceDataset",
    "frozen_subset_fingerprints",
    "freeze_configured_batches",
    "freeze_below_median_alternatives",
    "freeze_candidate_index_alternatives",
    "freeze_random_batch",
    "read_candidate_indices",
    "sha256_file",
    "write_csv_atomic",
    "write_json_atomic",
]
