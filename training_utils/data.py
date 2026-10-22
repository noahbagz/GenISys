"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import copy


import hashlib


import json


from dataclasses import dataclass, field


from pathlib import Path


import numpy as np


import pandas as pd


from training_utils.schema import V1p1ParameterSchema


from training_utils.subsystems import GLOBAL_LABEL_COLUMN, LOCAL_LABEL_COLUMN, SUBSYSTEM_ORDER, SYSTEM_LABEL_COLUMN, parameter_subsets as granular_parameter_subsets


LOCAL_LABELS_BY_SUBSYSTEM = {
    name: (LOCAL_LABEL_COLUMN[name],) for name in SUBSYSTEM_ORDER
}


@dataclass(frozen=True)
class PreparedFirstExperimentData:
    """In-memory form of the frozen split, labels, and scaled coordinates."""

    model_values: np.ndarray
    scaled_values: np.ndarray
    row_table: pd.DataFrame
    parameter_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    metric_lower: np.ndarray
    metric_upper: np.ndarray
    parameter_subsets: dict[str, np.ndarray]
    global_condition: np.ndarray
    system_condition: np.ndarray
    local_conditions: dict[str, np.ndarray]
    local_pass_labels: dict[str, np.ndarray]
    global_pass_labels: np.ndarray
    system_pass_labels: np.ndarray
    quantile_normalizer: object | None = None
    _normalizer_cache: dict[tuple[object, ...], object] = field(default_factory=dict)

    def model_normalizer(
        self,
        indices: np.ndarray | None = None,
        *,
        population: str | None = None,
        unknown_label_policy: str = "include",
        bounds_scope: str = "population",
    ):
        """Fit population ranks within either population or known-train bounds."""

        if self.quantile_normalizer is None:
            raise RuntimeError("Prepared data has no fitted quantile normalizer.")

        source = self.quantile_normalizer
        if population is not None:
            population_key = ("population", population, unknown_label_policy, bounds_scope)
            if population_key not in self._normalizer_cache:
                from models.diffusion import Data_Normalizer

                positions = self.generator_positions(
                    population,
                    unknown_label_policy=unknown_label_policy,
                )
                values = self.model_values[positions]
                bounds_values = (
                    self.model_values[self.classifier_positions("train")]
                    if bounds_scope == "known_training" else values
                )
                fitted = Data_Normalizer(
                    bounds_values.min(axis=0), bounds_values.max(axis=0),
                    len(positions), "quantile",
                )
                fitted.fit_Data(values)
                self._normalizer_cache[population_key] = fitted
            source = self._normalizer_cache[population_key]

        if indices is None:
            return source

        key = (population, unknown_label_policy, bounds_scope, *(int(index) for index in indices))
        if key not in self._normalizer_cache:
            from models.diffusion import Data_Normalizer

            subset = np.asarray(indices, dtype=int)
            row_count = (
                len(self.generator_positions(
                    population,
                    unknown_label_policy=unknown_label_policy,
                ))
                if population is not None else int(self.train_mask.sum())
            )
            view = Data_Normalizer(
                source.X_LL_Scaled[subset], source.X_UL_Scaled[subset],
                row_count, "quantile",
            )
            view.normalizer = copy.copy(source.normalizer)
            view.normalizer.quantiles_ = source.normalizer.quantiles_[:, subset].copy()
            view.normalizer.n_features_in_ = len(subset)
            self._normalizer_cache[key] = view
        return self._normalizer_cache[key]

    def scaled_values_for_population(
        self,
        population: str,
        *,
        unknown_label_policy: str = "include",
        bounds_scope: str = "population",
    ) -> np.ndarray:
        """Encode unified rows once in the generator population's coordinates."""

        key = ("scaled_values", population, unknown_label_policy, bounds_scope)
        if key not in self._normalizer_cache:
            self._normalizer_cache[key] = (
                self.model_normalizer(
                    population=population,
                    unknown_label_policy=unknown_label_policy,
                    bounds_scope=bounds_scope,
                ).transform_Data(
                    self.model_values
                )
            )
        return self._normalizer_cache[key]

    @property
    def train_mask(self) -> np.ndarray:
        return self.row_table["Split"].eq("train").to_numpy()

    @property
    def known_mask(self) -> np.ndarray:
        return self.row_table["Labels_Known"].astype(bool).to_numpy()

    @property
    def feasible_mask(self) -> np.ndarray:
        return self.row_table["Verified_Global_Feasible"].astype(bool).to_numpy()

    def generator_positions(
        self,
        population: str,
        *,
        parent_fraction: float = 1.0,
        subset_seed: int = 0,
        unknown_label_policy: str = "include",
        matched_row_count: int | None = None,
    ) -> np.ndarray:
        """Return a nested parent subset for one generator population."""

        keep = self.train_mask
        if population == "feasible_only":
            keep = keep & self.feasible_mask
        elif population != "full_data":
            raise ValueError(f"Unknown generator population {population!r}.")

        if unknown_label_policy == "exclude":
            keep = keep & self.known_mask
        elif unknown_label_policy != "include":
            raise ValueError(
                f"Unknown unknown_label_policy {unknown_label_policy!r}."
            )

        if not 0.0 < float(parent_fraction) <= 1.0:
            raise ValueError("parent_fraction must be in (0, 1].")
        if float(parent_fraction) < 1.0:
            parents = np.sort(
                self.row_table.loc[keep, "Source_Structure_Index"]
                .astype(np.int64)
                .unique()
            )
            random = np.random.default_rng(int(subset_seed))
            ranked = parents.copy()
            random.shuffle(ranked)
            count = max(1, int(np.ceil(float(parent_fraction) * len(ranked))))
            keep = keep & self.row_table["Source_Structure_Index"].isin(
                ranked[:count]
            ).to_numpy()

        if matched_row_count is not None:
            selected = self.row_table.loc[keep, ["Source_Structure_Index"]]
            counts = selected["Source_Structure_Index"].value_counts(sort=False)
            parents = counts.index.to_numpy(dtype=np.int64)
            random = np.random.default_rng(int(subset_seed))
            random.shuffle(parents)
            cumulative = counts.loc[parents].cumsum().to_numpy()
            parent_count = int(np.argmin(np.abs(cumulative - int(matched_row_count)))) + 1
            keep = keep & self.row_table["Source_Structure_Index"].isin(
                parents[:parent_count]
            ).to_numpy()
        return np.flatnonzero(keep)

    def parent_balancing_probabilities(self, positions: np.ndarray) -> np.ndarray:
        """Give every selected parent equal expected batch representation."""

        parents = self.row_table.iloc[positions]["Source_Structure_Index"]
        counts = parents.value_counts()
        weights = parents.map(lambda value: 1.0 / counts.loc[value]).to_numpy(float)
        return weights / weights.sum()

    def classifier_positions(self, split: str = "train") -> np.ndarray:
        """Return known-label rows from one frozen parent split."""

        split_mask = self.row_table["Split"].eq(split).to_numpy()
        return np.flatnonzero(split_mask & self.known_mask)

    def local_feasible_positions(
        self,
        subsystem: str,
        *,
        split: str = "train",
    ) -> np.ndarray:
        """Return complete designs whose named subsystem has a known pass label.

        These rows support the locally-feasible implicit side experiment.  A
        selected design may fail other subsystems or whole-system constraints;
        only the target subsystem's label controls inclusion.
        """

        split_mask = self.row_table["Split"].eq(split).to_numpy()
        local_pass = self.local_pass_labels[subsystem].reshape(-1).astype(bool)
        return np.flatnonzero(split_mask & self.known_mask & local_pass)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assign_parent_splits(
    table: pd.DataFrame,
    *,
    seed: int,
) -> dict[int, str]:
    """Assign parent families with coarse outcome stratification."""

    working = table.loc[
        :,
        [
            "Source_Structure_Index",
            "Evaluation_Status",
            "Verified_Global_Feasible",
            "V1p1_Constraint_Fail_Count",
        ],
    ].copy()
    working["Completed"] = (
        working["Evaluation_Status"].fillna("").astype(str).str.lower().eq("completed")
    )
    working["V1p1_Constraint_Fail_Count"] = pd.to_numeric(
        working["V1p1_Constraint_Fail_Count"],
        errors="coerce",
    )
    parents = working.groupby("Source_Structure_Index", as_index=False).agg(
        All_Completed=("Completed", "all"),
        Any_Feasible=("Verified_Global_Feasible", "any"),
        Median_Violation_Count=("V1p1_Constraint_Fail_Count", "median"),
    )
    maximum = parents["Median_Violation_Count"].max()
    filled = parents["Median_Violation_Count"].fillna(
        0.0 if pd.isna(maximum) else maximum + 1.0
    )
    parents["Violation_Bin"] = pd.qcut(
        filled.rank(method="first"),
        q=min(5, len(parents)),
        labels=False,
        duplicates="drop",
    ).astype(int)
    parents["Stratum"] = (
        parents["All_Completed"].astype(int).astype(str)
        + "_"
        + parents["Any_Feasible"].astype(int).astype(str)
        + "_"
        + parents["Violation_Bin"].astype(str)
    )

    random = np.random.default_rng(seed)
    assignments = []
    fractions = np.asarray([0.70, 0.15, 0.15])
    split_names = np.asarray(["train", "validation", "test"])
    for _, group in parents.groupby("Stratum", sort=True):
        group_values = group["Source_Structure_Index"].to_numpy(dtype=np.int64)
        random.shuffle(group_values)
        raw_counts = fractions * len(group_values)
        counts = np.floor(raw_counts).astype(int)
        remainder = len(group_values) - int(counts.sum())
        order = np.argsort(-(raw_counts - counts), kind="stable")
        counts[order[:remainder]] += 1
        start = 0
        for split, count in zip(split_names, counts, strict=True):
            assignments.extend(
                (int(parent), str(split))
                for parent in group_values[start : start + count]
            )
            start += count

    assigned = pd.DataFrame(assignments, columns=["Parent", "Split"])
    target_counts = np.floor(fractions * len(parents)).astype(int)
    remainder = len(parents) - int(target_counts.sum())
    fractional_order = np.argsort(
        -(fractions * len(parents) - target_counts),
        kind="stable",
    )
    target_counts[fractional_order[:remainder]] += 1
    targets = dict(zip(split_names, target_counts, strict=True))

    # Singleton strata can overfill train. Move the minimum number of complete
    # parent groups needed to recover exact global split counts.
    for receiving in ("validation", "test", "train"):
        while int(assigned["Split"].eq(receiving).sum()) < int(targets[receiving]):
            counts = assigned["Split"].value_counts().to_dict()
            donors = [
                split
                for split in split_names
                if counts.get(split, 0) > int(targets[split])
            ]
            donor = max(
                donors,
                key=lambda split: counts.get(split, 0) - int(targets[split]),
            )
            candidates = assigned.index[assigned["Split"].eq(donor)].to_numpy()
            assigned.loc[int(random.choice(candidates)), "Split"] = receiving

    return dict(
        zip(
            assigned["Parent"].astype(int),
            assigned["Split"].astype(str),
            strict=True,
        )
    )


def _condition_from_pass_values(
    passed: np.ndarray,
    known: np.ndarray,
) -> np.ndarray:
    """Encode known labels as [pass, fail]; leave unknown labels missing."""

    passed = np.asarray(passed, dtype=np.float32)
    known = np.asarray(known, dtype=bool).reshape(-1, 1)
    if passed.ndim == 1:
        passed = passed.reshape(-1, 1)

    # Unknown rows are excluded from conditional-generator training. NaN
    # prevents them from acquiring an accidental third condition code.
    pass_values = np.where(known, passed, np.nan)
    fail_values = np.where(known, 1.0 - passed, np.nan)

    # Keep the pair for each named label adjacent in the condition vector.
    return np.stack(
        (pass_values, fail_values),
        axis=2,
    ).reshape(len(passed), -1)


def feasible_condition_target(label_count: int) -> np.ndarray:
    """Return the [pass=1, fail=0] target for each requested label."""

    return np.tile(
        np.asarray([1.0, 0.0], dtype=np.float32),
        int(label_count),
    )


def prepare_first_experiment_data(
    *,
    unified_root: str | Path,
    artifact_root: str | Path,
    split_seed: int,
    force: bool = False,
) -> PreparedFirstExperimentData:
    """Build or reload the shared data artifacts used by all fifteen methods."""

    unified_root = Path(unified_root)
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    training_path = unified_root / "Training_Rows.csv.gz"
    labels_path = unified_root / "Granular_20_Subsystem_Labels.csv.gz"
    split_path = artifact_root / "Shared_Split_Rows.csv.gz"
    scaler_path = artifact_root / "Shared_Training_MinMax_Scaler.csv"
    summary_path = artifact_root / "Data_Preparation_Summary.json"

    schema = V1p1ParameterSchema()
    schema.schema_table().to_csv(
        artifact_root / "Model_Parameter_Schema.csv",
        index=False,
    )
    metadata_columns = (
        "Population",
        "Index",
        "Source_Design_Index",
        "Source_Structure_Index",
        "Training_Eligible_RP",
        "Evaluation_Status",
        "Global_V1p1_Feasible",
        "V1p1_Constraint_Fail_Count",
        "Verified_Global_Feasible",
    )
    use_columns = [*metadata_columns, *schema.full_names]
    table = pd.read_csv(training_path, usecols=use_columns)

    label_columns = [LOCAL_LABEL_COLUMN[name] for name in SUBSYSTEM_ORDER]
    labels = pd.read_csv(
        labels_path,
        usecols=[
            "Population",
            "Index",
            "Labels_Known",
            GLOBAL_LABEL_COLUMN,
            SYSTEM_LABEL_COLUMN,
            *label_columns,
        ],
    )
    table = table.merge(
        labels,
        on=["Population", "Index"],
        how="left",
        validate="one_to_one",
    )

    if table["Source_Structure_Index"].isna().any():
        raise ValueError("Every unified row must have a source structural parent.")

    # A current-evaluator label release carries the original parent split.
    # New output roots must import it instead of stratifying again after label
    # changes, which would leak a different split into the comparison.
    source_split_path = unified_root / "Shared_Split_Rows.csv.gz"
    existing_split_path = (
        split_path if split_path.exists() and not force
        else source_split_path if source_split_path.exists() else None
    )
    if existing_split_path is not None:
        frozen_split = pd.read_csv(existing_split_path)
        table = table.merge(
            frozen_split.loc[:, ["Population", "Index", "Split"]],
            on=["Population", "Index"],
            how="left",
            validate="one_to_one",
        )
        if table["Split"].isna().any():
            raise ValueError("The frozen split does not cover every unified row.")
        if existing_split_path == source_split_path:
            frozen_split.to_csv(split_path, index=False, compression="gzip")
    else:
        parent_split = _assign_parent_splits(table, seed=split_seed)
        table["Split"] = (
            table["Source_Structure_Index"].astype(np.int64).map(parent_split)
        )
        table.loc[
            :,
            [
                "Population",
                "Index",
                "Source_Design_Index",
                "Source_Structure_Index",
                "Split",
            ],
        ].to_csv(split_path, index=False, compression="gzip")

    full_values = table.loc[:, schema.full_names].to_numpy(dtype=np.float64)
    model_values = schema.encode(full_values)
    train_mask = table["Split"].eq("train").to_numpy()

    # Retain the historical shared transform for existing runs. The
    # C_ShipGen recipe fits an additional transform per training population
    # when its generator and classifier collections are built. Distance
    # metrics keep their separate feasible-reference min/max bounds.
    lower = np.min(model_values[train_mask], axis=0)
    upper = np.max(model_values[train_mask], axis=0)
    width = upper - lower
    if np.any(np.isclose(width, 0.0)):
        names = np.asarray(schema.model_names)[np.isclose(width, 0.0)].tolist()
        raise ValueError(f"Training-only scaler has constant controls: {names}.")
    # Fit the historical shared normal-output quantile map once. Each
    # population-specific map is fitted lazily on its training rows.
    from models.diffusion import Data_Normalizer

    quantile_normalizer = Data_Normalizer(
        lower, upper, int(train_mask.sum()), "quantile",
    )
    quantile_normalizer.fit_Data(model_values[train_mask])
    scaled_values = quantile_normalizer.transform_Data(model_values)

    scaler = pd.DataFrame(
        {
            "Model_Index": np.arange(len(schema.model_names)),
            "Parameter": schema.model_names,
            "Training_Min": lower,
            "Training_Max": upper,
        }
    )
    if scaler_path.exists() and not force:
        recorded = pd.read_csv(scaler_path)
        if not np.allclose(
            recorded[["Training_Min", "Training_Max"]].to_numpy(dtype=float),
            scaler[["Training_Min", "Training_Max"]].to_numpy(dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("The shared training-only scaler has changed.")
    else:
        scaler.to_csv(scaler_path, index=False)

    known = table["Labels_Known"].fillna(False).astype(bool).to_numpy()
    global_pass = (
        table[GLOBAL_LABEL_COLUMN].astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    system_pass = (
        table[SYSTEM_LABEL_COLUMN].astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    global_condition = _condition_from_pass_values(global_pass, known)
    system_condition = _condition_from_pass_values(system_pass, known)

    local_conditions = {}
    local_pass_labels = {}
    for subsystem in SUBSYSTEM_ORDER:
        column = LOCAL_LABEL_COLUMN[subsystem]
        pass_values = table.loc[:, [column]].apply(
            lambda column: column.map(
                lambda value: False if pd.isna(value) else bool(value)
            )
        ).to_numpy(dtype=np.float32)
        local_pass_labels[subsystem] = pass_values
        local_conditions[subsystem] = _condition_from_pass_values(
            pass_values,
            known,
        )

    parameter_subsets = granular_parameter_subsets(schema.model_names)

    reference_mask = train_mask & global_pass & known
    reference_values = model_values[reference_mask]
    metric_lower = np.min(reference_values, axis=0)
    metric_upper = np.max(reference_values, axis=0)
    metric_width = metric_upper - metric_lower
    if np.any(np.isclose(metric_width, 0.0)):
        names = np.asarray(schema.model_names)[
            np.isclose(metric_width, 0.0)
        ].tolist()
        raise ValueError(
            "Fully feasible training reference has constant controls: "
            f"{names}."
        )

    # Preserve the shared quantile-coordinate reference for generator and
    # classifier diagnostics. Distance metrics use the separate feasible-only
    # transform below, as specified by the experiment protocol.
    reference = pd.DataFrame(
        scaled_values[reference_mask],
        columns=schema.model_names,
    )
    reference.insert(
        0,
        "Index",
        table.loc[reference_mask, "Index"].to_numpy(),
    )
    reference.insert(
        0,
        "Population",
        table.loc[reference_mask, "Population"].to_numpy(),
    )
    reference.to_csv(
        artifact_root / "Feasible_Training_Reference_Scaled.csv.gz",
        index=False,
        compression="gzip",
    )

    metric_scaler = pd.DataFrame(
        {
            "Model_Index": np.arange(len(schema.model_names)),
            "Parameter": schema.model_names,
            "Feasible_Training_Min": metric_lower,
            "Feasible_Training_Max": metric_upper,
        }
    )
    metric_scaler.to_csv(
        artifact_root / "Feasible_Reference_MinMax_Scaler.csv",
        index=False,
    )

    metric_reference = pd.DataFrame(
        (reference_values - metric_lower) / metric_width,
        columns=schema.model_names,
    )
    metric_reference.insert(
        0,
        "Index",
        table.loc[reference_mask, "Index"].to_numpy(),
    )
    metric_reference.insert(
        0,
        "Population",
        table.loc[reference_mask, "Population"].to_numpy(),
    )
    metric_reference.to_csv(
        artifact_root / "Feasible_Training_Reference_Metric_Scaled.csv.gz",
        index=False,
        compression="gzip",
    )

    # Cluster distances use the 189 values actually submitted to the exact
    # evaluator after its CAD cleanup. The unified compiled rows provide the
    # corresponding complete feasible training reference. Fixed dimensions
    # contribute zero distance; generated values are never clipped.
    full_reference_values = full_values[reference_mask]
    full_metric_lower = np.min(full_reference_values, axis=0)
    full_metric_upper = np.max(full_reference_values, axis=0)
    full_constant = np.isclose(
        full_metric_upper, full_metric_lower, rtol=0.0, atol=1.0e-12,
    )
    full_metric_width = np.where(
        full_constant, 1.0, full_metric_upper - full_metric_lower,
    )
    full_reference_scaled = (full_reference_values - full_metric_lower) / full_metric_width
    full_reference_scaled[:, full_constant] = 0.0
    pd.DataFrame({
        "Full_Index": np.arange(len(schema.full_names)),
        "Parameter": schema.full_names,
        "Feasible_Training_Min": full_metric_lower,
        "Feasible_Training_Max": full_metric_upper,
        "Constant_In_Feasible_Reference": full_constant,
    }).to_csv(artifact_root / "Feasible_Reference_189_MinMax_Scaler.csv", index=False)
    pd.DataFrame(
        full_reference_scaled, columns=schema.full_names,
    ).to_csv(
        artifact_root / "Feasible_Training_Reference_189_Metric_Scaled.csv.gz",
        index=False, compression="gzip",
    )

    summary = {
        "schema_version": 2,
        "split_seed": int(split_seed),
        "row_count": int(len(table)),
        "training_row_count": int(train_mask.sum()),
        "known_label_training_row_count": int((train_mask & known).sum()),
        "feasible_training_row_count": int(reference_mask.sum()),
        "generator_normalization": "training-only C_ShipGen-style quantile map with normal output",
        "generator_quantile_count": int(quantile_normalizer.normalizer.n_quantiles_),
        "generator_weight_selection": "final_epoch_live",
        "distance_scaling": (
            "189 exact-evaluator submitted parameters scaled by the fully "
            "feasible training reference minimum and maximum; constant "
            "reference parameters contribute zero and generated values are not clipped"
        ),
        "model_coordinate_count": int(len(schema.model_names)),
        "subsystem_model_count": len(SUBSYSTEM_ORDER),
        "local_label_count": len(SUBSYSTEM_ORDER),
        "system_level_label_count": 1,
        "global_label_count": 1,
        "unknown_label_policy": (
            "The prepared table retains the known mask. Conditional full-data "
            "generators and classifiers use known-label rows only."
        ),
        "source_sha256": {
            str(training_path): _sha256_file(training_path),
            str(labels_path): _sha256_file(labels_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    row_table = table.loc[
        :,
        [
            "Population",
            "Index",
            "Source_Design_Index",
            "Source_Structure_Index",
            "Split",
            "Training_Eligible_RP",
            "Evaluation_Status",
            "Labels_Known",
            "Verified_Global_Feasible",
        ],
    ].copy()

    return PreparedFirstExperimentData(
        model_values=model_values,
        scaled_values=scaled_values,
        row_table=row_table,
        parameter_names=schema.model_names,
        lower=lower,
        upper=upper,
        metric_lower=metric_lower,
        metric_upper=metric_upper,
        parameter_subsets=parameter_subsets,
        global_condition=global_condition,
        system_condition=system_condition,
        local_conditions=local_conditions,
        local_pass_labels=local_pass_labels,
        global_pass_labels=global_pass.astype(np.float32).reshape(-1, 1),
        system_pass_labels=system_pass.astype(np.float32).reshape(-1, 1),
        quantile_normalizer=quantile_normalizer,
    )


__all__ = [
    "LOCAL_LABELS_BY_SUBSYSTEM",
    "PreparedFirstExperimentData",
    "SUBSYSTEM_ORDER",
    "feasible_condition_target",
    "prepare_first_experiment_data",
]


