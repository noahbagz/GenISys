"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import hashlib


import json


import os


from dataclasses import dataclass


from pathlib import Path


import numpy as np


import pandas as pd


from training_utils.schema import V1p1ParameterSchema


SCHEMA_NAME = "genisys.performance_guidance_data.v2"


RFR_QUANTILE_SCHEMA_NAME = "genisys.rfr_quantile_guidance_data.v3"


FIVE_OBJECTIVE_QUANTILE_SCHEMA_NAME = (
    "genisys.five_objective_quantile_guidance_data.v3"
)


SPLIT_TO_CODE = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class PerformanceGuidanceData:
    """Arrays and provenance needed to train a full-state performance critic."""

    controls: np.ndarray
    conditions: np.ndarray
    condition_known: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    raw_conditions: np.ndarray
    split_codes: np.ndarray
    rows: pd.DataFrame
    control_names: tuple[str, ...]
    condition_names: tuple[str, ...]
    target_names: tuple[str, ...]
    scalers: dict
    manifest: dict

    def positions(self, split: str) -> np.ndarray:
        """Return row positions for one frozen parent split."""

        return np.flatnonzero(self.split_codes == SPLIT_TO_CODE[split])

    def inverse_targets(self, scaled: np.ndarray) -> np.ndarray:
        """Convert critic targets from robust log space to physical units."""

        scaled = np.asarray(scaled, dtype=np.float64)
        center = np.asarray(self.scalers["targets"]["center"], dtype=np.float64)
        scale = np.asarray(self.scalers["targets"]["scale"], dtype=np.float64)
        transformed = scaled * scale + center
        return np.expm1(transformed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _write_csv_gzip_atomic(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_performance_guidance_data(
    output_root: str | Path,
) -> PerformanceGuidanceData:
    """Load one previously built versioned response artifact."""

    output_root = Path(output_root)
    with np.load(output_root / "Performance_Critic_Arrays.npz") as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    rows = pd.read_csv(output_root / "Performance_Critic_Rows.csv.gz")
    with (output_root / "Performance_Critic_Scalers.json").open(
        "r", encoding="utf-8"
    ) as stream:
        scalers = json.load(stream)
    with (output_root / "Performance_Data_Manifest.json").open(
        "r", encoding="utf-8"
    ) as stream:
        manifest = json.load(stream)
    if manifest.get("schema") not in {
        SCHEMA_NAME,
        RFR_QUANTILE_SCHEMA_NAME,
        FIVE_OBJECTIVE_QUANTILE_SCHEMA_NAME,
    }:
        raise ValueError("Performance training cache predates the unified-only data release.")
    if any("Dataset_100k/compiled" in path or "Dataset_100k_Repaired/compiled" in path
           for path in manifest.get("source_sha256", {})):
        raise ValueError("Performance training cache contains non-unified source paths.")

    return PerformanceGuidanceData(
        controls=arrays["controls"],
        conditions=arrays["conditions"],
        condition_known=arrays["condition_known"],
        targets=arrays["targets"],
        raw_targets=arrays["raw_targets"],
        raw_conditions=arrays["raw_conditions"],
        split_codes=arrays["split_codes"],
        rows=rows,
        control_names=tuple(scalers["controls"]["names"]),
        condition_names=tuple(scalers["conditions"]["names"]),
        target_names=tuple(scalers["targets"]["names"]),
        scalers=scalers,
        manifest=manifest,
    )


def _prepare_quantile_guidance_data(
    *,
    project_root: str | Path,
    output_root: str | Path,
    feasible_generator_checkpoint: str | Path,
    target_columns: tuple[str, ...],
    cache_schema: str,
    force: bool = False,
) -> PerformanceGuidanceData:
    """Build response critic data in the saved feasible-only DDPM coordinates.

    All rows, responses, and splits come from the unified release. The saved
    generator transform supplies only its fitted coordinate map; it is never
    refitted on the broader performance population.
    """

    import torch

    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    checkpoint_path = Path(feasible_generator_checkpoint).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    arrays_path = output_root / "Performance_Critic_Arrays.npz"
    rows_path = output_root / "Performance_Critic_Rows.csv.gz"
    scalers_path = output_root / "Performance_Critic_Scalers.json"
    audit_path = output_root / "Performance_Data_Audit.csv"
    manifest_path = output_root / "Performance_Data_Manifest.json"
    if manifest_path.exists() and not force:
        with manifest_path.open("r", encoding="utf-8") as stream:
            prior = json.load(stream)
        if prior.get("generator_checkpoint_sha256") != _sha256_file(checkpoint_path):
            raise ValueError("The saved generator coordinate map changed.")
        return load_performance_guidance_data(output_root)

    schema = V1p1ParameterSchema()
    unified_path = (
        project_root
        / "Ship_Systems_Designs_v1p1/Unified_Training_Dataset/Training_Rows.csv.gz"
    )
    split_path = (
        project_root
        / "Ship_Systems_Designs_v1p1/Unified_Training_Dataset/Shared_Split_Rows.csv.gz"
    )
    metadata = (
        "Population", "Index", "Source_Structure_Index", "Evaluation_Status",
        "Verified_Global_Feasible", "V1p1_Constraint_Fail_Count",
    )
    table = pd.read_csv(
        unified_path,
        usecols=list(dict.fromkeys([
            *metadata, *schema.full_names, *target_columns,
        ])),
    )
    split = pd.read_csv(
        split_path, usecols=["Population", "Index", "Split"],
    )
    table = table.merge(split, on=["Population", "Index"], validate="one_to_one")

    completed = table["Evaluation_Status"].astype(str).str.lower().eq("completed")
    finite_target = np.isfinite(
        table.loc[:, list(target_columns)].to_numpy(dtype=np.float64)
    ).all(axis=1)
    eligible = completed.to_numpy() & finite_target
    rows = table.loc[
        eligible, [*metadata, *target_columns, "Split"]
    ].reset_index(drop=True)
    rows.insert(0, "Row_Position", np.arange(len(rows), dtype=np.int64))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalizer = checkpoint["environments"][0]["data_normalizer"]
    controls_physical = schema.encode(
        table.loc[eligible, schema.full_names].to_numpy(dtype=np.float64)
    )
    controls = normalizer.transform_Data(controls_physical).astype(np.float32)

    # An RFR-only critic receives no clean physical design conditions: speed,
    # voyage length, dimensions, and cargo capacity remain generated variables.
    conditions = np.empty((len(rows), 0), dtype=np.float32)
    split_codes = rows["Split"].map(SPLIT_TO_CODE).to_numpy(dtype=np.int8)
    raw_targets = rows.loc[:, list(target_columns)].to_numpy(dtype=np.float64)
    transformed = np.log1p(raw_targets)
    train = split_codes == SPLIT_TO_CODE["train"]
    center = np.median(transformed[train], axis=0)
    q25 = np.quantile(transformed[train], 0.25, axis=0)
    q75 = np.quantile(transformed[train], 0.75, axis=0)
    scale = q75 - q25
    targets = ((transformed - center) / scale).astype(np.float32)

    # Boolean and integer endpoint values are intentional. Audit only true
    # continuous controls, grouped by feasibility and frozen parent split.
    continuous = np.asarray(
        [
            name not in schema.registry.boolean_names
            and name not in schema.registry.integer_names
            for name in schema.model_names
        ]
    )
    endpoint = np.abs(controls[:, continuous]) >= 5.19
    feasible = rows["Verified_Global_Feasible"].fillna(False).astype(bool).to_numpy()
    audit = pd.DataFrame(
        [
            {
                "Split": split_name,
                "Fully_Feasible": bool(feasibility),
                "Rows": int(selection.sum()),
                "Rows_With_Continuous_Endpoint": int(
                    endpoint[selection].any(axis=1).sum()
                ),
                "Continuous_Endpoint_Fraction": float(
                    endpoint[selection].mean()
                ),
            }
            for split_name, code in SPLIT_TO_CODE.items()
            for feasibility in (False, True)
            for selection in [(split_codes == code) & (feasible == feasibility)]
        ]
    )

    scalers = {
        "schema": cache_schema,
        "controls": {
            "method": "saved_feasible_only_quantile_normal",
            "names": list(schema.model_names),
            "generator_checkpoint_sha256": _sha256_file(checkpoint_path),
        },
        "conditions": {"method": "none", "names": []},
        "targets": {
            "transform": "log1p_then_training_median_iqr",
            "names": list(target_columns),
            "center": center.tolist(),
            "scale": scale.tolist(),
            "q25": q25.tolist(),
            "q75": q75.tolist(),
        },
    }
    _write_npz_atomic(
        arrays_path,
        controls=controls,
        conditions=conditions,
        condition_known=conditions,
        targets=targets,
        raw_targets=raw_targets,
        raw_conditions=conditions.astype(np.float64),
        split_codes=split_codes,
    )
    _write_csv_gzip_atomic(rows, rows_path)
    _write_json_atomic(scalers, scalers_path)
    audit.to_csv(audit_path, index=False)
    manifest = {
        "schema": cache_schema,
        "row_count": int(len(rows)),
        "control_count": int(controls.shape[1]),
        "condition_count": 0,
        "target_count": len(target_columns),
        "objective_names": list(target_columns),
        "condition_names": [],
        "split_counts": {
            name: int((split_codes == code).sum())
            for name, code in SPLIT_TO_CODE.items()
        },
        "source_sha256": {
            str(path.relative_to(project_root)): _sha256_file(path)
            for path in (unified_path, split_path)
        },
        "generator_checkpoint_sha256": _sha256_file(checkpoint_path),
        "artifact_sha256": {
            path.name: _sha256_file(path)
            for path in (arrays_path, rows_path, scalers_path, audit_path)
        },
        "data_policy": (
            "Completed unified rows only; frozen split; saved feasible-only "
            "generator quantile map; no clean design conditions."
        ),
    }
    _write_json_atomic(manifest, manifest_path)
    return load_performance_guidance_data(output_root)


def prepare_rfr_quantile_guidance_data(
    *,
    project_root: str | Path,
    output_root: str | Path,
    feasible_generator_checkpoint: str | Path,
    force: bool = False,
) -> PerformanceGuidanceData:
    """Build the first-arm RFR-only critic cache from unified designs."""

    return _prepare_quantile_guidance_data(
        project_root=project_root,
        output_root=output_root,
        feasible_generator_checkpoint=feasible_generator_checkpoint,
        target_columns=("Required_Freight_Rate_pounds_per_t",),
        cache_schema=RFR_QUANTILE_SCHEMA_NAME,
        force=force,
    )


