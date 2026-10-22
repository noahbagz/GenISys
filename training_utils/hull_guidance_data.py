"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import hashlib


import json


from pathlib import Path


import numpy as np


import pandas as pd


from tools.HullParameterization import Hull_Parameterization


from training_utils.subsystems import PARAMETERS_BY_SUBSYSTEM


from training_utils.schema import V1p1ParameterSchema


from parametric_repair_tools.hull_domain_repairs import HULL_CONSTRAINT_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


UNIFIED_ROOT = PROJECT_ROOT / "Ship_Systems_Designs_v1p1/Unified_Training_Dataset"


DEFAULT_OUTPUT = UNIFIED_ROOT / "Hull_Uniform_50k_v1"


HULL_SUBSYSTEMS = tuple(
    name for name in PARAMETERS_BY_SUBSYSTEM if name.startswith("Hull_")
)


HULL_CONTROLS = tuple(
    control
    for subsystem in HULL_SUBSYSTEMS
    for control in PARAMETERS_BY_SUBSYSTEM[subsystem]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constraint_map() -> pd.DataFrame:
    """Keep the established one-to-one map from hull equations to labels."""

    source = PROJECT_ROOT / "training_utils/subsystem_maps/Constraint_Map.csv"
    table = pd.read_csv(source)
    hull = table.loc[table["Constraint_Class"].eq("Hull")].copy()
    hull["Constraint_Name_Key"] = hull["Constraint_Name"].astype(str).str.strip()
    hull = hull.set_index("Constraint_Name_Key").loc[
        [name.strip() for name in HULL_CONSTRAINT_NAMES]
    ].reset_index(drop=True)
    if len(hull) != 49 or hull["Constraint_Name"].duplicated().any():
        raise ValueError("The frozen hull map must cover all 49 equations once.")
    if not set(hull["Subsystem"]).issubset(HULL_SUBSYSTEMS):
        raise ValueError("A hull equation maps outside the six hull subsystems.")
    hull.insert(0, "Constraint_Index", np.arange(49))
    hull.insert(1, "Bit_Column", [f"Hull_Constraint_{i:02d}_Bit" for i in range(49)])
    hull.insert(2, "Residual_Column", [f"Hull_Constraint_{i:02d}_Residual" for i in range(49)])
    return hull


def build_uniform_hull_data(
    *, count: int = 50_000, seed: int = 20260928,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict:
    """Keep every proposed hull, including an equation-construction failure."""

    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Hull release already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    # Training-split extrema prevent validation or test hulls from setting a
    # training distribution's sampling bounds.
    training_path = UNIFIED_ROOT / "Training_Rows.csv.gz"
    split_path = UNIFIED_ROOT / "Shared_Split_Rows.csv.gz"
    source = pd.read_csv(training_path, usecols=["Population", "Index", *HULL_CONTROLS])
    split = pd.read_csv(split_path, usecols=["Population", "Index", "Split"])
    source = source.merge(split, on=["Population", "Index"], validate="one_to_one")
    training = source.loc[source["Split"].eq("train"), HULL_CONTROLS]
    lower = training.min().to_numpy(np.float64)
    upper = training.max().to_numpy(np.float64)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Hull training bounds must be finite.")

    schema = V1p1ParameterSchema()
    full_hull_names = schema.full_names[:45]
    if len(HULL_CONTROLS) != 42 or set(full_hull_names) - set(HULL_CONTROLS) != {
        "Bc", "Beta", "bit_EP_T"
    }:
        raise ValueError("The established 42-control/45-input hull schema changed.")

    random = np.random.default_rng(seed)
    draws = random.uniform(lower, upper, size=(count, len(HULL_CONTROLS)))
    for index, name in enumerate(HULL_CONTROLS):
        if name in schema.registry.boolean_names:
            draws[:, index] = random.integers(0, 2, size=count)

    hull = np.empty((count, 45), dtype=np.float64)
    positions = {name: index for index, name in enumerate(full_hull_names)}
    for index, name in enumerate(HULL_CONTROLS):
        hull[:, positions[name]] = draws[:, index]
    hull[:, positions["Bc"]] = hull[:, positions["Bd"]]
    hull[:, positions["Beta"]] = 0.0
    hull[:, positions["bit_EP_T"]] = hull[:, positions["bit_EP_S"]]

    # Split synthetic proposals independently of their feasibility. The test
    # rows are held out from classifier fitting and calibration.
    permutation = random.permutation(count)
    split_names = np.empty(count, dtype=object)
    train_stop = int(0.70 * count)
    validation_stop = int(0.85 * count)
    split_names[permutation[:train_stop]] = "train"
    split_names[permutation[train_stop:validation_stop]] = "validation"
    split_names[permutation[validation_stop:]] = "test"

    mapping = _constraint_map()
    residuals = np.full((count, 49), np.nan, dtype=np.float64)
    bits = np.full((count, 49), pd.NA, dtype=object)
    errors = []
    for index, vector in enumerate(hull):
        try:
            with np.errstate(all="ignore"):
                values = np.asarray(
                    Hull_Parameterization(vector).input_Constraints(),
                    dtype=np.float64,
                )
            if values.shape != (49,) or not np.isfinite(values).all():
                raise ValueError("Hull constraints returned wrong shape or nonfinite values")
            residuals[index] = values
            bits[index] = values <= 0.0
            errors.append(("completed", "", ""))
        except Exception as error:
            # An exception supplies no trustworthy individual equation bits.
            # Keep the proposal and failure, never silently substitute 49
            # failed labels or resample to hide the difficult hull region.
            errors.append(("failed", type(error).__name__, str(error)))

    identifiers = pd.DataFrame({
        "Hull_ID": np.arange(count), "Split": split_names,
    })
    params = pd.concat(
        [identifiers, pd.DataFrame(hull, columns=full_hull_names)], axis=1,
    )
    bit_columns = mapping["Bit_Column"].tolist()
    bit_frame = pd.DataFrame(bits, columns=bit_columns).astype("boolean")
    bit_frame = pd.concat([identifiers, bit_frame], axis=1)
    residual_frame = pd.concat([
        identifiers,
        pd.DataFrame(residuals, columns=mapping["Residual_Column"]),
    ], axis=1)
    labels = identifiers.copy()
    completed = np.array([status == "completed" for status, _, _ in errors])
    for subsystem in HULL_SUBSYSTEMS:
        group_columns = mapping.loc[
            mapping["Subsystem"].eq(subsystem), "Bit_Column"
        ].tolist()
        labels[f"Feasible::Subsystem::{subsystem}"] = (
            bit_frame[group_columns].all(axis=1).astype("boolean")
        )
    labels["Feasible::Hull_Global"] = bit_frame[bit_columns].all(axis=1).astype("boolean")
    labels["Hull_Constraints_Known"] = completed
    labels["Evaluation_Status"] = [status for status, _, _ in errors]
    labels["Error_Type"] = [kind for _, kind, _ in errors]
    labels["Error_Message"] = [message for _, _, message in errors]

    bounds = pd.DataFrame({
        "Parameter": HULL_CONTROLS,
        "Training_Min": lower,
        "Training_Max": upper,
        "Sampling_Kind": [
            "uniform_boolean" if name in schema.registry.boolean_names else "uniform_continuous"
            for name in HULL_CONTROLS
        ],
    })
    output_frames = {
        "Hull_Parameters.csv.gz": params,
        "Hull_Constraint_Bits.csv.gz": bit_frame,
        "Hull_Constraint_Residuals.csv.gz": residual_frame,
        "Hull_Subsystem_Labels.csv.gz": labels,
        "Hull_Sampling_Bounds.csv": bounds,
        "Hull_Constraint_Map.csv": mapping,
    }
    for filename, frame in output_frames.items():
        frame.to_csv(output_root / filename, index=False)

    manifest = {
        "schema_version": 1,
        "description": "Uniform hull-only guidance data from unified training-split bounds",
        "sample_count": count,
        "seed": seed,
        "training_source_rows": int(len(training)),
        "hull_controls": len(HULL_CONTROLS),
        "hull_inputs": len(full_hull_names),
        "constraint_count": len(HULL_CONSTRAINT_NAMES),
        "completed": int(completed.sum()),
        "failed": int((~completed).sum()),
        "fully_hull_feasible": int(labels["Feasible::Hull_Global"].eq(True).sum()),
        "split_counts": labels["Split"].value_counts().to_dict(),
        "subsystem_label_counts": {
            name: {
                "pass": int(labels[f"Feasible::Subsystem::{name}"].eq(True).sum()),
                "fail": int(labels[f"Feasible::Subsystem::{name}"].eq(False).sum()),
                "unknown": int(labels[f"Feasible::Subsystem::{name}"].isna().sum()),
            }
            for name in HULL_SUBSYSTEMS
        },
        "sources": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in (
                training_path,
                split_path,
                PROJECT_ROOT / "training_utils/subsystem_maps/Constraint_Map.csv",
            )
        },
        "outputs": {name: _sha256(output_root / name) for name in output_frames},
    }
    (output_root / "Hull_Dataset_Manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


