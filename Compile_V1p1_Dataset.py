"""Compile restartable GenISys V1.1 shards without hiding missing records.

The compiler is intentionally separate from generation and evaluation.  It
never modifies or deletes a source shard.  Instead, it concatenates compatible
tables, derives convenient audited-constraint views, and writes explicit
acceptance checks plus a manifest of the exact source and rule files used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.ConstraintIdentity import (
    canonical_constraint_keys,
    collapse_constraint_aliases,
)
from tools.EconomicInputs import DEFAULT_ECONOMIC_INPUTS
from tools.LoadingConditionLoads import (
    LOAD_CLOSURE_RELATIVE_TOLERANCE,
    MASS_CLOSURE_RELATIVE_TOLERANCE,
)
from tools.V1p1Dataset import (
    HULLS_PER_STRUCTURE,
    NUMBER_OF_STRUCTURES,
    OPERATING_CONDITIONS_PER_HULL,
    POST_COUPLING_HULL_PARAMETER_NAMES,
    SYSTEMS_PER_STRUCTURE,
    derive_structural_bilge_radius_m,
    load_parameter_catalog,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "Ship_Systems_Designs_v1p1" / "Dataset_100k"
)

# These files define the input schema, regulatory interpretation, active V1.1
# evaluation path, and scenario values that materially affect compiled output.
# Keep the list explicit so each dataset manifest can bind its results to the
# exact source versions used for that run.
AUTHORITY_RELATIVE_PATHS = (
    # Runtime environment and the exact generation/evaluation/compilation
    # entry points used by the laptop and cluster workflows.
    "GenISys_Env.yml",
    "Generate_V1p1_Design_Batch.py",
    "Run_V1p1_System_Evaluation_Batch.py",
    "Run_ShipSystems_Eval_Parallel_v1p1.py",
    "Compile_V1p1_Dataset.py",
    "Validate_V1p1_Pipeline.py",
    "ShipSystems_V1p1_Batch.sh",
    "Run_V1p1_Laptop_Pipeline_Test.sh",

    # Ordered design schemas and runtime CSV inputs.
    "tools/ShipSystemDesignParameters.csv",
    "StructuralParameterList_V2.csv",
    "tools/Hull_constraints_list.csv",

    # Deterministic design-space sampling and CadQuery structural-element
    # generation.  CAD exporters and visualization-only modules are omitted
    # because the production path uses artifact_mode='elements_only'.
    "tools/V1p1Dataset.py",
    "tools/HullParameterization.py",
    "cad_query_tools/generate.py",
    "cad_query_tools/builder.py",
    "cad_query_tools/geometry.py",
    "cad_query_tools/identifiers.py",
    "cad_query_tools/model.py",
    "cad_query_tools/parameters.py",
    "cad_query_tools/properties.py",
    "cad_query_tools/stiffeners.py",

    # Active ship-system calculations, constraint integration, and failure
    # reporting used by Run_V1p1_System_Evaluation_Batch.py.
    "tools/ABS_Part3_Constraints.py",
    "tools/CargoAllocation.py",
    "tools/CargoHolds.py",
    "tools/ConstraintIntegration.py",
    "tools/ConstraintIdentity.py",
    "tools/EvaluationErrors.py",
    "tools/System_Feasibility_Constraints_v1p1.py",
    "tools/HatchCovers.py",
    "tools/HullDesign.py",
    "tools/Hydrostatics.py",
    "tools/ModifiedMichellCw.py",
    "tools/SOLASSubdivisionAndDetectors.py",
    "tools/StructureCalcs.py",
    "tools/Weights.py",

    # Regulatory interpretation and current implementation authority.
    "docs/ABS_Bulk_Carrier_Rule_Audit.csv",
    "docs/Hydrostatics_Damage_Stability_Flooding_Rule_Audit.csv",
    "docs/ABS_Structural_Rule_Assumptions.md",
    "docs/Hydrostatics_Damage_Stability_Flooding_Assumptions.md",
    "tools/Structure_Constraints_v1p1.py",
    "tools/Hydrostatics_v1p1.py",
    "tools/LoadingConditions.py",
    "tools/LoadingConditionLoads.py",
    "tools/UprightStability.py",
    "tools/Loading_Condition_Constraints_v1p1.py",
    "tools/ShipSystemDesign.py",
    "tools/EarlyDesignMasses.py",
    "tools/PoweringInputs.py",
    "tools/Powering_Constraints_v1p1.py",
    "tools/EEDI.py",
    "tools/EconomicInputs.py",
    "docs/GENISYS_V1P1_CONSTRAINT_OUTPUTS.md",
)

IDENTIFIER_COLUMNS = [
    "Index",
    "Structure_Index",
    "Hull_Variant_Index",
    "Operating_Condition_Index",
    "Hull_Candidate_Index",
    "Structure_Sample_Attempt",
]


def _read_shards(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate one ordered family of CSV shards."""

    tables = []

    for path in paths:
        try:
            tables.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            # Empty diagnostic shards are permitted.  They contribute no rows
            # and remain visible in the manifest's source-file inventory.
            continue

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True, sort=False)


def _csv_shard_paths(directory: Path, filename_pattern: str) -> list[Path]:
    """Find a shard family in ordinary or gzip-compressed CSV form."""

    plain_paths = list(directory.glob(filename_pattern))
    compressed_paths = list(directory.glob(f"{filename_pattern}.gz"))

    return sorted(plain_paths + compressed_paths)


def _write_csv_atomic(table: pd.DataFrame, output_path: Path) -> None:
    """Replace one complete compiled table without a partial final file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    compression = (
        {"method": "gzip", "compresslevel": 6, "mtime": 0}
        if output_path.suffix == ".gz"
        else None
    )
    table.to_csv(temporary_path, index=False, compression=compression)
    temporary_path.replace(output_path)


def _compiled_csv_path(
    compiled_directory: Path,
    filename: str,
    save_mode: str,
    compress_in_reduced_mode: bool = True,
) -> Path:
    """Return the mode-specific path for one compiled tabular artifact."""

    path = compiled_directory / filename

    if save_mode == "reduced" and compress_in_reduced_mode:
        return path.with_name(f"{path.name}.gz")

    return path


def _write_json_atomic(payload: dict, output_path: Path) -> None:
    """Replace one complete JSON document without a partial final file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _file_sha256_and_size(path: Path) -> tuple[str, int]:
    """Return one SHA-256 digest and the exact byte count hashed with it."""

    digest = hashlib.sha256()
    size_bytes = 0

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size_bytes += len(block)

    return digest.hexdigest(), size_bytes


def _manifest_path(path: Path) -> str:
    """Return a repository-relative path when possible, otherwise absolute.

    Production artifacts live below ``PROJECT_ROOT``.  The absolute fallback
    keeps the same helpers usable by focused temporary-directory tests and by
    intentionally external scenario files without weakening the production
    path convention.
    """

    path = Path(path).resolve()

    try:
        return str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _file_provenance(path: Path) -> dict:
    """Return stable path, byte size, and SHA-256 provenance for one file."""

    path = Path(path)
    digest, size_bytes = _file_sha256_and_size(path)

    return {
        "path": _manifest_path(path),
        "size_bytes": size_bytes,
        "sha256": digest,
    }


def _json_file_provenance(path: Path) -> dict:
    """Return file provenance plus parsed content for an invocation sidecar.

    Invocation files are written by the V1.1 generator/evaluator and contain
    the resolved execution configuration and optional-scenario provenance.
    Keeping their parsed content in the compiled manifest makes a single
    manifest sufficient for inspection, while the file hash binds that copy
    to the original sidecar.
    """

    path = Path(path)
    record = _file_provenance(path)

    try:
        record["content"] = json.loads(path.read_text(encoding="utf-8"))
        record["content_status"] = "parsed"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        # A malformed legacy or manually edited sidecar must remain visible in
        # provenance rather than making compilation discard all usable data.
        record["content"] = None
        record["content_status"] = "unreadable_or_invalid_json"
        record["content_error"] = f"{type(error).__name__}: {error}"

    return record


def _record_check(
    rows: list[dict],
    name: str,
    passed: bool,
    observed,
    expected,
    notes: str,
) -> None:
    """Append one machine-readable dataset acceptance check."""

    rows.append(
        {
            "Check": name,
            "Passed": bool(passed),
            "Observed": observed,
            "Expected": expected,
            "Notes": notes,
        }
    )


def _build_parameter_bound_violations(
    structures: pd.DataFrame,
    systems: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """List final generated parameters outside their published DOE bounds.

    Structure cleanup and hull/arrangement coupling both occur after the first
    random sample.  This table checks the values that were actually saved,
    which prevents a compiled dataset from silently describing a wider design
    space than its parameter catalog.
    """

    columns = [
        "Record_Type",
        "Structure_Index",
        "Index",
        "Parameter",
        "Value",
        "Lower_Bound",
        "Upper_Bound",
        "Violation",
    ]
    violation_tables = []

    def append_violations(
        source: pd.DataFrame,
        definitions: pd.DataFrame,
        record_type: str,
    ) -> None:
        """Append vectorized violations for one source-table family."""

        if source.empty:
            return

        for _, definition in definitions.iterrows():
            name = str(definition["Name"])

            if name not in source:
                continue

            lower = float(definition["LL"])
            upper = float(definition["UL"])
            tolerance = 1.0e-12*max(1.0, abs(lower), abs(upper))
            values = source[name].astype(float)
            below = values < lower - tolerance
            above = values > upper + tolerance

            for mask, direction in (
                (below, "below_lower_bound"),
                (above, "above_upper_bound"),
            ):
                if not bool(mask.any()):
                    continue

                selected = source.loc[mask].copy()
                table = pd.DataFrame(
                    {
                        "Record_Type": record_type,
                        "Structure_Index": selected["Structure_Index"].astype(
                            int
                        ),
                        "Index": (
                            selected["Index"].astype(int)
                            if "Index" in selected
                            else np.nan
                        ),
                        "Parameter": name,
                        "Value": selected[name].astype(float),
                        "Lower_Bound": lower,
                        "Upper_Bound": upper,
                        "Violation": direction,
                    }
                )
                violation_tables.append(table)

    structural_definitions = catalog.loc[
        (catalog["Class"] == "Structure")
        # T is intentionally derived separately for each operating condition;
        # its zero placeholder is not an accepted structure-level value.
        # Inner Side Shell Thickness has the disjoint cleaned domain {0} or
        # [10, 20] mm, which is checked explicitly below.
        & ~catalog["Name"].isin(
            ("T", "Inner Side Shell Thickness", "web frame w")
        )
    ]
    append_violations(
        structures,
        structural_definitions,
        "structure",
    )

    # A zero inner-side thickness is the intentional single-side sentinel.
    # Active inner-side shells retain the cleaned 10--20 mm interval.
    if (
        not structures.empty
        and "Inner Side Shell Thickness" in structures
    ):
        values = structures["Inner Side Shell Thickness"].astype(float)
        invalid = ~(
            np.isclose(values, 0.0, rtol=0.0, atol=1.0e-12)
            | values.between(10.0, 20.0, inclusive="both")
        )

        if bool(invalid.any()):
            selected = structures.loc[invalid].copy()
            violation_tables.append(
                pd.DataFrame(
                    {
                        "Record_Type": "structure",
                        "Structure_Index": selected[
                            "Structure_Index"
                        ].astype(int),
                        "Index": np.nan,
                        "Parameter": "Inner Side Shell Thickness",
                        "Value": selected[
                            "Inner Side Shell Thickness"
                        ].astype(float),
                        "Lower_Bound": 10.0,
                        "Upper_Bound": 20.0,
                        "Violation": "outside_zero_or_active_range",
                    }
                )
            )

    # The web-frame flange width is removed only when the inner-side shell is
    # active.  Preserve that relational cleaned-geometry contract instead of
    # treating its valid zero as a generic lower-bound failure.
    if (
        not structures.empty
        and {
            "Inner Side Shell Thickness",
            "web frame w",
        }.issubset(structures.columns)
    ):
        inner_side = structures["Inner Side Shell Thickness"].astype(float)
        web_frame_width = structures["web frame w"].astype(float)
        valid = (
            (inner_side >= 10.0)
            & np.isclose(web_frame_width, 0.0, rtol=0.0, atol=1.0e-12)
        ) | (
            np.isclose(inner_side, 0.0, rtol=0.0, atol=1.0e-12)
            & web_frame_width.between(300.0, 600.0, inclusive="both")
        )
        invalid = ~valid

        if bool(invalid.any()):
            selected = structures.loc[invalid].copy()
            violation_tables.append(
                pd.DataFrame(
                    {
                        "Record_Type": "structure",
                        "Structure_Index": selected[
                            "Structure_Index"
                        ].astype(int),
                        "Index": np.nan,
                        "Parameter": "web frame w",
                        "Value": selected["web frame w"].astype(float),
                        "Lower_Bound": 300.0,
                        "Upper_Bound": 600.0,
                        "Violation": "inconsistent_with_inner_side_shell",
                    }
                )
            )

    coupled_hull_definitions = catalog.loc[
        catalog["Name"].isin(POST_COUPLING_HULL_PARAMETER_NAMES)
    ]
    append_violations(
        systems,
        coupled_hull_definitions,
        "system_post_coupling",
    )

    # Every saved system row must preserve the directed hull-to-structure
    # coupling Rc -> R_b.  This catches a future regression that treats the
    # dimensionless hull ratio itself as metres or lets structure cleanup
    # overwrite the radius selected by the hull design module.
    coupling_columns = {"B", "Rc", "R_b", "Structure_Index", "Index"}

    if not systems.empty and coupling_columns.issubset(systems.columns):
        expected_r_b = pd.Series(
            derive_structural_bilge_radius_m(
                systems["B"].astype(float).to_numpy(),
                systems["Rc"].astype(float).to_numpy(),
            ),
            index=systems.index,
        )
        actual_r_b = systems["R_b"].astype(float)
        tolerance = 1.0e-10*np.maximum(1.0, np.abs(expected_r_b))
        mismatch = np.abs(actual_r_b - expected_r_b) > tolerance

        if bool(mismatch.any()):
            selected = systems.loc[mismatch].copy()
            selected_expected = expected_r_b.loc[mismatch]
            violation_tables.append(
                pd.DataFrame(
                    {
                        "Record_Type": "system_hull_structure_coupling",
                        "Structure_Index": selected[
                            "Structure_Index"
                        ].astype(int),
                        "Index": selected["Index"].astype(int),
                        "Parameter": "R_b = 0.5*B*Rc",
                        "Value": selected["R_b"].astype(float),
                        "Lower_Bound": selected_expected,
                        "Upper_Bound": selected_expected,
                        "Violation": "coupling_mismatch",
                    }
                )
            )

    if not violation_tables:
        return pd.DataFrame(columns=columns)

    return pd.concat(violation_tables, ignore_index=True)[columns].sort_values(
        ["Record_Type", "Structure_Index", "Index", "Parameter"],
        na_position="first",
    )


def _build_constraint_views(
    constraints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build wide rule bits and subsystem feasibility from long audit rows."""

    if constraints.empty:
        return pd.DataFrame(), pd.DataFrame()

    constraint_rows = constraints.copy()
    constraint_rows["Constraint_Key"] = canonical_constraint_keys(
        constraint_rows,
        audited=True,
    )
    has_constraint_result = constraint_rows["Constraint Bit"].notna()

    # Non-applicable requirements carry an explicit True bit under the current
    # contract.  Only partial calculations and diagnostic metrics remain
    # missing because they have no pass/fail interpretation.
    constraint_rows["Training_Constraint_Bit"] = constraint_rows[
        "Constraint Bit"
    ].astype(float)
    constraint_rows.loc[
        ~has_constraint_result,
        "Training_Constraint_Bit",
    ] = np.nan

    # Collapse reviewed aliases only in this derived view.  Metric and N/A
    # rows are then added back with NaN bits so their canonical columns remain
    # visible without entering feasibility totals.  Long-form source rows are
    # never deleted or rewritten.
    applicable_rows = constraint_rows.loc[has_constraint_result].copy()
    applicable_rows = collapse_constraint_aliases(
        applicable_rows,
        audited=True,
    )
    applicable_rows['Training_Constraint_Bit'] = applicable_rows[
        'Constraint Bit'
    ].astype(float)

    non_counted_rows = constraint_rows.loc[~has_constraint_result].copy()
    wide_source = pd.concat(
        [applicable_rows, non_counted_rows],
        ignore_index=True,
    )

    # Keep compilation available even when a malformed shard repeats a key.
    # A separate acceptance check below makes unexpected literal duplicates
    # visible.  Plain ``pivot`` preserves a rule column even when that rule is
    # not applicable to every generated design.
    unique_constraint_rows = wide_source.drop_duplicates(
        subset=["Index", "Constraint_Key"],
        keep="first",
    )
    wide = unique_constraint_rows.pivot(
        index="Index",
        columns="Constraint_Key",
        values="Training_Constraint_Bit",
    ).reset_index()
    wide.columns.name = None

    # Partial numeric requirement rows remain in the long audit table and in
    # the wide catalog as NaN, but only canonical Boolean comparisons define
    # subsystem/global feasibility.
    subsystem = (
        applicable_rows.groupby(["Index", "Class"], as_index=False)
        .agg(
            Applicable_Constraint_Count=("Constraint Bit", "size"),
            Passed_Constraint_Count=("Constraint Bit", "sum"),
            Subsystem_Feasible=("Constraint Bit", "all"),
        )
    )

    global_feasibility = (
        applicable_rows.groupby("Index", as_index=False)
        .agg(
            Applicable_Constraint_Count=("Constraint Bit", "size"),
            Passed_Constraint_Count=("Constraint Bit", "sum"),
            Subsystem_Feasible=("Constraint Bit", "all"),
        )
    )
    global_feasibility.insert(1, "Class", "ALL_AUDITED_CONSTRAINTS")
    subsystem = pd.concat(
        [subsystem, global_feasibility],
        ignore_index=True,
    )

    return wide.sort_values("Index"), subsystem.sort_values(["Index", "Class"])


def _build_v1p1_constraint_views(
    constraints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ML-ready views from the expanded legacy-format V1.1 rows.

    Calculated hydrostatic and flooding metrics deliberately have a blank
    ``Constraint Bit``.  They remain in the long table but are excluded from
    pass/fail aggregation.
    """

    if constraints.empty:
        return pd.DataFrame(), pd.DataFrame()

    evaluated = collapse_constraint_aliases(constraints)
    evaluated["Constraint_Key"] = evaluated[
        "Canonical Constraint Key"
    ]
    evaluated["Constraint Bit"] = (
        evaluated["Constraint Bit"].astype(bool)
    )

    unique_rows = evaluated.drop_duplicates(
        subset=["Index", "Constraint_Key"],
        keep="first",
    )
    wide = unique_rows.pivot(
        index="Index",
        columns="Constraint_Key",
        values="Constraint Bit",
    ).reset_index()
    wide.columns.name = None

    subsystem = (
        evaluated.groupby(["Index", "Class"], as_index=False)
        .agg(
            Evaluated_Constraint_Count=("Constraint Bit", "size"),
            Passed_Constraint_Count=("Constraint Bit", "sum"),
            Subsystem_Feasible=("Constraint Bit", "all"),
        )
    )
    global_feasibility = (
        evaluated.groupby("Index", as_index=False)
        .agg(
            Evaluated_Constraint_Count=("Constraint Bit", "size"),
            Passed_Constraint_Count=("Constraint Bit", "sum"),
            Subsystem_Feasible=("Constraint Bit", "all"),
        )
    )
    global_feasibility.insert(1, "Class", "ALL_V1P1_CONSTRAINTS")
    subsystem = pd.concat(
        [subsystem, global_feasibility],
        ignore_index=True,
    )

    return wide.sort_values("Index"), subsystem.sort_values(["Index", "Class"])


def _build_normalized_constraint_margins(
    constraints: pd.DataFrame,
) -> pd.DataFrame:
    """Return dimensionless residuals for every numeric V1.1 constraint.

    Every V1.1 minimum and maximum predicate uses the same residual sign
    convention: ``Constraint Eval <= 0`` passes.  Dividing by the larger
    magnitude of the threshold and observed value preserves that sign while
    making millimetre, metre, stress, and ratio checks comparable for dataset
    analysis.  A scale floor is used only for genuine zero/zero comparisons;
    Boolean predicates and metric-only rows intentionally retain ``NaN``.

    The individual per-design files are not changed.  They keep the exact
    eight-column legacy format requested for GenISys V1.1.
    """

    columns = [
        "Index",
        "Structure_Index",
        "Class",
        "Requirement",
        "Name",
        "Constraint_Key",
        "Threshold_Value",
        "Constraint_Value",
        "Constraint_Eval",
        "Normalization_Scale",
        "Normalized_Constraint_Eval",
        "Constraint_Bit",
        "Units",
    ]

    if constraints.empty:
        return pd.DataFrame(columns=columns)

    working = constraints.copy()
    threshold = pd.to_numeric(
        working["Threshold Value"],
        errors="coerce",
    )
    value = pd.to_numeric(
        working["Constraint Value"],
        errors="coerce",
    )
    residual = pd.to_numeric(
        working["Constraint Eval"],
        errors="coerce",
    )
    numeric = (
        threshold.notna()
        & value.notna()
        & residual.notna()
        & working["Constraint Bit"].notna()
        & ~working["Units"].astype(str).str.casefold().eq("boolean")
        & ~working["Units"].astype(str).str.casefold().eq("n/a")
    )
    working = working.loc[numeric].copy()
    threshold = threshold.loc[numeric].astype(float)
    value = value.loc[numeric].astype(float)
    residual = residual.loc[numeric].astype(float)

    scale = np.maximum.reduce(
        [
            np.abs(threshold.to_numpy()),
            np.abs(value.to_numpy()),
            np.full(len(working), 1.0e-12),
        ]
    )
    working["Constraint_Key"] = (
        working["Requirement"].astype(str)
        + "::"
        + working["Name"].astype(str)
    )
    working["Threshold_Value"] = threshold.to_numpy()
    working["Constraint_Value"] = value.to_numpy()
    working["Constraint_Eval"] = residual.to_numpy()
    working["Normalization_Scale"] = scale
    working["Normalized_Constraint_Eval"] = residual.to_numpy()/scale
    working["Constraint_Bit"] = (
        working["Constraint Bit"].astype(str).str.lower() == "true"
    )

    for identifier in ("Index", "Structure_Index"):
        if identifier not in working:
            working[identifier] = np.nan

    result = working.rename(columns={"Units": "Units"})[columns]

    return result.sort_values(["Index", "Class", "Requirement", "Name"])


def _build_subsystem_feasibility_wide(
    subsystem: pd.DataFrame,
    index_name: str = "Index",
) -> pd.DataFrame:
    """Pivot long subsystem feasibility into one Boolean column per class."""

    if subsystem.empty:
        return pd.DataFrame(columns=[index_name])

    unique = subsystem.drop_duplicates(
        subset=[index_name, "Class"],
        keep="first",
    ).copy()
    wide = unique.pivot(
        index=index_name,
        columns="Class",
        values="Subsystem_Feasible",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            column: f"Feasible::{column}"
            for column in wide.columns
            if column != index_name
        }
    )

    return wide.sort_values(index_name)


def _build_ml_design_matrix(
    systems: pd.DataFrame,
    scalar_results: pd.DataFrame,
    constraint_bits: pd.DataFrame,
    subsystem_feasibility: pd.DataFrame,
) -> pd.DataFrame:
    """Join inputs, scalar outputs, rule bits, and subsystem target labels.

    ``systems`` is the population authority and therefore drives a left join.
    Failed evaluations remain as rows with their explicit failure status and
    missing calculated fields.  This prevents a model-preparation step from
    accidentally selecting only successfully evaluated designs.
    """

    if systems.empty:
        return pd.DataFrame()

    result = systems.copy()

    for table in (
        scalar_results,
        constraint_bits,
        _build_subsystem_feasibility_wide(subsystem_feasibility),
    ):
        if table.empty or "Index" not in table:
            continue

        # Dataset identifiers already exist in System_Designs.csv.  Drop
        # duplicate copies before the one-to-one join so suffixes cannot make
        # the machine-learning schema ambiguous.
        joining = table.drop(
            columns=[
                column
                for column in IDENTIFIER_COLUMNS
                if column != "Index" and column in table
            ],
            errors="ignore",
        )
        result = result.merge(
            joining,
            on="Index",
            how="left",
            validate="one_to_one",
        )

    return result.sort_values("Index")


def _build_runtime_constraint_catalog(
    constraints: pd.DataFrame,
) -> pd.DataFrame:
    """Build one stable catalog of every expanded constraint and metric row."""

    columns = [
        "Constraint_Key",
        "Class",
        "Name",
        "Requirement",
        "Units",
        "Output_Type",
        "Observed_Row_Count",
        "Observed_System_Count",
        "Conditional_Runtime_Row",
    ]

    if constraints.empty:
        return pd.DataFrame(columns=columns)

    working = constraints.copy()
    working["Constraint_Key"] = (
        working["Requirement"].astype(str)
        + "::"
        + working["Name"].astype(str)
    )
    working["Output_Type"] = np.where(
        working["Constraint Bit"].notna(),
        "constraint",
        "metric",
    )
    total_systems = int(working["Index"].nunique())

    catalog = (
        working.groupby(
            [
                "Constraint_Key",
                "Class",
                "Name",
                "Requirement",
                "Units",
                "Output_Type",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            Observed_Row_Count=("Index", "size"),
            Observed_System_Count=("Index", "nunique"),
        )
    )
    catalog["Conditional_Runtime_Row"] = (
        catalog["Observed_System_Count"] < total_systems
    )

    return catalog[columns].sort_values(
        ["Output_Type", "Class", "Requirement", "Name"]
    )


def _unit_from_scalar_field(field_name: str) -> str:
    """Infer display units from the explicit V1.1 scalar-field suffix."""

    suffix_units = (
        ("_gCO2_per_t_nm", "g CO2/(t nm)"),
        ("_pounds_per_tonne", "GBP/t"),
        ("_pounds_per_t", "GBP/t"),
        ("_days_per_call", "day/call"),
        ("_Days_per_Voyage", "day/voyage"),
        ("Trips_per_Year", "1/year"),
        ("_m_per_s", "m/s"),
        ("_kg_per_kWh", "kg/kWh"),
        ("_t_per_day", "t/day"),
        ("_kg_per_m3", "kg/m^3"),
        ("_percent", "%"),
        ("_kg_m", "kg-m"),
        ("_Seconds", "s"),
        ("_Nm", "N-m"),
        ("_N", "N"),
        ("_kNm", "kN-m"),
        ("_kN", "kN"),
        ("_kW", "kW"),
        ("_nmi", "nmi"),
        ("_knots", "kn"),
        ("_deg", "deg"),
        ("_m4", "m^4"),
        ("_m3", "m^3"),
        ("_m2", "m^2"),
        ("_kg", "kg"),
        ("_t", "t"),
        ("_m", "m"),
        ("_pounds", "GBP"),
        ("_Days", "day"),
    )

    for suffix, units in suffix_units:
        if field_name.endswith(suffix):
            return units

    return "dimensionless"


def _build_scalar_result_catalog(
    scalar_results: pd.DataFrame,
) -> pd.DataFrame:
    """Describe the compiled scalar-result schema in machine-readable form."""

    columns = ["Field", "Role", "Units", "Data_Type"]

    if scalar_results.empty:
        return pd.DataFrame(columns=columns)

    metadata_fields = set(
        [
            "Evaluation_Status",
            "Failure_Stage",
            "Failure_Category",
            "Error_Type",
            "Error_Message",
            "Integration_Method",
        ]
        + [
            "Index",
            "Structure_Index",
            "Hull_Variant_Index",
            "Operating_Condition_Index",
            "Hull_Candidate_Index",
            "Structure_Sample_Attempt",
        ]
    )
    rows = []

    for field in scalar_results.columns:
        if (
            field in metadata_fields
            or field.endswith("_Status")
            or field.endswith("_Basis")
            or field.endswith("_Fields")
        ):
            role = "identifier_or_status"
        elif field.endswith("_Passed") or field.startswith("Global_"):
            role = "feasibility_flag"
        elif field.endswith("_Count") or field.endswith("_Iterations"):
            role = "diagnostic_count"
        elif field in DEFAULT_ECONOMIC_INPUTS:
            role = "scenario_input"
        else:
            role = "calculated_metric"

        rows.append(
            {
                "Field": field,
                "Role": role,
                "Units": _unit_from_scalar_field(field),
                "Data_Type": str(scalar_results[field].dtype),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def compile_dataset(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    number_of_structures: int = NUMBER_OF_STRUCTURES,
    save_mode: str = "all",
    start_structure: int = 0,
) -> dict[str, Path]:
    """Compile available V1.1 shards against an explicit population size.

    Production keeps the 5,000-structure default.  Laptop smoke tests pass a
    smaller expected count and an optional nonzero starting index so the same
    acceptance logic can validate independent laptop batches without
    weakening the production contract.
    """

    if save_mode not in {"all", "reduced"}:
        raise ValueError("save_mode must be either 'all' or 'reduced'.")

    output_root = Path(output_root)
    compiled_directory = output_root / "compiled"
    manifest_directory = output_root / "manifests"

    shard_families = {
        "structure_parameters": _csv_shard_paths(
            output_root / "structures" / "parameters",
            "Structure_Parameters_*.csv",
        ),
        "structure_status": _csv_shard_paths(
            output_root / "structures" / "parameters",
            "Structure_Status_*.csv",
        ),
        "structure_failures": _csv_shard_paths(
            output_root / "structures" / "failures",
            "Structure_Failures_*.csv",
        ),
        "system_designs": _csv_shard_paths(
            output_root / "systems" / "shards",
            "System_Designs_*.csv",
        ),
        "scalar_results": _csv_shard_paths(
            output_root / "evaluations" / "scalar_results",
            "System_Results_*.csv",
        ),
        "updated_parameters": _csv_shard_paths(
            output_root / "evaluations" / "updated_parameters",
            "Updated_System_Parameters_*.csv",
        ),
        "audited_constraints": _csv_shard_paths(
            output_root / "evaluations" / "constraints",
            "Audited_Constraints_*.csv",
        ),
        "expanded_constraints_v1p1": _csv_shard_paths(
            output_root / "evaluations" / "constraints_v1p1" / "shards",
            "Constraints_v1p1_*.csv",
        ),
        "legacy_constraints": _csv_shard_paths(
            output_root / "evaluations" / "legacy_constraints",
            "Legacy_Constraints_*.csv",
        ),
        "evaluation_failures": _csv_shard_paths(
            output_root / "evaluations" / "failures",
            "Evaluation_Failures_*.csv",
        ),
        "structure_properties": _csv_shard_paths(
            output_root / "evaluations" / "structure_properties",
            "Structure_Evaluations_*.csv",
        ),
        "weight_components": _csv_shard_paths(
            output_root / "evaluations" / "weight_components",
            "Weight_Components_*.csv",
        ),
        "compartment_loads": _csv_shard_paths(
            output_root / "evaluations" / "compartment_loads",
            "Compartment_Loads_*.csv",
        ),
        "compartment_condition_states": _csv_shard_paths(
            output_root / "evaluations" / "compartment_condition_states",
            "Compartment_Condition_States_*.csv",
        ),
        "bulkhead_locations": _csv_shard_paths(
            output_root / "evaluations" / "bulkhead_locations",
            "Bulkhead_Locations_*.csv",
        ),
        "hatch_covers": _csv_shard_paths(
            output_root / "evaluations" / "hatch_covers",
            "Hatch_Covers_*.csv",
        ),
        "loading_conditions": _csv_shard_paths(
            output_root / "evaluations" / "loading_conditions",
            "Loading_Conditions_*.csv",
        ),
        "loading_condition_components": _csv_shard_paths(
            output_root / "evaluations" / "loading_condition_components",
            "Loading_Condition_Components_*.csv",
        ),
        "upright_stability": _csv_shard_paths(
            output_root / "evaluations" / "upright_stability",
            "Upright_Stability_*.csv",
        ),
        "named_condition_loads": _csv_shard_paths(
            output_root / "evaluations" / "named_condition_loads",
            "Named_Condition_Loads_*.csv",
        ),
        "load_closure": _csv_shard_paths(
            output_root / "evaluations" / "load_closure",
            "Load_Closure_*.csv",
        ),
        "economic_performance": _csv_shard_paths(
            output_root / "evaluations" / "economic_performance",
            "Economic_Performance_*.csv",
        ),
    }
    tables = {
        family: _read_shards(paths)
        for family, paths in shard_families.items()
    }

    output_names = {
        "structure_parameters": "Structure_Parameters.csv",
        "structure_status": "Structure_Status.csv",
        "structure_failures": "Structure_Failures.csv",
        "system_designs": "System_Designs.csv",
        "scalar_results": "Ship_System_Performance.csv",
        "updated_parameters": "Updated_Ship_System_Parameters.csv",
        "audited_constraints": "Audited_Constraint_Evaluations.csv",
        "expanded_constraints_v1p1": "Constraint_Evaluations_v1p1.csv",
        "legacy_constraints": "Legacy_Constraint_Evaluations.csv",
        "evaluation_failures": "Evaluation_Failures.csv",
        "structure_properties": "Structure_Evaluations.csv",
        "weight_components": "Weight_Components.csv",
        "compartment_loads": "Compartment_Loads.csv",
        "compartment_condition_states": "Compartment_Condition_States.csv",
        "bulkhead_locations": "Bulkhead_Locations.csv",
        "hatch_covers": "Hatch_Covers.csv",
        "loading_conditions": "Loading_Conditions.csv",
        "loading_condition_components": "Loading_Condition_Components.csv",
        "upright_stability": "Upright_Stability.csv",
        "named_condition_loads": "Named_Condition_Loads.csv",
        "load_closure": "Load_Closure.csv",
        "economic_performance": "Economic_Performance.csv",
    }
    outputs = {}

    # Reduced mode keeps the authoritative compressed source shards for every
    # long-form table family.  Re-materializing those same rows in ``compiled``
    # would nearly double dataset size, so only population-level core tables
    # are copied there.  Derived ML/constraint products are still produced.
    reduced_materialized_families = {
        "structure_parameters",
        "structure_status",
        "structure_failures",
        "system_designs",
        "scalar_results",
        "updated_parameters",
        "evaluation_failures",
    }

    for family, output_name in output_names.items():
        if tables[family].empty and family == "legacy_constraints":
            continue

        if (
            save_mode == "reduced"
            and family not in reduced_materialized_families
        ):
            continue

        output_path = _compiled_csv_path(
            compiled_directory,
            output_name,
            save_mode,
        )
        _write_csv_atomic(tables[family], output_path)
        outputs[family] = output_path

    constraint_bits, subsystem_feasibility = _build_constraint_views(
        tables["audited_constraints"]
    )
    constraint_bits_path = _compiled_csv_path(
        compiled_directory,
        "Compiled_Audited_Constraint_Bits.csv",
        save_mode,
    )
    subsystem_path = _compiled_csv_path(
        compiled_directory,
        "Subsystem_Level_Feasibility.csv",
        save_mode,
    )
    _write_csv_atomic(constraint_bits, constraint_bits_path)
    _write_csv_atomic(subsystem_feasibility, subsystem_path)
    outputs["constraint_bits"] = constraint_bits_path
    outputs["subsystem_feasibility"] = subsystem_path

    v1p1_bits, v1p1_subsystem = _build_v1p1_constraint_views(
        tables["expanded_constraints_v1p1"]
    )
    v1p1_bits_path = _compiled_csv_path(
        compiled_directory,
        "Compiled_V1p1_Constraint_Bits.csv",
        save_mode,
    )
    v1p1_subsystem_path = _compiled_csv_path(
        compiled_directory,
        "V1p1_Subsystem_Level_Feasibility.csv",
        save_mode,
    )
    _write_csv_atomic(v1p1_bits, v1p1_bits_path)
    _write_csv_atomic(v1p1_subsystem, v1p1_subsystem_path)
    outputs["v1p1_constraint_bits"] = v1p1_bits_path
    outputs["v1p1_subsystem_feasibility"] = v1p1_subsystem_path

    v1p1_subsystem_wide = _build_subsystem_feasibility_wide(
        v1p1_subsystem
    )
    v1p1_subsystem_wide_path = _compiled_csv_path(
        compiled_directory,
        "V1p1_Subsystem_Feasibility_Bits.csv",
        save_mode,
    )
    _write_csv_atomic(v1p1_subsystem_wide, v1p1_subsystem_wide_path)
    outputs["v1p1_subsystem_feasibility_bits"] = (
        v1p1_subsystem_wide_path
    )

    checks = []
    structures = tables["structure_parameters"]
    systems = tables["system_designs"]
    scalar_results = tables["scalar_results"]
    constraints = tables["audited_constraints"]
    expanded_constraints = tables["expanded_constraints_v1p1"]
    parameter_catalog = load_parameter_catalog()

    # Publish analysis-ready views without changing the requested legacy
    # format of any individual design's constraint file.  The normalized long
    # table supports rule-margin studies, while the joined matrix retains the
    # complete intended population, including explicitly failed evaluations.
    normalized_margins = _build_normalized_constraint_margins(
        expanded_constraints
    )
    normalized_margins_path = _compiled_csv_path(
        compiled_directory,
        "Normalized_Constraint_Margins.csv",
        save_mode,
    )
    _write_csv_atomic(normalized_margins, normalized_margins_path)
    outputs["normalized_constraint_margins"] = normalized_margins_path

    ml_design_matrix = _build_ml_design_matrix(
        systems,
        scalar_results,
        v1p1_bits,
        v1p1_subsystem,
    )
    ml_design_matrix_path = _compiled_csv_path(
        compiled_directory,
        "ML_Design_Matrix.csv",
        save_mode,
    )
    _write_csv_atomic(ml_design_matrix, ml_design_matrix_path)
    outputs["ml_design_matrix"] = ml_design_matrix_path

    # Preserve every post-cleanup/post-coupling violation in a separate table,
    # even though a corrected production generator should normally make this
    # output empty.  This is especially useful when compiling older shards or
    # hand-edited laptop fixtures.
    parameter_bound_violations = _build_parameter_bound_violations(
        structures,
        systems,
        parameter_catalog,
    )
    parameter_bound_path = _compiled_csv_path(
        compiled_directory,
        "Parameter_Bound_Violations.csv",
        save_mode,
        compress_in_reduced_mode=False,
    )
    _write_csv_atomic(parameter_bound_violations, parameter_bound_path)
    outputs["parameter_bound_violations"] = parameter_bound_path

    # Copy the ordered input schema into the compiled dataset and derive the
    # two output catalogs from the rows/fields actually produced by this run.
    # These tables make the 100,000-row dataset self-describing without
    # duplicating prose documentation inside every design file.
    parameter_catalog_path = _compiled_csv_path(
        compiled_directory,
        "Parameter_Catalog.csv",
        save_mode,
        compress_in_reduced_mode=False,
    )
    _write_csv_atomic(parameter_catalog, parameter_catalog_path)
    outputs["parameter_catalog"] = parameter_catalog_path

    runtime_catalog = _build_runtime_constraint_catalog(
        expanded_constraints
    )
    runtime_catalog_path = _compiled_csv_path(
        compiled_directory,
        "Constraint_Catalog.csv",
        save_mode,
        compress_in_reduced_mode=False,
    )
    _write_csv_atomic(runtime_catalog, runtime_catalog_path)
    outputs["constraint_catalog"] = runtime_catalog_path

    scalar_catalog = _build_scalar_result_catalog(scalar_results)
    scalar_catalog_path = _compiled_csv_path(
        compiled_directory,
        "Metric_Catalog.csv",
        save_mode,
        compress_in_reduced_mode=False,
    )
    _write_csv_atomic(scalar_catalog, scalar_catalog_path)
    outputs["metric_catalog"] = scalar_catalog_path

    completed_status = (
        tables["structure_status"].loc[
            tables["structure_status"]["Status"] == "completed"
        ]
        if "Status" in tables["structure_status"]
        else pd.DataFrame()
    )
    _record_check(
        checks,
        "accepted_structure_count",
        len(structures) == number_of_structures,
        len(structures),
        number_of_structures,
        "One accepted structure-parameter row is required per Structure_Index.",
    )
    _record_check(
        checks,
        "completed_structure_status_count",
        len(completed_status) == number_of_structures,
        len(completed_status),
        number_of_structures,
        "Failed or unsubmitted structure shards remain visible here.",
    )

    structure_ids = (
        structures["Structure_Index"].astype(int)
        if "Structure_Index" in structures
        else pd.Series(dtype=int)
    )
    end_structure = start_structure + number_of_structures
    expected_structure_ids = set(range(start_structure, end_structure))
    observed_structure_ids = set(structure_ids.tolist())
    _record_check(
        checks,
        "complete_structure_index_set",
        observed_structure_ids == expected_structure_ids,
        len(observed_structure_ids),
        len(expected_structure_ids),
        (
            f"Missing={sorted(expected_structure_ids - observed_structure_ids)[:20]}; "
            f"unexpected={sorted(observed_structure_ids - expected_structure_ids)[:20]}"
        ),
    )
    _record_check(
        checks,
        "unique_structure_indices",
        not structure_ids.duplicated().any(),
        int(structure_ids.duplicated().sum()),
        0,
        "Observed value is the number of duplicate Structure_Index rows.",
    )

    structural_parameter_names = parameter_catalog.loc[
        lambda table: table["Class"] == "Structure",
        "Name",
    ].tolist()
    duplicate_structure_vectors = (
        int(structures.duplicated(subset=structural_parameter_names).sum())
        if not structures.empty
        else 0
    )
    _record_check(
        checks,
        "unique_structural_parameter_vectors",
        duplicate_structure_vectors == 0,
        duplicate_structure_vectors,
        0,
        "Observed value is the number of exact duplicate 120-value structures.",
    )

    intended_system_count = number_of_structures * SYSTEMS_PER_STRUCTURE
    system_ids = (
        systems["Index"].astype(int)
        if "Index" in systems
        else pd.Series(dtype=int)
    )
    _record_check(
        checks,
        "system_design_count",
        len(systems) == intended_system_count,
        len(systems),
        intended_system_count,
        (
            f"The intended population is {number_of_structures} x "
            f"{HULLS_PER_STRUCTURE} x "
            f"{OPERATING_CONDITIONS_PER_HULL}."
        ),
    )
    _record_check(
        checks,
        "unique_system_indices",
        not system_ids.duplicated().any(),
        int(system_ids.duplicated().sum()),
        0,
        "Observed value is the number of duplicate Index rows.",
    )
    first_system_index = start_structure*SYSTEMS_PER_STRUCTURE
    expected_system_ids = set(
        range(first_system_index, first_system_index + intended_system_count)
    )
    observed_system_ids = set(system_ids.tolist())
    _record_check(
        checks,
        "complete_system_index_set",
        observed_system_ids == expected_system_ids,
        len(observed_system_ids),
        len(expected_system_ids),
        (
            f"Missing={sorted(expected_system_ids - observed_system_ids)[:20]}; "
            f"unexpected={sorted(observed_system_ids - expected_system_ids)[:20]}"
        ),
    )

    if not systems.empty:
        # Apply the published identifier and draft equations vectorially.  The
        # production table contains 100,000 rows, so compilation must not
        # reload the parameter catalog or invoke a Python callback per row.
        expected_indices = (
            systems["Structure_Index"].astype(int) * SYSTEMS_PER_STRUCTURE
            + systems["Hull_Variant_Index"].astype(int)
            * OPERATING_CONDITIONS_PER_HULL
            + systems["Operating_Condition_Index"].astype(int)
        )
        identifier_errors = int((systems["Index"] != expected_indices).sum())
        systems_per_structure = systems.groupby("Structure_Index").size()
        expected_drafts = (
            systems["Draft_frac"].to_numpy(dtype=float)
            * systems["Dd"].to_numpy(dtype=float)
            * systems["LOA"].to_numpy(dtype=float)
        )
        draft_errors = int(
            (
                ~np.isclose(
                    systems["T"].to_numpy(dtype=float),
                    expected_drafts,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            ).sum()
        )
        parameter_names = parameter_catalog["Name"].tolist()
        duplicate_parameter_rows = int(
            systems.duplicated(subset=parameter_names).sum()
        )

        hull_group_columns = ["Structure_Index", "Hull_Variant_Index"]
        hull_group_sizes = systems.groupby(hull_group_columns).size()
        hulls_per_structure = systems.groupby("Structure_Index")[
            "Hull_Variant_Index"
        ].nunique()
        operating_values_in_range = bool(
            systems["Hull_Variant_Index"].between(
                0,
                HULLS_PER_STRUCTURE - 1,
            ).all()
            and systems["Operating_Condition_Index"].between(
                0,
                OPERATING_CONDITIONS_PER_HULL - 1,
            ).all()
        )

        # Every five-row group is one unchanged hull with only the three legacy
        # operating coordinates (and their derived dimensional T) varying.
        stable_parameter_names = [
            name
            for name in parameter_names
            if name not in {"Draft_frac", "Fn_ship", "Voyage_Range", "T"}
        ]
        altered_hull_groups = int(
            systems.groupby(hull_group_columns)[stable_parameter_names]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
            .sum()
        )
    else:
        identifier_errors = 0
        systems_per_structure = pd.Series(dtype=int)
        draft_errors = 0
        duplicate_parameter_rows = 0
        hull_group_sizes = pd.Series(dtype=int)
        hulls_per_structure = pd.Series(dtype=int)
        operating_values_in_range = False
        altered_hull_groups = 0

    _record_check(
        checks,
        "deterministic_identifier_formula",
        identifier_errors == 0,
        identifier_errors,
        0,
        "Index = structure*20 + hull*5 + operating condition.",
    )
    _record_check(
        checks,
        "twenty_systems_per_structure",
        (
            len(systems_per_structure) == number_of_structures
            and bool((systems_per_structure == SYSTEMS_PER_STRUCTURE).all())
        ),
        int((systems_per_structure != SYSTEMS_PER_STRUCTURE).sum()),
        0,
        "Observed value is the number of structures without exactly 20 rows.",
    )
    _record_check(
        checks,
        "constraint_draft_lineage",
        draft_errors == 0,
        draft_errors,
        0,
        "T must equal Draft_frac * Dd * LOA for every system row.",
    )
    _record_check(
        checks,
        "unique_complete_parameter_vectors",
        duplicate_parameter_rows == 0,
        duplicate_parameter_rows,
        0,
        "Identifier columns are excluded from the duplicate-design check.",
    )
    _record_check(
        checks,
        "four_hulls_and_five_operating_points",
        (
            operating_values_in_range
            and len(hulls_per_structure) == number_of_structures
            and bool((hulls_per_structure == HULLS_PER_STRUCTURE).all())
            and bool(
                (
                    hull_group_sizes
                    == OPERATING_CONDITIONS_PER_HULL
                ).all()
            )
        ),
        (
            f"hull_groups={len(hull_group_sizes)}; "
            f"wrong_group_sizes="
            f"{int((hull_group_sizes != OPERATING_CONDITIONS_PER_HULL).sum())}"
        ),
        (
            f"{number_of_structures * HULLS_PER_STRUCTURE} hull groups, "
            f"{OPERATING_CONDITIONS_PER_HULL} rows each"
        ),
        "Also requires hull indices 0-3 and operating indices 0-4.",
    )
    _record_check(
        checks,
        "legacy_operating_copy_contract",
        altered_hull_groups == 0,
        altered_hull_groups,
        0,
        (
            "Within each five-row hull group, only Draft_frac, Fn_ship, "
            "Voyage_Range, and derived T may vary."
        ),
    )
    _record_check(
        checks,
        "post_coupling_parameter_bounds",
        parameter_bound_violations.empty,
        len(parameter_bound_violations),
        0,
        (
            "Final CadQuery-cleaned structural values and coupled LOA/Bd/Dd/"
            "Bc/Rc values must remain inside the published parameter bounds. "
            "See Parameter_Bound_Violations.csv for exact records."
        ),
    )

    completed_evaluations = (
        scalar_results.loc[
            scalar_results["Evaluation_Status"] == "completed"
        ]
        if "Evaluation_Status" in scalar_results
        else pd.DataFrame()
    )
    _record_check(
        checks,
        "one_scalar_result_per_system",
        len(scalar_results) == len(systems)
        and not scalar_results.get("Index", pd.Series(dtype=int)).duplicated().any(),
        len(scalar_results),
        len(systems),
        "Failed evaluations must remain as labeled scalar-result rows.",
    )
    _record_check(
        checks,
        "audited_constraints_exist_for_completed_evaluations",
        (
            len(completed_evaluations) == 0
            or (
                not constraints.empty
                and set(completed_evaluations["Index"].astype(int))
                <= set(constraints["Index"].astype(int))
            )
        ),
        constraints["Index"].nunique() if "Index" in constraints else 0,
        len(completed_evaluations),
        "Every completed system requires at least one audit-qualified row.",
    )
    _record_check(
        checks,
        "expanded_constraints_exist_for_completed_evaluations",
        (
            len(completed_evaluations) == 0
            or (
                not expanded_constraints.empty
                and set(completed_evaluations["Index"].astype(int))
                <= set(expanded_constraints["Index"].astype(int))
            )
        ),
        (
            expanded_constraints["Index"].nunique()
            if "Index" in expanded_constraints
            else 0
        ),
        len(completed_evaluations),
        "Every completed system requires expanded V1.1 constraint rows.",
    )

    evaluation_failures = tables["evaluation_failures"]
    classified_failure_count = (
        int(evaluation_failures["Failure_Category"].notna().sum())
        if "Failure_Category" in evaluation_failures
        else 0
    )
    _record_check(
        checks,
        "every_evaluation_failure_is_classified",
        (
            evaluation_failures.empty
            or classified_failure_count == len(evaluation_failures)
        ),
        classified_failure_count,
        len(evaluation_failures),
        (
            "Failure categories distinguish physical infeasibility, method-"
            "domain rejection, data/I/O failure, and numerical or code failure."
        ),
    )

    numerical_failure_count = (
        int(
            evaluation_failures["Failure_Category"]
            .astype(str)
            .eq("numerical_or_implementation_failure")
            .sum()
        )
        if "Failure_Category" in evaluation_failures
        else len(evaluation_failures)
    )
    _record_check(
        checks,
        "no_numerical_or_implementation_failures",
        numerical_failure_count == 0,
        numerical_failure_count,
        0,
        (
            "Physical or method-domain rejection may be valid data; an "
            "unresolved numerical/code failure is not an accepted dataset row."
        ),
    )

    if (
        not completed_evaluations.empty
        and "Cargo_Optimization_Success" in completed_evaluations
    ):
        unresolved_cargo_searches = int(
            (~completed_evaluations["Cargo_Optimization_Success"].astype(bool))
            .sum()
        )
    else:
        unresolved_cargo_searches = 0

    _record_check(
        checks,
        "completed_cargo_searches_are_determinate",
        unresolved_cargo_searches == 0,
        unresolved_cargo_searches,
        0,
        (
            "The deterministic capacity-extreme cargo search must return a "
            "finite best allocation for every completed evaluation."
        ),
    )

    _record_check(
        checks,
        "ml_design_matrix_retains_every_system",
        (
            len(ml_design_matrix) == len(systems)
            and not ml_design_matrix.get(
                "Index",
                pd.Series(dtype=int),
            ).duplicated().any()
        ),
        len(ml_design_matrix),
        len(systems),
        "ML_Design_Matrix.csv is population-driven and may not drop failures.",
    )

    completed_ids = set(completed_evaluations.get(
        "Index",
        pd.Series(dtype=int),
    ).astype(int))
    detail_requirements = {
        "structure_properties": "structural-property rows",
        "weight_components": "component-weight rows",
        "compartment_loads": "cargo/tank state rows",
        "compartment_condition_states": (
            "condition-specific cargo/tank load-state rows"
        ),
        "bulkhead_locations": "bulkhead-location rows",
        "hatch_covers": "hatch-cover mass and geometry rows",
        "loading_conditions": "named loading-condition rows",
        "loading_condition_components": (
            "loading-condition component-state rows"
        ),
        "upright_stability": (
            "condition-specific equilibrium and upright-stability rows"
        ),
        "named_condition_loads": (
            "condition-specific still-water load and closure rows"
        ),
        "load_closure": "load-closure diagnostic rows",
        "economic_performance": (
            "economic-scenario provenance and annual-cost-breakdown rows"
        ),
    }

    for family, description in detail_requirements.items():
        detail_table = tables[family]
        detail_ids = set(
            detail_table.get("Index", pd.Series(dtype=int)).astype(int)
        )
        _record_check(
            checks,
            f"{family}_exist_for_completed_evaluations",
            not completed_ids or completed_ids <= detail_ids,
            len(detail_ids),
            len(completed_ids),
            f"Every completed system requires {description}.",
        )

    expected_loading_condition_ids = {
        "full_cargo_departure",
        "full_cargo_arrival",
    }
    named_condition_loads = tables["named_condition_loads"]

    if completed_ids and not named_condition_loads.empty:
        named_loads_for_completed = named_condition_loads.loc[
            named_condition_loads["Index"].astype(int).isin(completed_ids)
        ]
        condition_sets = named_loads_for_completed.groupby("Index")[
            "Loading_Condition_ID"
        ].agg(lambda values: set(values.astype(str)))
        condition_counts = named_loads_for_completed.groupby("Index").size()
        complete_condition_sets = bool(
            len(condition_sets) == len(completed_ids)
            and (condition_counts == 2).all()
            and condition_sets.apply(
                lambda observed: observed == expected_loading_condition_ids
            ).all()
        )
        unsolved_named_conditions = int(
            (~named_loads_for_completed["Equilibrium_Solved"].astype(bool))
            .sum()
        )
        trim_result_count = int(
            named_loads_for_completed["Trim_Constraint_Passed"]
            .notna()
            .sum()
        )
    else:
        complete_condition_sets = not completed_ids
        unsolved_named_conditions = len(completed_ids)
        trim_result_count = 0

    _record_check(
        checks,
        "two_named_full_load_conditions_per_completed_system",
        complete_condition_sets,
        (
            len(named_condition_loads)
            if not named_condition_loads.empty
            else 0
        ),
        2*len(completed_ids),
        (
            "Every completed system requires full-load departure and "
            "full-load arrival."
        ),
    )
    _record_check(
        checks,
        "named_condition_longitudinal_equilibrium_solved",
        unsolved_named_conditions == 0,
        unsolved_named_conditions,
        0,
        (
            "Every named condition must solve simultaneous displacement and "
            "longitudinal first-moment equilibrium."
        ),
    )
    _record_check(
        checks,
        "named_condition_trim_results_are_complete",
        trim_result_count == 2*len(completed_ids),
        trim_result_count,
        2*len(completed_ids),
        (
            "HDF-ASM-033 is evaluated for both named full-load conditions. A "
            "False physical trim result is valid labeled design data and "
            "does not make dataset compilation itself fail."
        ),
    )

    # HDF-ASM-035: unlike a failed physical trim predicate, a failed closure
    # gate means the numerical evaluation is invalid.  Make these fractions
    # compilation acceptance checks as well as per-design legacy-format rows
    # so a broken integration cannot be hidden inside a large compiled table.
    mass_closure_column = "Mass_Residual_Fraction"
    load_closure_columns = [
        "Net_Vertical_Force_Residual_Fraction",
        "Net_Vertical_First_Moment_Residual_Fraction",
        "Aft_Shear_Force_Residual_Fraction",
        "Aft_Bending_Moment_Residual_Fraction",
    ]
    closure_columns_present = (
        mass_closure_column in named_condition_loads.columns
        and set(load_closure_columns) <= set(named_condition_loads.columns)
    )

    if closure_columns_present and not named_condition_loads.empty:
        completed_closure = named_condition_loads.loc[
            named_condition_loads["Index"].astype(int).isin(completed_ids)
        ]
        mass_residuals = np.abs(
            completed_closure[mass_closure_column].to_numpy(dtype=float)
        )
        load_residuals = np.abs(
            completed_closure[load_closure_columns].to_numpy(dtype=float)
        )
        # Count each invalid residual once.  The combined masks also retain a
        # useful failure count if a future corrupt artifact contains infinity:
        # infinity is both non-finite and over tolerance, but it is one failed
        # closure value rather than two independent failures.
        mass_closure_failure_count = int(
            (
                (~np.isfinite(mass_residuals))
                | (mass_residuals > MASS_CLOSURE_RELATIVE_TOLERANCE)
            ).sum()
        )
        load_closure_failure_count = int(
            (
                (~np.isfinite(load_residuals))
                | (load_residuals > LOAD_CLOSURE_RELATIVE_TOLERANCE)
            ).sum()
        )
    else:
        mass_closure_failure_count = 2*len(completed_ids)
        load_closure_failure_count = 4*2*len(completed_ids)

    _record_check(
        checks,
        "named_condition_relative_mass_closure",
        (
            not completed_ids
            or (
                closure_columns_present
                and mass_closure_failure_count == 0
            )
        ),
        mass_closure_failure_count,
        0,
        "HDF-ASM-035 requires relative mass residual <= 1e-8.",
    )
    _record_check(
        checks,
        "named_condition_relative_load_closure",
        (
            not completed_ids
            or (
                closure_columns_present
                and load_closure_failure_count == 0
            )
        ),
        load_closure_failure_count,
        0,
        (
            "HDF-ASM-035 requires relative force, first-moment, aft-shear, "
            "and aft-bending residuals <= 1e-6."
        ),
    )

    individual_constraint_files = sorted(
        (
            output_root
            / "evaluations"
            / "constraints_v1p1"
            / "individual"
        ).glob("Structure_*/Ship_*_Constraints_Evaluations.csv")
    )
    _record_check(
        checks,
        "one_individual_constraint_file_per_system",
        len(individual_constraint_files) == len(systems),
        len(individual_constraint_files),
        len(systems),
        (
            "Completed designs contain expanded rows; failed designs contain "
            "one legacy-format failure row with the design index and cause."
        ),
    )

    if not constraints.empty:
        constraint_keys = (
            constraints["Audit ID"].astype(str)
            + "::"
            + constraints["Name"].astype(str)
        )
        duplicate_constraint_rows = int(
            pd.DataFrame(
                {
                    "Index": constraints["Index"],
                    "Constraint_Key": constraint_keys,
                }
            ).duplicated().sum()
        )
    else:
        duplicate_constraint_rows = 0

    _record_check(
        checks,
        "unique_constraint_key_per_system",
        duplicate_constraint_rows == 0,
        duplicate_constraint_rows,
        0,
        "Constraint key is Audit ID plus runtime constraint name.",
    )

    if not expanded_constraints.empty:
        expanded_keys = (
            expanded_constraints["Requirement"].astype(str)
            + "::"
            + expanded_constraints["Name"].astype(str)
        )
        duplicate_v1p1_rows = int(
            pd.DataFrame(
                {
                    "Index": expanded_constraints["Index"],
                    "Constraint_Key": expanded_keys,
                }
            ).duplicated().sum()
        )
    else:
        duplicate_v1p1_rows = 0

    _record_check(
        checks,
        "unique_v1p1_constraint_key_per_system",
        duplicate_v1p1_rows == 0,
        duplicate_v1p1_rows,
        0,
        "V1.1 constraint key is Requirement plus runtime Name.",
    )

    checks_table = pd.DataFrame(checks)
    checks_path = compiled_directory / "Compilation_Checks.csv"
    _write_csv_atomic(checks_table, checks_path)
    outputs["compilation_checks"] = checks_path

    authority_files = [
        PROJECT_ROOT / relative_path
        for relative_path in AUTHORITY_RELATIVE_PATHS
    ]
    array_files = sorted(
        (output_root / "evaluations" / "arrays").glob(
            "Numerical_Arrays_*.npz"
        )
    )
    structural_element_files = sorted(
        (output_root / "structures" / "elements").glob(
            "Structure_*_Structural_Elements.csv"
        )
    )
    invocation_files = {
        "generation": sorted(
            (
                output_root
                / "manifests"
                / "invocations"
                / "generation"
            ).glob("Generation_*.json")
        ),
        "evaluation": sorted(
            (
                output_root
                / "manifests"
                / "invocations"
                / "evaluation"
            ).glob("Evaluation_*.json")
        ),
    }

    # Retain the established simple path lists for downstream consumers.  The
    # parallel provenance mappings add content identity without changing that
    # backwards-compatible manifest contract.
    source_shard_paths = {
        family: [_manifest_path(path) for path in paths]
        for family, paths in shard_families.items()
    }
    source_shard_provenance = {
        family: [_file_provenance(path) for path in paths]
        for family, paths in shard_families.items()
    }
    authority_file_provenance = {
        _manifest_path(path): _file_provenance(path)
        for path in authority_files
    }
    invocation_expected_counts = {
        "generation": len(shard_families["system_designs"]),
        "evaluation": len(shard_families["scalar_results"]),
    }
    invocation_provenance_summary = {}

    for stage, paths in invocation_files.items():
        expected_count = invocation_expected_counts[stage]
        observed_count = len(paths)

        if observed_count == expected_count:
            status = "complete"
        elif observed_count == 0:
            status = "legacy_sidecars_unavailable"
        else:
            status = "incomplete_or_stale_sidecar_set"

        invocation_provenance_summary[stage] = {
            "expected_count": expected_count,
            "observed_count": observed_count,
            "status": status,
        }

    manifest = {
        "dataset_version": "GenISys V1.1",
        "save_mode": save_mode,
        "intended_population": {
            "start_structure_index": start_structure,
            "end_structure_index_exclusive": end_structure,
            "structures": number_of_structures,
            "hulls_per_structure": HULLS_PER_STRUCTURE,
            "operating_conditions_per_hull": (
                OPERATING_CONDITIONS_PER_HULL
            ),
            "systems": intended_system_count,
            "first_system_index": first_system_index,
            "last_system_index": (
                first_system_index + intended_system_count - 1
            ),
        },
        "observed_rows": {
            family: int(len(table))
            for family, table in tables.items()
        },
        "all_acceptance_checks_passed": bool(checks_table["Passed"].all()),
        "source_shards": source_shard_paths,
        "source_shard_provenance": source_shard_provenance,
        "authority_file_sha256": {
            path: provenance["sha256"]
            for path, provenance in authority_file_provenance.items()
        },
        "authority_file_provenance": authority_file_provenance,
        "structural_element_provenance": [
            _file_provenance(path)
            for path in structural_element_files
        ],
        "array_shards": [_manifest_path(path) for path in array_files],
        "array_shard_provenance": [
            _file_provenance(path)
            for path in array_files
        ],
        "invocation_provenance": {
            stage: [_json_file_provenance(path) for path in paths]
            for stage, paths in invocation_files.items()
        },
        "invocation_provenance_summary": invocation_provenance_summary,
        "notes": [
            "Compilation does not delete, move, or modify source shards.",
            "Numerical NPZ shards remain sharded to avoid one oversized file.",
            "Source CSV shards, structural-element inputs, numerical NPZ "
            "shards, and available invocation sidecars include SHA-256 and "
            "byte-size provenance.",
            "Invocation provenance is empty for legacy runs created before "
            "the generation/evaluation sidecars were introduced.",
            "An audit row is not a runtime constraint unless it is emitted in "
            "Audited_Constraint_Evaluations.csv.",
            "Every expected system has one exact eight-column individual "
            "constraint file, including systems that fail evaluation.",
            (
                "Reduced mode keeps canonical long-form source shards gzip-"
                "compressed and does not duplicate their complete row sets "
                "in the compiled directory."
                if save_mode == "reduced"
                else "All mode materializes complete compiled long-form CSVs."
            ),
        ],
    }
    manifest_path = manifest_directory / "Dataset_Manifest.json"
    _write_json_atomic(manifest, manifest_path)
    outputs["manifest"] = manifest_path

    return outputs


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile GenISys V1.1 shards and write acceptance checks."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--number-of-structures",
        type=int,
        default=NUMBER_OF_STRUCTURES,
        help=(
            "Expected accepted structure count. Production default is 5000; "
            "use 10 for the documented laptop pipeline test."
        ),
    )
    parser.add_argument(
        "--start-structure",
        type=int,
        default=0,
        help=(
            "First expected Structure_Index. Use 10 for a second 200-design "
            "laptop batch numbered 200 through 399."
        ),
    )
    parser.add_argument(
        "--save-mode",
        choices=("all", "reduced"),
        default="all",
        help=(
            "Materialize all long-form compiled CSVs ('all') or preserve "
            "compressed canonical shards and write only core/derived "
            "compiled products ('reduced')."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    outputs = compile_dataset(
        output_root=args.output_root,
        number_of_structures=args.number_of_structures,
        save_mode=args.save_mode,
        start_structure=args.start_structure,
    )

    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
