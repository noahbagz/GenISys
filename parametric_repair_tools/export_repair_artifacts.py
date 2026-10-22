"""Export reproducible repair registries and experiment input snapshots.

This utility is intentionally read-only with respect to the GenISys source
dataset and the frozen development/validation selections.  Its only write
target is ``parametric_repair_tools/artifacts`` (or an explicitly supplied
output directory).

Run from the repository root with the required project environment::

    /opt/anaconda3/envs/GenISys_Env/bin/python \
        -m parametric_repair_tools.export_repair_artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .configuration import PROJECT_ROOT, RepairConfiguration
from .dataset import (
    SourceDataset,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)
from .parameters import (
    CARGO_FRACTION_NAMES,
    FIXED_VALUES,
    INACTIVE_NAMES,
    ParameterRegistry,
)
from .registry import ConstraintRepairRegistry
from .evaluation import failed_constraint_rows


DEFAULT_ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "parametric_repair_tools" / "artifacts"
)


# The combined 0--399 pilot catalog is immutable experiment input.  It cannot
# contain evaluator identities introduced after that dataset was produced, so
# the exporter appends current declarations with zero observations instead of
# rewriting the historical catalog or pretending the new row was measured.
CURRENT_DECLARED_CONSTRAINTS = (
    {
        "Class": "Powering Method Domain",
        "Name": (
            "Holtrop run-length to waterline-length equation domain"
        ),
        "Requirement": (
            "Holtrop and Mennen (1982), implemented form-factor equation; "
            "HDF-ASM-036 numerical clearance"
        ),
        "Units": "dimensionless",
        "Output_Type": "constraint",
        "Conditional_Runtime_Row": True,
    },
)


# These dispositions record the user's decisions made on 2026-09-14.  The
# evidence tables remain historical: resolving an evaluator issue does not
# mutate the rows that originally revealed it.
CODE_REVIEW_DISPOSITIONS = {
    "CRF-001": {
        "status": "resolved_implemented",
        "disposition": "confirmed_code_error",
        "detail": (
            "Replaced local-versus-whole-ship extent comparisons with "
            "continuous-double-bottom project-basis evidence."
        ),
    },
    "CRF-002": {
        "status": "resolved_implemented",
        "disposition": "superseded_by_mandatory_member_contract",
        "detail": (
            "Every new or repaired V1.1 design requires an inner-side shell "
            "with a 10 mm minimum modeled thickness. Historical absent-member "
            "rows retain the evaluator's zero/blank N/A representation."
        ),
    },
    "CRF-003": {
        "status": "resolved_implemented",
        "disposition": "confirmed_code_error",
        "detail": (
            "Duplicate rows remain visible for traceability, but exact "
            "formula aliases share one canonical feasibility count."
        ),
    },
    "CRF-004": {
        "status": "resolved_implemented",
        "disposition": "confirmed_applicability_error",
        "detail": (
            "The shear-strake web-aspect row is retained as not applicable "
            "and removed from active feasibility."
        ),
    },
    "CRF-005": {
        "status": "resolved_implemented",
        "disposition": "confirmed_applicability_error",
        "detail": (
            "HDF-083 rows are zero-valued, blank-bit diagnostics and do not "
            "enter feasibility counts."
        ),
    },
    "CRF-006": {
        "status": "resolved_with_deferred_ballast_integration",
        "disposition": "confirmed_applicability_error",
        "detail": (
            "Cargo is optimized for departure trim only and the separate "
            "arrival trim row is removed. Physical compensating-ballast "
            "allocation awaits HDF-OPEN-020; arrival load bending moment is "
            "deferred from the current repair-pipeline scope."
        ),
    },
    "CRF-007": {
        "status": "resolved_implemented",
        "disposition": "confirmed_code_error",
        "detail": (
            "Generated structural-element geometry is the spacing authority; "
            "repairs may change counts and must regenerate that geometry."
        ),
    },
    "CRF-008": {
        "status": "resolved_implemented",
        "disposition": "confirmed_code_error",
        "detail": (
            "Web-frame section modulus is evaluated from the actual generated "
            "web-frame section."
        ),
    },
    "CRF-009": {
        "status": "resolved_implemented",
        "disposition": "confirmed_applicability_error",
        "detail": (
            "Relative section-modulus checks are limited to the two member "
            "types supported by the cited ABS text. Each valid legacy/atomic "
            "alias pair is retained but canonical-counted once; 17 "
            "unsupported member rows are excluded."
        ),
    },
    "CRF-010": {
        "status": "resolved_with_separate_open_method_domain",
        "disposition": "confirmed_code_error",
        "detail": (
            "The generated bulb height is Hbb*WL. The separate strict "
            "T-1.5*hb bulb-pressure denominator remains a hard method-domain "
            "rejection pending an approved clearance/repair contract."
        ),
    },
    "CRF-011": {
        "status": "resolved_implemented",
        "disposition": "confirmed_code_error",
        "detail": (
            "Lr/LWL >= 0.01 is an explicit structured method-domain row with "
            "declared Ls, Lb, and Draft_frac repair controls."
        ),
    },
}


def _relative_path(path: str | Path) -> str:
    """Use repository-relative paths when possible for portable manifests."""

    resolved = Path(path).resolve()

    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _effective_ownership(name: str, raw_role: str) -> str:
    """Resolve canonicalization ownership beyond the raw catalog datatype."""

    if name in {"Bc", "bit_EP_T"}:
        return "coupled_alias"

    if name == "Db":
        return "independent_control_with_geometry_guard"

    if raw_role == "derived":
        return "derived_by_owning_module"

    if raw_role == "solver_state":
        return "owned_by_evaluator_solver"

    if raw_role == "fixed":
        return "fixed_by_project_scope"

    if raw_role == "inactive":
        return "inactive_out_of_scope"

    return "independent_repair_control"


def _canonical_relationship(name: str) -> str:
    """Return the exact relationship that owns a non-independent coordinate."""

    relationships = {
        "B": "B = LOA * Bd",
        "D": "D = LOA * Dd",
        "T": "T = Draft_frac * D = Draft_frac * Dd * LOA",
        "R_b": "R_b = 0.5 * B * Rc",
        "L_3h": (
            "L_3h = 3 * LOA * (1 - sum(end-region fractions)) / "
            "(Num_Cargo_Holds * (1 - 2*l_overhang))"
        ),
        "Bc": "Bc = Bd",
        "bit_EP_T": "bit_EP_T = bit_EP_S",
        "num top girders": "num top girders = num bottom girders",
        "num intermediate deck stiffeners": (
            "num intermediate deck stiffeners = "
            "num intermediate bottom stiffeners"
        ),
        "Db": "Db >= 1000 * R_b (conditional geometric guard)",
    }

    if name in CARGO_FRACTION_NAMES:
        return "Reset to 0 before exact cargo-allocation solver evaluation"

    if name in FIXED_VALUES:
        return f"Fixed to {FIXED_VALUES[name]:.12g} by approved project scope"

    if name in INACTIVE_NAMES:
        return "Fixed to 0 because the parameter block is out of scope"

    return relationships.get(name, "")


def _parameter_registry(parameters: ParameterRegistry) -> pd.DataFrame:
    """Build a complete 189-coordinate ownership and bounds snapshot."""

    table = parameters.role_table().copy()
    table.insert(0, "Coordinate_Position", np.arange(len(table), dtype=int))
    table["Effective_Repair_Ownership"] = [
        _effective_ownership(str(name), str(role))
        for name, role in zip(
            table["Name"],
            table["Repair_Role"],
            strict=True,
        )
    ]
    table["Direct_Repair_Eligible"] = table[
        "Effective_Repair_Ownership"
    ].isin(
        {
            "independent_repair_control",
            "independent_control_with_geometry_guard",
        }
    )
    table["Canonical_Relationship"] = table["Name"].map(
        _canonical_relationship
    )
    table["Sampling_Bounds_May_Be_Exceeded"] = ~(
        np.isclose(table["LL"], table["Repair_Lower_Limit"])
        & np.isclose(table["UL"], table["Repair_Upper_Limit"])
    )
    table["Fixed_Value"] = table["Name"].map(FIXED_VALUES)
    return table.sort_values("Coordinate_Position").reset_index(drop=True)


def _parameter_couplings() -> pd.DataFrame:
    """List every canonical dependency in explicit machine-readable form."""

    rows: list[dict[str, object]] = []

    def add(
        destination: str,
        source_parameters: tuple[str, ...],
        owner: str,
        relationship: str,
        enforcement: str,
        note: str = "",
    ) -> None:
        rows.append(
            {
                "Application_Order": len(rows),
                "Destination_Parameter": destination,
                "Source_Parameters": "; ".join(source_parameters),
                "Owning_Component": owner,
                "Canonical_Relationship": relationship,
                "Enforcement_Mode": enforcement,
                "Note": note,
            }
        )

    # Scope is restored before geometric relationships are evaluated.
    for name, value in FIXED_VALUES.items():
        add(
            name,
            (),
            "approved GenISys project scope",
            f"{name} = {value:.12g}",
            "unconditional overwrite",
        )

    for name in sorted(INACTIVE_NAMES):
        add(
            name,
            (),
            "approved GenISys project scope",
            f"{name} = 0",
            "unconditional overwrite",
            "The associated parameter block is out of scope.",
        )

    add(
        "Bc",
        ("Bd",),
        "current hull/structure coupling",
        "Bc = Bd",
        "unconditional overwrite",
    )
    add(
        "bit_EP_T",
        ("bit_EP_S",),
        "current endpoint-topology convention",
        "bit_EP_T = bit_EP_S",
        "unconditional overwrite",
    )
    add(
        "num top girders",
        ("num bottom girders",),
        "CadQuery structure generator",
        "num top girders = num bottom girders",
        "unconditional overwrite",
    )
    add(
        "num intermediate deck stiffeners",
        ("num intermediate bottom stiffeners",),
        "CadQuery structure generator",
        (
            "num intermediate deck stiffeners = "
            "num intermediate bottom stiffeners"
        ),
        "unconditional overwrite",
    )

    for name in CARGO_FRACTION_NAMES:
        add(
            name,
            (),
            "exact cargo-allocation solver",
            f"{name} = 0 before evaluation; solver writes evaluated value",
            "pre-evaluation reset",
        )

    add(
        "B",
        ("LOA", "Bd"),
        "hull design module",
        "B = LOA * Bd",
        "unconditional derivation",
    )
    add(
        "D",
        ("LOA", "Dd"),
        "hull design module",
        "D = LOA * Dd",
        "unconditional derivation",
    )
    add(
        "R_b",
        ("B", "Rc"),
        "hull-to-structure coupling",
        "R_b = 0.5 * B * Rc",
        "unconditional derivation",
    )
    add(
        "L_3h",
        (
            "LOA",
            "frac_CrashBlkhd",
            "frac_AftPeakBlkhd",
            "frac_lazarette_L",
            "frac_engineRoom_L",
            "frac_fuel_tank_L",
            "Num_Cargo_Holds",
            "l_overhang",
        ),
        "system arrangement and structure coupling",
        (
            "L_3h = 3 * LOA * (1 - sum(end-region fractions)) / "
            "(Num_Cargo_Holds * (1 - 2*l_overhang))"
        ),
        "unconditional derivation",
    )
    add(
        "T",
        ("LOA", "Dd", "Draft_frac"),
        "loading-condition / hydrostatics coupling",
        "T = Draft_frac * D = Draft_frac * Dd * LOA",
        "unconditional derivation",
    )
    add(
        "Db",
        ("R_b",),
        "structure geometric compatibility guard",
        "Db = max(Db, 1000 * R_b)",
        "conditional lower-bound overwrite",
    )

    return pd.DataFrame(rows)


def _catalog_with_current_declarations(catalog: pd.DataFrame) -> pd.DataFrame:
    """Combine immutable observations with post-pilot evaluator identities.

    The two observation-count columns always describe the old 0--399 pilot.
    A zero in an appended row therefore means "not present in that immutable
    dataset," not "the current evaluator can never emit this constraint."
    """

    combined = catalog.copy()
    combined["Catalog_Provenance"] = "immutable_0_399_pilot_observation"
    combined["Declared_By_Current_Evaluator"] = False

    for declaration in CURRENT_DECLARED_CONSTRAINTS:
        constraint_key = (
            f"{declaration['Requirement']}::{declaration['Name']}"
        )
        existing = combined["Constraint_Key"].astype(str).eq(constraint_key)

        if existing.any():
            # A future catalog may already contain the identity.  Preserve its
            # measured counts while recording that it is also part of the
            # current evaluator contract.
            combined.loc[
                existing,
                "Declared_By_Current_Evaluator",
            ] = True
            continue

        appended = {
            "Constraint_Key": constraint_key,
            **declaration,
            "Observed_Row_Count": 0,
            "Observed_System_Count": 0,
            "Catalog_Provenance": (
                "current_evaluator_declaration_not_observed_in_"
                "immutable_0_399_pilot"
            ),
            "Declared_By_Current_Evaluator": True,
        }
        combined = pd.concat(
            [combined, pd.DataFrame([appended], columns=combined.columns)],
            ignore_index=True,
        )

    return combined


def _frozen_batches(
    experiment_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the immutable development and validation selection artifacts."""

    selections = []
    fingerprints = []

    for directory in (
        experiment_root / "development_batch",
        experiment_root / "validation_batch",
    ):
        selection_path = directory / "selected_source_designs.csv"
        manifest_path = directory / "selection_manifest.json"

        if not selection_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Frozen batch artifacts are incomplete under {directory}."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        table = pd.read_csv(selection_path)
        table["Batch_Directory"] = _relative_path(directory)
        table["Selection_File_SHA256"] = sha256_file(selection_path)
        table["Selection_Manifest_SHA256"] = sha256_file(manifest_path)
        selections.append(table)

        for label, artifact in sorted(
            manifest["source_fingerprints"].items()
        ):
            source_path = Path(str(artifact["path"]))
            fingerprints.append(
                {
                    "Batch_Kind": manifest["batch_kind"],
                    "Source_Artifact": label,
                    "Source_Path": _relative_path(source_path),
                    "Recorded_Size_Bytes": int(artifact["size_bytes"]),
                    "Recorded_SHA256": str(artifact["sha256"]),
                    "Current_File_Exists": source_path.exists(),
                    "Current_SHA256_Matches": (
                        source_path.exists()
                        and sha256_file(source_path) == str(artifact["sha256"])
                    ),
                }
            )

    selection_table = pd.concat(selections, ignore_index=True).sort_values(
        ["Batch_Kind", "Selection_Order"]
    )
    fingerprint_table = pd.DataFrame(fingerprints).sort_values(
        ["Batch_Kind", "Source_Artifact"]
    )
    return (
        selection_table.reset_index(drop=True),
        fingerprint_table.reset_index(drop=True),
    )


def _constraint_code_review_evidence(
    dataset: SourceDataset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bind resolved review findings to their immutable pilot evidence."""

    constraints = dataset.all_constraints().copy()
    name = constraints["Name"].fillna("").astype(str)
    result_class = constraints["Class"].fillna("").astype(str)
    duplicate_names = {
        "Min Center Bottom Girder Thickness",
        "MVR center-girder plate thickness amidships",
        "Min Floor Thickness",
        "Min Bottom Floor Thickness",
        "MVR solid-floor plate thickness",
        "Max Frame Spacing",
        "MVR maximum transverse-frame spacing",
        "Min Inner Bottom Grab-Service Thickness",
        "Inner bottom plating - grab loading",
        "Min Lower Hopper Grab-Service Thickness",
        "Lower hopper plating - grab loading",
        "Max Allowable Uniform Deck Loading",
        "Max Allowable Uniform Bottom Loading",
        "Min Relative Deck Stringer Section Modulus",
        "Strength-deck longitudinal section modulus",
        "Min Relative Deck Frame Section Modulus",
        "Ordinary strength-deck transverse-beam section modulus",
    }
    predicates = {
        "CRF-001": name.isin(
            {
                "SOLAS double-bottom forward extent",
                "SOLAS double-bottom aft extent",
            }
        ),
        "CRF-002": name.eq("Min Inner Side Shell Thickness"),
        "CRF-003": name.isin(duplicate_names),
        "CRF-004": name.eq("Shear_Strake_Aspect_Ratio"),
        "CRF-005": result_class.eq("Flooding Diagnostic"),
        # CRF-006 concerned the separate arrival predicate.  Departure trim
        # remains an approved GenISys design constraint and must not be folded
        # into the superseded-arrival evidence set.
        "CRF-006": (
            result_class.eq("Loading Condition Trim")
            & name.eq("Full-cargo arrival one-percent-draft trim")
        ),
        "CRF-007": (
            name.str.contains("Deck Girder|Deck_Girder", case=False)
            | name.str.contains("Bulkhead Stiffener", case=False)
        ),
        "CRF-008": name.str.contains("Web.?Frame", case=False, regex=True),
        "CRF-009": name.str.startswith("Min Relative "),
    }
    evidence_tables = []

    for finding_id, predicate in predicates.items():
        selected = constraints.loc[predicate].copy()
        selected.insert(0, "Finding_ID", finding_id)
        selected.insert(1, "Evidence_Source", "V1.1 constraint output")
        evidence_tables.append(selected)

    failures = dataset.failures.copy()
    holtrop = failures.loc[
        failures["Error_Message"]
        .fillna("")
        .str.contains("bulb-pressure denominator", case=False)
    ].copy()

    if not holtrop.empty:
        holtrop_evidence = pd.DataFrame(
            {
                "Finding_ID": "CRF-010",
                "Evidence_Source": "V1.1 evaluation failure",
                "Index": holtrop["Index"].astype(int),
                "Structure_Index": holtrop["Structure_Index"],
                "Hull_Variant_Index": holtrop["Hull_Variant_Index"],
                "Operating_Condition_Index": holtrop[
                    "Operating_Condition_Index"
                ],
                "Class": "Evaluation Failure",
                "Name": "Holtrop bulb-pressure denominator failure",
                "Requirement": "Current implemented Holtrop mathematical domain",
                "Threshold Value": np.nan,
                "Constraint Value": holtrop["Error_Message"],
                "Units": "",
                "Constraint Eval": np.nan,
                "Constraint Bit": False,
            }
        )
        evidence_tables.append(holtrop_evidence)

    # Before CRF-011 was corrected, a negative Holtrop run length was raised
    # before the ordinary method-domain rows were written.  Preserve that old
    # exception as historical evidence even though the current evaluator now
    # emits the normalized Lr/LWL predicate explicitly.
    negative_run_length = failures.loc[
        failures["Error_Message"]
        .fillna("")
        .str.contains(
            "Holtrop run length must be finite and positive",
            case=False,
            regex=False,
        )
    ].copy()

    if not negative_run_length.empty:
        error_detail = negative_run_length["Error_Message"].astype(str)
        reported_run_length = pd.to_numeric(
            error_detail
            .str.strip()
            .str.rstrip(".")
            .str.extract(r"received\s+(\S+)\s*$", expand=False),
            errors="raise",
        )

        if not reported_run_length.lt(0.0).all():
            raise ValueError(
                "CRF-011 evidence includes a non-negative reported run "
                "length; review the failure classifier before exporting."
            )

        run_length_evidence = pd.DataFrame(
            {
                "Finding_ID": "CRF-011",
                "Evidence_Source": "V1.1 evaluation failure",
                "Index": negative_run_length["Index"].astype(int),
                "Structure_Index": negative_run_length["Structure_Index"],
                "Hull_Variant_Index": negative_run_length[
                    "Hull_Variant_Index"
                ],
                "Operating_Condition_Index": negative_run_length[
                    "Operating_Condition_Index"
                ],
                "Class": "Evaluation Failure",
                "Name": "Negative Holtrop run length",
                "Requirement": (
                    "Superseded pre-CRF-011 Holtrop mathematical-domain "
                    "exception; current contract requires Lr/LWL >= 0.01"
                ),
                "Threshold Value": 0.0,
                "Constraint Value": reported_run_length,
                "Units": "m",
                "Constraint Eval": np.nan,
                "Constraint Bit": False,
                "Evidence_Detail": error_detail,
            }
        )
        evidence_tables.append(run_length_evidence)

    evidence = pd.concat(evidence_tables, ignore_index=True, sort=False)
    evidence = evidence.sort_values(
        ["Finding_ID", "Index", "Name"],
        kind="stable",
    ).reset_index(drop=True)
    evidence["Evidence_Status"] = (
        "historical_pre_disposition_0_399_pilot_evidence"
    )
    summary_rows = []

    for finding_id, decision in CODE_REVIEW_DISPOSITIONS.items():
        rows = evidence.loc[evidence["Finding_ID"].eq(finding_id)]
        indices = sorted(rows["Index"].dropna().astype(int).unique())
        false_bits = rows["Constraint Bit"].astype(str).str.lower().eq("false")
        summary_rows.append(
            {
                "Finding_ID": finding_id,
                "Finding_Status": decision["status"],
                "Approved_Disposition": decision["disposition"],
                "Disposition_Date": "2026-09-14",
                "Current_Implementation_Detail": decision["detail"],
                "Evidence_Row_Count": int(len(rows)),
                "Affected_Design_Count": int(len(indices)),
                "Observed_Failed_Row_Count": int(false_bits.sum()),
                "Affected_Design_Indices": ";".join(map(str, indices)),
            }
        )

    return evidence, pd.DataFrame(summary_rows)


def _frozen_batch_unresolved_failures(
    dataset: SourceDataset,
    selections: pd.DataFrame,
    registry: ConstraintRepairRegistry,
) -> pd.DataFrame:
    """List immutable source failures not routed to parameter repair.

    Some rows are now resolved as superseded, diagnostic, or not applicable;
    their presence here records the old source state and does not reopen the
    associated code-review finding.
    """

    rows = []

    for _, selected in selections.iterrows():
        parent_index = int(selected["Index"])
        failed = failed_constraint_rows(dataset.constraints(parent_index))

        for _, failure in failed.iterrows():
            rule = registry.classify(failure)

            if rule.status in {"repairable", "repairable_approximate_coupled"}:
                continue

            rows.append(
                {
                    "Batch_Kind": selected["Batch_Kind"],
                    "Selection_Order": int(selected["Selection_Order"]),
                    "Parent_Index": parent_index,
                    "Requirement": failure.get("Requirement"),
                    "Class": failure.get("Class"),
                    "Name": failure.get("Name"),
                    "Threshold_Value": failure.get("Threshold Value"),
                    "Constraint_Value": failure.get("Constraint Value"),
                    "Constraint_Eval": failure.get("Constraint Eval"),
                    "Repair_Status": rule.status,
                    "Repair_Family": rule.family,
                    "Repair_Note": rule.note,
                    "Evidence_Status": (
                        "historical_immutable_source_row_currently_not_"
                        "parameter_repaired"
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["Batch_Kind", "Selection_Order", "Repair_Status", "Name"],
        kind="stable",
    ).reset_index(drop=True)


def export_artifacts(output_directory: str | Path) -> dict[str, Path]:
    """Export the complete reproducible repair-interface snapshot."""

    configuration = RepairConfiguration()
    parameters = ParameterRegistry(configuration)
    repair_registry = ConstraintRepairRegistry()
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    catalog_path = (
        Path(configuration.source_root)
        / "compiled"
        / "Constraint_Catalog.csv"
    )

    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)

    catalog = pd.read_csv(catalog_path)

    # Classify the immutable observed identities and any current evaluator
    # identities added since that dataset was frozen.  Appended declarations
    # retain zero historical observations and carry an explicit provenance
    # label in the exported registry.
    runtime_catalog = _catalog_with_current_declarations(catalog)
    runtime_registry = repair_registry.export_runtime_registry(runtime_catalog)
    runtime_registry = runtime_registry.sort_values(
        ["Output_Type", "Constraint_Key"]
    ).reset_index(drop=True)
    constraint_registry = runtime_registry.loc[
        runtime_registry["Output_Type"].eq("constraint")
    ].reset_index(drop=True)

    summary = (
        runtime_registry.groupby(
            ["Output_Type", "Repair_Status", "Repair_Method"],
            dropna=False,
        )
        .size()
        .rename("Registry_Row_Count")
        .reset_index()
        .sort_values(["Output_Type", "Repair_Status", "Repair_Method"])
        .reset_index(drop=True)
    )
    frozen_batches, source_fingerprints = _frozen_batches(
        Path(configuration.experiment_root)
    )
    source_dataset = SourceDataset(configuration.source_root, parameters)
    review_evidence, review_summary = _constraint_code_review_evidence(
        source_dataset
    )
    frozen_unresolved = _frozen_batch_unresolved_failures(
        source_dataset,
        frozen_batches,
        repair_registry,
    )

    tables = {
        "parameter_registry.csv": _parameter_registry(parameters),
        "parameter_couplings.csv": _parameter_couplings(),
        "constraint_repair_registry.csv": constraint_registry,
        "evaluated_output_repair_registry.csv": runtime_registry,
        "repair_registry_summary.csv": summary,
        "frozen_batch_registry.csv": frozen_batches,
        "source_input_fingerprints.csv": source_fingerprints,
        "constraint_code_review_evidence.csv": review_evidence,
        "constraint_code_review_summary.csv": review_summary,
        "frozen_batch_unresolved_failures.csv": frozen_unresolved,
    }
    paths: dict[str, Path] = {}

    for filename, table in tables.items():
        path = output_directory / filename
        write_csv_atomic(table, path)
        paths[filename] = path

    configuration_path = output_directory / "repair_configuration.json"
    write_json_atomic(
        {
            "schema_version": 1,
            "configuration_fingerprint": configuration.fingerprint(),
            "configuration": configuration.to_dict(),
        },
        configuration_path,
    )
    paths[configuration_path.name] = configuration_path

    manifest_rows = []

    for filename, path in sorted(paths.items()):
        row_count = None
        column_count = None

        if path.suffix == ".csv":
            table = tables[filename]
            row_count = int(len(table))
            column_count = int(len(table.columns))

        manifest_rows.append(
            {
                "path": filename,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_count": row_count,
                "column_count": column_count,
            }
        )

    manifest_path = output_directory / "artifact_manifest.json"
    write_json_atomic(
        {
            "schema_version": 1,
            "generator": (
                "parametric_repair_tools.export_repair_artifacts"
            ),
            "determinism_policy": (
                "The exporter draws no new randomness and stores no timestamp; "
                "it snapshots already-frozen seeds and selections, so stable "
                "source files and repair code produce byte-identical CSV/JSON "
                "artifacts."
            ),
            "constraint_catalog_source": _relative_path(catalog_path),
            "constraint_catalog_sha256": sha256_file(catalog_path),
            "runtime_registry_policy": (
                "Immutable pilot observations are preserved. Current "
                "evaluator identities absent from that pilot are appended "
                "with zero observed rows/systems and explicit provenance."
            ),
            "configuration_fingerprint": configuration.fingerprint(),
            "artifacts": manifest_rows,
        },
        manifest_path,
    )
    paths[manifest_path.name] = manifest_path
    return paths


def main() -> None:
    """Parse the optional output path and export all repair artifacts."""

    parser = argparse.ArgumentParser(
        description="Export reproducible GenISys repair registries."
    )
    parser.add_argument(
        "--output-directory",
        default=str(DEFAULT_ARTIFACT_DIRECTORY),
        help="Directory receiving CSV/JSON snapshots.",
    )
    arguments = parser.parse_args()
    paths = export_artifacts(arguments.output_directory)

    print(
        f"Exported {len(paths)} reproducible repair artifacts to "
        f"{Path(arguments.output_directory).resolve()}"
    )


if __name__ == "__main__":
    main()
