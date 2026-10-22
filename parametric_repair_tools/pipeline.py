"""One-pass stochastic equation-based repair orchestration.

Each selected source design already has a saved V1.1 constraint evaluation.
Those source failures are mapped to parameters once, one repaired 189-value
vector is produced, and that vector is regenerated and evaluated once. The
final result is saved as dataset evidence and is never fed back into repair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd

from .configuration import RepairConfiguration
from .dataset import (
    SourceDataset,
    freeze_below_median_alternatives,
    freeze_configured_batches,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)
from .evaluation import (
    canonical_evaluated_constraint_count,
    canonical_failed_constraint_rows,
    evaluate_repaired_vector,
    failed_constraint_rows,
    normalized_violation_score,
    vector_hash,
)
from .parameters import ParameterRegistry
from .registry import ConstraintRepairRegistry
from .repair_methods import EquationRepairMethods, MarginLedger


ACTION_OUTPUT_COLUMNS = [
    "Parent_Index",
    "Alternative_Number",
    "Repaired_Index",
    "Repaired_Structure_Index",
    "Batch_Kind",
    "Selection_Order",
    "Batch_Seed",
    "Design_Repair_Seed",
    "Repair_Action_Sequence",
    "Parameter",
    "Old_Value",
    "Proposed_Value",
    "Evaluated_Value",
    "Repair_Method",
    "Repair_Family",
    "Constraint_Keys",
    "Sampled_Margins",
    "Stochastic_Choices",
    "Equation",
    "Repair_Lower_Limit",
    "Repair_Upper_Limit",
    "At_Repair_Domain_Limit",
    "Outside_Original_Sampling_Bounds",
    "Source_Constraint_Evidence",
    "Final_Constraint_Evidence",
    "Source_Failure_Count",
    "Final_Failure_Count",
    "Original_Vector_SHA256",
    "Proposed_Vector_SHA256",
    "Final_Vector_SHA256",
    "Final_Status",
]


@dataclass(frozen=True)
class RepairDesignSummary:
    """Compact outcome for one source, one repair, and one evaluation."""

    schema_version: int
    repair_method: str
    batch_kind: str
    selection_order: int
    parent_index: int
    alternative_number: int
    batch_seed: int
    design_repair_seed: int
    repaired_index: int
    repaired_structure_index: int
    repair_configuration_sha256: str
    original_vector_sha256: str
    proposed_vector_sha256: str
    final_vector_sha256: str
    source_reported_failure_count: int
    source_canonical_failure_count: int
    source_evaluated_constraint_count: int
    source_violation_score: float
    proposed_parameter_change_count: int
    final_status: str
    final_failure_count: int
    final_evaluated_constraint_count: int
    final_violation_score: float
    fully_feasible: bool
    repair_passes: int
    repaired_design_evaluations: int
    result_directory: str


def _json_safe(value: Any) -> Any:
    """Convert NumPy, Path, and nonfinite values for stable JSON output."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float) and not np.isfinite(value):
        return None

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if value is None or (
        not isinstance(value, (str, bytes)) and bool(pd.isna(value))
    ):
        return None

    return value


def _read_csv_allow_empty(path: str | Path) -> pd.DataFrame:
    """Read an evaluator shard that may be headerless after a failure."""

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _committed_evaluation_status(
    committed_parameters: pd.DataFrame,
    failures: pd.DataFrame,
) -> str:
    """Classify the single repaired-design evaluation."""

    if len(committed_parameters) == 1 and failures.empty:
        return "completed"

    if not failures.empty and "Failure_Category" in failures:
        categories = set(
            failures["Failure_Category"].dropna().astype(str).str.strip()
        )

        if categories and categories.issubset(
            {"physical_infeasibility", "method_domain_rejection"}
        ):
            return "physical_rejection"

    return "evaluation_failure"


def _constraint_evidence(
    table: pd.DataFrame,
    keys: tuple[str, ...],
) -> str:
    """Serialize values for every constraint associated with an action."""

    if table.empty:
        return json.dumps({}, sort_keys=True)

    identities = (
        table["Requirement"].fillna("").astype(str)
        + "::"
        + table["Name"].fillna("").astype(str)
    )
    evidence = {}

    for key in keys:
        matches = table.loc[identities.eq(key)]

        if matches.empty:
            evidence[key] = None
            continue

        row = matches.iloc[0]
        evidence[key] = {
            "threshold": _json_safe(row.get("Threshold Value")),
            "value": _json_safe(row.get("Constraint Value")),
            "residual": _json_safe(row.get("Constraint Eval")),
            "passed": _json_safe(row.get("Constraint Bit")),
        }

    return json.dumps(evidence, sort_keys=True, separators=(",", ":"))


def _filter_evaluator_table(
    table: pd.DataFrame,
    repaired_index: int,
) -> pd.DataFrame:
    """Keep one repaired design from an ordinary evaluator shard."""

    if "Index" not in table:
        return table.iloc[0:0].copy()

    return table.loc[
        table["Index"].astype(int).eq(int(repaired_index))
    ].copy()


def _prepare_design_root(design_root: Path) -> None:
    """Create an empty result directory without overwriting prior evidence."""

    if design_root.exists() and any(design_root.iterdir()):
        raise FileExistsError(
            f"Repair results already exist under {design_root}. Archive or "
            "remove that generated parent directory before an intentional rerun."
        )

    design_root.mkdir(parents=True, exist_ok=True)


def _final_unresolved_table(
    constraints: pd.DataFrame,
    registry: ConstraintRepairRegistry,
) -> pd.DataFrame:
    """Classify failures remaining after the single repaired evaluation."""

    failed = failed_constraint_rows(constraints)
    metadata_columns = [
        "Constraint_Key",
        "Repair_Method",
        "Repair_Family",
        "Controlling_Parameters",
        "Repair_Priority",
        "Repair_Status",
        "Repair_Note",
    ]
    columns = list(dict.fromkeys([*constraints.columns, *metadata_columns]))

    if failed.empty:
        return pd.DataFrame(columns=columns)

    rows = []

    for _, row in failed.iterrows():
        rule = registry.classify(row)
        status = rule.status

        if status in {"repairable", "repairable_approximate_coupled"}:
            status = "violation_remaining_after_one_pass"

        rows.append(
            {
                **row.to_dict(),
                "Constraint_Key": (
                    f"{row.get('Requirement', '')}::{row.get('Name', '')}"
                ),
                "Repair_Method": rule.method,
                "Repair_Family": rule.family,
                "Controlling_Parameters": "; ".join(rule.parameters),
                "Repair_Priority": int(rule.priority),
                "Repair_Status": status,
                "Repair_Note": rule.note,
            }
        )

    return pd.DataFrame(rows).reindex(columns=columns)


def _runtime_provenance() -> dict[str, Any]:
    """Describe the environment and seeded NumPy randomness."""

    package_versions = {}

    for distribution in ("cadquery", "numpy", "pandas", "scipy"):
        try:
            package_versions[distribution] = importlib_metadata.version(
                distribution
            )
        except importlib_metadata.PackageNotFoundError:
            package_versions[distribution] = None

    generator = np.random.default_rng(0)
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "package_versions": package_versions,
        "random_number_generator": {
            "library": "numpy",
            "generator_api": "numpy.random.Generator",
            "bit_generator": type(generator.bit_generator).__name__,
            "margin_draw": "Generator.uniform(lower, upper)",
            "fuel_repair_draw": "none; fuel repair uses the sampled margin",
        },
    }


def _implementation_paths(project_root: Path) -> list[Path]:
    """Return Python sources that can affect repair or evaluation."""

    paths: set[Path] = set()

    for relative_root in (
        "parametric_repair_tools",
        "tools",
        "cad_query_tools",
    ):
        source_root = project_root / relative_root

        if source_root.exists():
            paths.update(source_root.rglob("*.py"))

    evaluator_runner = project_root / "Run_V1p1_System_Evaluation_Batch.py"

    if evaluator_runner.exists():
        paths.add(evaluator_runner)

    return sorted(
        (path for path in paths if path.is_file()),
        key=lambda path: path.relative_to(project_root).as_posix(),
    )


def _artifact_record(path: str | Path) -> dict[str, Any]:
    """Return a hash record for one saved artifact."""

    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _build_final_output_manifest(
    *,
    normal_output_paths: dict[str, Path],
    design_root: Path,
    committed_elements: Path | None,
    copied_elements: Path,
) -> dict[str, dict[str, Any]]:
    """Hash evaluator outputs and the compact one-pass design artifacts."""

    manifest = {
        label: _artifact_record(path)
        for label, path in normal_output_paths.items()
        if Path(path).exists()
    }

    for path in sorted(design_root.rglob("*")):
        if not path.is_file() or path.name == "final_output_manifest.json":
            continue

        relative = path.relative_to(design_root).as_posix()
        manifest[f"design_artifact::{relative}"] = _artifact_record(path)

    if (
        committed_elements is not None
        and committed_elements.exists()
        and copied_elements.exists()
    ):
        record = _artifact_record(copied_elements)
        record["source_path"] = str(committed_elements.resolve())
        record["source_sha256"] = sha256_file(committed_elements)
        manifest["final_structural_elements"] = record

    return manifest


def _build_batch_output_manifest(
    batch_root: Path,
) -> dict[str, dict[str, Any]]:
    """Hash batch inputs, summaries, and per-design manifests."""

    filenames = (
        "selected_source_designs.csv",
        "original_189_parameter_vectors.csv.gz",
        "original_constraint_results.csv.gz",
        "original_performance_metrics.csv.gz",
        "original_evaluation_failures.csv.gz",
        "selection_manifest.json",
        "repair_run_manifest.json",
        "parameter_repair_registry.csv",
        "constraint_repair_registry.csv",
        "batch_repair_summary.csv",
        "batch_outcome.json",
    )
    manifest = {}

    for filename in filenames:
        path = batch_root / filename

        if path.exists():
            manifest[f"batch_artifact::{filename}"] = _artifact_record(path)

    for path in sorted(
        (batch_root / "results").glob(
            "parent_*/**/final_output_manifest.json"
        )
    ):
        relative_parent = path.parent.relative_to(
            batch_root / "results"
        ).as_posix()
        manifest[f"design_manifest::{relative_parent}"] = _artifact_record(path)

    return manifest


def _require_validation_prerequisite(
    experiment_root: Path,
    target: int,
) -> None:
    """Keep validation closed until the declared development test succeeds."""

    outcome_path = experiment_root / "development_batch" / "batch_outcome.json"

    if not outcome_path.exists():
        raise RuntimeError(
            "Run the complete one-pass development batch before validation."
        )

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))

    if (
        int(outcome.get("design_count", 0)) != 10
        or int(outcome.get("fully_feasible_count", 0)) < int(target)
    ):
        raise RuntimeError(
            f"Validation requires at least {int(target)} fully feasible "
            "development designs out of 10."
        )


class RepairPipeline:
    """Create and evaluate one-pass repaired variants from frozen sources."""

    REPAIRED_INDEX_BASE = {
        "development": 9_100_000,
        "validation": 9_200_000,
        "below_median": 9_300_000,
        "candidate_indices": 9_400_000,
    }

    def __init__(
        self,
        configuration: RepairConfiguration | None = None,
    ):
        self.configuration = configuration or RepairConfiguration()
        self.parameters = ParameterRegistry(self.configuration)
        self.dataset = SourceDataset(
            self.configuration.source_root,
            self.parameters,
        )
        self.constraint_registry = ConstraintRepairRegistry()
        self.methods = EquationRepairMethods(
            self.parameters,
            self.constraint_registry,
            self.configuration,
        )
        self.experiment_root = Path(self.configuration.experiment_root)

    def repair_one(
        self,
        selection: pd.Series,
        batch_kind: str,
        *,
        batch_root_override: str | Path | None = None,
    ) -> RepairDesignSummary:
        """Repair the saved source failures once and evaluate the result once.

        ``batch_root_override`` lets a cluster worker create the ordinary
        evaluator artifacts in node-local scratch.  The repair configuration,
        random seed, equations, and configuration fingerprint remain
        unchanged; only the temporary storage location is redirected.
        """

        parent_index = int(selection["Index"])
        selection_order = int(selection["Selection_Order"])
        alternative_number = int(selection.get("Alternative_Number", 1))
        design_seed = int(selection["Design_Repair_Seed"])
        repaired_index = (
            self.REPAIRED_INDEX_BASE[batch_kind] + selection_order
        )
        repaired_structure_index = repaired_index
        if batch_root_override is None:
            batch_root = self.experiment_root / f"{batch_kind}_batch"
        else:
            batch_root = Path(batch_root_override)

        design_root = batch_root / "results" / f"parent_{parent_index:06d}"

        # Development and validation preserve their established directory
        # contract.  Population repair nests alternatives beneath the shared
        # parent so independent proposals cannot overwrite one another.
        if batch_kind in {"below_median", "candidate_indices"}:
            design_root = (
                design_root / f"alternative_{alternative_number:02d}"
            )

        _prepare_design_root(design_root)

        original_vector = self.dataset.vector(parent_index)
        source_constraints = self.dataset.constraints(parent_index)
        source_identifiers = self.dataset.identifiers(parent_index)
        source_failure_count = int(
            len(canonical_failed_constraint_rows(source_constraints))
        )
        source_evaluated_count = canonical_evaluated_constraint_count(
            source_constraints
        )
        source_score = float(normalized_violation_score(source_constraints))
        original_hash = vector_hash(original_vector)

        # One persistent seed supplies all stochastic satisfaction margins.
        # Fuel repair uses its sampled margin but has no discrete method draw.
        # There is exactly one proposal and no repair/evaluation loop.
        ledger = MarginLedger(design_seed, self.configuration)
        proposal = self.methods.propose(
            original_vector,
            source_constraints,
            ledger,
        )
        proposed_hash = vector_hash(proposal.vector)

        write_csv_atomic(
            pd.DataFrame(
                [
                    {
                        "Parent_Index": parent_index,
                        "Alternative_Number": alternative_number,
                        **self.parameters.row_from_vector(original_vector),
                    }
                ]
            ),
            design_root / "source_189_parameter_vector.csv.gz",
        )
        write_csv_atomic(
            source_constraints,
            design_root / "source_constraint_evaluations.csv.gz",
        )
        write_csv_atomic(
            pd.DataFrame(
                [
                    {
                        "Parent_Index": parent_index,
                        "Alternative_Number": alternative_number,
                        **self.parameters.row_from_vector(proposal.vector),
                    }
                ]
            ),
            design_root / "proposed_189_parameter_vector.csv.gz",
        )
        write_csv_atomic(
            proposal.unmapped,
            design_root / "unresolved_source_constraints.csv.gz",
        )
        write_csv_atomic(
            self.parameters.sampling_bound_exceedances(proposal.vector),
            design_root / "proposed_sampling_bound_exceedances.csv",
        )
        margin_table = ledger.table()
        margin_table.insert(0, "Parent_Index", parent_index)
        margin_table.insert(1, "Alternative_Number", alternative_number)
        write_csv_atomic(
            margin_table,
            design_root / "sampled_repair_decisions.csv",
        )

        final_output_paths: dict[str, Path] = {}
        submitted_vector, _ = self.parameters.canonicalize(proposal.vector)
        committed_elements = None
        evaluation_error = None

        try:
            (
                final_output_paths,
                submitted_vector,
                committed_elements,
            ) = evaluate_repaired_vector(
                proposal.vector,
                output_root=batch_root / "final_evaluations",
                repaired_index=repaired_index,
                repaired_structure_index=repaired_structure_index,
                source_identifiers=source_identifiers,
                configuration=self.configuration,
                parameters=self.parameters,
            )
        except Exception as error:
            evaluation_error = {
                "Error_Type": type(error).__name__,
                "Error_Message": str(error),
                "Traceback": traceback.format_exc(),
            }
            write_json_atomic(
                evaluation_error,
                design_root / "evaluation_failure.json",
            )

        if final_output_paths:
            committed_parameters = _filter_evaluator_table(
                _read_csv_allow_empty(final_output_paths["updated_parameters"]),
                repaired_index,
            )
            final_constraints = _filter_evaluator_table(
                _read_csv_allow_empty(
                    final_output_paths["expanded_constraints_v1p1"]
                ),
                repaired_index,
            )
            final_failures = _filter_evaluator_table(
                _read_csv_allow_empty(final_output_paths["failures"]),
                repaired_index,
            )
            final_performance = _filter_evaluator_table(
                _read_csv_allow_empty(final_output_paths["scalar_results"]),
                repaired_index,
            )
            final_status = _committed_evaluation_status(
                committed_parameters,
                final_failures,
            )
        else:
            committed_parameters = pd.DataFrame()
            final_constraints = pd.DataFrame()
            final_performance = pd.DataFrame()
            final_failures = pd.DataFrame([evaluation_error or {}])
            final_status = "evaluation_failure"

        if len(committed_parameters) == 1:
            final_vector = self.parameters.vector_from_row(
                committed_parameters.iloc[0]
            )
        else:
            final_vector = submitted_vector.copy()

        final_failure_count = int(
            len(canonical_failed_constraint_rows(final_constraints))
        )
        final_evaluated_count = canonical_evaluated_constraint_count(
            final_constraints
        )
        final_score = float(normalized_violation_score(final_constraints))
        final_hash = vector_hash(final_vector)
        fully_feasible = bool(
            final_status == "completed"
            and final_evaluated_count > 0
            and final_failure_count == 0
        )

        write_csv_atomic(
            pd.DataFrame(
                [
                    {
                        "Parent_Index": parent_index,
                        "Alternative_Number": alternative_number,
                        **self.parameters.row_from_vector(final_vector),
                    }
                ]
            ),
            design_root / "final_189_parameter_vector.csv.gz",
        )
        write_csv_atomic(
            final_constraints,
            design_root / "final_constraint_evaluations.csv.gz",
        )
        write_csv_atomic(
            final_failures,
            design_root / "final_evaluation_failures.csv.gz",
        )
        write_csv_atomic(
            final_performance,
            design_root / "final_performance_metrics.csv.gz",
        )
        write_csv_atomic(
            _final_unresolved_table(
                final_constraints,
                self.constraint_registry,
            ),
            design_root / "final_unresolved_constraints.csv.gz",
        )
        write_csv_atomic(
            self.parameters.sampling_bound_exceedances(final_vector),
            design_root / "final_sampling_bound_exceedances.csv",
        )

        copied_elements = design_root / "final_structural_elements.csv"

        if committed_elements is not None and committed_elements.exists():
            shutil.copy2(committed_elements, copied_elements)

        action_rows = []

        for sequence, change in enumerate(proposal.changes, start=1):
            choices = []

            for key in change.constraint_keys:
                record = ledger.draws.get(key, {})
                choice = record.get("Stochastic_Choice")

                if choice:
                    choices.append(f"{key}={choice}")

            action_rows.append(
                {
                    "Parent_Index": parent_index,
                    "Alternative_Number": alternative_number,
                    "Repaired_Index": repaired_index,
                    "Repaired_Structure_Index": repaired_structure_index,
                    "Batch_Kind": batch_kind,
                    "Selection_Order": selection_order,
                    "Batch_Seed": int(selection["Batch_Seed"]),
                    "Design_Repair_Seed": design_seed,
                    "Repair_Action_Sequence": sequence,
                    "Parameter": change.parameter,
                    "Old_Value": change.old_value,
                    "Proposed_Value": change.proposed_value,
                    "Evaluated_Value": float(
                        final_vector[self.parameters.index[change.parameter]]
                    ),
                    "Repair_Method": change.repair_method,
                    "Repair_Family": change.repair_family,
                    "Constraint_Keys": " | ".join(change.constraint_keys),
                    "Sampled_Margins": " | ".join(
                        f"{value:.17g}" for value in change.sampled_margins
                    ),
                    "Stochastic_Choices": " | ".join(choices),
                    "Equation": change.equation,
                    "Repair_Lower_Limit": change.repair_lower_limit,
                    "Repair_Upper_Limit": change.repair_upper_limit,
                    "At_Repair_Domain_Limit": change.at_repair_domain_limit,
                    "Outside_Original_Sampling_Bounds": (
                        change.outside_sampling_bounds
                    ),
                    "Source_Constraint_Evidence": _constraint_evidence(
                        source_constraints,
                        change.constraint_keys,
                    ),
                    "Final_Constraint_Evidence": _constraint_evidence(
                        final_constraints,
                        change.constraint_keys,
                    ),
                    "Source_Failure_Count": source_failure_count,
                    "Final_Failure_Count": final_failure_count,
                    "Original_Vector_SHA256": original_hash,
                    "Proposed_Vector_SHA256": proposed_hash,
                    "Final_Vector_SHA256": final_hash,
                    "Final_Status": final_status,
                }
            )

        write_csv_atomic(
            pd.DataFrame(action_rows, columns=ACTION_OUTPUT_COLUMNS),
            design_root / "repair_actions.csv.gz",
        )

        summary = RepairDesignSummary(
            schema_version=5,
            repair_method="one_pass_stochastic_equation_repair",
            batch_kind=batch_kind,
            selection_order=selection_order,
            parent_index=parent_index,
            alternative_number=alternative_number,
            batch_seed=int(selection["Batch_Seed"]),
            design_repair_seed=design_seed,
            repaired_index=repaired_index,
            repaired_structure_index=repaired_structure_index,
            repair_configuration_sha256=self.configuration.fingerprint(),
            original_vector_sha256=original_hash,
            proposed_vector_sha256=proposed_hash,
            final_vector_sha256=final_hash,
            source_reported_failure_count=int(
                selection["Original_V1p1_Constraint_Fail_Count"]
            ),
            source_canonical_failure_count=source_failure_count,
            source_evaluated_constraint_count=source_evaluated_count,
            source_violation_score=source_score,
            proposed_parameter_change_count=len(proposal.changes),
            final_status=final_status,
            final_failure_count=final_failure_count,
            final_evaluated_constraint_count=final_evaluated_count,
            final_violation_score=final_score,
            fully_feasible=fully_feasible,
            repair_passes=1,
            repaired_design_evaluations=1,
            result_directory=str(design_root.resolve()),
        )
        write_json_atomic(
            _json_safe(asdict(summary)),
            design_root / "design_summary.json",
        )
        write_json_atomic(
            _json_safe(
                _build_final_output_manifest(
                    normal_output_paths=final_output_paths,
                    design_root=design_root,
                    committed_elements=committed_elements,
                    copied_elements=copied_elements,
                )
            ),
            design_root / "final_output_manifest.json",
        )
        return summary

    def run_batch(
        self,
        batch_kind: str,
        *,
        stop_after: int | None = None,
    ) -> pd.DataFrame:
        """Run one repair and one evaluation for each frozen source."""

        if batch_kind not in self.REPAIRED_INDEX_BASE:
            raise ValueError(
                "batch_kind must be development, validation, below_median, "
                "or candidate_indices."
            )

        # The validation batch is also the fixed ten-design regression set for
        # testing changes to the constraint and repair formulations.  Run that
        # batch directly; a prior development result is not a prerequisite for
        # this engineering pipeline test.
        #
        # The former formal-study gate is intentionally not called here:
        # _require_validation_prerequisite(...)

        if batch_kind == "candidate_indices":
            raise ValueError(
                "candidate_indices batches are partitioned by "
                "run_candidate_cluster_array and cannot use run_batch."
            )

        if batch_kind == "below_median":
            batch_root = self.experiment_root / "below_median_batch"
            selection = freeze_below_median_alternatives(
                self.dataset,
                batch_root,
                seed=self.configuration.below_median_batch_seed,
                alternatives_per_source=(
                    self.configuration.alternatives_per_below_median_source
                ),
                batch_kind="below_median",
            ).sort_values("Selection_Order")
        else:
            batches = freeze_configured_batches(self.configuration)
            selection = batches[batch_kind].sort_values("Selection_Order")
            batch_root = self.experiment_root / f"{batch_kind}_batch"

        if stop_after is not None:
            if not 1 <= int(stop_after) <= len(selection):
                raise ValueError("stop_after must be within the frozen batch.")

            selection = selection.iloc[: int(stop_after)]

        for _, selected in selection.iterrows():
            design_root = (
                batch_root
                / "results"
                / f"parent_{int(selected['Index']):06d}"
            )

            if batch_kind == "below_median":
                design_root = (
                    design_root
                    / f"alternative_{int(selected['Alternative_Number']):02d}"
                )

            _prepare_design_root(design_root)

        project_root = Path(__file__).resolve().parents[1]
        write_json_atomic(
            _json_safe(
                {
                    "schema_version": 4,
                    "method": "one_pass_stochastic_equation_repair",
                    "configuration": self.configuration.to_dict(),
                    "configuration_sha256": self.configuration.fingerprint(),
                    "batch_kind": batch_kind,
                    "selected_parent_indices": (
                        selection["Index"].astype(int).tolist()
                    ),
                    "runtime_provenance": _runtime_provenance(),
                    "repair_contract": {
                        "source_evaluations": (
                            "saved V1.1 rows from the immutable 0-399 pool"
                        ),
                        "repair_passes_per_design": 1,
                        "repaired_evaluations_per_design": 1,
                        "alternatives_per_source": (
                            self.configuration
                            .alternatives_per_below_median_source
                            if batch_kind == "below_median"
                            else 1
                        ),
                        "post_evaluation_repair": False,
                        "performance_metrics_used_for_repair": False,
                        "fuel_repair_method": (
                            "hold fuel-tank geometry fixed; reduce Fn_ship "
                            "first, then Voyage_Range only if the speed repair "
                            "floor prevents the required fuel reduction"
                        ),
                    },
                    "implementation_fingerprints": {
                        str(path.relative_to(project_root)): {
                            "sha256": sha256_file(path),
                            "size_bytes": path.stat().st_size,
                        }
                        for path in _implementation_paths(project_root)
                    },
                }
            ),
            batch_root / "repair_run_manifest.json",
        )
        write_csv_atomic(
            self.parameters.role_table(),
            batch_root / "parameter_repair_registry.csv",
        )
        write_csv_atomic(
            self.constraint_registry.export_runtime_registry(
                self.dataset.constraint_catalog
            ),
            batch_root / "constraint_repair_registry.csv",
        )

        summaries = [
            asdict(self.repair_one(row, batch_kind))
            for _, row in selection.iterrows()
        ]
        table = pd.DataFrame(summaries)
        write_csv_atomic(table, batch_root / "batch_repair_summary.csv")

        feasible_count = int(table["fully_feasible"].sum())
        target = int(self.configuration.development_feasible_target)
        parent_count = int(table["parent_index"].nunique())
        is_ten_design_test = batch_kind in {"development", "validation"}
        write_json_atomic(
            _json_safe(
                {
                    "batch_kind": batch_kind,
                    "design_count": int(len(table)),
                    "source_design_count": parent_count,
                    "alternatives_per_source": (
                        self.configuration
                        .alternatives_per_below_median_source
                        if batch_kind == "below_median"
                        else 1
                    ),
                    "fully_feasible_count": feasible_count,
                    "success_rate": float(table["fully_feasible"].mean()),
                    "target": (
                        f"at least {target} of 10 fully feasible designs"
                        if is_ten_design_test
                        else "report all below-median repair outcomes"
                    ),
                    "target_met": bool(
                        len(table) == 10 and feasible_count >= target
                    ),
                }
            ),
            batch_root / "batch_outcome.json",
        )
        write_json_atomic(
            _json_safe(_build_batch_output_manifest(batch_root)),
            batch_root / "batch_output_manifest.json",
        )
        return table


__all__ = [
    "ACTION_OUTPUT_COLUMNS",
    "RepairDesignSummary",
    "RepairPipeline",
]
