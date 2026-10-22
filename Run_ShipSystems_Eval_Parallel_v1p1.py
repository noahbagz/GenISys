"""Generate and evaluate one restartable GenISys V1.1 cluster shard.

This worker follows the legacy ``Run_ShipSystems_Eval_Parallel.py`` contract:
one SLURM array task supplies one integer task ID.  Unlike the legacy worker,
the V1.1 worker first generates its assigned CadQuery structures and their
system-design rows, then evaluates that same local shard with the current
constraint, performance, loading-condition, and flooding calculations.

By default, the 250-task production array divides the complete population as
follows:

    20 structures/task x 4 hulls/structure x 5 conditions/hull
    = 400 system designs/task

    250 tasks x 400 system designs/task = 100,000 system designs

Every task writes uniquely named, atomic shard files.  This worker does not
compile the dataset; compilation is intentionally performed after the SLURM
array has completed.

Environment-variable overrides allow the same worker to run a smaller cluster
acceptance dataset without changing the production defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from Generate_V1p1_Design_Batch import (
    DEFAULT_MAXIMUM_STRUCTURE_ATTEMPTS,
    generate_design_batch,
)
from Run_V1p1_System_Evaluation_Batch import (
    _load_optional_json_mapping_with_provenance,
    evaluate_system_shard,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "Ship_Systems_Designs_v1p1" / "Dataset_100k"
)

NUMBER_OF_STRUCTURES = int(
    os.environ.get("GENISYS_V1P1_NUMBER_OF_STRUCTURES", "5000")
)
STRUCTURES_PER_TASK = int(
    os.environ.get("GENISYS_V1P1_STRUCTURES_PER_TASK", "20")
)
MAXIMUM_STRUCTURE_ATTEMPTS = int(
    os.environ.get(
        "GENISYS_V1P1_MAXIMUM_STRUCTURE_ATTEMPTS",
        str(DEFAULT_MAXIMUM_STRUCTURE_ATTEMPTS),
    )
)
WRITE_PER_DESIGN_REVIEW_FILES = (
    os.environ.get("GENISYS_WRITE_PER_DESIGN_REVIEW_FILES", "0") == "1"
)
SAVE_MODE = os.environ.get("GENISYS_V1P1_SAVE_MODE", "all")


def structure_range_for_task(task_id: int) -> tuple[int, int]:
    """Return the half-open structure range owned by one array task."""

    start_structure = int(task_id) * STRUCTURES_PER_TASK
    end_structure = min(
        start_structure + STRUCTURES_PER_TASK,
        NUMBER_OF_STRUCTURES,
    )

    return start_structure, end_structure


def _load_scenario_from_environment(variable_name: str):
    """Load one optional scenario JSON and preserve its file provenance."""

    scenario_path = os.environ.get(variable_name) or None

    return _load_optional_json_mapping_with_provenance(scenario_path)


def main(task_id: int) -> None:
    """Generate and evaluate the structure range assigned to ``task_id``."""

    start_structure, end_structure = structure_range_for_task(task_id)

    # Keep the output-root override simple so a cluster run can be redirected
    # without changing the source file.  The default remains Dataset_100k.
    output_root = Path(
        os.environ.get("GENISYS_V1P1_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    )

    print(f"Array task: {task_id}")
    print(f"Structure range: [{start_structure}, {end_structure})")
    print(f"Output root: {output_root}")
    print(f"Save mode: {SAVE_MODE}")
    print(f"Maximum structure attempts: {MAXIMUM_STRUCTURE_ATTEMPTS}")

    # Generate the assigned CadQuery structures, four hull variants per structure,
    # and five operating conditions per hull.  The returned system-design CSV
    # is the only input shard evaluated by this worker.
    generation_outputs = generate_design_batch(
        start_structure=start_structure,
        end_structure=end_structure,
        output_root=output_root,
        maximum_structure_attempts=MAXIMUM_STRUCTURE_ATTEMPTS,
        save_mode=SAVE_MODE,
        # A failed structure is a local generation outcome. Preserve it in
        # Structure_Status/Structure_Failures, skip its nonexistent systems,
        # and still evaluate every accepted sibling in this task.
        raise_on_failed_structure=False,
    )

    system_shard = generation_outputs["system_designs"]

    # Optional scenario files use the same environment-variable interface as
    # the earlier V1.1 launcher.  When absent, current built-in assumptions are
    # used and recorded explicitly in the evaluation provenance sidecar.
    powering_inputs, powering_provenance = (
        _load_scenario_from_environment("GENISYS_POWERING_INPUTS_JSON")
    )
    early_mass_inputs, early_mass_provenance = (
        _load_scenario_from_environment(
            "GENISYS_EARLY_DESIGN_MASS_INPUTS_JSON"
        )
    )
    economic_inputs, economic_provenance = (
        _load_scenario_from_environment("GENISYS_ECONOMIC_INPUTS_JSON")
    )

    # Evaluate all 400 system rows in this production shard. The existing
    # binary-search flooding diagnostic remains active for production.
    evaluation_outputs = evaluate_system_shard(
        input_csv=system_shard,
        output_root=output_root,
        run_flooding_diagnostic=True,
        write_per_design_review_files=WRITE_PER_DESIGN_REVIEW_FILES,
        save_mode=SAVE_MODE,
        powering_inputs=powering_inputs,
        early_design_mass_inputs=early_mass_inputs,
        economic_inputs=economic_inputs,
        scenario_provenance={
            "powering_inputs": powering_provenance,
            "early_design_mass_inputs": early_mass_provenance,
            "economic_inputs": economic_provenance,
        },
    )

    print(f"Generated system shard: {system_shard}")
    print(
        "Evaluated result shard: "
        f"{evaluation_outputs['scalar_results']}"
    )


if __name__ == "__main__":
    array_task_id = int(sys.argv[1])
    main(array_task_id)
