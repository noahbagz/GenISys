#!/usr/bin/env python3
"""Compile repaired task rows into the available V1.1 dataset table families.

The repair worker retained vectors, scalar results, exact constraint rows, and
failures. It did not retain the structural elements, numerical arrays, or
long-form engineering detail shards from the ordinary generation workflow.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import sys

from tools.ConstraintIdentity import _CANONICAL_ALIASES_BY_NAME


ROOT = Path(__file__).resolve().parent / "Ship_Systems_Designs_v1p1"
SOURCE = ROOT / "Dataset_100k" / "compiled"
REPAIRED = ROOT / "repair_experiments" / "Dataset_100k_Repaired"
IDENTIFIERS = (
    "Index", "Structure_Index", "Hull_Variant_Index",
    "Operating_Condition_Index", "Hull_Candidate_Index",
    "Structure_Sample_Attempt",
)
MARGIN_HEADER = (
    "Index", "Structure_Index", "Class", "Requirement", "Name",
    "Constraint_Key", "Threshold_Value", "Constraint_Value",
    "Constraint_Eval", "Normalization_Scale",
    "Normalized_Constraint_Eval", "Constraint_Bit", "Units",
)
CATALOG_HEADER = (
    "Constraint_Key", "Class", "Name", "Requirement", "Units",
    "Output_Type", "Observed_Row_Count", "Observed_System_Count",
    "Conditional_Runtime_Row",
)


def _header(path: Path) -> list[str]:
    """Read one CSV header without loading its rows."""

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return next(csv.reader(stream))


def _bit(value) -> bool | None:
    """Use the same blank/true/false meanings as the V1.1 compiler."""

    if value is None or str(value).strip() == "":
        return None
    return str(value).strip().lower() in {"true", "1", "1.0"}


def _number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _constraint_key(row: dict) -> str:
    name = str(row.get("Name", ""))
    return _CANONICAL_ALIASES_BY_NAME.get(
        name, f"{row.get('Requirement', '')}::{name}"
    )


def _task_rows(paths: list[Path], expected_header: list[str]):
    """Stream task rows with one shared positional schema."""

    for path in paths:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            if next(reader) != expected_header:
                raise ValueError(f"Repair task schema differs: {path}")
            for row in reader:
                if len(row) != len(expected_header):
                    raise ValueError(f"Malformed repair row: {path}")
                yield row


def compile_tables(repair_root: Path, source_root: Path, task_count: int) -> None:
    """Make regular-format tables from the data the repair worker retained."""

    csv.field_size_limit(sys.maxsize)
    batch = repair_root / "candidate_indices_batch"
    paths = [
        batch / "task_results" / f"Repaired_Designs_Task_{i:04d}.csv.gz"
        for i in range(task_count)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} task shards: {missing[:3]}")

    selection_path = batch / "selected_source_designs.csv"
    with selection_path.open(newline="", encoding="utf-8") as stream:
        expected_orders = {
            int(row["Selection_Order"]) for row in csv.DictReader(stream)
        }

    source_header = _header(paths[0])
    positions = {name: i for i, name in enumerate(source_header)}
    design_header = _header(source_root / "System_Designs.csv.gz")
    performance_header = _header(source_root / "Ship_System_Performance.csv.gz")
    parameter_names = design_header[len(IDENTIFIERS):]
    with (source_root / "Parameter_Catalog.csv").open(newline="", encoding="utf-8") as stream:
        bounds = [
            (row["Name"], float(row["LL"]), float(row["UL"]))
            for row in csv.DictReader(stream)
        ]

    # Inspect exact constraint rows first. The wide repair columns include
    # metrics, while the ordinary bit table includes evaluated rules only.
    bit_keys = set()
    class_names = set()
    observed_orders = set()
    observed_indices = set()
    for row in _task_rows(paths, source_header):
        index = int(row[positions["Index"]])
        order = int(row[positions["Selection_Order"]])
        if index in observed_indices or order in observed_orders:
            raise ValueError(f"Duplicate repaired Index or Selection_Order: {index}")
        observed_indices.add(index)
        observed_orders.add(order)

        for constraint in json.loads(row[positions["Constraint_Evaluations_JSON"]]):
            if _bit(constraint.get("Constraint Bit")) is not None:
                bit_keys.add(_constraint_key(constraint))
                class_names.add(str(constraint.get("Class", "")))

    if observed_orders != expected_orders:
        raise ValueError(
            "Repair shards do not cover the frozen selection: "
            f"missing={len(expected_orders - observed_orders)}, "
            f"unexpected={len(observed_orders - expected_orders)}"
        )

    bit_header = ["Index", *sorted(bit_keys)]
    feasible_header = [
        "Index", "Feasible::ALL_V1P1_CONSTRAINTS",
        *(f"Feasible::{name}" for name in sorted(class_names)),
    ]
    matrix_header = [
        *design_header,
        *(name for name in performance_header if name not in IDENTIFIERS),
        *bit_header[1:],
        *feasible_header[1:],
    ]
    output = repair_root / "compiled"
    output.mkdir(parents=True, exist_ok=True)
    temporary = []
    # Each design is visited once, so a last-seen index gives distinct-system
    # counts without retaining millions of index/key pairs in memory.
    catalog_counts: dict[tuple, list] = defaultdict(lambda: [0, 0, None])
    failure_count = 0
    systems_with_constraints = 0
    bound_violation_count = 0

    def open_csv(stack: ExitStack, name: str, header: list[str] | tuple[str, ...]):
        path = output / name
        temp = output / f".{name}.tmp"
        temporary.append((temp, path))
        if name.endswith(".gz"):
            stream = stack.enter_context(
                gzip.open(temp, "wt", newline="", encoding="utf-8", compresslevel=3)
            )
        else:
            stream = stack.enter_context(open(temp, "w", newline="", encoding="utf-8"))
        writer = csv.writer(stream)
        writer.writerow(header)
        return writer

    try:
        with ExitStack() as stack:
            writers = {
                "design": open_csv(stack, "System_Designs.csv.gz", design_header),
                "updated": open_csv(stack, "Updated_Ship_System_Parameters.csv.gz", design_header),
                "performance": open_csv(stack, "Ship_System_Performance.csv.gz", performance_header),
                "bits": open_csv(stack, "Compiled_V1p1_Constraint_Bits.csv.gz", bit_header),
                "subsystem": open_csv(stack, "V1p1_Subsystem_Level_Feasibility.csv.gz", [
                    "Index", "Class", "Evaluated_Constraint_Count",
                    "Passed_Constraint_Count", "Subsystem_Feasible",
                ]),
                "feasible": open_csv(stack, "V1p1_Subsystem_Feasibility_Bits.csv.gz", feasible_header),
                "margins": open_csv(stack, "Normalized_Constraint_Margins.csv.gz", MARGIN_HEADER),
                "failures": open_csv(stack, "Evaluation_Failures.csv.gz", _header(source_root / "Evaluation_Failures.csv.gz")),
                "matrix": open_csv(stack, "ML_Design_Matrix.csv.gz", matrix_header),
                "bounds": open_csv(stack, "Parameter_Bound_Violations.csv", [
                    "Record_Type", "Structure_Index", "Index", "Parameter",
                    "Value", "Lower_Bound", "Upper_Bound", "Violation",
                ]),
                "lineage": open_csv(stack, "Repair_Lineage.csv.gz", [
                    "Index", "Parent_Index", "Alternative_Number",
                    "Selection_Order", "Task_ID", "Final_Status", "Fully_Feasible",
                ]),
            }
            failure_header = _header(source_root / "Evaluation_Failures.csv.gz")

            for row_number, row in enumerate(_task_rows(paths, source_header), 1):
                index = int(row[positions["Index"]])
                value = lambda name: row[positions[name]]
                performance = [value(f"Performance::{name}") for name in performance_header]
                if not performance[0]:
                    performance[0] = index
                if not performance[1]:
                    performance[1] = value("Repaired_Structure_Index")
                if not performance[performance_header.index("Evaluation_Status")]:
                    performance[performance_header.index("Evaluation_Status")] = "failed"
                identifiers = performance[:len(IDENTIFIERS)]
                identifiers[0] = index
                identifiers[1] = value("Repaired_Structure_Index")
                design = [*identifiers, *(value(name) for name in parameter_names)]

                writers["design"].writerow(design)
                writers["updated"].writerow(design)
                writers["performance"].writerow(performance)
                writers["lineage"].writerow([
                    index, value("Parent_Index"), value("Alternative_Number"),
                    value("Selection_Order"), value("Task_ID"),
                    value("Final_Status"), value("Fully_Feasible"),
                ])

                # Repair can move inputs beyond the original DOE bounds.
                # Record those values with the regular compiled table schema.
                for name, lower, upper in bounds:
                    actual = _number(value(name))
                    if actual is None:
                        continue
                    tolerance = 1.0e-12 * max(1.0, abs(lower), abs(upper))
                    if actual < lower - tolerance or actual > upper + tolerance:
                        writers["bounds"].writerow([
                            "system", identifiers[1], index, name, actual,
                            lower, upper,
                            "below_lower_bound" if actual < lower else "above_upper_bound",
                        ])
                        bound_violation_count += 1

                # Collapse only explicitly reviewed aliases for bit and
                # feasibility counts. Preserve every exact row for margins.
                bits: dict[str, bool] = {}
                class_bits: dict[str, dict[str, bool]] = defaultdict(dict)
                constraints = json.loads(value("Constraint_Evaluations_JSON"))
                if constraints:
                    systems_with_constraints += 1
                for constraint in constraints:
                    name = str(constraint.get("Name", ""))
                    requirement = str(constraint.get("Requirement", ""))
                    key = f"{requirement}::{name}"
                    constraint_class = str(constraint.get("Class", ""))
                    units = str(constraint.get("Units", ""))
                    bit = _bit(constraint.get("Constraint Bit"))
                    catalog_key = (
                        key, constraint_class, name, requirement, units,
                        "constraint" if bit is not None else "metric",
                    )
                    counts = catalog_counts[catalog_key]
                    counts[0] += 1
                    if counts[2] != index:
                        counts[1] += 1
                        counts[2] = index

                    if bit is not None:
                        canonical = _constraint_key(constraint)
                        bits[canonical] = bits.get(canonical, True) and bit
                        prior = class_bits[constraint_class].get(canonical, True)
                        class_bits[constraint_class][canonical] = prior and bit

                    threshold = _number(constraint.get("Threshold Value"))
                    actual = _number(constraint.get("Constraint Value"))
                    residual = _number(constraint.get("Constraint Eval"))
                    if (
                        bit is not None and threshold is not None
                        and actual is not None and residual is not None
                        and units.casefold() not in {"boolean", "n/a"}
                    ):
                        scale = max(abs(threshold), abs(actual), 1.0e-12)
                        writers["margins"].writerow([
                            index, identifiers[1], constraint_class,
                            requirement, name, key, threshold, actual,
                            residual, scale, residual / scale, bit, units,
                        ])

                total = list(bits.values())
                if total:
                    writers["subsystem"].writerow([
                        index, "ALL_V1P1_CONSTRAINTS", len(total),
                        sum(total), all(total),
                    ])
                for class_name, class_values in class_bits.items():
                    values = list(class_values.values())
                    writers["subsystem"].writerow([
                        index, class_name, len(values), sum(values), all(values),
                    ])

                bit_row = [index, *(bits.get(key, "") for key in bit_header[1:])]
                feasible_row = [
                    index,
                    all(total) if total else "",
                    *(all(class_bits[name].values()) if name in class_bits else ""
                      for name in sorted(class_names)),
                ]
                writers["bits"].writerow(bit_row)
                writers["feasible"].writerow(feasible_row)
                writers["matrix"].writerow([
                    *design, *performance[len(IDENTIFIERS):],
                    *bit_row[1:], *feasible_row[1:],
                ])

                for failure in json.loads(value("Evaluation_Failures_JSON")):
                    writers["failures"].writerow(
                        failure.get(name, "") for name in failure_header
                    )
                    failure_count += 1

                if row_number % 10000 == 0:
                    print(f"Compiled {row_number}/{len(expected_orders)} repairs", flush=True)

        # Catalogs and checks are generated after all compressed writers close.
        with ExitStack() as stack:
            catalog = open_csv(stack, "Constraint_Catalog.csv", CATALOG_HEADER)
            for key, (count, system_count, _) in sorted(catalog_counts.items()):
                catalog.writerow([
                    *key, count, system_count,
                    system_count < systems_with_constraints,
                ])

            checks = open_csv(stack, "Compilation_Checks.csv", [
                "Check", "Passed", "Observed", "Expected", "Notes",
            ])
            checks.writerow([
                "complete_repaired_selection", True, len(observed_orders),
                len(expected_orders), "Every frozen alternative has one task row.",
            ])
            checks.writerow([
                "unique_repaired_indices", True, len(observed_indices),
                len(expected_orders), "Repaired Index is unique across tasks.",
            ])
            checks.writerow([
                "parameter_bound_violations_recorded", True,
                bound_violation_count, "recorded",
                "Repair proposals may exceed original DOE bounds; see the violation table.",
            ])
            checks.writerow([
                "unavailable_source_families", False,
                "structure status/elements; audited constraints; detail shards; arrays",
                "available", "The repair worker did not retain these artifacts.",
            ])

        for name in ("Parameter_Catalog.csv", "Metric_Catalog.csv"):
            src = source_root / name
            dst = output / name
            temp = output / f".{name}.tmp"
            shutil.copyfile(src, temp)
            temporary.append((temp, dst))

        for temp, dst in temporary:
            os.replace(temp, dst)
    except Exception:
        for temp, _ in temporary:
            temp.unlink(missing_ok=True)
        raise

    print(f"Compiled {len(observed_indices)} repaired designs and {failure_count} failures")
    print(f"Output: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, default=REPAIRED)
    parser.add_argument("--source-compiled", type=Path, default=SOURCE)
    parser.add_argument("--task-count", type=int, default=250)
    args = parser.parse_args()
    compile_tables(args.repair_root, args.source_compiled, args.task_count)


if __name__ == "__main__":
    main()
