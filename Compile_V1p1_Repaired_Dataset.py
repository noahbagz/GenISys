#!/usr/bin/env python3
"""Merge task-level repaired-design rows into one compiled CSV file."""

from __future__ import annotations

import argparse
import csv
import gzip
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPAIR_ROOT = (
    PROJECT_ROOT
    / "Ship_Systems_Designs_v1p1"
    / "repair_experiments"
    / "Dataset_100k_Repaired"
)


def parse_arguments() -> argparse.Namespace:
    """Read the repair root and expected SLURM task count."""

    parser = argparse.ArgumentParser(
        description=(
            "Compile one-row-per-repair SLURM task shards into one file."
        )
    )
    parser.add_argument(
        "--repair-root",
        default=str(DEFAULT_REPAIR_ROOT),
    )
    parser.add_argument("--task-count", type=int, default=250)
    return parser.parse_args()


def compile_repaired_dataset(
    repair_root: str | Path,
    *,
    task_count: int,
) -> Path:
    """Stream all task CSVs into one atomically committed gzip CSV."""

    repair_root = Path(repair_root).resolve()
    batch_root = repair_root / "candidate_indices_batch"
    task_root = batch_root / "task_results"
    expected_paths = [
        task_root / f"Repaired_Designs_Task_{task_id:04d}.csv.gz"
        for task_id in range(int(task_count))
    ]
    missing = [str(path) for path in expected_paths if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} repaired task shards; first missing: "
            f"{missing[:10]}"
        )

    selection_path = batch_root / "selected_source_designs.csv"

    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Frozen repaired-design selection was not found: {selection_path}"
        )

    selection = pd.read_csv(
        selection_path,
        usecols=["Selection_Order"],
    )
    expected_orders = set(selection["Selection_Order"].astype(int))

    compiled_root = repair_root / "compiled"
    compiled_root.mkdir(parents=True, exist_ok=True)
    output_path = compiled_root / "Repaired_Designs.csv.gz"
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    reference_header: list[str] | None = None
    observed_orders: set[int] = set()
    repaired_indices: set[int] = set()
    row_count = 0

    try:
        with gzip.open(temporary_path, "wt", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)

            for path in expected_paths:
                with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
                    reader = csv.reader(stream)

                    try:
                        header = next(reader)
                    except StopIteration as error:
                        raise ValueError(f"Empty repaired task shard: {path}") from error

                    if reference_header is None:
                        reference_header = header
                        writer.writerow(header)
                    elif header != reference_header:
                        raise ValueError(
                            "Repaired task shards do not share one schema: "
                            f"{path}."
                        )

                    index_position = header.index("Index")
                    order_position = header.index("Selection_Order")

                    for row in reader:
                        if len(row) != len(header):
                            raise ValueError(
                                f"Malformed repaired row in {path}: expected "
                                f"{len(header)} fields, observed {len(row)}."
                            )

                        repaired_index = int(float(row[index_position]))
                        selection_order = int(float(row[order_position]))

                        if repaired_index in repaired_indices:
                            raise ValueError(
                                f"Duplicate repaired Index {repaired_index}."
                            )

                        if selection_order in observed_orders:
                            raise ValueError(
                                "Duplicate repaired Selection_Order "
                                f"{selection_order}."
                            )

                        repaired_indices.add(repaired_index)
                        observed_orders.add(selection_order)
                        writer.writerow(row)
                        row_count += 1

        if observed_orders != expected_orders:
            missing_orders = sorted(expected_orders - observed_orders)[:20]
            unexpected_orders = sorted(observed_orders - expected_orders)[:20]
            raise ValueError(
                "Compiled repair rows do not match the frozen selection; "
                f"missing={missing_orders}, unexpected={unexpected_orders}."
            )

        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise

    print(f"Compiled repaired designs: {row_count}")
    print(f"Compiled output: {output_path}")
    return output_path


def main() -> int:
    """Run the repaired-data compiler."""

    arguments = parse_arguments()

    if int(arguments.task_count) < 1:
        raise ValueError("--task-count must be at least one.")

    compile_repaired_dataset(
        arguments.repair_root,
        task_count=int(arguments.task_count),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
