#!/usr/bin/env python3
"""Command-line entry point for frozen GenISys repair experiments."""

from __future__ import annotations

import argparse

from .configuration import RepairConfiguration
from .dataset import freeze_configured_batches
from .pipeline import RepairPipeline


def parse_arguments() -> argparse.Namespace:
    """Expose only experiment controls that do not alter repair equations."""

    parser = argparse.ArgumentParser(
        description=(
            "Repair a frozen GenISys V1.1 source population through CadQuery "
            "and the complete system evaluator."
        )
    )
    parser.add_argument(
        "--batch",
        choices=("development", "validation", "below_median"),
        default="development",
        help=(
            "development and validation use frozen ten-design samples; "
            "below_median selects every completed partially feasible source "
            "strictly below the completed-population median and creates two "
            "alternative repairs per source"
        ),
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help=(
            "Run only the first N already-frozen designs. This is a smoke-test "
            "switch; it never redraws or replaces the batch."
        ),
    )
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Commit/reload both disjoint random batches without evaluating.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested reproducible experiment and print its summary."""

    arguments = parse_arguments()
    configuration = RepairConfiguration()

    if arguments.freeze_only:
        batches = freeze_configured_batches(configuration)

        for kind, table in batches.items():
            indices = ", ".join(table["Index"].astype(int).astype(str))
            print(f"{kind}: {indices}")

        return 0

    summary = RepairPipeline(configuration).run_batch(
        arguments.batch,
        stop_after=arguments.stop_after,
    )
    print(summary.to_string(index=False))
    feasible = int(summary["fully_feasible"].sum())
    print(f"Fully feasible: {feasible}/{len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
