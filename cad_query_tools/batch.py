"""Explicit, restart-friendly batch generation for 120-column design tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .generate import generate_structure_files, select_structural_parameter_table


def generate_batch(
    input_csv: str | Path,
    output_directory: str | Path,
    start_index: int = 0,
    end_index: int | None = None,
    ship_id_prefix: str = "CadQuery_Design",
    file_format: str = "igs",
    clean_parameters: bool = False,
    artifact_mode: str = "full",
) -> dict[str, Path]:
    """Generate an explicit half-open index range and record every result."""

    table = select_structural_parameter_table(pd.read_csv(input_csv))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stop = len(table) if end_index is None else min(int(end_index), len(table))
    # Store only rows owned by this half-open shard.  The previous batch path
    # allocated the full input length and wrote zero rows outside the requested
    # range, making a partial cluster shard look like failed structure data.
    updated_rows = []
    failures = []
    timings = []

    for row_index in range(int(start_index), stop):
        ship_id = f"{ship_id_prefix}_{row_index}"

        try:
            result = generate_structure_files(
                table.iloc[row_index].to_numpy(dtype=float, copy=True),
                output_directory=output,
                ship_id=ship_id,
                file_format=file_format,
                clean_parameters=clean_parameters,
                parameter_names=table.columns.tolist(),
                artifact_mode=artifact_mode,
            )
            updated_rows.append(
                {
                    "row_index": row_index,
                    "ship_id": ship_id,
                    **dict(
                        zip(
                            table.columns,
                            result["cleaned_parameters"],
                            strict=True,
                        )
                    ),
                }
            )
            timings.append(
                {
                    "row_index": row_index,
                    "ship_id": ship_id,
                    "status": "success",
                    **{
                        key: value
                        for key, value in result["timing"].items()
                        if isinstance(value, (int, float, str, bool))
                    },
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "row_index": row_index,
                    "ship_id": ship_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            timings.append(
                {
                    "row_index": row_index,
                    "ship_id": ship_id,
                    "status": "failed",
                }
            )

    updated_path = output / f"{ship_id_prefix}_Parameters_Updated.csv"
    failures_path = output / f"{ship_id_prefix}_Failures.csv"
    timings_path = output / f"{ship_id_prefix}_CadQuery_Timings.csv"
    updated_columns = ["row_index", "ship_id", *table.columns]
    pd.DataFrame(updated_rows, columns=updated_columns).to_csv(
        updated_path,
        index=False,
    )
    pd.DataFrame(
        failures,
        columns=("row_index", "ship_id", "error_type", "error_message"),
    ).to_csv(failures_path, index=False)
    pd.DataFrame(timings).to_csv(timings_path, index=False)

    return {
        "updated_parameters_csv": updated_path,
        "failures_csv": failures_path,
        "timings_csv": timings_path,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a range of ship structures without Rhino."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--ship-id-prefix", default="CadQuery_Design")
    parser.add_argument(
        "--file-format",
        choices=("igs", "iges", "step", "stp", "both"),
        default="igs",
    )
    parser.add_argument("--clean-parameters", action="store_true")
    parser.add_argument(
        "--elements-only",
        action="store_true",
        help="Write structural-element CSVs without CAD or per-design sidecars.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    reports = generate_batch(
        input_csv=args.input_csv,
        output_directory=args.output_directory,
        start_index=args.start_index,
        end_index=args.end_index,
        ship_id_prefix=args.ship_id_prefix,
        file_format=args.file_format,
        clean_parameters=args.clean_parameters,
        artifact_mode=("elements_only" if args.elements_only else "full"),
    )

    for label, path in reports.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
