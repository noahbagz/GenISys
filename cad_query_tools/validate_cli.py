"""CLI for repeatable generated-versus-Rhino artifact comparisons."""

from __future__ import annotations

import argparse
import json

from .validation import ElementComparisonConfig, validate_files_against_reference


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CadQuery CAD/CSV outputs with a Rhino reference pair."
    )
    parser.add_argument("--generated-cad", required=True)
    parser.add_argument("--generated-csv", required=True)
    parser.add_argument("--reference-cad", required=True)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=1.0e-6,
        help="Default absolute tolerance for geometry and element fields.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=1.0e-6,
        help="Default relative tolerance for geometry and element fields.",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip the headless isometric and orthographic PNG artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()

    # Keep the CLI scalar controls simple. Python callers can use the config's
    # per-field mappings when individual derived properties need looser limits.
    comparison_config = ElementComparisonConfig(
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    reports = validate_files_against_reference(
        generated_cad_path=args.generated_cad,
        generated_csv_path=args.generated_csv,
        reference_cad_path=args.reference_cad,
        reference_csv_path=args.reference_csv,
        output_directory=args.output_directory,
        report_name=args.report_name,
        comparison_config=comparison_config,
        write_previews=not args.no_previews,
    )
    print(json.dumps({key: str(path) for key, path in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
