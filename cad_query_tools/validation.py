"""Repeatable comparisons against Rhino-generated IGES and CSV references.

The Rhino UUID columns cannot be used to establish equivalence because the
CadQuery generator deliberately issues stable logical identifiers. Element
matching therefore uses structural meaning and geometry metadata instead:

1. group rows by ``Class``, ``Type``, and the class of ``Parent_Struct``;
2. assign rows within each group from location and nominal dimensions; and
3. compare every numeric field with explicit absolute and relative tolerances.

The matching and comparison reports retain both object IDs only so a failed
row can be traced back to its source artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from math import isclose
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .exporters import load_iges, load_step
from .model import RHINO_CSV_COLUMNS, ShipStructure


DEFAULT_MATCH_NUMERIC_FIELDS = (
    "x_loc",
    "y_loc",
    "z_loc",
    "L1",
    "L2",
    "Thickness",
    # Disconnected members on opposite sides of a hatch can share every
    # projected legacy location/dimension field. Their geometric centroids
    # provide the ID-independent tie-break needed to avoid crossing the pair.
    "x_centroid_pannel",
    "y_centroid_pannel",
    "z_centroid_pannel",
)


DEFAULT_COMPARE_NUMERIC_FIELDS = (
    "x_loc",
    "y_loc",
    "z_loc",
    "x_dir",
    "y_dir",
    "z_dir",
    "rot",
    "L1",
    "L2",
    "Thickness",
    "area_pannel",
    "volume_pannel",
    "x_centroid_pannel",
    "y_centroid_pannel",
    "z_centroid_pannel",
    "area_cx",
    "I_11",
    "I_22",
    "J_p",
    "centroid_cx_11",
    "centroid_cx_22",
    "I_11_1D",
    "I_22_1D",
    "J_p_1D",
)


@dataclass(frozen=True)
class ElementComparisonConfig:
    """Tolerance and field selections for element-level CSV comparison.

    Scalar tolerances establish the default acceptance check. Per-field
    mappings override them where a derived property needs a different error
    allowance. Match assignment uses the same tolerances only to normalize
    fields with different units; acceptance is evaluated after assignment.
    """

    absolute_tolerance: float = 1.0e-6
    relative_tolerance: float = 1.0e-6
    absolute_tolerances: Mapping[str, float] = field(default_factory=dict)
    relative_tolerances: Mapping[str, float] = field(default_factory=dict)
    match_numeric_fields: tuple[str, ...] = DEFAULT_MATCH_NUMERIC_FIELDS
    compare_numeric_fields: tuple[str, ...] = DEFAULT_COMPARE_NUMERIC_FIELDS
    resolve_parent_to_class: bool = True
    missing_numeric_penalty: float = 1.0e12

    def tolerance_for(self, metric: str) -> tuple[float, float]:
        """Return ``(atol, rtol)`` for one numeric metric."""

        absolute = float(
            self.absolute_tolerances.get(metric, self.absolute_tolerance)
        )
        relative = float(
            self.relative_tolerances.get(metric, self.relative_tolerance)
        )
        return absolute, relative


def _safe_relative_error(actual: float, reference: float) -> float:
    """Calculate a signed relative error while handling a zero reference."""

    if reference == 0.0:
        return 0.0 if actual == 0.0 else float("inf")

    return (actual - reference) / abs(reference)


def _absolute_relative_error(actual: float, reference: float) -> float:
    """Calculate a non-negative relative error for acceptance reports."""

    return abs(_safe_relative_error(actual, reference))


def _bounding_box(shape: cq.Shape) -> dict[str, float]:
    box = shape.BoundingBox()
    return {
        "xmin": box.xmin,
        "xmax": box.xmax,
        "ymin": box.ymin,
        "ymax": box.ymax,
        "zmin": box.zmin,
        "zmax": box.zmax,
        "x_length": box.xlen,
        "y_length": box.ylen,
        "z_length": box.zlen,
    }


def _numeric_value(value: Any) -> float | None:
    """Convert one CSV cell to a finite float, or return ``None``."""

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric) or not np.isfinite(float(numeric)):
        return None

    return float(numeric)


def _normalized_text(value: Any) -> str:
    """Normalize categorical CSV values without conflating missing and text."""

    if pd.isna(value):
        return "<MISSING>"

    return str(value).strip()


def _parent_signatures(
    frame: pd.DataFrame,
    resolve_parent_to_class: bool,
) -> list[str]:
    """Express each parent as a semantic class where the CSV permits it.

    Rhino parent values embed coordinates and Rhino-era names, while CadQuery
    parents can contain deterministic logical IDs. Both normally point to a
    row in the same CSV. Resolving that row to its ``Class`` preserves the
    parent relationship without relying on either ID convention.
    """

    if not resolve_parent_to_class:
        return [_normalized_text(value) for value in frame["Parent_Struct"]]

    id_to_classes: dict[str, set[str]] = {}

    for object_id, element_class in zip(
        frame["Object_ID"],
        frame["Class"],
        strict=False,
    ):
        normalized_id = _normalized_text(object_id)

        if normalized_id == "<MISSING>":
            continue

        id_to_classes.setdefault(normalized_id, set()).add(
            _normalized_text(element_class)
        )

    signatures = []

    for parent in frame["Parent_Struct"]:
        normalized_parent = _normalized_text(parent)
        parent_classes = id_to_classes.get(normalized_parent)

        if parent_classes:
            signatures.append("|".join(sorted(parent_classes)))
        else:
            # ``Hull`` and ``Bracket`` are intentional non-row parents.
            signatures.append(normalized_parent)

    return signatures


def _prepared_rows(
    frame: pd.DataFrame,
    config: ElementComparisonConfig,
) -> pd.DataFrame:
    """Add stable positional and semantic keys without changing input data."""

    prepared = frame.copy()
    prepared["__row_position"] = np.arange(len(frame), dtype=int)
    prepared["__source_index"] = list(frame.index)
    prepared["__class_key"] = [
        _normalized_text(value) for value in frame["Class"]
    ]
    prepared["__type_key"] = [
        _normalized_text(value) for value in frame["Type"]
    ]
    prepared["__parent_key"] = _parent_signatures(
        frame,
        resolve_parent_to_class=config.resolve_parent_to_class,
    )
    return prepared


def _match_cost_matrix(
    generated_group: pd.DataFrame,
    reference_group: pd.DataFrame,
    config: ElementComparisonConfig,
) -> np.ndarray:
    """Return vectorized location/dimension costs for one semantic group."""

    generated_count = len(generated_group)
    reference_count = len(reference_group)
    total_cost = np.zeros((generated_count, reference_count), dtype=float)
    fields_used = np.zeros((generated_count, reference_count), dtype=int)

    for metric in config.match_numeric_fields:
        generated_values = pd.to_numeric(
            generated_group.get(metric),
            errors="coerce",
        ).to_numpy(dtype=float)
        reference_values = pd.to_numeric(
            reference_group.get(metric),
            errors="coerce",
        ).to_numpy(dtype=float)
        generated_grid = generated_values[:, np.newaxis]
        reference_grid = reference_values[np.newaxis, :]
        generated_valid = np.isfinite(generated_grid)
        reference_valid = np.isfinite(reference_grid)
        both_missing = ~generated_valid & ~reference_valid
        one_missing = generated_valid ^ reference_valid
        both_valid = generated_valid & reference_valid
        fields_used += ~both_missing
        total_cost[one_missing] += config.missing_numeric_penalty

        absolute_tolerance, relative_tolerance = config.tolerance_for(metric)

        # A floor of one metre keeps zero-valued location coordinates from
        # dominating the assignment solely because their relative scale is 0.
        value_scale = np.maximum(
            np.maximum(np.abs(generated_grid), np.abs(reference_grid)),
            1.0,
        )
        normalizer = np.maximum(
            np.maximum(absolute_tolerance, relative_tolerance * value_scale),
            np.finfo(float).eps,
        )
        normalized_difference = (
            np.abs(generated_grid - reference_grid) / normalizer
        )
        total_cost[both_valid] += normalized_difference[both_valid]

    # Groups with no available numeric match fields remain interchangeable.
    # Division is restricted to populated cells to avoid warnings and NaNs.
    costs = np.zeros_like(total_cost)
    populated = fields_used > 0
    costs[populated] = total_cost[populated] / fields_used[populated]

    return costs


def match_structural_elements(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    config: ElementComparisonConfig | None = None,
) -> pd.DataFrame:
    """Match generated and Rhino rows without using either artifact's IDs.

    Assignment is solved independently for every semantic
    ``Class``/``Type``/parent group. A rectangular assignment pairs as many
    rows as possible; surplus rows are emitted explicitly as
    ``generated_only`` or ``reference_only``.
    """

    comparison_config = config or ElementComparisonConfig()
    generated_rows = _prepared_rows(generated, comparison_config)
    reference_rows = _prepared_rows(reference, comparison_config)
    key_columns = ("__class_key", "__type_key", "__parent_key")

    generated_groups = {
        key: group
        for key, group in generated_rows.groupby(
            list(key_columns),
            dropna=False,
            sort=False,
        )
    }
    reference_groups = {
        key: group
        for key, group in reference_rows.groupby(
            list(key_columns),
            dropna=False,
            sort=False,
        )
    }

    records: list[dict[str, Any]] = []
    match_number = 0

    for key in sorted(set(generated_groups) | set(reference_groups)):
        generated_group = generated_groups.get(key)
        reference_group = reference_groups.get(key)
        generated_positions = (
            generated_group["__row_position"].astype(int).tolist()
            if generated_group is not None
            else []
        )
        reference_positions = (
            reference_group["__row_position"].astype(int).tolist()
            if reference_group is not None
            else []
        )
        paired_generated: set[int] = set()
        paired_reference: set[int] = set()

        if generated_positions and reference_positions:
            costs = _match_cost_matrix(
                generated_group,
                reference_group,
                comparison_config,
            )

            generated_assignment, reference_assignment = linear_sum_assignment(
                costs
            )

            for generated_index, reference_index in zip(
                generated_assignment,
                reference_assignment,
                strict=True,
            ):
                generated_position = generated_positions[int(generated_index)]
                reference_position = reference_positions[int(reference_index)]
                generated_row = generated_rows.iloc[generated_position]
                reference_row = reference_rows.iloc[reference_position]
                paired_generated.add(generated_position)
                paired_reference.add(reference_position)

                records.append(
                    {
                        "match_id": match_number,
                        "match_status": "matched",
                        "generated_position": generated_position,
                        "reference_position": reference_position,
                        "generated_index": generated_row["__source_index"],
                        "reference_index": reference_row["__source_index"],
                        "generated_object_id": generated_row.get("Object_ID"),
                        "reference_object_id": reference_row.get("Object_ID"),
                        "Class": key[0],
                        "Type": key[1],
                        "parent_signature": key[2],
                        "generated_parent": generated_row.get("Parent_Struct"),
                        "reference_parent": reference_row.get("Parent_Struct"),
                        "matching_cost": float(
                            costs[int(generated_index), int(reference_index)]
                        ),
                    }
                )
                match_number += 1

        for generated_position in generated_positions:
            if generated_position in paired_generated:
                continue

            generated_row = generated_rows.iloc[generated_position]
            records.append(
                {
                    "match_id": match_number,
                    "match_status": "generated_only",
                    "generated_position": generated_position,
                    "reference_position": None,
                    "generated_index": generated_row["__source_index"],
                    "reference_index": None,
                    "generated_object_id": generated_row.get("Object_ID"),
                    "reference_object_id": None,
                    "Class": key[0],
                    "Type": key[1],
                    "parent_signature": key[2],
                    "generated_parent": generated_row.get("Parent_Struct"),
                    "reference_parent": None,
                    "matching_cost": None,
                }
            )
            match_number += 1

        for reference_position in reference_positions:
            if reference_position in paired_reference:
                continue

            reference_row = reference_rows.iloc[reference_position]
            records.append(
                {
                    "match_id": match_number,
                    "match_status": "reference_only",
                    "generated_position": None,
                    "reference_position": reference_position,
                    "generated_index": None,
                    "reference_index": reference_row["__source_index"],
                    "generated_object_id": None,
                    "reference_object_id": reference_row.get("Object_ID"),
                    "Class": key[0],
                    "Type": key[1],
                    "parent_signature": key[2],
                    "generated_parent": None,
                    "reference_parent": reference_row.get("Parent_Struct"),
                    "matching_cost": None,
                }
            )
            match_number += 1

    return pd.DataFrame(records)


def compare_structural_elements_elementwise(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    config: ElementComparisonConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare matched rows field-by-field and report all unmatched rows."""

    comparison_config = config or ElementComparisonConfig()
    matches = match_structural_elements(
        generated,
        reference,
        config=comparison_config,
    )
    details: list[dict[str, Any]] = []
    pair_results: dict[int, bool] = {}

    for match in matches.to_dict(orient="records"):
        common = {
            "match_id": int(match["match_id"]),
            "match_status": match["match_status"],
            "generated_index": match["generated_index"],
            "reference_index": match["reference_index"],
            "generated_object_id": match["generated_object_id"],
            "reference_object_id": match["reference_object_id"],
            "Class": match["Class"],
            "Type": match["Type"],
            "parent_signature": match["parent_signature"],
            "generated_parent": match["generated_parent"],
            "reference_parent": match["reference_parent"],
            "matching_cost": match["matching_cost"],
        }

        if match["match_status"] != "matched":
            details.append(
                {
                    **common,
                    "metric": "__row__",
                    "generated": None,
                    "reference": None,
                    "absolute_error": None,
                    "relative_error": None,
                    "signed_relative_error": None,
                    "absolute_tolerance": None,
                    "relative_tolerance": None,
                    "comparison_status": "unmatched_row",
                    "passed": False,
                    "pair_passed": False,
                }
            )
            pair_results[int(match["match_id"])] = False
            continue

        generated_row = generated.iloc[int(match["generated_position"])]
        reference_row = reference.iloc[int(match["reference_position"])]
        field_records: list[dict[str, Any]] = []

        for metric in comparison_config.compare_numeric_fields:
            generated_value = _numeric_value(generated_row.get(metric))
            reference_value = _numeric_value(reference_row.get(metric))
            absolute_tolerance, relative_tolerance = (
                comparison_config.tolerance_for(metric)
            )

            if generated_value is None and reference_value is None:
                comparison_status = "both_missing"
                absolute_error = None
                relative_error = None
                signed_relative_error = None
                passed = True
            elif generated_value is None or reference_value is None:
                comparison_status = "missing_value"
                absolute_error = None
                relative_error = None
                signed_relative_error = None
                passed = False
            else:
                comparison_status = "compared"
                absolute_error = abs(generated_value - reference_value)
                relative_error = _absolute_relative_error(
                    generated_value,
                    reference_value,
                )
                signed_relative_error = _safe_relative_error(
                    generated_value,
                    reference_value,
                )
                passed = isclose(
                    generated_value,
                    reference_value,
                    abs_tol=absolute_tolerance,
                    rel_tol=relative_tolerance,
                )

            field_records.append(
                {
                    **common,
                    "metric": metric,
                    "generated": generated_value,
                    "reference": reference_value,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "signed_relative_error": signed_relative_error,
                    "absolute_tolerance": absolute_tolerance,
                    "relative_tolerance": relative_tolerance,
                    "comparison_status": comparison_status,
                    "passed": bool(passed),
                }
            )

        pair_passed = all(record["passed"] for record in field_records)
        pair_results[int(match["match_id"])] = pair_passed

        for field_record in field_records:
            field_record["pair_passed"] = pair_passed
            details.append(field_record)

    detail_frame = pd.DataFrame(details)
    matched_pairs = int((matches["match_status"] == "matched").sum())
    generated_only = int((matches["match_status"] == "generated_only").sum())
    reference_only = int((matches["match_status"] == "reference_only").sum())
    compared_metrics = int(
        (detail_frame.get("comparison_status") == "compared").sum()
    )
    failed_metrics = int(
        (
            (
                detail_frame.get("comparison_status").isin(
                    ["compared", "missing_value"]
                )
            )
            & (~detail_frame.get("passed").astype(bool))
        ).sum()
    )
    matched_pair_passes = sum(
        1
        for match_id, passed in pair_results.items()
        if passed
        and matches.loc[matches["match_id"] == match_id, "match_status"].iloc[0]
        == "matched"
    )

    summary = {
        "generated_rows": int(len(generated)),
        "reference_rows": int(len(reference)),
        "matched_pairs": matched_pairs,
        "generated_only_rows": generated_only,
        "reference_only_rows": reference_only,
        "compared_numeric_values": compared_metrics,
        "failed_numeric_values": failed_metrics,
        "matched_pairs_passed": int(matched_pair_passes),
        "matched_pairs_failed": int(matched_pairs - matched_pair_passes),
        "all_rows_matched": generated_only == 0 and reference_only == 0,
        "all_numeric_values_passed": failed_metrics == 0,
        "passed": (
            generated_only == 0
            and reference_only == 0
            and failed_metrics == 0
            and matched_pair_passes == matched_pairs
        ),
        "absolute_tolerance": comparison_config.absolute_tolerance,
        "relative_tolerance": comparison_config.relative_tolerance,
        "absolute_tolerance_overrides": {
            str(metric): float(value)
            for metric, value in comparison_config.absolute_tolerances.items()
        },
        "relative_tolerance_overrides": {
            str(metric): float(value)
            for metric, value in comparison_config.relative_tolerances.items()
        },
        "match_numeric_fields": list(comparison_config.match_numeric_fields),
        "compare_numeric_fields": list(comparison_config.compare_numeric_fields),
        "parent_matching": (
            "resolved_parent_class"
            if comparison_config.resolve_parent_to_class
            else "exact_parent_text"
        ),
        "object_ids_used_for_matching": False,
    }

    return detail_frame, summary


def _comparison_row(
    *,
    category: str,
    metric: str,
    generated_value: float,
    reference_value: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Build one consistently formatted scalar comparison row."""

    return {
        "category": category,
        "metric": metric,
        "generated": generated_value,
        "reference": reference_value,
        "absolute_error": abs(generated_value - reference_value),
        "relative_error": _absolute_relative_error(
            generated_value,
            reference_value,
        ),
        "signed_relative_error": _safe_relative_error(
            generated_value,
            reference_value,
        ),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "passed": isclose(
            generated_value,
            reference_value,
            abs_tol=absolute_tolerance,
            rel_tol=relative_tolerance,
        ),
    }


def compare_geometry(
    generated_shape: cq.Shape,
    reference_shape: cq.Shape,
    absolute_tolerance: float = 1.0e-6,
    relative_tolerance: float = 1.0e-6,
) -> pd.DataFrame:
    """Compare validity, topology counts, areas, and bounding-box dimensions."""

    generated_box = _bounding_box(generated_shape)
    reference_box = _bounding_box(reference_shape)
    rows: list[dict[str, Any]] = []

    for metric in generated_box:
        rows.append(
            _comparison_row(
                category="bounding_box",
                metric=metric,
                generated_value=generated_box[metric],
                reference_value=reference_box[metric],
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        )

    scalar_metrics = {
        "surface_area": (generated_shape.Area(), reference_shape.Area()),
        "face_count": (len(generated_shape.Faces()), len(reference_shape.Faces())),
        "edge_count": (len(generated_shape.Edges()), len(reference_shape.Edges())),
        "is_valid": (int(generated_shape.isValid()), int(reference_shape.isValid())),
    }

    for metric, (generated_value, reference_value) in scalar_metrics.items():
        rows.append(
            _comparison_row(
                category="topology",
                metric=metric,
                generated_value=float(generated_value),
                reference_value=float(reference_value),
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        )

    return pd.DataFrame(rows)


def compare_structural_elements_csv(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    absolute_tolerance: float = 1.0e-6,
    relative_tolerance: float = 1.0e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare schemas, class counts, and class-level physical aggregates."""

    generated_columns = list(generated.columns)
    reference_columns = list(reference.columns)
    schema = {
        "generated_columns": generated_columns,
        "reference_columns": reference_columns,
        "expected_columns": RHINO_CSV_COLUMNS,
        "generated_matches_expected": generated_columns == RHINO_CSV_COLUMNS,
        "reference_matches_expected": reference_columns == RHINO_CSV_COLUMNS,
        "same_column_order": generated_columns == reference_columns,
        "generated_object_ids_unique": bool(generated["Object_ID"].is_unique),
        "reference_object_ids_unique": bool(reference["Object_ID"].is_unique),
    }

    class_names = sorted(
        set(generated["Class"].dropna().astype(str))
        | set(reference["Class"].dropna().astype(str))
    )
    rows: list[dict[str, Any]] = []

    for class_name in class_names:
        generated_class = generated[generated["Class"].astype(str) == class_name]
        reference_class = reference[reference["Class"].astype(str) == class_name]

        for metric in ("row_count", "area_pannel", "volume_pannel"):
            if metric == "row_count":
                generated_value = float(len(generated_class))
                reference_value = float(len(reference_class))
            else:
                generated_value = float(
                    pd.to_numeric(generated_class[metric], errors="coerce").sum()
                )
                reference_value = float(
                    pd.to_numeric(reference_class[metric], errors="coerce").sum()
                )

            row = _comparison_row(
                category="class_aggregate",
                metric=metric,
                generated_value=generated_value,
                reference_value=reference_value,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            row["Class"] = class_name
            rows.append(row)

    return pd.DataFrame(rows), schema


def normalize_legacy_physical_rows(
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse Rhino's mesh-split bottom-frame rows to physical members.

    ``correct_1D_Elements`` wrote one structural CSV row for every mesh curve
    segment while leaving both rows tied to the same ``struct_rsObject``.
    Geometry and weight are physical-member quantities, so the CadQuery port
    emits one row per repeated structural UUID. Other Rhino duplicate names,
    notably independently clipped deck pieces, are deliberately preserved.
    """

    normalized = reference.copy()
    bottom_mask = (
        normalized["Class"].astype(str).str.strip()
        == "Transverse_Bottom_Frame"
    )
    valid_structural_id = normalized["struct_rsObject"].notna()
    bottom_frames = normalized.loc[bottom_mask & valid_structural_id]
    repeated_structures = bottom_frames[
        bottom_frames.duplicated("struct_rsObject", keep=False)
    ]
    identity_columns = {"mesh_rsObject", "Object_ID"}
    payload_columns = [
        column
        for column in reference.columns
        if column not in identity_columns
    ]
    payloads_identical = True

    for _, group in repeated_structures.groupby(
        "struct_rsObject",
        dropna=False,
        sort=False,
    ):
        if any(
            group[column].nunique(dropna=False) > 1
            for column in payload_columns
        ):
            payloads_identical = False
            break

    duplicate_mask = bottom_mask & valid_structural_id & normalized.duplicated(
        "struct_rsObject",
        keep="first",
    )
    normalized = normalized.loc[~duplicate_mask].reset_index(drop=True)
    group_count = int(repeated_structures["struct_rsObject"].nunique())
    rows_removed = int(duplicate_mask.sum())

    summary = {
        "strategy": (
            "Collapse repeated Transverse_Bottom_Frame rows sharing one "
            "Rhino struct_rsObject; preserve all other rows."
        ),
        "reference_rows_before": int(len(reference)),
        "reference_rows_after": int(len(normalized)),
        "bottom_frame_physical_groups_collapsed": group_count,
        "bottom_frame_mesh_rows_removed": rows_removed,
        "duplicate_payloads_identical_except_mesh_and_object_ids": bool(
            payloads_identical
        ),
        "normalization_is_intentional_cadquery_difference": True,
    }

    return normalized, summary


def screen_reference_unit_consistency(
    reference_shape: cq.Shape,
    reference_csv: pd.DataFrame,
    absolute_tolerance: float = 1.0e-3,
    relative_tolerance: float = 2.0e-2,
    scale_detection_tolerance: float = 5.0e-2,
) -> dict[str, Any]:
    """Check CAD longitudinal extent against the Side Shell CSV length.

    A ratio near 1000 (or its reciprocal) is labelled explicitly. Missing
    Side Shell data is an indeterminate, fail-safe result rather than a silent
    pass because the CAD file's unit convention then cannot be established.
    """

    x_length = float(reference_shape.BoundingBox().xlen)

    if "Class" not in reference_csv or "L1" not in reference_csv:
        return {
            "status": "unable_to_screen",
            "passed": False,
            "reason": "Reference CSV does not contain Class and L1 columns.",
            "cad_bbox_x_length": x_length,
            "side_shell_l1": None,
            "cad_to_csv_length_ratio": None,
            "suspected_scale_factor": None,
        }

    side_shell = reference_csv[
        reference_csv["Class"].astype(str).str.strip() == "Side_Shell"
    ]
    lengths = pd.to_numeric(side_shell["L1"], errors="coerce")
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]

    if lengths.empty:
        return {
            "status": "unable_to_screen",
            "passed": False,
            "reason": "Reference CSV contains no positive Side_Shell L1 value.",
            "cad_bbox_x_length": x_length,
            "side_shell_l1": None,
            "cad_to_csv_length_ratio": None,
            "suspected_scale_factor": None,
        }

    side_shell_l1 = float(lengths.median())
    ratio = x_length / side_shell_l1
    absolute_error = abs(x_length - side_shell_l1)
    relative_error = _absolute_relative_error(x_length, side_shell_l1)
    passed = isclose(
        x_length,
        side_shell_l1,
        abs_tol=absolute_tolerance,
        rel_tol=relative_tolerance,
    )
    suspected_scale_factor: float | None = None
    status = "consistent"
    reason = "CAD x extent and Side_Shell L1 use a consistent metre scale."

    if not passed:
        status = "inconsistent"
        reason = "CAD x extent does not agree with the Side_Shell L1 metadata."

        if isclose(
            ratio,
            1000.0,
            abs_tol=0.0,
            rel_tol=scale_detection_tolerance,
        ):
            suspected_scale_factor = 1000.0
            status = "suspected_1000x_cad_scale"
            reason = (
                "CAD x extent is approximately 1000 times Side_Shell L1; "
                "the reference CAD artifact is likely encoded at the wrong scale."
            )
        elif isclose(
            ratio,
            0.001,
            abs_tol=0.0,
            rel_tol=scale_detection_tolerance,
        ):
            suspected_scale_factor = 0.001
            status = "suspected_0.001x_cad_scale"
            reason = (
                "CAD x extent is approximately 0.001 times Side_Shell L1; "
                "the reference CAD artifact is likely encoded at the wrong scale."
            )

    return {
        "status": status,
        "passed": bool(passed),
        "reason": reason,
        "cad_bbox_x_length": x_length,
        "side_shell_l1": side_shell_l1,
        "side_shell_rows_used": int(len(lengths)),
        "cad_to_csv_length_ratio": ratio,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "suspected_scale_factor": suspected_scale_factor,
    }


def load_cad(path: str | Path) -> cq.Shape:
    """Load a supported reference CAD file using its suffix."""

    cad_path = Path(path)
    suffix = cad_path.suffix.lower()

    if suffix in {".igs", ".iges"}:
        return load_iges(cad_path)

    if suffix in {".step", ".stp"}:
        return load_step(cad_path)

    raise ValueError(f"Unsupported CAD reference suffix: {cad_path.suffix}")


def _write_validation_reports(
    generated_shape: cq.Shape,
    reference_shape: cq.Shape,
    generated_csv: pd.DataFrame,
    reference_csv: pd.DataFrame,
    output: Path,
    name: str,
    comparison_config: ElementComparisonConfig,
    write_previews: bool,
) -> dict[str, Path]:
    """Write every report shared by in-memory and file-based validation."""

    geometry_comparison = compare_geometry(
        generated_shape,
        reference_shape,
        absolute_tolerance=comparison_config.absolute_tolerance,
        relative_tolerance=comparison_config.relative_tolerance,
    )
    csv_comparison, schema = compare_structural_elements_csv(
        generated_csv,
        reference_csv,
        absolute_tolerance=comparison_config.absolute_tolerance,
        relative_tolerance=comparison_config.relative_tolerance,
    )
    element_comparison, element_summary = (
        compare_structural_elements_elementwise(
            generated_csv,
            reference_csv,
            config=comparison_config,
        )
    )
    unit_screen = screen_reference_unit_consistency(
        reference_shape,
        reference_csv,
    )
    physical_reference_csv, normalization_summary = (
        normalize_legacy_physical_rows(reference_csv)
    )
    physical_csv_comparison, _ = compare_structural_elements_csv(
        generated_csv,
        physical_reference_csv,
        absolute_tolerance=comparison_config.absolute_tolerance,
        relative_tolerance=comparison_config.relative_tolerance,
    )
    physical_element_comparison, physical_element_summary = (
        compare_structural_elements_elementwise(
            generated_csv,
            physical_reference_csv,
            config=comparison_config,
        )
    )

    geometry_path = output / f"{name}_geometry_comparison.csv"
    csv_path = output / f"{name}_structural_csv_comparison.csv"
    element_path = output / f"{name}_element_comparison.csv"
    schema_path = output / f"{name}_schema_summary.json"
    element_summary_path = output / f"{name}_element_match_summary.json"
    unit_screen_path = output / f"{name}_reference_unit_screen.json"
    physical_csv_path = output / f"{name}_physical_structural_csv_comparison.csv"
    physical_element_path = output / f"{name}_physical_element_comparison.csv"
    physical_element_summary_path = (
        output / f"{name}_physical_element_match_summary.json"
    )
    normalization_summary_path = (
        output / f"{name}_reference_normalization_summary.json"
    )

    geometry_comparison.to_csv(geometry_path, index=False)
    csv_comparison.to_csv(csv_path, index=False)
    element_comparison.to_csv(element_path, index=False)
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    element_summary_path.write_text(
        json.dumps(element_summary, indent=2),
        encoding="utf-8",
    )
    unit_screen_path.write_text(
        json.dumps(unit_screen, indent=2),
        encoding="utf-8",
    )
    physical_csv_comparison.to_csv(physical_csv_path, index=False)
    physical_element_comparison.to_csv(physical_element_path, index=False)
    physical_element_summary_path.write_text(
        json.dumps(physical_element_summary, indent=2),
        encoding="utf-8",
    )
    normalization_summary_path.write_text(
        json.dumps(normalization_summary, indent=2),
        encoding="utf-8",
    )

    reports = {
        "geometry_comparison": geometry_path,
        "structural_csv_comparison": csv_path,
        "element_comparison": element_path,
        "schema_summary": schema_path,
        "element_match_summary": element_summary_path,
        "reference_unit_screen": unit_screen_path,
        "physical_structural_csv_comparison": physical_csv_path,
        "physical_element_comparison": physical_element_path,
        "physical_element_match_summary": physical_element_summary_path,
        "reference_normalization_summary": normalization_summary_path,
    }

    if write_previews:
        # Importing here keeps Matplotlib out of non-rendering validation paths.
        from .rendering import (
            write_geometry_previews,
            write_transverse_section_preview,
        )

        preview_directory = output / "previews"
        generated_previews = write_geometry_previews(
            generated_shape,
            preview_directory,
            prefix=f"{name}_generated",
        )
        reference_previews = write_geometry_previews(
            reference_shape,
            preview_directory,
            prefix=f"{name}_reference",
        )
        generated_section = write_transverse_section_preview(
            generated_shape,
            preview_directory / f"{name}_generated_section.png",
            title=f"{name} generated — midship transverse section",
        )
        reference_section = write_transverse_section_preview(
            reference_shape,
            preview_directory / f"{name}_reference_section.png",
            title=f"{name} reference — midship transverse section",
        )

        for view, path in generated_previews.items():
            reports[f"generated_preview_{view}"] = path

        for view, path in reference_previews.items():
            reports[f"reference_preview_{view}"] = path

        reports["generated_preview_section"] = generated_section
        reports["reference_preview_section"] = reference_section

    return reports


def validate_against_reference(
    structure: ShipStructure,
    reference_cad_path: str | Path,
    reference_csv_path: str | Path,
    output_directory: str | Path,
    report_name: str | None = None,
    *,
    comparison_config: ElementComparisonConfig | None = None,
    write_previews: bool = True,
) -> dict[str, Path]:
    """Write geometry, element, unit, schema, and preview reports."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    name = report_name or structure.ship_id

    return _write_validation_reports(
        generated_shape=structure.shape(),
        reference_shape=load_cad(reference_cad_path),
        generated_csv=structure.structural_elements_dataframe(),
        reference_csv=pd.read_csv(reference_csv_path),
        output=output,
        name=name,
        comparison_config=comparison_config or ElementComparisonConfig(),
        write_previews=write_previews,
    )


def validate_files_against_reference(
    generated_cad_path: str | Path,
    generated_csv_path: str | Path,
    reference_cad_path: str | Path,
    reference_csv_path: str | Path,
    output_directory: str | Path,
    report_name: str,
    *,
    comparison_config: ElementComparisonConfig | None = None,
    write_previews: bool = True,
) -> dict[str, Path]:
    """Write the same reports when generated artifacts are already on disk."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    return _write_validation_reports(
        generated_shape=load_cad(generated_cad_path),
        reference_shape=load_cad(reference_cad_path),
        generated_csv=pd.read_csv(generated_csv_path),
        reference_csv=pd.read_csv(reference_csv_path),
        output=output,
        name=report_name,
        comparison_config=comparison_config or ElementComparisonConfig(),
        write_previews=write_previews,
    )
