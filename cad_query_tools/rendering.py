"""Headless, repeatable wireframe previews for CadQuery validation reports."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Callable

import cadquery as cq
import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt


# Force a noninteractive backend and a writable cache before pyplot is loaded.
# This keeps the renderer usable from tests, CI, and terminal-only workflows.
_matplotlib_cache = Path(tempfile.gettempdir()) / "genisys-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


Projection = Callable[[np.ndarray], np.ndarray]


def _top(points: np.ndarray) -> np.ndarray:
    """Project along vertical ``z`` onto the longitudinal/transverse plane."""

    return points[:, [0, 1]]


def _side(points: np.ndarray) -> np.ndarray:
    """Project along transverse ``y`` onto the longitudinal/vertical plane."""

    return points[:, [0, 2]]


def _bow(points: np.ndarray) -> np.ndarray:
    """Project along longitudinal ``x`` as viewed from the bow."""

    return points[:, [1, 2]]


def _stern(points: np.ndarray) -> np.ndarray:
    """Project along longitudinal ``x`` from the opposite direction."""

    projected = points[:, [1, 2]].copy()
    projected[:, 0] *= -1.0
    return projected


def _isometric(points: np.ndarray) -> np.ndarray:
    """Apply a fixed orthographic isometric projection."""

    # The two basis vectors are orthonormal and keep global +z upward.
    screen_right = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    screen_up = np.array([-1.0, 1.0, 2.0]) / np.sqrt(6.0)
    return np.column_stack((points @ screen_right, points @ screen_up))


_VIEW_DEFINITIONS: dict[str, tuple[Projection, str, str]] = {
    "isometric": (_isometric, "projected horizontal (m)", "projected vertical (m)"),
    "top": (_top, "x (m)", "y (m)"),
    "side": (_side, "x (m)", "z (m)"),
    "bow": (_bow, "y (m)", "z (m)"),
    "stern": (_stern, "-y (m)", "z (m)"),
}


def _edge_sort_key(edge: cq.Edge) -> tuple[float, ...]:
    """Return stable bounds so repeated renders use the same draw order."""

    bounds = edge.BoundingBox()
    return (
        bounds.xmin,
        bounds.ymin,
        bounds.zmin,
        bounds.xmax,
        bounds.ymax,
        bounds.zmax,
    )


def _sample_edge(edge: cq.Edge, curved_edge_points: int) -> np.ndarray:
    """Sample a topology edge into an XYZ polyline."""

    point_count = 2 if edge.geomType() == "LINE" else curved_edge_points

    try:
        points, _ = edge.sample(point_count)
    except (RuntimeError, ValueError):
        # Degenerate reference edges occasionally reject curve sampling. Their
        # endpoints still provide a useful, explicit preview segment.
        points = [edge.startPoint(), edge.endPoint()]

    coordinates = np.asarray([point.toTuple() for point in points], dtype=float)

    if len(coordinates) == 1:
        coordinates = np.vstack((coordinates, coordinates))

    return coordinates


def wireframe_segments(
    shape: cq.Shape,
    curved_edge_points: int = 20,
) -> list[np.ndarray]:
    """Return stable XYZ polylines for all unique topology edges in a shape."""

    return [
        _sample_edge(edge, curved_edge_points)
        for edge in sorted(shape.Edges(), key=_edge_sort_key)
    ]


def _set_equal_limits(axis, projected_segments: list[np.ndarray]) -> None:
    """Use equal scale and a fixed margin without depending on autoscale state."""

    all_points = np.vstack(projected_segments)
    minimum = np.min(all_points, axis=0)
    maximum = np.max(all_points, axis=0)
    centre = (minimum + maximum) / 2.0
    span = max(float(np.max(maximum - minimum)), 1.0e-9)
    half_span = span * 0.525

    axis.set_xlim(centre[0] - half_span, centre[0] + half_span)
    axis.set_ylim(centre[1] - half_span, centre[1] + half_span)
    axis.set_aspect("equal", adjustable="box")


def _write_projected_preview(
    segments: list[np.ndarray],
    output_path: str | Path,
    view: str,
    *,
    title: str | None = None,
    dpi: int = 160,
) -> Path:
    """Project already sampled edges and write one deterministic PNG."""

    try:
        projection, x_label, y_label = _VIEW_DEFINITIONS[view]
    except KeyError as exc:
        options = ", ".join(_VIEW_DEFINITIONS)
        raise ValueError(f"Unsupported preview view {view!r}; use {options}") from exc

    path = Path(output_path)

    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")

    path.parent.mkdir(parents=True, exist_ok=True)

    if not segments:
        raise ValueError("Cannot render a geometry preview without topology edges")

    projected_segments = [projection(segment) for segment in segments]
    figure, axis = plt.subplots(figsize=(8.0, 5.5), dpi=dpi)
    collection = LineCollection(
        projected_segments,
        colors="#17324d",
        linewidths=0.45,
        alpha=0.9,
    )
    axis.add_collection(collection)
    _set_equal_limits(axis, projected_segments)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title or f"CadQuery geometry — {view}")
    axis.grid(True, color="#d7dde3", linewidth=0.35)
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=dpi,
        facecolor="white",
        metadata={"Software": "GenISys CadQuery validation"},
    )
    plt.close(figure)
    return path


def write_geometry_preview(
    shape: cq.Shape,
    output_path: str | Path,
    view: str = "isometric",
    *,
    title: str | None = None,
    curved_edge_points: int = 20,
    dpi: int = 160,
) -> Path:
    """Write one deterministic PNG wireframe using a fixed camera projection."""

    segments = wireframe_segments(
        shape,
        curved_edge_points=curved_edge_points,
    )
    return _write_projected_preview(
        segments,
        output_path,
        view,
        title=title,
        dpi=dpi,
    )


def write_geometry_previews(
    shape: cq.Shape,
    output_directory: str | Path,
    prefix: str,
    views: tuple[str, ...] = ("isometric", "top", "side", "bow", "stern"),
    *,
    curved_edge_points: int = 20,
    dpi: int = 160,
) -> dict[str, Path]:
    """Write the standard isometric and orthographic validation views."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # Sampling OpenCascade edges is the expensive operation. Reuse one stable
    # topology pass across every camera instead of sampling five times.
    segments = wireframe_segments(
        shape,
        curved_edge_points=curved_edge_points,
    )

    for view in views:
        written[view] = _write_projected_preview(
            segments,
            output / f"{prefix}_{view}.png",
            view,
            title=f"{prefix} — {view}",
            dpi=dpi,
        )

    return written


def transverse_section_shape(
    shape: cq.Shape,
    x_position: float | None = None,
) -> tuple[cq.Shape, float]:
    """Intersect geometry with a transverse plane and return its section.

    The default station is the midpoint of the model's longitudinal bounds.
    A true OpenCascade section is used instead of projecting the entire model,
    so the resulting image exposes the members and plates actually crossing
    that station.
    """

    bounds = shape.BoundingBox()
    station = (
        (bounds.xmin + bounds.xmax) / 2.0
        if x_position is None
        else float(x_position)
    )

    # Construct a finite plane comfortably larger than the transverse model.
    # Plane coordinates are symmetric because the precise local axis choice
    # for a normal in +x is not material to the intersection.
    plane_half_size = 2.0 * max(bounds.ylen, bounds.zlen, 1.0)
    plane = gp_Pln(gp_Pnt(station, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
    plane_face = BRepBuilderAPI_MakeFace(
        plane,
        -plane_half_size,
        plane_half_size,
        -plane_half_size,
        plane_half_size,
    ).Face()
    operation = BRepAlgoAPI_Section(shape.wrapped, plane_face)
    operation.Build()

    if not operation.IsDone():
        raise RuntimeError(f"Unable to calculate transverse section at x={station}")

    section = cq.Shape.cast(operation.Shape())

    if not section.Edges():
        raise ValueError(f"Transverse section at x={station} contains no edges")

    return section, station


def write_transverse_section_preview(
    shape: cq.Shape,
    output_path: str | Path,
    x_position: float | None = None,
    *,
    title: str | None = None,
    curved_edge_points: int = 20,
    dpi: int = 160,
) -> Path:
    """Write a true transverse cut-section preview in the ``y-z`` plane."""

    section, station = transverse_section_shape(shape, x_position=x_position)

    return write_geometry_preview(
        section,
        output_path,
        view="bow",
        title=title or f"Transverse section at x={station:.3f} m",
        curved_edge_points=curved_edge_points,
        dpi=dpi,
    )
