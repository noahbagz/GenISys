"""Unit-safe IGES and STEP import/export helpers.

CadQuery models in this project use numeric coordinates in metres. CadQuery's
high-level STEP writer can declare that directly. OpenCascade's IGES writer,
however, treats incoming topology coordinates as millimetres before applying
its requested file unit. A scaled export-only copy therefore keeps IGES files
declared in metres without changing the in-memory model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cadquery as cq
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.IGESCAFControl import IGESCAFControl_Writer
from OCP.IGESControl import (
    IGESControl_Controller,
    IGESControl_Reader,
    IGESControl_Writer,
)
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.XmlXCAFDrivers import XmlXCAFDrivers
from OCP.XSControl import XSControl_WorkSession

from .model import ShipStructure


_FORMAT_ALIASES = {
    "igs": "iges",
    "iges": "iges",
    "stp": "step",
    "step": "step",
    "both": "both",
}


def _as_shape(
    structure: ShipStructure | cq.Shape | cq.Workplane,
    *,
    mesh: bool = False,
) -> cq.Shape:
    """Normalize the supported public inputs to a CadQuery shape."""

    if isinstance(structure, ShipStructure):
        return structure.shape(mesh=mesh)

    if isinstance(structure, cq.Workplane):
        value = structure.val()

        if not isinstance(value, cq.Shape):
            raise TypeError("The CadQuery Workplane does not contain a shape")

        return value

    if isinstance(structure, cq.Shape):
        return structure

    raise TypeError("structure must be a ShipStructure, cadquery.Shape, or Workplane")


def _normalized_base(output_path: str | Path) -> Path:
    """Remove any supported CAD suffix so format selection owns extensions."""

    path = Path(output_path)

    if path.suffix.lower() in {".igs", ".iges", ".stp", ".step"}:
        path = path.with_suffix("")

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_iges(shape: cq.Shape, output_path: Path) -> None:
    """Write metre-coordinate topology as a metre-declared IGES file."""

    # OCP assumes topology coordinates are millimetres. Scaling a copy by
    # 1000 causes a model coordinate of 1.0 m to be encoded as 1.0 m in IGES.
    export_shape = shape.scale(1000.0)
    writer = IGESControl_Writer("M", 0)

    if not writer.AddShape(export_shape.wrapped):
        raise RuntimeError("OpenCascade rejected the shape for IGES export")

    if not writer.Write(str(output_path)):
        raise RuntimeError(f"OpenCascade failed to write IGES file: {output_path}")


def _named_shapes(
    structure: ShipStructure,
    *,
    mesh: bool,
) -> list[tuple[str, cq.Shape]]:
    """Collect each separately named structural or mesh export entity."""

    named = []

    for element in structure.elements:
        if mesh:
            shape = element.mesh_shape
            name = element.mesh_object_name
        else:
            shape = element.shape
            name = element.structural_object_name

        if shape is not None and name is not None:
            named.append((name, shape))

    return named


def _new_xcaf_document() -> TDocStd_Document:
    """Create the OpenCascade document used to carry IGES entity names."""

    application = XCAFApp_Application.GetApplication_s()
    XmlXCAFDrivers.DefineFormat_s(application)
    document = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    application.InitDocument(document)

    return document


def _write_named_iges(
    named_shapes: Iterable[tuple[str, cq.Shape]],
    output_path: Path,
) -> None:
    """Write separately named metre-coordinate entities to one IGES file."""

    document = _new_xcaf_document()
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    shape_tool.SetAutoNaming_s(False)

    for name, shape in named_shapes:
        # Keep scaling export-only, exactly as in the unnamed IGES path.
        export_shape = shape.scale(1000.0)
        label = shape_tool.AddShape(export_shape.wrapped, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))

    # A selected IGES norm is required before using the unit-aware XCAF
    # constructor.  An uninitialized work session can otherwise write INF
    # coordinates while still reporting success.
    IGESControl_Controller.Init_s()
    work_session = XSControl_WorkSession()

    if not work_session.SelectNorm("IGES"):
        raise RuntimeError("OpenCascade could not initialize the IGES work session")

    writer = IGESCAFControl_Writer(work_session, "M")
    writer.SetNameMode(True)

    # XCAF transfer reads this process-wide setting as the unit of the source
    # topology.  A preceding metre-target import can leave it set to ``M``;
    # force the millimetre convention expected by the scaled export copy.
    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")

    if not writer.Transfer(document):
        raise RuntimeError("OpenCascade rejected the named shapes for IGES export")

    if not writer.Write(str(output_path)):
        raise RuntimeError(f"OpenCascade failed to write IGES file: {output_path}")


def _write_step(shape: cq.Shape, output_path: Path) -> None:
    """Write a STEP file whose model and output units are metres."""

    shape.exportStep(str(output_path), unit="M", outputUnit="M")


def save_structure(
    structure: ShipStructure | cq.Shape | cq.Workplane,
    output_path: str | Path,
    file_format: str = "igs",
    *,
    mesh: bool = False,
) -> list[Path]:
    """Save structural geometry as IGES, STEP, or both.

    Parameters
    ----------
    structure:
        Generated :class:`ShipStructure` or a CadQuery shape/workplane.
    output_path:
        File stem or a path with any supported CAD suffix.
    file_format:
        ``"igs"``/``"iges"`` (default), ``"step"``/``"stp"``, or ``"both"``.
    mesh:
        Export the mesh representation of a :class:`ShipStructure`.  This is
        ignored for an already selected CadQuery shape/workplane.

    Returns
    -------
    list[pathlib.Path]
        Paths written, in IGES-then-STEP order when ``both`` is selected.
    """

    try:
        normalized_format = _FORMAT_ALIASES[file_format.lower()]
    except KeyError as exc:
        options = ", ".join(sorted(_FORMAT_ALIASES))
        raise ValueError(f"Unsupported file format {file_format!r}; use {options}") from exc

    shape = _as_shape(structure, mesh=mesh)
    base = _normalized_base(output_path)
    written: list[Path] = []

    if normalized_format in {"iges", "both"}:
        iges_path = base.with_suffix(".igs")

        if isinstance(structure, ShipStructure):
            _write_named_iges(
                _named_shapes(structure, mesh=mesh),
                iges_path,
            )
        else:
            _write_iges(shape, iges_path)

        written.append(iges_path)

    if normalized_format in {"step", "both"}:
        step_path = base.with_suffix(".step")
        _write_step(shape, step_path)
        written.append(step_path)

    return written


def load_iges(input_path: str | Path) -> cq.Shape:
    """Load a metre-declared IGES file into metre-coordinate CadQuery topology."""

    # Constructing the first IGES reader initializes OpenCascade's controller
    # defaults and can overwrite an earlier target unit.  Conversely, an XCAF
    # name-reading session can leave a previously initialized controller bound
    # to millimetres.  Set the target on both sides of construction so either
    # call order imports metre-declared geometry into metre model coordinates.
    Interface_Static.SetCVal_s("xstep.cascade.unit", "M")
    reader = IGESControl_Reader()
    Interface_Static.SetCVal_s("xstep.cascade.unit", "M")
    status = reader.ReadFile(str(Path(input_path)))

    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"Could not read IGES file: {input_path}")

    if reader.TransferRoots() < 1:
        raise RuntimeError(f"IGES file contains no transferable roots: {input_path}")

    return cq.Shape.cast(reader.OneShape())


def load_step(input_path: str | Path) -> cq.Shape:
    """Load a STEP file using metres as CadQuery's numeric model unit."""

    # The first STEP reader can initialize OpenCascade controller defaults
    # after a caller has set the target unit.  Set it again after construction
    # so a first import and every later import both return metre coordinates.
    Interface_Static.SetCVal_s("xstep.cascade.unit", "M")
    reader = STEPControl_Reader()
    Interface_Static.SetCVal_s("xstep.cascade.unit", "M")
    status = reader.ReadFile(str(Path(input_path)))

    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"Could not read STEP file: {input_path}")

    if reader.TransferRoots() < 1:
        raise RuntimeError(f"STEP file contains no transferable roots: {input_path}")

    return cq.Shape.cast(reader.OneShape())


def bounding_box_lengths(shape: cq.Shape) -> tuple[float, float, float]:
    """Return x/y/z extents in metres for validation reports and tests."""

    box = shape.BoundingBox()
    return box.xlen, box.ylen, box.zlen
