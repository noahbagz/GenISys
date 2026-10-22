"""Write a viewable ship-system scene and repeatable, headless PNG views.

The full hull remains a mesh. CadQuery components are tessellated only for
display; neither the GLB nor the PNGs participate in structural mass or CG
calculations. GLB nodes retain each supplied component name and use metres.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import tempfile
from dataclasses import dataclass, replace

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray


# Matplotlib's Agg renderer works without the OpenGL context required by
# VTK's native rendering window on headless macOS machines.
_cache_dir = Path(tempfile.gettempdir()) / "genisys-matplotlib"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection


_VALID_STYLES = {"shaded", "wireframe"}
_VALID_VIEWS = {"isometric", "top", "side", "longitudinal_cutaway", "transverse_cutaway"}
_HULL_BLUE = (0.045, 0.18, 0.30)
_HULL_RED = (0.79, 0.24, 0.22)


@dataclass(frozen=True)
class _Mesh:
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    color: tuple[float, float, float]
    opacity: float


def _polydata_mesh(polydata: vtk.vtkPolyData, name: str,
                   color: tuple[float, float, float], opacity: float) -> _Mesh:
    """Copy a triangle-only VTK surface into NumPy arrays."""

    vertices = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float, copy=True)
    triangles = vtk_to_numpy(
        polydata.GetPolys().GetConnectivityArray()
    ).reshape(-1, 3).astype(np.int32, copy=True)

    return _Mesh(name, vertices, triangles, color, opacity)


def _read_hull(path: Path, waterline_z: float) -> tuple[list[_Mesh], list[_Mesh], np.ndarray]:
    """Split the source hull at the design waterline for consistent materials."""

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()

    if polydata.GetNumberOfPolys() == 0:
        raise ValueError(f"Hull STL contains no triangles: {path}")

    points = vtk_to_numpy(polydata.GetPoints().GetData())
    hull_bounds = np.stack((points.min(axis=0), points.max(axis=0)), axis=1)
    waterline = vtk.vtkPlane()
    waterline.SetOrigin(0.0, 0.0, waterline_z)
    waterline.SetNormal(0.0, 0.0, 1.0)
    full_hull = []
    image_hull = []

    for name, color, keep_below in (
        ("hull_above_waterline", _HULL_BLUE, False),
        ("hull_below_waterline", _HULL_RED, True),
    ):
        clip = vtk.vtkClipPolyData()
        clip.SetInputData(polydata)
        clip.SetClipFunction(waterline)
        clip.SetInsideOut(keep_below)
        clip.Update()
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(clip.GetOutputPort())
        triangles.Update()
        surface = triangles.GetOutput()
        full_mesh = _polydata_mesh(surface, name, color, 1.0)
        full_hull.append(full_mesh)

        # Decimate only the PNG copy; the GLB keeps the complete clipped mesh.
        if surface.GetNumberOfPolys() > 18_000:
            reducer = vtk.vtkQuadricDecimation()
            reducer.SetInputData(surface)
            reducer.SetTargetReduction(1.0 - 18_000 / surface.GetNumberOfPolys())
            reducer.Update()
            image_hull.append(_polydata_mesh(reducer.GetOutput(), name, color, 1.0))
        else:
            image_hull.append(full_mesh)

    return full_hull, image_hull, hull_bounds


def _tessellate_part(part: dict) -> _Mesh:
    """Tessellate one already positioned CadQuery display component."""

    vertices, triangles = part["shape"].tessellate(0.15)
    coordinates = np.asarray([vertex.toTuple() for vertex in vertices], dtype=float)

    return _Mesh(
        str(part["name"]),
        coordinates,
        np.asarray(triangles, dtype=np.int32).reshape(-1, 3),
        tuple(float(value) for value in part["color"]),
        float(part["opacity"]),
    )


def _vertex_normals(mesh: _Mesh) -> np.ndarray:
    """Average adjacent triangle normals for ordinary GLB surface lighting."""

    corners = mesh.vertices[mesh.triangles]
    face_normals = np.cross(
        corners[:, 1] - corners[:, 0],
        corners[:, 2] - corners[:, 0],
    )
    vertex_normals = np.zeros_like(mesh.vertices)

    for corner in range(3):
        np.add.at(vertex_normals, mesh.triangles[:, corner], face_normals)

    lengths = np.linalg.norm(vertex_normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    return vertex_normals / lengths[:, None]


def _write_glb(meshes: list[_Mesh], output_path: Path) -> Path:
    """Write a named, material-colored binary glTF scene without OpenGL."""

    document: dict = {
        "asset": {"version": "2.0", "generator": "GenISys system scene"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{
            "name": "GenISys ship system (metres)",
            # GLB viewers expect Y up. Preserve native X/Y/Z in buffers while
            # rotating the display root so GenISys +Z is visually upward.
            "rotation": [-0.7071067811865475, 0.0, 0.0, 0.7071067811865476],
            "children": [],
            "extras": {"source_axes": "GenISys X/Y/Z, Z up", "units": "m"},
        }],
        "meshes": [],
        "materials": [],
        "bufferViews": [],
        "accessors": [],
    }
    binary = bytearray()

    def add_accessor(values: np.ndarray, *, target: int, component_type: int,
                     value_type: str, bounds: bool = False) -> int:
        """Append an aligned binary block and its glTF view/accessor records."""

        binary.extend(b"\x00" * (-len(binary) % 4))
        offset = len(binary)
        binary.extend(values.tobytes(order="C"))
        view_index = len(document["bufferViews"])
        document["bufferViews"].append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": values.nbytes,
            "target": target,
        })
        accessor = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": len(values),
            "type": value_type,
        }

        if bounds:
            accessor["min"] = values.min(axis=0).astype(float).tolist()
            accessor["max"] = values.max(axis=0).astype(float).tolist()

        accessor_index = len(document["accessors"])
        document["accessors"].append(accessor)
        return accessor_index

    for mesh in meshes:
        if len(mesh.triangles) == 0:
            continue

        positions = np.ascontiguousarray(mesh.vertices, dtype="<f4")
        normals = np.ascontiguousarray(_vertex_normals(mesh), dtype="<f4")
        indices = np.ascontiguousarray(mesh.triangles.reshape(-1), dtype="<u4")

        position_accessor = add_accessor(
            positions, target=34962, component_type=5126,
            value_type="VEC3", bounds=True,
        )
        normal_accessor = add_accessor(
            normals, target=34962, component_type=5126,
            value_type="VEC3",
        )
        index_accessor = add_accessor(
            indices, target=34963, component_type=5125,
            value_type="SCALAR",
        )

        material_index = len(document["materials"])
        document["materials"].append({
            "name": mesh.name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [*mesh.color, mesh.opacity],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.82,
            },
            "alphaMode": "BLEND" if mesh.opacity < 1.0 else "OPAQUE",
            "doubleSided": True,
        })
        scene_mesh_index = len(document["meshes"])
        document["meshes"].append({
            "name": mesh.name,
            "primitives": [{
                "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
                "indices": index_accessor,
                "material": material_index,
                "mode": 4,
            }],
        })
        document["nodes"].append({"name": mesh.name, "mesh": scene_mesh_index})
        document["nodes"][0]["children"].append(len(document["nodes"]) - 1)

    binary.extend(b"\x00" * (-len(binary) % 4))
    document["buffers"] = [{"byteLength": len(binary)}]
    json_bytes = json.dumps(document, separators=(",", ":"),
                            sort_keys=True).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary)

    with output_path.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, total_length))
        stream.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        stream.write(json_bytes)
        stream.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        stream.write(binary)

    return output_path


def _view_basis(view: str, camera_vector: tuple[float, float, float] | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed screen-right, screen-up, and toward-camera unit vectors."""

    if view == "top":
        return (np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]))

    if view == "side":
        return (np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]))

    # The bow is at low X. A lower starboard camera keeps the stern at left,
    # the bow at right, and shows the hold depth and transverse framing.
    default_camera = {
        "isometric": (-0.55, 1.0, 0.35),
        "longitudinal_cutaway": (-0.75, 1.0, 0.38),
        "transverse_cutaway": (-1.0, 0.85, 0.72),
    }
    toward_camera = np.array(
        camera_vector if camera_vector is not None else default_camera[view],
        dtype=float,
    )
    toward_camera /= np.linalg.norm(toward_camera)
    screen_right = np.cross([0.0, 0.0, 1.0], toward_camera)
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(toward_camera, screen_right)
    return screen_right, screen_up, toward_camera


def _cut_hull(mesh: _Mesh, hull_bounds: np.ndarray, view: str,
              transverse_x: float) -> _Mesh:
    """Remove near-side shell facets for the longitudinal hull window."""

    if not mesh.name.startswith("hull_") or view != "longitudinal_cutaway":
        return mesh

    centres = mesh.vertices[mesh.triangles].mean(axis=1)
    x_min, x_max = hull_bounds[0]
    z_min, z_max = hull_bounds[2]
    length = x_max - x_min
    depth = z_max - z_min

    removed = (
        (centres[:, 0] > x_min + 0.08 * length)
        & (centres[:, 0] < x_max - 0.08 * length)
        & (centres[:, 1] > 0.0)
        & (centres[:, 2] > z_min + 0.13 * depth)
    )

    return replace(mesh, triangles=mesh.triangles[~removed])


def _clip_mesh_aft_of_station(mesh: _Mesh, station_x: float) -> _Mesh:
    """Intersect triangles with an X plane and retain the aft half."""

    if len(mesh.triangles) == 0:
        return mesh

    # VTK clips crossing facets at their actual intersections. Selecting
    # whole triangles by centroid left long structural facets protruding
    # through the transverse section as visible streaks.
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32), deep=True
    ))
    cells = vtk.vtkCellArray()
    offsets = np.arange(0, 3 * (len(mesh.triangles) + 1), 3, dtype=np.int64)
    connectivity = mesh.triangles.astype(np.int64).ravel()
    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(cells)

    plane = vtk.vtkPlane()
    plane.SetOrigin(station_x, 0.0, 0.0)
    plane.SetNormal(1.0, 0.0, 0.0)
    clip = vtk.vtkClipPolyData()
    clip.SetInputData(surface)
    clip.SetClipFunction(plane)
    clip.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(clip.GetOutputPort())
    triangles.Update()

    if triangles.GetOutput().GetNumberOfPolys() == 0:
        return replace(mesh, triangles=np.empty((0, 3), dtype=np.int32))

    return _polydata_mesh(triangles.GetOutput(), mesh.name, mesh.color, mesh.opacity)


def _view_meshes(meshes: list[_Mesh], hull_bounds: np.ndarray, view: str,
                 transverse_x: float) -> list[_Mesh]:
    """Expose empty cargo holds and structural planes within cutaway scenes."""

    if view not in {"longitudinal_cutaway", "transverse_cutaway"}:
        # Exterior plates use an opaque hull. Keep internal display solids in
        # the GLB while omitting them from this surface-only screenshot.
        return [mesh for mesh in meshes if (
            mesh.name.startswith("hull_")
            or mesh.name.startswith("hatch_")
            or mesh.name == "deck_display_plane"
            or mesh.name == "superstructure_display_box"
        )]

    visible = []

    for mesh in meshes:
        if mesh.name.startswith("hull_"):
            if view == "longitudinal_cutaway" and mesh.name == "hull_below_waterline":
                # Keep a faint floor for spatial context, without the red
                # surface that reads as a fuel load level in the open hold.
                visible.append(replace(
                    _cut_hull(mesh, hull_bounds, view, transverse_x),
                    opacity=0.28,
                ))
                continue
            visible.append(_cut_hull(mesh, hull_bounds, view, transverse_x))
        elif (mesh.name.startswith("hatch_") or mesh.name.endswith("_envelope")
              or mesh.name == "deck_display_plane"):
            # Omit covers and full-depth fuel envelopes so they do not hide
            # the empty cargo holds or the representative structure.
            continue
        elif mesh.name.startswith("structure_"):
            # The system view shows the continuous inner bottom and complete
            # compartment stations. The native three-hold structural detail
            # has its own enlarged paper panel.
            continue
        elif mesh.name.startswith("bulkhead_"):
            # Opaque bulkheads make the system subdivision readable in the
            # open starboard cutaway.
            visible.append(replace(mesh, color=(0.72, 0.69, 0.62), opacity=1.0))
        else:
            visible.append(mesh)

    if view == "transverse_cutaway":
        # Cut every displayed component at the same station. Clipping only
        # the hull left unsupported bulkhead planes floating ahead of the cut.
        return [_clip_mesh_aft_of_station(mesh, transverse_x) for mesh in visible]

    return visible


def _write_image(meshes: list[_Mesh], hull_bounds: np.ndarray,
                 path: Path, style: str, view: str,
                 publication: bool = False,
                 camera_vector: tuple[float, float, float] | None = None,
                 dpi_override: int | None = None) -> Path:
    """Project the scene in software and draw depth-ordered mesh facets."""

    screen_right, screen_up, toward_camera = _view_basis(view, camera_vector)
    projection = np.stack((screen_right, screen_up), axis=1)
    projected_faces = []
    face_depths = []
    face_colors = []
    face_layers = []

    for mesh in meshes:
        triangles = mesh.triangles

        if len(triangles) == 0:
            continue

        corners = mesh.vertices[triangles]
        face_points = corners @ projection
        depths = corners.mean(axis=1) @ toward_camera
        normals = np.cross(corners[:, 1] - corners[:, 0],
                           corners[:, 2] - corners[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0.0] = 1.0
        normals /= lengths[:, None]

        # Fixed ambient and directional light distinguish sloping shell from
        # flat decks without tying the result to a GUI or machine graphics.
        light = np.array([0.30, 0.35, 0.89])
        light /= np.linalg.norm(light)
        brightness = 0.58 + 0.42 * np.abs(normals @ light)
        # Quiet materials separate the physical hull, hatch envelopes, and
        # internal structure in grayscale as well as in color. These shades
        # affect screenshots only, never the source geometry or saved GLB.
        material = mesh.color
        if publication:
            if mesh.name == "hull_above_waterline":
                material = (0.22, 0.29, 0.33)
            elif mesh.name == "hull_below_waterline":
                material = ((0.65, 0.68, 0.68)
                            if view == "longitudinal_cutaway"
                            else (0.49, 0.32, 0.29))
            elif mesh.name.startswith("hatch_"):
                material = (0.53, 0.62, 0.64)
            elif mesh.name == "deck_display_plane":
                material = (0.67, 0.70, 0.70)
            elif mesh.name == "superstructure_display_box":
                material = (0.72, 0.75, 0.74)

        colors = np.tile(np.asarray([*material, mesh.opacity]), (len(triangles), 1))
        colors[:, :3] *= brightness[:, None]

        if style == "wireframe":
            colors[:, 3] = 0.55 if mesh.name.startswith("hull_") else 0.82

        projected_faces.append(face_points)
        face_depths.append(depths)
        face_colors.append(colors)
        layer = (2 if mesh.name == "superstructure_display_box" else
                 1 if mesh.name.startswith("hatch_") else 0)
        face_layers.append(np.full(len(triangles), layer, dtype=np.int8))

    points = np.concatenate(projected_faces)
    depths = np.concatenate(face_depths)
    colors = np.concatenate(face_colors)
    layers = np.concatenate(face_layers)
    # Hatch covers and the white house sit above the illustrative deck.
    # Draw them after the large deck polygons to avoid painter-order stripes.
    order = np.lexsort((depths, layers))

    # Set one fixed image geometry per view so image dimensions and framing
    # do not change when the user selects shaded or wireframe rendering.
    # Use only vertices that still belong to drawn triangles. A transverse
    # cut may leave unused bow vertices in each mesh; including them would
    # reserve blank page width after the section is clipped.
    all_vertices = np.concatenate([
        mesh.vertices[np.unique(mesh.triangles)]
        for mesh in meshes if len(mesh.triangles) > 0
    ])
    extents = all_vertices @ projection
    low = extents.min(axis=0)
    high = extents.max(axis=0)
    span = np.maximum(high - low, 1.0)
    margin = (0.018 if publication else 0.055) * max(span)

    # Publication mode changes framing and raster quality, never geometry.
    size = (10, 10 * (span[1] + 2 * margin) / (span[0] + 2 * margin))
    dpi = dpi_override if dpi_override is not None else (600 if publication else 120)
    figure, axis = plt.subplots(figsize=size if publication else (14.4, 8.1), dpi=dpi)
    background = "white" if publication else "#f8fafc"
    figure.patch.set_facecolor(background)
    axis.set_facecolor(background)

    if style == "shaded":
        collection = PolyCollection(
            points[order], facecolors=colors[order], edgecolors="none",
            linewidths=0.0, antialiaseds=False,
        )
    else:
        collection = PolyCollection(
            points[order], facecolors="none", edgecolors=colors[order],
            linewidths=0.20, antialiaseds=False,
        )

    axis.add_collection(collection)
    axis.set_xlim(low[0] - margin, high[0] + margin)
    axis.set_ylim(low[1] - margin, high[1] + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    if publication:
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    else:
        axis.set_title(f"GenISys ship system  |  {view}  |  {style}",
                       loc="left", fontsize=14, color="#17324d", pad=16)
        figure.tight_layout(pad=1.2)
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor(),
                   metadata={"Software": "GenISys system scene renderer"})
    plt.close(figure)
    return path


def render_and_export_scene(
    parts: list[dict],
    hull_stl_path: Path,
    output_dir: Path,
    waterline_z: float,
    styles: tuple[str, ...] = ("shaded", "wireframe"),
    views: tuple[str, ...] = ("isometric", "top", "side", "longitudinal_cutaway", "transverse_cutaway"),
    transverse_x: float | None = None,
    publication: bool = False,
) -> dict[str, Path]:
    """Export one combined GLB and the requested repeatable PNG views.

    Each part supplies ``name``, ``shape`` (a CadQuery shape), ``color`` as
    three 0–1 floats, and ``opacity`` as one 0–1 float. The return mapping
    contains ``scene_glb`` and keys such as ``shaded_isometric_png``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unsupported_styles = set(styles) - _VALID_STYLES
    unsupported_views = set(views) - _VALID_VIEWS

    if unsupported_styles or unsupported_views:
        raise ValueError(
            f"Unsupported scene styles {sorted(unsupported_styles)} "
            f"or views {sorted(unsupported_views)}"
        )

    full_hull, image_hull, hull_bounds = _read_hull(Path(hull_stl_path), waterline_z)
    part_meshes = [_tessellate_part(part) for part in parts]
    full_meshes = [*full_hull, *part_meshes]
    image_meshes = [*image_hull, *part_meshes]
    section_x = float(transverse_x) if transverse_x is not None else float(hull_bounds[0].mean())

    written = {"scene_glb": _write_glb(full_meshes, output_dir / "scene.glb")}

    for view in views:
        if view in {"longitudinal_cutaway", "transverse_cutaway"}:
            cut_meshes = _view_meshes(full_meshes, hull_bounds, view, section_x)
            written[f"{view}_glb"] = _write_glb(cut_meshes, output_dir / f"{view}.glb")

    for style in styles:
        for view in views:
            key = f"{style}_{view}_png"
            view_meshes = _view_meshes(image_meshes, hull_bounds, view, section_x)
            written[key] = _write_image(
                view_meshes, hull_bounds,
                output_dir / f"{style}_{view}.png", style, view, publication,
            )

    return written
