"""Legacy-compatible plate and idealized one-dimensional section properties."""

from __future__ import annotations

import math
from collections import defaultdict

from .model import ShipStructure, StructuralElement


def _surface_properties(element: StructuralElement) -> None:
    """Populate area, volume, centroid, and rectangular section properties."""

    metadata = element.metadata
    # Joined polycurve sweeps are exported as one face to match Rhino's IGES
    # topology. OpenCascade approximates that joined face and can shift its
    # area centroid by millimetres. Rhino calculated CSV properties from the
    # exact pre-export segments, so use the retained analytical shape here.
    property_shape = (
        element.property_shape
        if element.property_shape is not None
        else element.shape
    )
    area = float(property_shape.Area())
    centroid = property_shape.Center()
    length_2 = float(metadata.get("L2") or 0.0)
    thickness = float(metadata.get("Thickness") or 0.0)
    rotation = math.radians(float(metadata.get("rot") or 0.0))

    metadata["area_pannel"] = area
    metadata["volume_pannel"] = area * thickness
    metadata["x_centroid_pannel"] = centroid.x
    metadata["y_centroid_pannel"] = centroid.y
    metadata["z_centroid_pannel"] = centroid.z
    metadata["area_cx"] = length_2 * thickness

    metadata["I_11"] = (
        (length_2 * thickness / 12.0)
        * (
            thickness**2 * math.cos(rotation) ** 2
            + length_2**2 * math.sin(rotation) ** 2
        )
    )
    metadata["I_22"] = (
        (length_2 * thickness / 12.0)
        * (
            thickness**2 * math.sin(rotation) ** 2
            + length_2**2 * math.cos(rotation) ** 2
        )
    )

    if length_2 == 0.0:
        metadata["J_p"] = 0.0
    else:
        metadata["J_p"] = (
            length_2
            * thickness**3
            * (1.0 / 16.0)
            * (16.0 / 3.0 - 3.36 * thickness / length_2)
            * (1.0 - (thickness / length_2) ** 4 / 12.0)
        )

    x_direction = float(metadata.get("x_dir") or 0.0)
    y_direction = float(metadata.get("y_dir") or 0.0)
    z_direction = float(metadata.get("z_dir") or 0.0)
    x_location = float(metadata.get("x_loc") or 0.0)
    y_location = float(metadata.get("y_loc") or 0.0)
    z_location = float(metadata.get("z_loc") or 0.0)

    metadata["centroid_cx_11"] = (
        x_direction * z_location
        + y_direction * x_location
        + z_direction * x_location
        + math.cos(rotation) * length_2 / 2.0
    )
    metadata["centroid_cx_22"] = (
        x_direction * y_location
        + y_direction * z_location
        + z_direction * y_location
        + math.sin(rotation) * length_2 / 2.0
    )

    metadata["I_11_1D"] = None
    metadata["I_22_1D"] = None
    metadata["J_p_1D"] = None


def _calculate_group_1d_properties(elements: list[StructuralElement]) -> None:
    """Combine linked web/flange rectangles using the legacy line-axis formula."""

    web = next(
        (item for item in elements if item.metadata.get("Class") != "Stiffener_Flange"),
        None,
    )
    flange = next(
        (item for item in elements if item.metadata.get("Class") == "Stiffener_Flange"),
        None,
    )

    if web is not None and bool(web.metadata.get("bit_1D_element")):
        target = web
        contributors = [web] + ([flange] if flange is not None else [])
    elif flange is not None:
        # A flange on a 2D girder remains its own 1D line element.
        target = flange
        contributors = [flange]
    else:
        return

    anchor = target.metadata.get("_mesh_anchor")

    if anchor is None:
        return

    x_direction = float(target.metadata.get("x_dir") or 0.0)
    y_direction = float(target.metadata.get("y_dir") or 0.0)
    z_direction = float(target.metadata.get("z_dir") or 0.0)
    inertia_11 = 0.0
    inertia_22 = 0.0
    torsion = 0.0

    for contributor in contributors:
        metadata = contributor.metadata
        distance_11 = (
            x_direction * (anchor[2] - metadata["centroid_cx_11"])
            + y_direction * (anchor[0] - metadata["centroid_cx_11"])
            + z_direction * (anchor[0] - metadata["centroid_cx_11"])
        )
        distance_22 = (
            x_direction * (anchor[1] - metadata["centroid_cx_22"])
            + y_direction * (anchor[2] - metadata["centroid_cx_22"])
            + z_direction * (anchor[1] - metadata["centroid_cx_22"])
        )

        inertia_11 += metadata["I_11"] + metadata["area_cx"] * distance_22**2
        inertia_22 += metadata["I_22"] + metadata["area_cx"] * distance_11**2
        torsion += metadata["J_p"]

    target.metadata["I_11_1D"] = inertia_11
    target.metadata["I_22_1D"] = inertia_22
    target.metadata["J_p_1D"] = torsion


def calculate_properties(structure: ShipStructure) -> ShipStructure:
    """Populate all derived fields expected by existing evaluation scripts."""

    groups: defaultdict[str, list[StructuralElement]] = defaultdict(list)

    for element in structure.elements:
        _surface_properties(element)
        section_group = element.metadata.get("_section_group")

        if section_group is not None:
            groups[str(section_group)].append(element)

    for elements in groups.values():
        _calculate_group_1d_properties(elements)

    return structure
