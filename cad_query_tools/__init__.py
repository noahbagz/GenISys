"""Rhino-independent ship-structure generation with CadQuery.

All public geometry coordinates are expressed in metres, matching the Rhino
models and structural-element CSV files already stored in this repository.
"""

from .builder import CadQueryStructureBuilder, build_structure
from .exporters import load_iges, load_step, save_structure
from .model import ShipStructure, StructuralElement
from .visualization_proxies import (
    build_deckhouse_visualization_proxy,
    build_hatch_cover_visualization_proxies,
    build_system_visualization_proxy,
)

__all__ = [
    "CadQueryStructureBuilder",
    "ShipStructure",
    "StructuralElement",
    "build_structure",
    "load_iges",
    "load_step",
    "save_structure",
    "build_deckhouse_visualization_proxy",
    "build_hatch_cover_visualization_proxies",
    "build_system_visualization_proxy",
]
