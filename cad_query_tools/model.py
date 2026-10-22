"""In-memory geometry and metadata shared by all CadQuery builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cadquery as cq
import pandas as pd


RHINO_CSV_COLUMNS = [
    "struct_rsObject",
    "mesh_rsObject",
    "Object_ID",
    "Color",
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
    "Lightening_Holes",
    "bit_1D_element",
    "Class",
    "Type",
    "Parent_Struct",
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
]


@dataclass(slots=True)
class StructuralElement:
    """One independently identifiable structural surface or member.

    ``shape`` is the structural CAD face set. ``mesh_shape`` is the idealized
    centre-line or mid-surface representation used by the legacy meshing path.
    ``property_shape`` may retain an exact segmented analytical surface while
    ``shape`` uses one joined face for Rhino-compatible IGES topology. Metadata
    retains the established Rhino CSV column names for downstream
    compatibility, even though no Rhino UUID is produced.
    """

    object_id: str
    shape: cq.Shape
    metadata: dict[str, Any]
    mesh_shape: cq.Shape | None = None
    property_shape: cq.Shape | None = None
    side: str = "starboard-half"
    sequence: int = 0

    @property
    def structural_object_name(self) -> str:
        """Return the name assigned to this entity in structural CAD files."""

        return self.object_id

    @property
    def mesh_object_name(self) -> str | None:
        """Return the matching name assigned to its mesh entity, if present."""

        if self.mesh_shape is None:
            return None

        return f"{self.object_id}:mesh"

    def csv_row(self) -> dict[str, Any]:
        """Return a row using the legacy structural-element schema."""

        row = {column: None for column in RHINO_CSV_COLUMNS}
        row.update(self.metadata)

        # CadQuery has no Rhino document UUID.  These deterministic names are
        # also embedded in the corresponding IGES entities, which preserves
        # the legacy CSV-to-CAD lookup contract without a Rhino document.
        row["struct_rsObject"] = self.structural_object_name
        row["mesh_rsObject"] = self.mesh_object_name
        row["Object_ID"] = self.object_id

        return row


@dataclass
class ShipStructure:
    """A generated half-ship structure plus its source and timing metadata."""

    ship_id: str
    elements: list[StructuralElement] = field(default_factory=list)
    source_parameters: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def add(self, element: StructuralElement) -> StructuralElement:
        """Append one element and return it for convenient builder use."""

        self.elements.append(element)
        return element

    def shape(self, mesh: bool = False) -> cq.Shape:
        """Combine all available element shapes into one exportable compound."""

        shapes = []

        for element in self.elements:
            candidate = element.mesh_shape if mesh else element.shape

            if candidate is not None:
                shapes.append(candidate)

        if not shapes:
            raise ValueError("The ship structure does not contain exportable geometry")

        return cq.Compound.makeCompound(shapes)

    def structural_elements_dataframe(self) -> pd.DataFrame:
        """Return structural metadata in the established 33-column order."""

        rows = [element.csv_row() for element in self.elements]
        return pd.DataFrame(rows, columns=RHINO_CSV_COLUMNS)

    def write_structural_elements_csv(self, output_path: str | Path) -> Path:
        """Write the downstream-compatible structural-elements CSV."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.structural_elements_dataframe().to_csv(path, index=False)
        return path

    def manifest_dataframe(
        self,
        exported_files: list[str | Path] | None = None,
    ) -> pd.DataFrame:
        """Describe the deterministic mapping between IDs, rows, and CAD files."""

        exported = "|".join(str(Path(item)) for item in (exported_files or []))
        rows = []

        for row_index, element in enumerate(self.elements):
            metadata = element.metadata
            rows.append(
                {
                    "Object_ID": element.object_id,
                    "element_type": metadata.get("Class"),
                    "source_csv_row": row_index,
                    "cad_shape_name": element.structural_object_name,
                    "x_loc": metadata.get("x_loc"),
                    "y_loc": metadata.get("y_loc"),
                    "z_loc": metadata.get("z_loc"),
                    "side": element.side,
                    "sequence": element.sequence,
                    "exported_file": exported,
                }
            )

        return pd.DataFrame(rows)

    def write_manifest(
        self,
        output_path: str | Path,
        exported_files: list[str | Path] | None = None,
    ) -> Path:
        """Write the ID-to-geometry sidecar manifest."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_dataframe(exported_files).to_csv(path, index=False)
        return path
