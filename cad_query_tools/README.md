# CadQuery structure and system visualization

This package generates the bulkcarrier midship structure with CadQuery and
writes the structural-element tables consumed by the ship evaluator.

Geometry uses metres, with x longitudinal, y transverse, and z vertical.
The native three-hold structural model is the positive-y half ship. The
existing structural calculations account for the other half. Plate thickness
is table metadata; the neutral CAD surfaces are not thickened solids.

Generate a structure from an explicit parameter-table row:

```bash
python -m cad_query_tools.generate \
  --input-csv /path/to/parameters.csv --row-index 0 \
  --output-directory outputs/cad/design_0 --ship-id design_0 --file-format both
```

The input must contain the existing named structural parameter columns.
Already-cleaned parameter rows are used as saved. Use `--clean-parameters`
only for raw rows that require the existing cleanup procedure.

The default format is IGES; `--file-format both` also writes STEP.
`--elements-only` builds the structure and writes its structural-element table
without CAD/mesh exports. This is the path used during dataset construction.

Regenerate a full-system display scene for a completed random-population row
from the downloaded unified dataset:

```bash
python -m cad_query_tools.visualize_system \
  --population random --indices 99 --style shaded
```

The Python API `create_system_visualization` additionally accepts an
`evaluation_root` for imported generated-design evaluation records. Full-system
scenes require saved updated parameters, structural elements, and evaluation
detail tables. A 189-parameter sample table alone does not supply all scene
inputs. Display proxies and the publication view options preserve the existing
geometry conventions and do not modify evaluator results.

Use each module's `--help` for its supported output and view options.
