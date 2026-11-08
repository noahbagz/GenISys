# Manuscript figures

These graphics match the current GenISys manuscript, synchronized on 8 November
2026. Included manuscript assets are copied byte-for-byte; available PNG/SVG
alternatives are retained for repository display and editing. No figure is
regenerated during synchronization.

| Figure | Contents | Generation task | Manuscript file |
| --- | --- | --- | --- |
| 1 | Framework, dataset, and integrated-system generation | `overview` | `fig_1.pdf` |
| 2 | Sequential and simultaneous subsystem diffusion | `architecture` | `fig_2.pdf` |
| 3 | Feasibility-guidance comparison | `guidance` | `fig_3.pdf` |
| 4 | Training populations and generated feasible designs; 512 per model | `embedding` | `fig_4.png` |
| 5 | Integrated ship and structural representation | `structure` | `fig_5.pdf` |
| 6 | Cost reduction-guided sampling results | `guidance` | `fig_6.pdf` |
| A1 | Constraint-violation distributions, including the near-zero detail | `violations` | `fig_A1.png` |
| C1 | Loading, shear, and bending examples | `loads` | `fig_C1.png` |
| H1 | Selected evaluated feasible designs | `feasible-designs` | `fig_H1.png` |

Run a task through `eval_utils/generate_figures.py`, using `GenISys_Env` and the
research workspace described in [the evaluation guide](../eval_utils/README.md).
These tasks write to the workspace manuscript; copy the resulting approved
assets here when updating the repository. Retired and supplementary figures
are not included in this directory.

[The figure manifest](../docs/provenance/figure_manifest.json) records each
source path, hash, and whether its format is included in the manuscript.
