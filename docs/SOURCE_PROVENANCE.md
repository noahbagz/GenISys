# Source provenance

The authoritative AI source is the Ada export
`genisys_model_training_experiments_architecture_20261022`. Missing exported
dependencies were recovered from the matching completed Ada retraining bundle
and verified against the snapshot's source hashes. The copy manifest identifies
the exact original file and SHA-256 for each retained source.

The reusable publication modules retain Ada's definitions under shorter paths:

| Ada implementation | Publication location |
| --- | --- |
| `TabDiffusionTools.py` | `models/diffusion.py` |
| `first_experiment_models.py` | `models/generators.py` |
| `performance_guidance_critic.py` | `models/critics.py` |
| Publication `range_sampling.py` | `models/sampling.py` |
| Publication `NoiseStream` and `load_classifier` | `models/noise.py`, `models/classifiers.py` |
| Reusable data, schema, training, and checkpoint functions | `training_utils/` |
| Evaluation bridge and distance/statistical functions | `eval_utils/` |

Only needed definitions are retained from mixed-purpose orchestration modules.
Their function bodies are unchanged apart from import and repository-root path
locations. Original definitions and their internal terminology remain intact.
The paper settings merge Ada's production defaults with its saved
`first_experiment_cshipgen_hull_tdim256_clf128_cap_v4.json` recipe; obsolete
experiment-matrix and package-output locations are omitted.

`run_genisys.py` is new orchestration around those existing functions. It selects
one architecture/component, supplies storage paths, loads guidance assets,
invokes the retained fixed-range sampler, and optionally calls the existing
evaluator. It does not include historical experiment matrices, preflight
contracts, statistical-study launchers, or duplicate source archives. Fresh
checkpoints use the publication module names; historical Ada artifact bundles
are not a supported drop-in input to this smaller workflow.

The source audit checks retained definition syntax and file hashes. Its result
is in `provenance/source_validation.json`; per-file origins and module locations
are in `provenance/source_manifest.json` and `provenance/module_locations.json`.
No original workspace source was edited or deleted. Folders containing `OLD`
and Rhino implementation/comparison tools are excluded.

## Validation on 22 October 2026

- The source audit found no hash, retained-definition, global-constant, syntax,
  or internal-import errors. It checked 130 retained Ada definitions.
- Imports of the reusable model/training/evaluation modules, ship evaluator,
  and CadQuery generator resolved from this publication copy.
- An isolated CPU fixture built all three architectures. With smaller test
  networks and a one-epoch test recipe stored outside this repository, DDPM
  training, checkpoint reload, sampling, 133-to-189 decoding, and the exact
  5,000-nmi range lock worked. Guidance was disabled for this sample.
- The same one-epoch HCDDPM fixture stopped at the unchanged raw-sampling
  stability gate because its sample contained non-finite coordinates. That
  gate was preserved; SCDDPM training did not execute after the stop.
- No complete CUDA run, full classifier/critic training, RFR-guided production,
  ship-evaluation batch, or CAD rendering was executed for this reorganization.

These checks establish source preservation and limited interface coverage.
They are not validation of Ada's saved models or reproduction of paper results.
