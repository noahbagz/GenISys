# GenISys environment audit

Audited on 22 October 2026 using
`/opt/anaconda3/envs/GenISys_Env/bin/python` (Python 3.12.10).
No packages were installed, removed, or updated.

All 491 Conda pins match the installed Conda records. All 37 pip pins match
effective Python distribution versions. Conda additionally retains a NumPy
1.26.4 record whose metadata file is missing; the effective pip installation
is NumPy 2.2.6, matching the YAML's pip pin.

The specification matches its listed versions, but the installed dependencies
are not mutually consistent. `python -m pip check` exits 1 with these findings:

| Package | Reported dependency issue |
| --- | --- |
| AutoGluon Multimodal 1.3.0 | Missing `nvidia-ml-py3` |
| spaCy 3.8.5 | Requires Thinc >=8.3.4,<8.4.0; installed 8.3.2 |
| CatBoost 1.2.7 | Requires NumPy <2.0; installed 2.2.6 |
| AutoGluon Multimodal 1.3.0 | Requires Transformers >=4.38.0,<4.50; installed 4.51.3 |
| GluonTS 0.16.1 | Requires NumPy <2.2; installed 2.2.6 |
| AutoGluon Timeseries 1.3.0 | Requires CoreForecast >=0.0.12,<0.0.16; installed 0.0.16 |
| AutoGluon Timeseries 1.3.0 | Requires Transformers >=4.38.0,<4.50; installed 4.51.3 |

The YAML was copied unchanged. Selecting dependency upgrades/downgrades would
change the execution environment and requires a separate decision; no such
changes were made as part of relocating code.

Full evidence: [`provenance/environment_audit.json`](provenance/environment_audit.json).
To check installed dependency consistency, run `python -m pip check` inside
`GenISys_Env`. The original full pin-audit script remains in the workspace's
publication-preparation backup; it is outside this runtime repository.
