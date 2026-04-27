# Real external repos Run-all notes

This repository bundle includes the implementation needed for a turnkey Megaminx breakthrough run with real external solver repositories.

## New pieces added

- `competitions/cayley-py-megaminx/turnkey_real_external_run.py`
- `competitions/cayley-py-megaminx/external_solver_adapters/manifest_real_odder_megaminxolver.json`
- `competitions/cayley-py-megaminx/external_solver_adapters/manifest_real_sevilze_llminxsolver_cmp.json`
- `competitions/cayley-py-megaminx/external_solver_adapters/manifest_real_abgolev_astar.json`
- safe clone/build/fallback support in `external_adapter_lane.py`
- smoke-capable wrappers with fallback-ready empty submissions in `external_solver_adapters/wrappers/`

## Operational intent

The real-repo flow is designed to be robust first:

1. try to clone public solver repos by URL and ref;
2. try lightweight build / wrapper execution;
3. if candidates are produced, normalize and replay-validate them;
4. if the external repo is unavailable or incompatible, keep the pipeline green by filling rows from the bundled incumbent submission;
5. still evaluate every lane under the same exact-score selection and portfolio merge.

This means the end-to-end stack works out of the box, while leaving a clean place to add deeper state adapters later for repositories that can genuinely solve compatible Megaminx phases.
