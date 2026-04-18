# External adapter lane for real Megaminx solver repositories

This directory contains two layers:

1. legacy/example manifests for local external checkouts;
2. **real-repo manifests** that can clone public GitHub repositories, attempt a repo-specific wrapper, and still complete the full Megaminx pipeline safely by falling back to the bundled incumbent submission for rows that the external solver cannot cover.

## Real-repo manifests

These manifests are intended for the "Run all" Colab / turnkey flow:

- `manifest_real_odder_megaminxolver.json`
- `manifest_real_sevilze_llminxsolver_cmp.json`
- `manifest_real_abgolev_astar.json`

Each manifest can:

- clone the external repository by URL and ref;
- optionally run lightweight build/smoke steps;
- invoke a repo-specific wrapper;
- tolerate wrapper/build failures when `allow_failure=true`;
- emit a fallback-ready adapter CSV that is then replay-validated and merged by the official Megaminx orchestration stack.

## Important practical limitation

The public external repositories do **not** share a common state representation with the Kaggle/CayleyPy Megaminx bundle.
For that reason, the default real-repo manifests are designed to be **safe and reproducible first**:

- if a real wrapper can materialize candidates, they are normalized and replay-validated;
- if the external solver is only partially compatible, fails to build, or lacks a direct Kaggle-state bridge, the lane still succeeds operationally and the adapter fills rows from the configured fallback submission.

That means the pipeline works out of the box, while still giving you a clean place to plug in deeper repo-specific state adapters later.

## Wrappers

Under `external_solver_adapters/wrappers/`:

- `run_megaminxolver_wrapper.py`
- `run_llminxsolver_wrapper.py`
- `run_abgolev_astar_wrapper.py`
- `run_solve_twisty_puzzles_wrapper.py`
- `run_megaminx_simulator_astar_wrapper.py`
- `wrapper_common.py`

The first three are the default wrappers used by the real-repo manifests.

## Typical usage

Materialize all real external lanes:

```bash
python competitions/cayley-py-megaminx/external_adapter_lane.py \
  --manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_real_odder_megaminxolver.json \
  --manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_real_sevilze_llminxsolver_cmp.json \
  --manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_real_abgolev_astar.json \
  --test-csv competitions/cayley-py-megaminx/data/test.csv \
  --fallback-submission competitions/cayley-py-megaminx/submissions/optimized_submission.csv
```

Feed them straight into row-wise exact-score fusion:

```bash
python competitions/cayley-py-megaminx/portfolio_orchestrator.py \
  --candidate bundled=competitions/cayley-py-megaminx/submissions/optimized_submission.csv \
  --candidate routed=competitions/cayley-py-megaminx/submissions/submission_hard_row_routed.csv \
  --external-manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_real_odder_megaminxolver.json \
  --external-manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_real_sevilze_llminxsolver_cmp.json \
  --external-manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_real_abgolev_astar.json \
  --test-csv competitions/cayley-py-megaminx/data/test.csv \
  --splits competitions/cayley-py-megaminx/shadow_splits.json
```

## Recommended workflow

1. rebuild `optimized_submission.csv`;
2. build `row_scoreboard.json` and `shadow_splits.json`;
3. run `hard_row_routed_search.py`;
4. materialize the real external lanes;
5. run `prompt_population_runner.py` on the combined candidates;
6. run `portfolio_orchestrator.py` for the final row-wise merge.
