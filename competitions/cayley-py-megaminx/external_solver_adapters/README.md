# External adapter lane manifests for `cayley-py-megaminx`

This directory now contains **runnable wrapper-backed manifests** for concrete external Megaminx repositories.

## What changed

Instead of pointing manifests at hypothetical `export_candidates.py` scripts inside every upstream repo,
this directory now ships **ready wrapper scripts** that:

1. know the expected checkout folder for a concrete upstream repo;
2. try the most likely entrypoints for that repo;
3. fall back to already-produced candidate files if the repo keeps outputs on disk;
4. always emit a Kaggle-style `initial_state_id,path` CSV;
5. feed that CSV into the official `external_adapter_lane.py`, which then normalizes notation,
   replay-validates every row, and routes the lane into exact-score orchestration.

## Included wrapper-backed manifests

- `manifest_megaminxolver.example.json`
- `manifest_llminxsolver.example.json`
- `manifest_solve_twisty_puzzles.example.json`
- `manifest_megaminx_simulator_astar.example.json`

## Included wrappers

Under `external_solver_adapters/wrappers/`:

- `run_megaminxolver_wrapper.py`
- `run_llminxsolver_wrapper.py`
- `run_solve_twisty_puzzles_wrapper.py`
- `run_megaminx_simulator_astar_wrapper.py`
- `wrapper_common.py`

Each wrapper expects the matching upstream checkout under `external/...` by default, but the repo path can
always be overridden with `--repo`.

## Supported flow

1. A wrapper probes a concrete external repo checkout.
2. It either runs a known command or picks up an already-produced candidate file.
3. The wrapper writes a Kaggle-style submission CSV.
4. `external_adapter_lane.py` normalizes the notation to the official move names from `puzzle_info.json`.
5. Each row is replay-validated against the official generators and `central_state`.
6. Invalid or missing rows fall back to the configured bundled submission.
7. The resulting adapterized CSV is fed into:
   - `portfolio_orchestrator.py` for row-wise best-of-lanes fusion
   - `prompt_population_runner.py` for shadow-split exact-score selection

## Placeholders available in command manifests

- `{repo_root}`
- `{comp_dir}`
- `{test_csv}`
- `{output_csv}`
- `{work_dir}`
- `{label}`

## Typical usage

### 1. Materialize one external lane

```bash
python competitions/cayley-py-megaminx/external_adapter_lane.py \
  --manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_megaminxolver.example.json \
  --test-csv competitions/cayley-py-megaminx/data/test.csv \
  --fallback-submission competitions/cayley-py-megaminx/submissions/optimized_submission.csv
```

### 2. Feed the external lane directly into exact-score orchestration

```bash
python competitions/cayley-py-megaminx/portfolio_orchestrator.py \
  --candidate bundled=competitions/cayley-py-megaminx/submissions/optimized_submission.csv \
  --external-manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_megaminxolver.example.json \
  --test-csv competitions/cayley-py-megaminx/data/test.csv \
  --splits competitions/cayley-py-megaminx/shadow_splits.json
```

### 3. Run shadow-split exact-score selection with multiple external lanes

```bash
python competitions/cayley-py-megaminx/prompt_population_runner.py \
  --candidate bundled=competitions/cayley-py-megaminx/submissions/optimized_submission.csv \
  --external-manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_megaminxolver.example.json \
  --external-manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_llminxsolver.example.json \
  --splits competitions/cayley-py-megaminx/shadow_splits.json
```

## Notes on upstream variability

Public Megaminx repositories do not expose a single standard CLI, and some of them are difficult to discover or
inspect reliably from automated web search in this environment. The wrappers therefore use **autodiscovery**:
first try a small list of repo-specific commands, then fall back to likely output files such as `submission.csv`,
`out/megaminx_candidates.csv`, or `*.jsonl` candidate dumps.

That makes the manifests runnable **out of the box** for the expected local checkout layout, while still keeping
all final acceptance under the official adapter lane and exact-score portfolio selection.
