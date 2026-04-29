# CUBE444 baseline integration audit

This archive integrates a working baseline for `cayley-py-444-cube` into the prompt-sweep repository.

## Baseline artifacts

- `competitions/cayley-py-444-cube/baselines/cube444_baseline_solver.py` — pipeline-compatible `solve(vec)` baseline and CSV builder.
- `competitions/cayley-py-444-cube/baselines/baseline_submission.csv` — valid baseline submission copied from official sample paths.
- `competitions/cayley-py-444-cube/solve_module.py` — replaced with the same working baseline for direct default use.
- `pipeline_registry.py` — default `baseline_solver` for `cayley-py-444-cube` now points at the baseline artifact.

## Required behavior for prompt variants

Prompt variants that improve from a baseline now explicitly instruct agents to use the approved baseline solver/submission as the validity anchor. A candidate path may replace a baseline path only if it is strictly shorter and replay-valid against `puzzle_info.json` with convention `new[i] = old[perm[i]]`.

## Competition-specific score floor

The baseline has 1043 rows and total move count 525714. It is valid but non-optimal, matching Kaggle's statement that sample paths can be used as a valid submission while remaining highly non-optimal.
