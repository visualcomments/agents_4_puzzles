# 2026-05-14 Megaminx notebook-process self-improvement baseline

## Added

- New Megaminx prompt variant: `notebook_process_self_improvement`.
- `pipeline_cli.py` now accepts `--prompt-variant notebook_process_self_improvement`.
- `prompt_self_improver.py` now has a notebook-process distillation directive for derived rounds.
- New prompt files:
  - `competitions/cayley-py-megaminx/prompts/user_prompt_notebook_process_self_improvement.txt`
  - `competitions/cayley-py-megaminx/prompts/custom_prompts_notebook_process_self_improvement.json`
- New design note:
  - `competitions/cayley-py-megaminx/NOTEBOOK_PROCESS_SELF_IMPROVING_PROMPTS.md`
- New reproducibility helper:
  - `competitions/cayley-py-megaminx/notebook_process_baseline_builder.py`

## Baseline update

The Megaminx baseline now uses a notebook-process distilled exact local optimizer:

- source optimized score: 414305;
- new baseline score: 414166;
- saved moves: 139;
- improved rows: 64;
- regressed rows: 0;
- solved bundled rows: 1001/1001.

Updated artifacts:

- `competitions/cayley-py-megaminx/solve_module.py`
- `competitions/cayley-py-megaminx/data/optimized_lookup.json`
- `competitions/cayley-py-megaminx/data/optimized_stats.json`
- `competitions/cayley-py-megaminx/submissions/optimized_submission.csv`
- `competitions/cayley-py-megaminx/submissions/submission.csv`
- `competitions/cayley-py-megaminx/submissions/notebook_process_depth5_submission.csv`

A backup of the previous optimized submission is stored at:

- `competitions/cayley-py-megaminx/submissions/optimized_submission.pre_notebook_process.csv`

## Rationale

The uploaded notebooks improve submissions by repeatedly turning fixed puzzle artifacts into candidate paths, validating candidates by exact replay, and aggregating shortest verified results. The new prompt type makes that process explicit for future self-improving code-generation rounds while forbidding runtime notebook dependencies such as PyTorch/XLA, Kaggle credentials, GPU/TPU, and leaderboard probing.
