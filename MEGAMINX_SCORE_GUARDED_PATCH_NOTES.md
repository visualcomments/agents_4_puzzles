# Megaminx score_guarded patch

This patch adds a stronger prompt-engineering path for `cayley-py-megaminx`.

## What changed
- Added new prompt variant: `score_guarded`.
- Extended `prompt_self_improver.py` with richer failure-signal analysis, plateau detection, and stronger directive selection.
- Added new directive families: `score_regression_guard`, `semantic_equivalence_replay`, `validator_triad_recheck`, `compile_first_then_optimize`, `policy_ablation_search`.
- Updated the Colab runner defaults to use `score_guarded`, sane improvement-round defaults, optional evaluation shard params, and reproducible run-config logging.
- Kept all secrets removed from notebook defaults.
