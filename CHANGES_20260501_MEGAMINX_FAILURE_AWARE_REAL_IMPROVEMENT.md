# Megaminx failure-aware self-improvement hardening (2026-05-01)

This patch turns the `failure_aware_self_improvement` prompt loop from a broad prompt paraphrasing layer into a more executable, row-targeted improvement loop.

## What changed

- Added AST-aware solver capability audit so the self-improver detects lookup-first architecture such as `_best_lookup`, `_LOOKUP_CACHE`, and submission-backed lookup wrappers instead of relying only on fragile token searches.
- Added row-profile memory ingestion from `optimized_submission.v3.profiles.json` and fallback parsing of `optimized_submission.csv`.
- Injected hard-row target IDs into the generated prompts, with a concrete goal: improve at least one listed hard row, regress zero rows, and expose row-level evidence.
- Added directive-specific executable evidence checks for selected directives such as `multi_policy_sweep`, `hard_row_routing`, `per_row_delta_acceptance`, and `score_regression_guard`.
- Strengthened plateau/no-novelty analysis with multi-axis stagnation signals: identical solver/submission, zero improved rows, row regressions, provider fallback, and submission digest novelty.
- Replaced fixed `search_policy_v3.py` thresholds with adaptive routing that can use path-length percentile, historical saved-moves-per-second, and regression risk while preserving legacy behavior when disabled.

## Why

The previous full failure-aware Megaminx run completed technically but accepted 0/12 rounds: candidates were either identical solvers or produced identical submissions with zero improved rows. This patch forces the next prompt round to use exact row-level memory and executable evidence, making no-op wrappers much harder to treat as plausible improvements.

## Smoke validation

Executed with Python 3:

```bash
python3 -m py_compile \
  competitions/cayley-py-megaminx/failure_aware_self_improvement/capability_audit.py \
  competitions/cayley-py-megaminx/failure_aware_self_improvement/row_profile_memory.py \
  competitions/cayley-py-megaminx/failure_aware_self_improvement/directive_evidence.py \
  competitions/cayley-py-megaminx/search_policy_v3.py \
  competitions/cayley-py-megaminx/prompt_self_improver.py
```

A smoke `build_round_prompt_bundle` call confirmed:

- `uses_lookup_first=True` for `megaminx_best_tested_solver.py`;
- `inspection_mode=ast_plus_capability_probe`;
- row profile memory found hard rows, starting with `1000, 989, 998, 996, 985`;
- directive evidence checks are emitted into round meta.
