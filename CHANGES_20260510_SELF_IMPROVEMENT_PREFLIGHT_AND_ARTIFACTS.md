# 2026-05-10 self-improvement preflight, artifacts, and novelty fixes

## Summary

This patch hardens the Megaminx self-improvement runner after two 2026-05-09 runs produced submissions path-wise identical to the 414305 baseline.

## Changes

- Added code-envelope model preflight to reject models that answer ping but cannot emit strict JSON/Python solver candidates.
- Added `--require-code-envelope`, `--full-detail`, `--skip-code-preflight`, `--allow-non-code-models`, and `--code-probe-timeout` controls.
- Lowered inline self-improvement runner default `--max-iters` from `100000` to `12`.
- Added scenario artifact collection so future run ZIPs include prompt rounds, candidate archives, solver candidates, prompt evolution JSON, baseline snapshots, and best submissions.
- Added post-lookup novelty instructions to the Megaminx prompt improver: lookup replay is a fallback oracle, not an improvement.
- Fixed adaptive row-routing percentile scale in `search_improver_v3.py` by passing `total_rows=top_k` into `RowFeatures`.
- Preserved g4f registry/fallback model order after de-duplication instead of applying a hard-coded preference order before capability filtering.

## Validation

- `python -m py_compile` on patched modules succeeded.
- `python -m pytest -q tests_test_g4f_model_check.py tests_test_megaminx_prompt_self_improver.py tests_test_megaminx_search_v3.py` passed with `16 passed`.
