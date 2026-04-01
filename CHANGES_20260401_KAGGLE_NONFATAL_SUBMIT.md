# CHANGES 2026-04-01 — non-fatal Kaggle submit handling

## Problem

A validated solver candidate could be treated as a failed improvement round when the live Kaggle submission step failed afterwards, especially when:

- no Kaggle credentials were available in the runtime,
- `--submit` was enabled during `--keep-improving`, and
- `_kaggle_submit()` raised `SystemExit`.

This caused logs like:

- candidate saved and validated,
- submission build completed,
- Kaggle auth failed,
- round marked as failed and excluded from normal improvement flow.

## Root cause

The submit path used `SystemExit` for operational submission failures. In Python, `SystemExit` inherits from `BaseException`, not `Exception`, so ordinary `except Exception:` blocks do not catch it.

As a result, a Kaggle submission failure escaped the submit hook and poisoned the improvement round, even though solver generation/validation had already succeeded.

## Fix

### 1) Early submission availability detection

Added `_resolve_kaggle_submit_availability(...)`:

- explicit missing `--kaggle-json` is still treated as a user-facing configuration error,
- environment credentials and default `~/.kaggle/...` discovery are supported,
- if no credentials are available and no explicit path was provided, live submit is skipped non-fatally with a warning.

### 2) Non-fatal per-round submit failures

Inside `cmd_run()` / `_validated_round_hook(...)`:

- Kaggle submission now runs in a local `try/except` block,
- `SystemExit` and ordinary exceptions are converted into a structured non-fatal submit report,
- the validated solver remains eligible for local scoring and acceptance.

### 3) Non-fatal final submit failures

Final live submit no longer crashes the whole run when the solver and submission CSV were already produced successfully.

### 4) Backward-compatible robustness

`cmd_run()` now tolerates partially mocked/test args objects by using safe `getattr(..., default)` access for `keep_improving` and `improvement_rounds` where needed.

## Files changed

- `pipeline_cli.py`
- `tests_test_submission_pipeline.py`
- `tests_test_keep_improving_cli.py`

## Validation

Passed:

- `pytest -q tests_test_submission_pipeline.py tests_test_keep_improving_cli.py tests_test_kaggle_auth.py tests_test_kaggle_preflight.py tests_test_embedded_kaggle_submit.py`

Result:

- `35 passed`
