# 2026-04-25 — run_log list payload normalization

Fixed `colab/megaminx_prompt_sweep_kaggle_feedback.py` crash:

```text
AttributeError: 'list' object has no attribute 'get'
```

Cause: some `pipeline_cli.py --run-log` outputs are JSON lists of stage/event records,
while the prompt-sweep runner expected a single dict summary and called `.get()`.

Change:
- `read_json()` now returns `Any` instead of assuming dict-only JSON.
- Added `normalize_run_payload(payload)`.
- The runner now normalizes list-shaped run logs into a dict with:
  - `raw_run_log_type`
  - `raw_run_log_len`
  - `raw_run_log_events`
  - reconstructed `stages` where stage/name/step fields are available.
- Updated embedded `RUNNER_SOURCE` in Colab notebooks so rerunning notebook cells writes the fixed runner.

Validation:
- `py_compile` passes for the patched runner.
- `normalize_run_payload()` was smoke-tested with a list-shaped run log.
- Runner dry-run now completes without this payload-shape crash.
