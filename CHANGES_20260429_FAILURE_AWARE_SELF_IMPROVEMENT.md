# 2026-04-29 - Failure-aware self-improving prompt loop

Added a failure-aware Megaminx self-improvement mode.

## Files changed

- `competitions/cayley-py-megaminx/prompt_self_improver.py`
  - Adds failure-autopsy helpers.
  - Adds repair-oriented directives.
  - Injects failed candidate summaries and bounded code excerpts into the next prompt round.
  - Writes `failure_repair_report` into prompt-round metadata.
- `competitions/cayley-py-megaminx/prompts/user_prompt_failure_aware_self_improvement.txt`
- `competitions/cayley-py-megaminx/prompts/custom_prompts_failure_aware_self_improvement.json`
- `pipeline_cli.py`
  - Adds `failure_aware_self_improvement` to allowed prompt variants.
- `pancake_prompt_sweep_pipeline_FULL.py`
  - Adds the variant to the sweep lists where possible.
- `tests_test_megaminx_failure_aware_prompt_self_improver.py`
  - Verifies that failed candidate information reaches the next prompt and metadata.

## Why

The previous loop could continue after failed generated code, but the next prompt mostly saw a short error string. This patch turns failed code into structured training signal so the model is guided toward progressively more working and more improving code: repair first, validate, then improve.
