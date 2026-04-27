# 2026-04-26 Prompt, success-gate, and Kaggle submit hardening

## What changed

- Hardened the Megaminx prompt sweep runner so a prompt/model attempt is successful only if it creates both:
  - an importable generated solver artifact, and
  - a valid non-empty submission CSV with no `UNSOLVED` rows, no blank rows, and no CSV parsing errors.
- Added explicit fallback artifact detection for sample-submission copies, offline baseline recovery, provider failures, and LLM-generation failure markers. These attempts remain in `failed_scripts` and do not increment success metrics.
- Added `--require-submit-success` with default `true`. When `--submit` is enabled, a round is reclassified as failed unless Kaggle upload is confirmed by the submit path.
- Preserved the current Kaggle CLI submit shape: `kaggle competitions submit <competition> -f <csv> -m <message>`.
- Strengthened all Megaminx user and custom prompt variants with explicit anti-fallback, replay-verification, rollback-safe optimization, and valid-submission requirements.
- Added a new auto-discoverable `submission_guarded` prompt bundle for strict solver/submission generation.

## Operational notes

- Fallback outputs are still copied for debugging, but they are not counted as successful scripts.
- A generated CSV alone is not enough to pass the sweep gate. The solver artifact and CSV content must both pass the stricter checks.
- Actual Kaggle submission still requires valid Kaggle credentials and accepted competition rules in the runtime environment.
