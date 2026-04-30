# Failure-aware self-improvement hardening

This patch improves the Megaminx `failure_aware_self_improvement` prompt variant.

## What changed

- Extended failure bucketing beyond `error` to include `failure_kind`, `rejection_reasons`, metrics, provider/fallback status, and `novelty_report`.
- Split coarse validation/runtime failures into more actionable buckets:
  - `illegal_move`
  - `replay_mismatch`
  - `compile_or_import`
  - `no_per_row_improvement`
  - `score_regression`
  - `provider_or_fallback`
- Treat no-novelty and no-per-row-improvement outcomes as failures rather than generic validated-but-not-selected rounds.
- Suppress failed-code excerpts when a no-novelty candidate is identical to the incumbent, avoiding prompt anchoring on the baseline.
- Added bucket-specific repair strategies and compact failure context injection into planner/coder/fixer custom prompts.
- Added support for alternate failed solver path fields such as `solver_path`, `candidate_path`, and `generated_solver_path`.
- Explicitly included `failure_aware_self_improvement` in Megaminx prompt sweep and breakthrough variant lists.
- Expanded regression tests for illegal moves, no-per-row improvement, no-novelty excerpt suppression, custom prompt failure context, and `solver_path`.

## Validation

The patched `prompt_self_improver.py` was parsed with `ast.parse`, imported from the patched archive, and smoke-tested against a no-per-row-improvement failure history. The smoke test verified that the failure is classified as `no_per_row_improvement`, not counted as `validated_not_selected`, and selects `patch_fresh_lane_split` together with the no-novelty/per-row repair directives.
