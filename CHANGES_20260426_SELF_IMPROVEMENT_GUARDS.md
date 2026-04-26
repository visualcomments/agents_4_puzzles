# 2026-04-26 — Strict self-improvement guards for Megaminx prompt sweep

This update implements the follow-up findings from the failed-script archive analysis.

## Added

- `megaminx_guarded_sweep.py`: standalone guarded sweep runner that calls `pipeline_cli.py run` with strict improvement gates and stores success/failure artifacts.
- `user_prompt_strict_self_improvement.txt` and `custom_prompts_strict_self_improvement.json`: prompt bundle focused on measurable novelty, per-row improvement, and anti-fallback behavior.
- Pipeline CLI flags:
  - `--reject-identical-candidates / --no-reject-identical-candidates`
  - `--write-per-row-delta / --no-write-per-row-delta`
- Per-row delta artifacts for locally scored improvement rounds:
  - `<candidate>.round<N>.per_row_delta.csv`
  - `<candidate>.round<N>.per_row_delta.summary.json`
- Solver SHA256 and submission digest lineage in improvement reports.

## Changed

- Self-improvement candidates are rejected when they are identical to the current best solver or produce an identical submission digest.
- Candidates with zero improved rows in per-row delta are rejected as `no_per_row_improvement`.
- Adaptive baseline promotion now refuses identical solver/submission fingerprints and candidates with no per-row improvement.
- Submission scoring/statistics now count actual dot-delimited Megaminx moves instead of treating each row as one token.
- Prompt self-improver now adds directives for:
  - no-novelty rejection;
  - per-row delta acceptance;
  - provider preflight/no fallback promotion.

## Failure classes preserved as failed

The following should remain failed attempts and should not count as successful scripts:

- provider credential failures;
- provider timeout / no working providers;
- sample submission fallback;
- offline baseline fallback;
- optimized submission replay without novelty;
- identical solver hash;
- identical submission digest;
- zero improved rows;
- Kaggle submit failure when `--submit` is enabled.

## Recommended guarded run

```bash
python megaminx_guarded_sweep.py \
  --competition cayley-py-megaminx \
  --variants strict_self_improvement,score_guarded,hard_row_routed,exact_score_population \
  --models g4f:gpt-4o-mini \
  --output-root runs/guarded_prompt_sweep \
  --improvement-rounds 3 \
  --no-submit
```

For a live Kaggle run, add `--submit --kaggle-json /path/to/kaggle.json`.
