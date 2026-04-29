# Failure-aware self-improving prompts for Megaminx

This patch adds a Megaminx prompt-improvement mode that learns from generated code that failed compilation, runtime checks, validator replay, submission checks, novelty checks, or local score acceptance.

## What changed

- `prompt_self_improver.py` now reads the most recent failed candidate path from the improvement history when available.
- The next generated round receives a compact failure autopsy: failure bucket, error/rejection reasons, failed solver fingerprint, feature flags, and a bounded failed-code excerpt.
- New directives steer the model through a repair ladder: compile/import, `solve(vec)` contract, script-mode JSON, official move legality, exact replay validation, and then bounded improvement.
- The repair prompt explicitly preserves promising algorithmic deltas from failed code behind rollback guards instead of blindly returning to the incumbent baseline.
- A new prompt bundle variant is available: `failure_aware_self_improvement`.

## Recommended command

```bash
python pipeline_cli.py run cayley-py-megaminx \
  --prompt-variant failure_aware_self_improvement \
  --keep-improving \
  --self-improve-prompts \
  --improvement-rounds 5
```

## Intended behavior after a failed generated solver

The next round should move monotonically through these gates:

1. failed candidate code;
2. code that compiles and imports;
3. code that preserves `solve(vec) -> (moves, sorted_array)` and JSON stdout;
4. code that emits only official move names;
5. code that passes exact replay validation;
6. non-fallback code with row-wise no-regression acceptance;
7. code with at least one bounded measurable improvement.

This keeps the self-improvement loop from repeatedly producing ambitious but invalid rewrites, while still forcing the solver to move beyond baseline-only fallback behavior.
