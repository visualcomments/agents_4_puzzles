# Prompt variants for CayleyPy 4x4x4 cube

Подробный пошаговый runbook со всеми основными сценариями запуска находится в `competitions/cayley-py-444-cube/CUBE444_PIPELINES_RUNBOOK.md`.

Available prompt bundles:

Baseline-backed variants inject `competitions/cayley-py-444-cube/4x4x4 cube_best_tested_solver.py` as the baseline code. The `regular` variant is true from-scratch mode and does not include any baseline section in the model prompt.

- `regular` — from-scratch, score-seeking prompt bundle with no injected baseline code and bounded creative search for a lower bundled move-count score.
- `improved` — stronger optimization prompt focused on 4x4x4 cube-specific path compression.
- `dataset_adapted` — competition-safe prompt bundle that explicitly steers the agents toward the fixed-depth word-table / multi-pass local-DP strategy adapted from the inspected 4x4x4 cube datasets.
- `heuristic_boosted` — stronger prompt bundle focused on multi-order commuting normalization, per-row best-of-fixed-candidates selection, and bidirectional bounded local rewrites.
- `master_hybrid` — master prompt bundle that synthesizes the repository architecture with stronger exact effect atlases, multi-policy candidate sweeps, and small-support commutator/conjugate mining for local semantic rewrites only.
- `score_guarded` — stricter bounded hybrid focused on semantic-equivalence replay, explicit anti-regression fallback, and safer score-seeking local rewrites.
- `algorithmic_population` — new algorithmic self-improvement bundle that frames each round as a bounded population search problem with exact evaluator shards, patch-vs-fresh deterministic lanes, Pareto-style candidate selection, and lineage-friendly staging for future rounds.
- `portfolio_orchestrated` — exact row-level portfolio fusion bundle focused on multi-lane orchestration and winner-take-best asset promotion.
- `hard_row_routed` — bundle that pushes compute toward the hardest rows first and emphasizes saved moves per CPU-hour.
- `exact_score_population` — bundle that turns prompt evolution into exact-score population search on deterministic dev/holdout shards.

Examples:

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-444-cube \
  --out generated/solve_4x4x4 cube_regular.py \
  --prompt-variant regular \
  --keep-improving
```

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-444-cube \
  --out generated/solve_4x4x4 cube_improved.py \
  --prompt-variant improved
```

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-444-cube \
  --out generated/solve_4x4x4 cube_dataset_adapted.py \
  --prompt-variant dataset_adapted
```

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-444-cube \
  --out generated/solve_4x4x4 cube_heuristic_boosted.py \
  --prompt-variant heuristic_boosted
```

```bash
python pipeline_cli.py run \
  --competition cayley-py-444-cube \
  --output competitions/cayley-py-444-cube/submissions/submission.csv \
  --prompt-variant heuristic_boosted
```

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-444-cube \
  --out generated/solve_4x4x4 cube_master_hybrid.py \
  --prompt-variant master_hybrid
```

```bash
python pipeline_cli.py run \
  --competition cayley-py-444-cube \
  --output competitions/cayley-py-444-cube/submissions/submission.csv \
  --prompt-variant master_hybrid
```


```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-444-cube \
  --out generated/solve_4x4x4 cube_algorithmic_population.py \
  --prompt-variant algorithmic_population \
  --keep-improving \
  --self-improve-prompts
```

```bash
python pipeline_cli.py run \
  --competition cayley-py-444-cube \
  --output competitions/cayley-py-444-cube/submissions/submission.csv \
  --prompt-variant algorithmic_population \
  --keep-improving \
  --self-improve-prompts
```

## Output format contract

All 4x4x4 cube prompt bundles now require the coder/fixer reply to use the shared `code_response.v2` JSON envelope instead of returning a raw ```python``` fenced block. This keeps the prompt bundles aligned with the repository-wide structured extraction path and reduces accidental leakage of prose into `solve_module.py`.
