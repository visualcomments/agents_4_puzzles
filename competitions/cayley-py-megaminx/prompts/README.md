# Prompt variants for CayleyPy Megaminx

Подробный пошаговый runbook со всеми основными сценариями запуска находится в `competitions/cayley-py-megaminx/MEGAMINX_PIPELINES_RUNBOOK.md`.

Available prompt bundles:

Baseline-backed variants inject `competitions/cayley-py-megaminx/megaminx_best_tested_solver.py` as the baseline code. The `regular` variant is true from-scratch mode and does not include any baseline section in the model prompt.

- `regular` — from-scratch, score-seeking prompt bundle with no injected baseline code and bounded creative search for a lower bundled move-count score.
- `improved` — stronger optimization prompt focused on Megaminx-specific path compression.
- `dataset_adapted` — competition-safe prompt bundle that explicitly steers the agents toward the fixed-depth word-table / multi-pass local-DP strategy adapted from the inspected Megaminx datasets.
- `heuristic_boosted` — stronger prompt bundle focused on multi-order commuting normalization, per-row best-of-fixed-candidates selection, and bidirectional bounded local rewrites.
- `master_hybrid` — master prompt bundle that synthesizes the repository architecture with stronger exact effect atlases, multi-policy candidate sweeps, and small-support commutator/conjugate mining for local semantic rewrites only.
- `score_guarded` — stricter bounded hybrid focused on semantic-equivalence replay, explicit anti-regression fallback, and safer score-seeking local rewrites.
- `algorithmic_population` — new algorithmic self-improvement bundle that frames each round as a bounded population search problem with exact evaluator shards, patch-vs-fresh deterministic lanes, Pareto-style candidate selection, and lineage-friendly staging for future rounds.

Examples:

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_regular.py \
  --prompt-variant regular \
  --keep-improving
```

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_improved.py \
  --prompt-variant improved
```

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_dataset_adapted.py \
  --prompt-variant dataset_adapted
```

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_heuristic_boosted.py \
  --prompt-variant heuristic_boosted
```

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission.csv \
  --prompt-variant heuristic_boosted
```

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_master_hybrid.py \
  --prompt-variant master_hybrid
```

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission.csv \
  --prompt-variant master_hybrid
```


```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_algorithmic_population.py \
  --prompt-variant algorithmic_population \
  --keep-improving \
  --self-improve-prompts
```

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission.csv \
  --prompt-variant algorithmic_population \
  --keep-improving \
  --self-improve-prompts
```

## Output format contract

All Megaminx prompt bundles now require the coder/fixer reply to use the shared `code_response.v2` JSON envelope instead of returning a raw ```python``` fenced block. This keeps the prompt bundles aligned with the repository-wide structured extraction path and reduces accidental leakage of prose into `solve_module.py`.
