# Prompt variants for CayleyPy Megaminx

Available prompt bundles:

- `regular` — conservative baseline-improvement prompt using the working optimized solver as the baseline.
- `improved` — stronger optimization prompt focused on Megaminx-specific path compression.
- `dataset_adapted` — competition-safe prompt bundle that explicitly steers the agents toward the fixed-depth word-table / multi-pass local-DP strategy adapted from the inspected Megaminx datasets.
- `heuristic_boosted` — stronger prompt bundle focused on multi-order commuting normalization, per-row best-of-fixed-candidates selection, and bidirectional bounded local rewrites.
- `master_hybrid` — master prompt bundle that synthesizes the repository architecture with stronger exact effect atlases, multi-policy candidate sweeps, and small-support commutator/conjugate mining for local semantic rewrites only.

Examples:

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_regular.py \
  --prompt-variant regular
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
