# Prompt variants for CayleyPy Megaminx

Available prompt bundles:

- `regular` — conservative baseline-improvement prompt using the working optimized solver as the baseline.
- `improved` — stronger optimization prompt focused on Megaminx-specific path compression.
- `dataset_adapted` — competition-safe prompt bundle that explicitly steers the agents toward the fixed-depth word-table / multi-pass local-DP strategy adapted from the inspected Megaminx datasets.

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
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission.csv \
  --prompt-variant dataset_adapted
```
