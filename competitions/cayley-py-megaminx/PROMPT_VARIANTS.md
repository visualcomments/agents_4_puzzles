# Megaminx prompt variants

This competition now supports multiple prompt bundles that all start from the working bundled-data Megaminx solver:

- `regular` — safer/conservative improvement prompt.
- `improved` — more aggressive Megaminx-specific optimization prompt.
- `dataset_adapted` — a stronger prompt bundle aligned with the inspected Megaminx datasets and tuned for fixed-depth local word optimization.
- `structured` — structured planner/coder/fixer prompt package with bounded JSON planning.
- `heuristic_boosted` — explicit multi-order commuting normalization + per-row best-of-fixed-candidates + bidirectional local rewrite prompt bundle.

CLI usage:

```bash
python pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx.py --prompt-variant regular
python pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx.py --prompt-variant improved
python pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx.py --prompt-variant dataset_adapted
python pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx.py --prompt-variant structured
python pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx.py --prompt-variant heuristic_boosted
python pipeline_cli.py run --competition cayley-py-megaminx --output competitions/cayley-py-megaminx/submissions/submission.csv --prompt-variant heuristic_boosted
```

Explicit `--prompt-file` and `--custom-prompts` still override the variant mechanism.
