# Megaminx prompt variants

This competition now supports two prompt bundles that both start from the working optimized Megaminx baseline solver:

- `regular` — safer/conservative improvement prompt.
- `improved` — more aggressive Megaminx-specific optimization prompt.

CLI usage:

```bash
python pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx.py --prompt-variant regular
python pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx.py --prompt-variant improved
python pipeline_cli.py run --competition cayley-py-megaminx --output competitions/cayley-py-megaminx/submissions/submission.csv --prompt-variant improved
```

Explicit `--prompt-file` and `--custom-prompts` still override the variant mechanism.
