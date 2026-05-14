# Megaminx prompt variants

This competition now supports multiple prompt bundles. Baseline-backed variants start from `megaminx_best_tested_solver.py`; the `regular` bundle is from-scratch and does not receive baseline code in the prompt.

- `regular` — from-scratch prompt with no injected baseline code.
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

- `portfolio_orchestrated`
- `hard_row_routed`
- `exact_score_population`

## Notebook-process self-improvement

```bash
python pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx_notebook_process.py --prompt-variant notebook_process_self_improvement --keep-improving --self-improve-prompts
```


## Chat-breakthrough self-improvement

This variant uses the chat export as a source of breakthrough scenarios, but converts them into a safe artifact contract instead of trusting private or TPU-only outputs. It preserves the local notebook-process baseline and adds exact replay, legal-move checks, row-wise shorter-only acceptance, and traceable min-merge of optional CSV artifacts.

```bash
python pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx_chat_breakthrough.py --prompt-variant chat_breakthrough_self_improvement --keep-improving --self-improve-prompts
```
