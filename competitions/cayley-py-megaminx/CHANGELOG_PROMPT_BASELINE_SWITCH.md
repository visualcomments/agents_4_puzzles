# Megaminx prompt baseline switch

- Baseline-backed prompt bundles now inject `megaminx_best_tested_solver.py` instead of the older generic `solve_module.py` baseline.
- The `regular` prompt bundle remains from-scratch and now omits any baseline section from planner/coder/fixer prompts.
- Offline fallback still keeps the best-tested solver available when LLM generation fails.
