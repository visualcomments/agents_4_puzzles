# Prompt variants

- `regular` — Produce a clean, valid baseline-compatible solver first; prioritize correctness over score.
- `improved` — Improve over the approved baseline using guarded shortening and row-level replacement.
- `dataset_adapted` — Exploit the exact official dataset: 1000 known test states, 48-position permutations, 12 generators, and sample baseline paths.
- `structured` — Use a structured research-plan-implement-validate loop with explicit replay checks.
- `heuristic_boosted` — Add heuristic search, inverse-distance estimates, cancellation, and local rewrite improvements over the baseline.
- `master_hybrid` — Combine multiple safe strategies and only accept improvements that beat the approved baseline for each row.
- `neighbour_model_hybrid` — Use neighbour/graph-model style ideas to propose shorter patches but keep baseline fallback for every row.
- `score_guarded` — Score-guard every replacement: if a candidate path is invalid or not shorter, keep the baseline path.
- `algorithmic_population` — Generate a population of algorithmic candidates per row and choose the shortest replay-valid path.
- `portfolio_orchestrated` — Orchestrate a portfolio of solvers and simplifiers, always falling back to baseline per row.
- `hard_row_routed` — Route long/hard baseline rows to deeper search while leaving easy rows baseline-valid.
- `exact_score_population` — Track exact total move tokens and accept only score-improving, fully validated submissions.
