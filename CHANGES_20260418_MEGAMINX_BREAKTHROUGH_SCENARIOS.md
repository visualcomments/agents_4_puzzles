# CHANGES 2026-04-18 — Megaminx breakthrough scenarios

This patch adds three concrete self-improvement scenarios requested for the Megaminx bundle:

- `portfolio_orchestrated` — exact row-level portfolio fusion and lineage-first asset promotion.
- `hard_row_routed` — hard-tail search routing with saved-moves-per-CPU-hour reporting.
- `exact_score_population` — exact dev/holdout score-driven prompt population selection.

## New tools

- `competitions/cayley-py-megaminx/row_scoreboard.py`
- `competitions/cayley-py-megaminx/portfolio_orchestrator.py`
- `competitions/cayley-py-megaminx/hard_row_routed_search.py`
- `competitions/cayley-py-megaminx/prompt_population_runner.py`

## Prompt-layer changes

- Added prompt bundles for `portfolio_orchestrated`, `hard_row_routed`, and `exact_score_population`.
- Expanded `prompt_self_improver.py` with directives for portfolio orchestration, hard-row routing, exact-metric acceptance, and shadow-split benchmarking.
- Extended `pipeline_cli.py` prompt-variant choices to expose the new bundles.

## Notebook changes

The standalone notebook now exposes the new prompt variants and a `BREAKTHROUGH_SCENARIO` selector so the same Colab run can optionally execute the new post-processing lanes.
