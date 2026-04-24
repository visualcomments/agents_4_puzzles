# Changes 2026-04-24 — Megaminx g4f fallback models + analytics

Added to the Colab Megaminx prompt sweep pipeline:

1. Ordered g4f model fallback support.
   - `--models` can now be a comma/semicolon/newline-separated list.
   - `--model-fallbacks` can explicitly override `--models`.
   - For each prompt variant, the runner tries one model at a time and stops at the first model that produces a valid submission CSV.
   - Failed model attempts are still logged and copied to `failed_scripts/`.

2. Analytics artifacts included in the auto-downloaded result zip.
   - `analytics/analysis_report.md`
   - `analytics/prompt_variant_summary.csv`
   - `analytics/score_timeline.csv`
   - `analytics/strategy_transitions.csv`
   - `analytics/self_improving_prompt_lineage.csv`
   - `analytics/self_improving_prompt_signals.csv` and `.jsonl`
   - `analytics/model_fallback_attempts.csv`
   - `analytics/model_fallback_summary.csv`
   - `analytics/successful_vs_failed_scripts.csv`

3. Notebook-side analytics section.
   - Displays prompt summaries, self-improving prompt lineage, and model fallback attempts.
   - Generates PNG charts when data is available.
   - Saves `analytics/notebook_analytics_addendum.md` so the final zip contains the extra analysis.
