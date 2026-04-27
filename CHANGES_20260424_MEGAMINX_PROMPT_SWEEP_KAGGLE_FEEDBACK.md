# 2026-04-24 Megaminx prompt sweep + Kaggle feedback

Added a Colab-ready full pipeline that tests all Megaminx prompt bundles and records detailed statistics.

Highlights:

- new outer runner: `colab/megaminx_prompt_sweep_kaggle_feedback.py`;
- new notebook: `colab/agents_4_puzzles_megaminx_full_prompt_sweep_kaggle_feedback.ipynb`;
- complete per-prompt logs, `run_log.json`, solver copies, submission stats, prompt manifests, and strategy history;
- successful and failed generated scripts are copied to separate folders;
- optional Kaggle submission after each successful generated submission;
- public-score polling from Kaggle submission history;
- score-aware strategy controller:
  - score improvement promotes the current lineage and prioritizes breakthrough prompt families next;
  - validated but score-flat runs increase breakthrough pressure;
  - repeated failures temporarily back off to guarded prompt variants.

No Kaggle secrets are included. The previous notebook pattern with inline credentials has been replaced by blank/upload-based credential handling.
