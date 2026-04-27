# 2026-04-27 — 444-cube prompt sweep pipeline

Added a full `cayley-py-444-cube` prompt-sweep package adapted from the Megaminx pipeline.

## Added

- `cube444_prompt_sweep_pipeline.py` standalone launcher.
- `colab/cube444_prompt_sweep_kaggle_feedback.py` inner Kaggle-feedback runner.
- Full prompt variants under `competitions/cayley-py-444-cube/prompts/`.
- `CUBE444_PROMPT_SWEEP_KAGGLE_FEEDBACK_RUNBOOK.md`.

## Key adaptation details

- Defaults now use competition slug `cayley-py-444-cube`.
- Output root now defaults to `runs/cube444_prompt_sweep`.
- Prompt guards now enforce 96-length states, `initial_state_id,path`, official generator names, and replay-validated promotion.
- Megaminx-specific order-5 assumptions were replaced with generator-order derivation from official 4x4x4 permutations.
