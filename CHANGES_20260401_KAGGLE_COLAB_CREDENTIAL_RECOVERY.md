# 2026-04-01 — Kaggle Colab credential recovery fix

## Problem

A common Colab workflow uploads a Kaggle credentials file with an arbitrary name such as
`kaggle_zuruck.json`, but later notebook cells still hardcode:

- `cp kaggle.json ~/.kaggle/`
- `--kaggle-json ~/.kaggle/kaggle.json`

This caused live submit to fail before the pipeline could reach Kaggle, even though a valid
credentials file had already been uploaded.

## Fix

- Added best-effort autodiscovery of a single valid Kaggle credentials file in notebook/Colab
  working directories when the explicit path is missing.
- `cmd_run()` now uses the resolved/recovered credentials path for both improvement-round submits
  and final submit.
- Added a fixed Colab notebook example at `colab/agents_4_puzzles_3_test_fixed.ipynb`.

## Safety

Autodiscovery only recovers when there is exactly one best candidate. If the situation is ambiguous,
it does not guess and the previous explicit error remains.
