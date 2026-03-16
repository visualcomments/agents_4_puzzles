# Megaminx optimization notes

This competition bundle now includes a Megaminx-specific offline optimization pass that improves the bundled `sample_submission.csv` before it is used by the baseline solver.

## What changed

1. The solver now prefers `data/optimized_lookup.json` when it exists.
2. A reproducible asset builder was added: `build_optimized_assets.py`.
3. A ready-to-submit file is bundled at `submissions/optimized_submission.csv`.

## Optimization pipeline

The optimized lookup is built from the official bundled `test.csv`, `sample_submission.csv`, and `puzzle_info.json` using three deterministic stages:

1. **Commutation-aware word reduction**
   - collapse repeated turns on the same face modulo the Megaminx face order (5)
   - sort commuting face turns into a canonical order so additional cancellations can surface
2. **Exact short-window replacement**
   - precompute all distinct states reachable from identity within depth 4
   - run dynamic programming over each path with a local window of 10 moves
   - replace any window whose net permutation has a shorter exact representative
3. **Final commutation cleanup**
   - run the commutation-aware reducer one more time after local substitutions

## Result on bundled assets

- original bundled score: **500572**
- optimized bundled score: **416067**
- improvement: **84505** fewer moves

These stats are also stored in `data/optimized_stats.json`.
