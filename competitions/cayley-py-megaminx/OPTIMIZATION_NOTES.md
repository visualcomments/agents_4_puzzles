# Megaminx optimization notes

This competition bundle now includes a stronger Megaminx-specific offline optimization pass that improves the bundled `sample_submission.csv` before it is used by the baseline solver.

## What changed

1. The solver prefers `data/optimized_lookup.json` when it exists.
2. `build_optimized_assets.py` rebuilds the optimized lookup and submission deterministically from the official competition bundle.
3. The local word optimizer was upgraded using bounded ideas adapted from inspected Megaminx datasets:
   - stronger fixed-depth exact short-word tables,
   - cached compact permutation effects,
   - order-5 face-power normalization,
   - multi-pass local dynamic programming.
4. A ready-to-submit file is bundled at `submissions/optimized_submission.csv`.

## Optimization pipeline

The optimized lookup is built from the official bundled `test.csv`, `sample_submission.csv`, and `puzzle_info.json` using deterministic stages:

1. **Commutation-aware word reduction**
   - collapse repeated turns on the same face modulo the Megaminx face order (5),
   - sort commuting face turns into a canonical order derived from the official generator permutations.

2. **Exact short-window replacement**
   - precompute all distinct local move effects reachable from identity within depth 5,
   - cache move permutations as `bytes`,
   - run dynamic programming over each path with a local window of 12 moves.

3. **Second optimization pass**
   - rerun the local dynamic-programming stage once more,
   - finish with another commutation-aware cleanup.

## Result on bundled assets

- original bundled score: **500572**
- previous bundled optimized score in this repo: **416067**
- refreshed bundled optimized score: **415075**
- incremental improvement over the previous optimized bundle: **992** fewer moves
- total improvement over the original sample submission: **85497** fewer moves

These stats are also stored in `data/optimized_stats.json`.
