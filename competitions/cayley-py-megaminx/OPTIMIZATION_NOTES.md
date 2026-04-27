# Megaminx optimization notes

This competition bundle includes a Megaminx-specific offline optimization stack that improves the bundled `sample_submission.csv` before it is used by the baseline solver.

## What changed

1. The solver prefers `data/optimized_lookup.json` when it exists.
2. `build_optimized_assets.py` rebuilds the optimized lookup and submission deterministically from the official competition bundle.
3. The local word optimizer uses:
   - fixed-depth exact short-word tables,
   - cached compact permutation effects,
   - order-5 face-power normalization,
   - multi-pass local dynamic programming.
4. `v3` adds an optional search layer on top of the current bundled best submission.

## Optimization pipeline

### Stage 1: deterministic word rewrite

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

### Stage 2: v3 bounded search (optional)

`search_improver_v3.py` is a post-optimizer, not a replacement solver.

It uses the current bundled best path as an upper bound, then for selected long rows:
- runs cheap `pre-opt`,
- classifies the row into `skip`, `light`, or `aggressive`,
- attempts bounded segment rewrites using beam search,
- optionally uses CayleyPy when installed,
- otherwise falls back to an internal beam + local meet-in-the-middle neighborhood,
- accepts a candidate only if it is shorter and still solves the puzzle.

## Suggested usage

Rebuild deterministic assets only:

```bash
cd competitions/cayley-py-megaminx
python build_optimized_assets.py
```

Rebuild and then run the v3 improver on the longest rows:

```bash
cd competitions/cayley-py-megaminx
python build_optimized_assets.py \
  --search-version v3 \
  --search-top-k 150 \
  --search-aggressive-time-budget-per-row 0.75
```

Run `v3` directly:

```bash
cd competitions/cayley-py-megaminx
python search_improver_v3.py \
  --submission submissions/optimized_submission.csv \
  --out submissions/submission_search_improved_v3.csv \
  --top-k 150
```
