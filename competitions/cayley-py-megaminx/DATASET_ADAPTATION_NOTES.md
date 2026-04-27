# Dataset adaptation notes for `cayley-py-megaminx`

This folder was updated after a targeted review of the uploaded repository and the bundled Megaminx-related datasets.

## What was inspected

The most relevant external Megaminx datasets for this competition section were:

- `MegaminXolver-master.zip`
- `llminxsolver-cmp-main.zip`
- `llminxsolver-master.zip`
- `solve-twisty-puzzles-main.zip`
- `Megaminx-Simulator-AStar-main.zip`
- `megaminx-solver-main (2).zip`

Several other archives were inspected at the README / file-layout level and rejected as less suitable because they were GUI-only, trainer-only, small educational A* prototypes, or incompatible with the full 120-sticker competition representation.

## What was adapted

### 1. Fixed-depth exact local effect tables
Inspired by the meet-in-the-middle / hash-table style used in `MegaminXolver-master`, the competition solver now uses a stronger **fixed-depth exact short-word table** for local path compression.

Adaptation for this repo:
- keep the approach purely deterministic;
- keep it bounded to a constant radius;
- use it only for rewriting already-valid bundled solution paths, not for risky generic solving.

### 2. Cached compact local move effects
Inspired by pruning-table / coordinate-heavy last-layer solvers such as `llminxsolver-cmp-main` and `llminxsolver-master`, the updated solver caches generator effects as `bytes` and reuses them across local optimization passes.

Adaptation for this repo:
- no external binaries;
- no Rust/Java dependency;
- only the practical caching and bounded local-table idea was reused.

### 3. Dodecahedral order-5 / cycle awareness
The generic dodecahedron simulator in `solve-twisty-puzzles-main` was useful as a consistency check that the Megaminx face moves should be treated as **order-5 cycles** and that commutation should respect the actual dodecahedral adjacency structure.

Adaptation for this repo:
- commutation is derived directly from the official `puzzle_info.json` permutations;
- repeated face powers are normalized modulo 5.

## What was intentionally not adapted

- Full A* / IDA* solving from educational solver repos:
  these implementations were either too shallow, too puzzle-specific, too slow for the 1001-row bundle, or incompatible with the official competition representation.
- Human-layer / phase solvers:
  useful for interactive solving, but not a good fit for a reproducible Kaggle bundle optimizer over official move words.
- GUI / trainer projects:
  useful for notation and visualization, but not for the competition pipeline.

## Resulting competition-side changes

- stronger `solve_module.py` path optimizer;
- updated `build_optimized_assets.py`;
- new prompt bundle: `dataset_adapted`;
- updated prompt documentation;
- refreshed optimized submission / lookup assets.
