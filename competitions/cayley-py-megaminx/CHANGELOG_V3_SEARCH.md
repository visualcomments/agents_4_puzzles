# Megaminx v3 search update

This update adds a full offline `v3` post-optimizer for the bundled Megaminx competition assets.

## Added

- `search_improver_v3.py`
  - tiered policy (`skip` / `light` / `aggressive`),
  - per-row profiling,
  - optional CayleyPy backend,
  - internal fallback beam + local MITM neighborhood,
  - segment rewrite search on top of the bundled baseline.
- `search_policy_v3.py`
  - row triage for expensive search.
- `cayley_adapter.py`
  - isolates optional `cayleypy` integration.
- `tests_test_megaminx_search_v3.py`
  - smoke test for non-degrading behavior.

## Updated

- `solve_module.py`
  - now exposes reusable helpers for path parsing, optimization, validation and trajectory reconstruction.
- `build_optimized_assets.py`
  - now supports `--search-version v3` and can run the v3 improver during asset rebuilds.

## Design intent

`v3` keeps the repo-safe fallback behavior:
- start from current bundled best-known paths,
- run cheap deterministic `pre-opt`,
- only then try bounded search,
- accept candidate only if it is shorter **and** validates to `central_state`.
