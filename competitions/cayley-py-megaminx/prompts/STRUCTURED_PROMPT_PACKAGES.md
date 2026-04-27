# Structured prompt packages for Megaminx

This repository now drives planner -> coder -> fixer with a structured planning contract before code generation.

## What changed first in `run_perm_pipeline.py`

The highest-leverage edits are now concentrated in these functions:

1. `generate_plan_candidates`
   - planner is asked for a JSON plan instead of loose prose
   - strategy families rotate across four bounded optimization variants
2. `build_initial_codegen_prompt`
   - coder receives a sectioned prompt with strategy package, planner JSON, planner summary, and the known-good baseline
3. `try_generate_with_model`
   - baseline code is passed on the first coder attempt for baseline-backed bundles; the `regular` from-scratch bundle omits the baseline section entirely
   - planner payload is forwarded into the fixer loop
4. `_run_fixer_loop`
   - fixer receives planner JSON, planner summary, baseline code, current code, and the failure report in a sectioned repair prompt

## Planner JSON schema

Common required fields:

- `strategy_family`
- `goal`
- `edit_targets`
- `must_preserve`
- `complexity_claim.precompute`
- `complexity_claim.per_row`
- `complexity_claim.why_polynomial`
- `proposed_changes`
- `validation_plan`
- `forbidden`
- optional: `patch_scope`, `notes`

## Variant A — stronger exact short-word table

- Family: `stronger_exact_table`
- Best for: improving constant-depth local exact replacement while keeping exact lookup first
- Edit targets:
  - `_short_word_data`
  - `_reduce_commuting_word`
  - `_optimize_word`

Planner focus:
- strengthen fixed-depth exact replacement tables
- canonicalize equivalent short effects before storing them
- reuse packed effects rather than widening search depth

## Variant B — bounded-window DP rewrite

- Family: `bounded_window_dp`
- Best for: stronger local path compression with fixed window sizes and pass counts
- Edit targets:
  - `_optimize_local_windows`
  - `_optimize_word`
  - `_compose_words`

Planner focus:
- bounded-window dynamic programming only
- memoize repeated local windows
- deterministic left-to-right canonicalization

## Variant C — bidirectional local replacement

- Family: `bidirectional_local_replacement`
- Best for: replacing local move windows via constant-radius forward/reverse effect tables
- Edit targets:
  - `_short_word_data`
  - `_best_local_rewrite`
  - `_optimize_word`

Planner focus:
- constant-radius forward and reverse local effects
- shortest equivalent-word replacement only
- never create a full-state frontier

## Variant D — offline parameter sweep

- Family: `offline_parameter_sweep`
- Best for: safe tuning over a tiny deterministic grid without changing solver architecture
- Edit targets:
  - `_short_word_data`
  - `_optimize_local_windows`
  - `solve`

Planner focus:
- expose a tiny fixed parameter grid
- evaluate candidates with deterministic local scores only
- select the best bounded candidate without instance-growing search

## Coder prompt structure

The coder now sees:

1. `## USER TASK`
2. `## STRATEGY PACKAGE`
3. `## PLANNER JSON`
4. `## PLANNER SUMMARY`
5. `## KNOWN-GOOD BASELINE SOLVER`
6. `## IMPLEMENTATION RULES`

## Fixer prompt structure

The fixer now sees:

1. `## USER TASK`
2. `## STRATEGY PACKAGE`
3. `## PLANNER JSON`
4. `## PLANNER SUMMARY`
5. `## CURRENT CODE`
6. `## FAILURE REPORT`
7. `## KNOWN-GOOD BASELINE`
8. `## REPAIR ORDER`

This keeps the repair step close to the accepted architecture and reduces the chance of a full rewrite.


## Code output envelope

All Megaminx prompt variants now require the coder and fixer to return one machine-readable JSON object with the shared `code_response.v2` contract:

- `version=code_response.v2`
- `artifact_type=python_module`
- `language=python`
- `filename=solve_module.py`
- `code=<full python file>`

The full solver must live only inside the `code` string. No prose may appear before or after the JSON object. This matches the structured-first extraction logic in the pipeline while keeping the legacy fenced/raw parser only as fallback compatibility.
