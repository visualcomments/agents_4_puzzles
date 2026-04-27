# 444-cube prompt sweep prompts

These prompt bundles target Kaggle `cayley-py-444-cube` only.

Verified official bundle facts:
- puzzle: 4x4x4
- state length: 96
- rows: 1043
- legal generators: 24
- submission schema: `initial_state_id,path`
- move separator: `.`
- replay convention: `new[i] = old[perm[i]]`

Prompt variants:
- `regular`
- `improved`
- `dataset_adapted`
- `structured`
- `heuristic_boosted`
- `master_hybrid`
- `neighbour_model_hybrid`
- `score_guarded`
- `algorithmic_population`
- `portfolio_orchestrated`
- `hard_row_routed`
- `exact_score_population`

All variants require strict replay validation to `central_state` and forbid `UNSOLVED`/blank/sample fallback outputs as successful generated scripts.
