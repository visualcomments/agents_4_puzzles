# CUBE444 competition compatibility audit

Checked against Kaggle competition slug: `cayley-py-444-cube`.

## Official bundle facts

Source archive: `cayley-py-444-cube.zip`

| File | SHA256 |
|---|---|
| `puzzle_info.json` | `2ae03f88108d9818315a1b36abf383bfab450bc6c1aeb6c0b6a2e502c8f05f48` |
| `test.csv` | `e9a734df2a63656477c2c6f9920f490188fdfa8a780027f1b91783e82b77419e` |
| `sample_submission.csv` | `9d28ba08e3eb930afc2b8386ddbe14fe40456d29df11bd9fc879a554b333d1e8` |

Parsed facts:
- `puzzle_info.name`: `4x4x4`
- `central_state` length: `96`
- `central_state` color counts: six colors x 16 stickers
- legal generator count: `24`
- legal generators: `f0, -f0, f1, -f1, f2, -f2, f3, -f3, r0, -r0, r1, -r1, r2, -r2, r3, -r3, d0, -d0, d1, -d1, d2, -d2, d3, -d3`
- `test.csv` rows: `1043`
- `sample_submission.csv` rows: `1043`
- submission schema: `initial_state_id,path`
- sample paths replay to `central_state` for all `1043` rows under convention `new[i] = old[perm[i]]`

## Fixes applied

1. Added a full 12-variant 444-cube prompt sweep pack:
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

2. Added `colab/cube444_prompt_sweep_kaggle_feedback.py` to the repository archive itself.
3. Hardened `competitions/cayley-py-444-cube/validate_solve_output.py`:
   - rejects `UNSOLVED` by default;
   - checks state length against official `central_state`;
   - checks all move tokens against official `generators`;
   - replays moves to `central_state`;
   - validates `sorted_array` equals the replayed final state.
4. Added `competitions/cayley-py-444-cube/validate_submission_csv.py`:
   - validates `initial_state_id,path`;
   - rejects blank/`UNSOLVED` rows;
   - replays every submitted path against official `test.csv` and `puzzle_info.json`.
5. Updated prompt-sweep runner success classification:
   - a generated attempt is successful only if the CSV is non-empty, contains no `UNSOLVED`/blank rows, and the full 444-cube submission replay validator passes.
6. Updated the 444-cube pipeline smoke vector to the official `central_state` of length `96`.

## Offline checks run

- Syntax check for:
  - `cube444_prompt_sweep_pipeline_checked.py`
  - `colab/cube444_prompt_sweep_kaggle_feedback.py`
  - `pipeline_cli.py`
  - `pipeline_registry.py`
  - `validate_solve_output.py`
  - `validate_submission_csv.py`
- `validate_submission_csv.py` on bundled `sample_submission.csv`: OK, `1043` rows, `96`-length state, `24` generators.
- Strict `validate_solve_output.py` rejects baseline `UNSOLVED`: OK.
- `validate_solve_output.py --allow-unsolved` accepts solved-state baseline only: OK.
- Exact lookup test solver validates row 0: OK.
- `pipeline_cli.py build-submission` with exact lookup test solver builds a full schema-valid CSV: OK.
- Full replay validation of that generated CSV: OK.

## Notes

The checked package is prepared for real prompt sweeps, but no live LLM generation or Kaggle submission was executed during this audit because that requires your active LLM/g4f providers and Kaggle credentials.
