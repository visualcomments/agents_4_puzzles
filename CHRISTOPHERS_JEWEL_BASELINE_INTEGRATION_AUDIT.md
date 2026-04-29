# Christopher's Jewel baseline integration audit

- Competition: `cayleypy-christophers-jewel`
- Official rows: 1000
- State length: 48
- Generator count: 12
- Submission columns: `initial_state_id,path`
- Baseline score floor: 500500 total move tokens
- Max baseline path length: 1000
- Baseline SHA256: `42bbceefd8ad32a7746fad6789f1a1af2f72b8d751e76065db422c332ebf764e`

Integrated artifacts:

- `competitions/cayleypy-christophers-jewel/baselines/jewel_baseline_solver.py`
- `competitions/cayleypy-christophers-jewel/baselines/baseline_submission.csv`
- `competitions/cayleypy-christophers-jewel/validate_submission_csv.py`
- `competitions/cayleypy-christophers-jewel/validate_solve_output.py`
- `colab/christophers_jewel_prompt_sweep_kaggle_feedback.py`

All baseline-dependent prompts explicitly require preserving the baseline path whenever a candidate replacement is invalid or not shorter.
