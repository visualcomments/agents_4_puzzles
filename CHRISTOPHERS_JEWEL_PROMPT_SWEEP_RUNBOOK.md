# Christopher's Jewel prompt sweep runbook

Smoke run without Kaggle submission:

```bash
python christophers_jewel_prompt_sweep_pipeline_baseline_integrated.py \
  --source-mode local_zip_path \
  --archive-path agents_4_puzzles_christophers_jewel_prompt_sweep_pipeline_baseline_integrated.zip \
  --no-submit-to-kaggle \
  --max-total-runs 2 \
  --run-name jewel_smoke
```

Full run with inline Kaggle credentials:

```bash
python christophers_jewel_prompt_sweep_pipeline_baseline_integrated.py \
  --source-mode local_zip_path \
  --archive-path agents_4_puzzles_christophers_jewel_prompt_sweep_pipeline_baseline_integrated.zip \
  --kaggle-credential-mode inline_json \
  --kaggle-json-inline '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}' \
  --run-name christophers_jewel_full_prompt_sweep
```
