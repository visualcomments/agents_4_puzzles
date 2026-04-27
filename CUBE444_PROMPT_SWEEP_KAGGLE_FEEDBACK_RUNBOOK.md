# 444-cube prompt sweep runbook

Use the checked standalone launcher with the checked repository archive.

## Smoke syntax/help check

```bash
python cube444_prompt_sweep_pipeline_checked.py --help
```

## Full run with inline Kaggle credentials

```bash
python cube444_prompt_sweep_pipeline_checked.py \
  --source-mode local_zip_path \
  --archive-path agents_4_puzzles_444_prompt_sweep_pipeline_checked.zip \
  --kaggle-credential-mode inline_json \
  --kaggle-json-inline '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}' \
  --run-name cube444_full_prompt_sweep_checked
```

## Safer smoke run without live Kaggle submit

```bash
python cube444_prompt_sweep_pipeline_checked.py \
  --source-mode local_zip_path \
  --archive-path agents_4_puzzles_444_prompt_sweep_pipeline_checked.zip \
  --no-submit-to-kaggle \
  --max-total-runs 2 \
  --run-name cube444_prompt_sweep_smoke
```

Important: strict generated-script validation rejects `UNSOLVED`, blank paths, invalid move tokens, and paths that do not replay to `central_state`.
