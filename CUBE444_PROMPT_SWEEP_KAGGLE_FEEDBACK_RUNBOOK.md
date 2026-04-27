# 444-cube prompt sweep with Kaggle feedback

Standalone launcher:

```bash
python cube444_prompt_sweep_pipeline.py \
  --source-mode local_zip_path \
  --archive-path agents_4_puzzles_444_prompt_sweep_pipeline.zip \
  --kaggle-credential-mode existing_path \
  --kaggle-json-path ~/.kaggle/kaggle.json
```

Fast local smoke run without Kaggle upload:

```bash
python cube444_prompt_sweep_pipeline.py \
  --source-mode local_zip_path \
  --archive-path agents_4_puzzles_444_prompt_sweep_pipeline.zip \
  --no-submit-to-kaggle \
  --max-total-runs 2 \
  --max-rows 5 \
  --run-name cube444_smoke
```

What was added over the original archive:

- full prompt-variant package for `competitions/cayley-py-444-cube/prompts/`;
- `colab/cube444_prompt_sweep_kaggle_feedback.py`;
- root-level `cube444_prompt_sweep_pipeline.py`;
- this runbook.

The runner keeps the Megaminx sweep behavior: model fallback, prompt variant sweep, strict success/failure classification, Kaggle submission polling, strategy feedback, successful/failed script manifests, and analytics artifacts.
