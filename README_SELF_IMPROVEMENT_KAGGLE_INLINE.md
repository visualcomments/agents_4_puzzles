# Self-improvement runner with inline Kaggle submit

This repository build includes a Colab notebook and a CLI runner for the Megaminx self-improvement scenarios.

## Files

- `colab/agents_4_puzzles_self_improvement_kaggle_inline_colab.ipynb`
- `scripts/run_self_improvement_scenarios_kaggle_inline.py`

## Basic usage

```bash
python scripts/run_self_improvement_scenarios_kaggle_inline.py \
  --scenario both \
  --allow-failures
```

## Kaggle dry-run

Use an environment variable for `kaggle.json` so the token is not stored in shell history or notebook output.

```bash
export KAGGLE_JSON_INLINE='{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}'

python scripts/run_self_improvement_scenarios_kaggle_inline.py \
  --scenario both \
  --auto-submit-kaggle \
  --kaggle-submit-dry-run \
  --allow-failures
```

Remove `--kaggle-submit-dry-run` only when you are ready to make a real Kaggle submission.

The submit guard compares the best successful non-baseline scenario with the baseline and submits only if it improves by at least `--submit-min-improvement` move-token by default.
