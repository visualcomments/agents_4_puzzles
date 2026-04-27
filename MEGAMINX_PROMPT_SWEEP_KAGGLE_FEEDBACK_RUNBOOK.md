# Megaminx prompt sweep with Kaggle feedback

This add-on provides a full Colab-oriented outer loop for `cayley-py-megaminx`:

- discovers and tests every Megaminx prompt bundle;
- stores stdout logs, `pipeline_cli.py` run logs, solver copies, submissions, prompt manifests, and strategy history;
- separates successful scripts from failed scripts;
- optionally submits every successful submission CSV to Kaggle;
- polls Kaggle submission history for public score;
- uses score movement to adjust the next prompt strategy.

## Main files

- `colab/megaminx_prompt_sweep_kaggle_feedback.py` — standard-library runner.
- `colab/agents_4_puzzles_megaminx_full_prompt_sweep_kaggle_feedback.ipynb` — Google Colab notebook.

## Typical Colab run

```bash
python colab/megaminx_prompt_sweep_kaggle_feedback.py \
  --repo-dir . \
  --competition cayley-py-megaminx \
  --kaggle-competition cayley-py-megaminx \
  --variants all \
  --models g4f:gpt-4o-mini \
  --keep-improving \
  --self-improve-prompts \
  --submit \
  --score-direction min
```

## Strategy feedback rule

After each successful script:

1. the runner submits the generated CSV when `--submit` is enabled;
2. it polls `kaggle competitions submissions <competition> -v -q`;
3. if score improves, the current solver/prompt lineage is promoted and the remaining queue is biased toward breakthrough variants;
4. if the script validates but score does not move, the queue is biased even more toward breakthrough variants to avoid plateauing on locally valid but leaderboard-flat code;
5. if scripts fail repeatedly, the queue temporarily backs off to safer guarded variants.

## Output structure

Each run writes:

- `run_index.jsonl` — complete per-round records;
- `strategy_history.jsonl` — prompt strategy decisions;
- `per_round_metrics.csv` — quantitative metrics table;
- `prompt_manifests/` — prompt bundle inventory;
- `successful_scripts/` — solver/log/submission copies for successful runs;
- `failed_scripts/` — solver/log copies for failed runs;
- `summary.json` — final aggregate summary;
- `<run_dir>.zip` — packaged results archive.

Kaggle credentials are intentionally not stored in the repository. Use `~/.kaggle/kaggle.json`, Colab upload, or environment variables.
