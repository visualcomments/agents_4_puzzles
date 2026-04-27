# 2026-04-26: Guarded sweep auto-install from GitHub

Updated `megaminx_guarded_sweep.py` so it can be launched from an empty working directory.

## What changed

- Added automatic checkout of `https://github.com/visualcomments/agents_4_puzzles.git` when the runner is not already inside an `agents_4_puzzles` repository.
- Added repository setup flags:
  - `--repo-url`
  - `--install-dir`
  - `--branch`
  - `--force-reinstall`
  - `--update-repo` / `--no-update-repo`
  - `--install-deps {auto,none,full}`
- Default dependency mode is `auto`:
  - installs `kaggle` only when submit/Kaggle credentials are requested and the module is missing;
  - installs `g4f` only when a `g4f:` model is requested and neither the module nor bundled `gpt4free/` is available.
- Copies the guarded runner into the cloned repo for reproducibility.
- Keeps Kaggle preflight before `--submit` runs.

## Example

```bash
python megaminx_guarded_sweep.py \
  --install-dir ./agents_4_puzzles \
  --competition cayley-py-megaminx \
  --variants algorithmic_population,dataset_adapted,exact_score_population,hard_row_routed,heuristic_boosted,improved,master_hybrid,neighbour_model_hybrid,portfolio_orchestrated,regular,score_guarded,strict_self_improvement,structured,submission_guarded \
  --models g4f:gpt-4o-mini \
  --output-root runs/full_prompt_sweep_submit \
  --run-name full_all_prompts_submit_$(date -u +%Y%m%d_%H%M%S) \
  --improvement-rounds 3 \
  --max-iters 8000 \
  --submit \
  --submit-via cli \
  --message-prefix "megaminx full guarded prompt sweep"
```
