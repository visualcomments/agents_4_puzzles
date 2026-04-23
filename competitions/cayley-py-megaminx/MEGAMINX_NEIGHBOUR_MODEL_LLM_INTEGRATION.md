# Megaminx neighbour-model + LLM integration

This repository now vendors the public Megaminx neighbour-model training snapshot under:

```text
tp/cayleypy-neighbour-model-training-main/
```

It is wired into the Megaminx module through:

- `competitions/cayley-py-megaminx/megaminx_neighbour_model_lane.py`
- `competitions/cayley-py-megaminx/megaminx_cayleypy_llm_hybrid_solver.py`
- prompt bundle variant `neighbour_model_hybrid`

## What the new lane does

The new lane loads the released Q checkpoint and tries to solve selected hard Megaminx rows from scratch.
It is used as a **score-improving candidate lane** on top of existing deterministic and LLM-generated submissions:

1. build / load deterministic candidates;
2. optionally generate LLM candidates through `pipeline_cli.py run`;
3. fuse candidates row-wise by shortest locally valid path;
4. optionally run the neighbour-model lane on the hardest fused rows;
5. optionally run `search_improver_v3` for final local rewrites.

That means the neighbour model now works **together** with LLM prompts instead of replacing them.

## Hybrid run example

```bash
python competitions/cayley-py-megaminx/megaminx_cayleypy_llm_hybrid_solver.py \
  --generate-llm \
  --llm-variants structured,heuristic_boosted,master_hybrid,neighbour_model_hybrid \
  --generate-neighbour-model \
  --neighbour-model-device cpu \
  --neighbour-model-max-rows 32 \
  --run-search-v3 \
  --out competitions/cayley-py-megaminx/submissions/submission_cayleypy_llm_neighbour_hybrid.csv
```

## Standalone neighbour-model lane example

```bash
python competitions/cayley-py-megaminx/megaminx_neighbour_model_lane.py \
  --submission competitions/cayley-py-megaminx/submissions/optimized_submission.csv \
  --out competitions/cayley-py-megaminx/submissions/submission_neighbour_model_lane.csv \
  --device cpu \
  --max-rows 32
```

## Notes

- `torch` remains an optional runtime dependency.
- If the neighbour-model repo cannot be found, the hybrid orchestrator keeps running and reports the lane error in stats.
- The neighbour-model lane never accepts a candidate unless it replays to the official Megaminx central state with legal official move names.
- The upstream public snapshot stores released `.pth` checkpoints via Git LFS. If your extracted archive still contains pointer stubs instead of real binaries, run:

```bash
python scripts/fetch_megaminx_neighbour_weights.py
```

or, if you keep the vendored repo as a normal Git clone with Git LFS enabled:

```bash
cd tp/cayleypy-neighbour-model-training-main
git lfs pull
```
