# Changes on 2026-03-22: monotonic prompt refinement + stronger Megaminx baseline

## What was wrong before

1. Hybrid planner refinement did **not** guarantee that a later plan/prompt was actually better than the earlier accepted one.
2. The Megaminx baseline worked, but the shipped optimized assets still left a small amount of score on the table.

## What changed

### 1) Monotonic prompt refinement in `AgentLaboratory/perm_pipeline/run_perm_pipeline.py`

Added:
- structured prompt/plan scoring beyond raw text length heuristics
- parent-aware refinement prompts that include:
  - previous accepted plan
  - previous planner JSON
  - archive failure memory
  - explicit instruction to return a **strictly better** plan
- an acceptance gate for refinement rounds:
  - same-signature plans are rejected
  - plans that do not beat the parent score by a minimum margin are rejected
- prompt history persistence to `<generated_solver_stem>_plan_history.json`

Effect:
- each accepted refinement round now corresponds to a prompt/plan that scored strictly higher than its parent under a deterministic rubric
- the pipeline no longer merely appends “different” plans; it only keeps accepted refinements

### 2) Stronger Megaminx precomputed assets

Updated:
- `competitions/cayley-py-megaminx/submissions/optimized_submission.csv`
- `competitions/cayley-py-megaminx/data/optimized_lookup.json`
- `competitions/cayley-py-megaminx/data/optimized_stats.json`
- `competitions/cayley-py-megaminx/build_optimized_assets.py`

New offline asset builder behavior:
- starts from the currently best shipped submission if present
- applies a deterministic bank of strong commuting-order policies
- also applies a fixed-seed per-row random commuting-order sweep
- only writes back if the result is not worse

Result:
- Megaminx baseline score is now **414305**

## Validation performed

- `pytest -q tests_test_cli_syntax.py tests_test_submission_pipeline.py tests_test_megaminx_monotonic_prompt_refinement.py tests_test_megaminx_superbaseline_assets.py`
- Megaminx validator run on a real bundled row
- full baseline score recomputed over all 1001 Megaminx states: `414305`
- end-to-end smoke run:
  - `python pipeline_cli.py run --competition cayley-py-megaminx --no-llm --output ...`

## Recommended usage

Use the strongest shipped prompt bundle together with the now-monotonic hybrid search:

```bash
python3 pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --prompt-variant heuristic_boosted \
  --output competitions/cayley-py-megaminx/submissions/submission_best.csv \
  --agent-models "planner=r1-1776;coder=r1-1776;fixer=r1-1776" \
  --search-mode hybrid \
  --plan-beam-width 4 \
  --frontier-width 12 \
  --archive-size 20 \
  --refine-rounds 3 \
  --max-iters 1000 \
  --g4f-async \
  --g4f-request-timeout 120 \
  --g4f-stop-at-python-fence \
  --print-generation \
  --print-generation-max-chars 16000 \
  --submit \
  --submit-via auto \
  --submit-competition cayley-py-megaminx \
  --message "heuristic_boosted hybrid monotonic prompt refinement" \
  --kaggle-json ~/.kaggle/kaggle.json
```
