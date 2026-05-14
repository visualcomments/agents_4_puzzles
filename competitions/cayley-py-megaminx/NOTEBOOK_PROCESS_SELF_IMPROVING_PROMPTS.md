# Notebook-process self-improving prompts for CayleyPy Megaminx

## Purpose

This prompt type converts the way the uploaded Kaggle notebooks improve solver code into an explicit self-improvement contract for `AgentLaboratory` rounds.

Earlier Megaminx prompt variants already asked for better bounded local word rewriting. The new variant adds a more concrete process model:

1. inspect fixed artifacts;
2. build or load a baseline;
3. create several bounded candidate lanes;
4. validate each lane by exact replay;
5. aggregate the shortest exact-valid row result;
6. persist lineage and score evidence;
7. reject fallback/no-novelty outputs.

The result is a prompt family that tells the model to improve code the way the notebooks improve submissions, while still producing a standard-library `solve_module.py` rather than a notebook that requires TPU/GPU/runtime training.

## Notebook observations distilled into prompt constraints

### Uploaded Megaminx beam notebooks

`cayleypy-megaminx-beam-shareable.ipynb` and `cayleypy-megaminx-beam-m-fr-v2.ipynb` use a production-style search pipeline:

- fixed config cell for pid range and symmetry count;
- embedded puzzle/test artifacts;
- pretrained value teacher and Q/student models from Kaggle artifacts;
- symmetry ensemble over rotations;
- Q-shortlist plus value reranking;
- duplicate hashing;
- per-rank partial JSON saves;
- final per-pid aggregation by shortest verified path.

For prompt evolution, the important pattern is not the TPU dependency. The important pattern is the row-wise candidate-generation / exact-verification / aggregation loop.

### Uploaded CayleyPy model notebooks

The CayleyPy training notebooks show the same algorithmic loop on smaller or related puzzles:

- convert official `puzzle_info.json` into the solver format;
- define the solved target;
- train or load a value model;
- train a neighbour/Q model from the value model;
- use beam search with inverse-backtracking suppression;
- write a Kaggle submission;
- validate path lengths and solved states.

For a repository baseline, this becomes a dependency-free candidate-lane contract: do not train in `solve_module.py`; instead distill “teacher/student” into exact short-effect tables and cheap deterministic shortlist policies.

## New prompt files

- `prompts/user_prompt_notebook_process_self_improvement.txt`
- `prompts/custom_prompts_notebook_process_self_improvement.json`

Run with:

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out competitions/cayley-py-megaminx/generated/solve_notebook_process.py \
  --prompt-variant notebook_process_self_improvement \
  --keep-improving \
  --self-improve-prompts
```

Build a submission from a generated solver with:

```bash
python pipeline_cli.py build-submission \
  --competition cayley-py-megaminx \
  --solver competitions/cayley-py-megaminx/generated/solve_notebook_process.py \
  --output competitions/cayley-py-megaminx/submissions/submission_notebook_process_generated.csv
```

## Baseline supplied with this patch

This patch also adds a concrete baseline for the new prompt type:

- `competitions/cayley-py-megaminx/solve_module.py`
- `competitions/cayley-py-megaminx/submissions/submission.csv`
- `competitions/cayley-py-megaminx/submissions/notebook_process_depth5_submission.csv`
- `competitions/cayley-py-megaminx/notebook_process_baseline_builder.py`

The helper can reproduce the baseline improvement from the preserved pre-patch submission:

```bash
python competitions/cayley-py-megaminx/notebook_process_baseline_builder.py \
  --source-submission competitions/cayley-py-megaminx/submissions/optimized_submission.pre_notebook_process.csv \
  --out competitions/cayley-py-megaminx/submissions/notebook_process_depth5_submission.csv \
  --depth 5 \
  --max-window 12 \
  --passes 3
```

The baseline keeps the existing optimized lookup first, then uses an exact short-effect atlas distilled from the notebook shortlist/rerank idea. The generated artifact uses:

- exact atlas depth: 5;
- local DP window: 12;
- passes: 3;
- row-wise acceptance: keep only strictly shorter exact-valid paths;
- validation result: 1001/1001 bundled rows solved;
- score improvement over the previous optimized repository submission: `414305 -> 414166`, saving 139 moves with zero row regressions.

## Why this is a distinct prompt type

The older `algorithmic_population` and `exact_score_population` variants are centered on prompt-population and score-shard language. This variant is centered on notebook-to-code distillation:

- model training in notebooks becomes deterministic local scorer artifacts in the solver;
- TPU/XLA beam search becomes fixed-radius exact local replacement;
- symmetry ensemble becomes safe candidate diversity only when exact permutation data is available;
- partial notebook saves become row-level lineage and rollback;
- final notebook aggregation becomes per-row exact-valid winner selection;
- notebook fallback outputs are explicitly disallowed as success.

This gives future self-improving rounds a more concrete code-improvement recipe than “try another heuristic”.
