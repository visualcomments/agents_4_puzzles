# Chat-breakthrough Megaminx scenario

This note records the extra Megaminx-specific scenario extracted from `ChatExport_2026-05-14.zip` and how it was converted into a safe repository/module change.

## Signals found in the chat export

Relevant Megaminx messages described several high-impact but non-portable or private-artifact routes:

- TPU teacher/student Q-shortlist beam notebooks, including model ensembles, symmetry expansion and chunked top-K beam scoring.
- Non-backtracking/history beam search, with discussion that history-aware pruning can materially increase solve probability.
- NISS and rescue-top-row passes, followed by row-wise min-merging of multiple CSV submissions.
- Private or locally generated CSV artifacts such as `tpu_qshort_sym_submission*.csv`, with a reported min-merge improvement in the chat transcript.
- Superflip centrality, coset decomposition, GAP/twsearch layers, and PDB-style features as promising future heuristic or validation targets.
- Pareto/consensus selection across multiple estimates instead of collapsing every candidate to one scalar too early.

The export did not include the strong TPU CSV files themselves. It contained references to them and scores, so the safe change is an artifact lane rather than an unverifiable hard-coded score claim.

## Implemented module change

`solve_module.py` and `solve_module_chat_breakthrough.py` now include `chat_breakthrough_artifact_lane_v1`:

1. Keep the bundled notebook-process optimized lookup as the default incumbent.
2. Optionally discover candidate submission CSVs in:
   - `MEGAMINX_CHAT_ARTIFACTS`
   - `/kaggle/input`
   - `competitions/cayley-py-megaminx/data/chat_breakthrough_artifacts`
   - `competitions/cayley-py-megaminx/chat_breakthrough_artifacts`
3. Parse rows with `initial_state_id` and `path`/`moves`/`solution` columns.
4. Reject illegal move names.
5. Replay every shorter candidate exactly from the bundled `test.csv` state to the official central state.
6. Accept only strictly shorter valid rows and keep per-row selected-lane traceability.
7. If no artifact is present, produce the same exact-valid local baseline score as before.

This turns the chat's most plausible breakthrough route into a reproducible repository interface: attach stronger external CSVs, then let the module validate and min-merge them safely.

## Local verification

No external breakthrough CSV was present in the upload, so local score is unchanged but the module is now ready to consume such artifacts safely.

- Rows: `1001`
- Solved rows: `1001`
- Local fallback score: `414166`
- Invalid rows: `0`
- Illegal rows: `0`
- Blank rows: `0`

## Usage

```bash
python competitions/cayley-py-megaminx/solve_module.py --build-submission competitions/cayley-py-megaminx/submissions/submission.csv
python competitions/cayley-py-megaminx/solve_module.py --chat-breakthrough-stats
```

With external artifacts:

```bash
export MEGAMINX_CHAT_ARTIFACTS=/path/to/tpu_or_niss_csvs
python competitions/cayley-py-megaminx/solve_module.py --build-submission competitions/cayley-py-megaminx/submissions/submission.csv
```

For self-improving prompt runs:

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_chat_breakthrough.py \
  --prompt-variant chat_breakthrough_self_improvement \
  --keep-improving \
  --self-improve-prompts
```
