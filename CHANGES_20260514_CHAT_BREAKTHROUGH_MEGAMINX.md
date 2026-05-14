# 2026-05-14 Megaminx chat-breakthrough artifact lane

Added a second Megaminx improvement based on `ChatExport_2026-05-14.zip`.

## New prompt variant

- `chat_breakthrough_self_improvement`
- Files:
  - `competitions/cayley-py-megaminx/prompts/user_prompt_chat_breakthrough_self_improvement.txt`
  - `competitions/cayley-py-megaminx/prompts/custom_prompts_chat_breakthrough_self_improvement.json`

This prompt variant asks agents to convert chat evidence into a safe artifact lane: optional CSV discovery, exact replay, and row-wise shorter-only merging of TPU/NISS/rescue/history-beam outputs.

## New module variant

- `competitions/cayley-py-megaminx/solve_module_chat_breakthrough.py`
- The default `competitions/cayley-py-megaminx/solve_module.py` now uses `BASELINE_VERSION = chat_breakthrough_artifact_lane_v1`.

The module remains standard-library-only and keeps the notebook-process lookup baseline when no external artifacts are available.

## Verification

Local no-artifact fallback:

- rows: `1001`
- solved_rows: `1001`
- score: `414166`
- invalid rows: `0`
- illegal rows: `0`
- blank rows: `0`

This change is designed to realize the chat's strongest scenario when external CSV artifacts are attached, without trusting any unverified artifact.
