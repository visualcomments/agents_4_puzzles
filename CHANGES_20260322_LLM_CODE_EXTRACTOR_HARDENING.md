# LLM code extractor hardening (2026-03-22)

## What changed
- Hardened Python extraction in `AgentLaboratory/perm_pipeline/run_perm_pipeline.py`.
- Hardened lightweight extraction in `llm-puzzles/CallLLM.py` used by model self-check.
- Added regression tests for LLM answers that mix prose and raw Python code without fences.

## Why
Some providers return a helpful preamble like `Content of solve_module.py`, bullet points, and a `Code starts here:` marker before the actual module. The old extractor could reject such outputs even when the Python file itself was valid.

## Result
The pipeline now scores fenced and raw candidates, tolerates leading prose, keeps shebang lines, and trims trailing explanatory text after the code body.
