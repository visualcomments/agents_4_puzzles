# Structured Megaminx tutorial (g4f-first)

This tutorial documents the structured planner -> coder -> fixer flow using **plain g4f model names** as the primary interface, matching the original repository style.

## Quick start

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_structured.py \
  --prompt-variant structured \
  --models gpt-4o-mini,claude-3.5-sonnet,deepseek-chat \
  --search-mode hybrid \
  --plan-beam-width 4 \
  --frontier-width 8 \
  --archive-size 12 \
  --refine-rounds 2 \
  --max-iters 3 \
  --print-generation
```

## Fast smoke run

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_smoke.py \
  --prompt-variant structured \
  --models gpt-4o-mini \
  --search-mode classic \
  --max-iters 2
```

## End-to-end submission build

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission_structured.csv \
  --prompt-variant structured \
  --models gpt-4o-mini,claude-3.5-sonnet,deepseek-chat \
  --search-mode hybrid \
  --plan-beam-width 4 \
  --frontier-width 8 \
  --archive-size 12 \
  --refine-rounds 2 \
  --max-iters 3 \
  --schema-check \
  --run-log competitions/cayley-py-megaminx/submissions/run_log_structured.json
```

## Per-agent model routing with g4f names

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_agent_split.py \
  --prompt-variant structured \
  --agent-models "planner=claude-3.5-sonnet;coder=deepseek-chat,qwen2.5-coder;fixer=gpt-4o-mini" \
  --search-mode hybrid \
  --plan-beam-width 4 \
  --frontier-width 8 \
  --archive-size 12 \
  --refine-rounds 2 \
  --max-iters 3
```

## Direct low-level pipeline entrypoint

```bash
python AgentLaboratory/perm_pipeline/run_perm_pipeline.py \
  --user-prompt-file competitions/cayley-py-megaminx/prompts/user_prompt_structured.txt \
  --custom-prompts competitions/cayley-py-megaminx/prompts/custom_prompts_structured.json \
  --baseline competitions/cayley-py-megaminx/solve_module.py \
  --validator competitions/cayley-py-megaminx/validate_solve_output.py \
  --out generated/solve_megaminx_direct.py \
  --models gpt-4o-mini,claude-3.5-sonnet,deepseek-chat \
  --search-mode hybrid \
  --plan-beam-width 4 \
  --frontier-width 8 \
  --archive-size 12 \
  --refine-rounds 2 \
  --max-iters 3 \
  --print-generation
```

## Output contract

- Coder/fixer responses now use a strict JSON code envelope instead of relying on markdown python fences.
- The canonical response shape is `{"version":"code_response.v2","artifact_type":"python_module","language":"python","filename":"solve_module.py","code":"..."}`.
- The pipeline extracts code from that JSON envelope first and only falls back to legacy fenced/raw-code heuristics for backward compatibility.

## Notes

- Bare model names are treated as **g4f** models.
- `--prompt-variant structured` now resolves to `user_prompt_structured.txt` and `custom_prompts_structured.json`.
- Explicit backends such as `g4fapi:*`, `local:*`, `ollama:*`, `vllm:*`, and `lmstudio:*` are still supported, but the examples above stay g4f-first to match the original archive style.
