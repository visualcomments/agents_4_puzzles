# Hybrid codegen architecture (2026-03-12)

## What changed

This repository now supports a stronger search architecture for solver synthesis:

- `hybrid` search mode in `AgentLaboratory/perm_pipeline/run_perm_pipeline.py`
- multi-plan beam search (`--plan-beam-width`)
- planner/coder frontier scheduling (`--frontier-width`)
- experiment-manager memory (`--archive-size`)
- planner refinement rounds driven by failed-attempt summaries (`--refine-rounds`)
- backend-diverse model ordering so local and remote families are mixed instead of tried strictly linearly

The previous linear flow (`planner -> coder -> fixer`) is still available via:

```bash
python AgentLaboratory/perm_pipeline/run_perm_pipeline.py --search-mode classic ...
```

The default is now:

```bash
--search-mode hybrid
```

## New model backends

`AgentLaboratory/inference.py` now supports these backend prefixes:

- `local:<hf_model_id>`
- `ollama:<model>`
- `vllm:<model>`
- `lmstudio:<model>`
- `openai-compatible:<model>`
- `g4fapi:<model>`
- `g4f:<model>`

### Examples

```bash
--models "ollama:qwen2.5-coder:7b,g4f:gpt-4o-mini"
```

```bash
--models "vllm:Qwen/Qwen2.5-Coder-7B-Instruct,local:Qwen/Qwen2.5-Coder-1.5B-Instruct"
```

```bash
--models "openai-compatible:deepseek-r1"
```

## g4f stability improvements

- ordered provider failover via `AGENTLAB_G4F_PROVIDER_LIST` / `G4F_PROVIDER_LIST`
- remembered last-good provider per model in-process
- g4f can also be used via an OpenAI-compatible server with `g4fapi:<model>`

Example:

```bash
export G4F_PROVIDER_LIST="Blackbox,PollinationsAI,Glider"
python pipeline_cli.py generate-solver \
  --competition cayleypy-rapapport-m2 \
  --out generated/solve_module.py \
  --models "g4f:gpt-4o-mini,g4f:aria"
```

## Useful environment variables

### Hybrid search

- `AGENTLAB_SEARCH_MODE`
- `AGENTLAB_PLAN_BEAM_WIDTH`
- `AGENTLAB_FRONTIER_WIDTH`
- `AGENTLAB_ARCHIVE_SIZE`
- `AGENTLAB_REFINE_ROUNDS`

### Local / OpenAI-compatible backends

- `AGENTLAB_OLLAMA_BASE_URL`
- `AGENTLAB_VLLM_BASE_URL`
- `AGENTLAB_LMSTUDIO_BASE_URL`
- `AGENTLAB_OPENAI_COMPAT_BASE_URL`
- `AGENTLAB_G4F_API_URL`

### g4f stability

- `AGENTLAB_G4F_PROVIDER_LIST`
- `G4F_PROVIDER_LIST`
- `G4F_PROVIDER`
- `AGENTLAB_G4F_USE_ASYNC`
- `AGENTLAB_G4F_REQUEST_TIMEOUT_S`
- `AGENTLAB_G4F_STOP_AT_PYTHON_FENCE`

## Validation performed

- full project pytest suite
- `pipeline_cli.py selftest`
- `pipeline_cli.py check-g4f-models --discover-only --json`
- Python compile checks for modified modules

## Notes

- `local:*` uses GPU in this Python process (when torch/transformers are configured).
- `ollama:*`, `vllm:*`, `lmstudio:*`, `openai-compatible:*`, and `g4fapi:*` may use GPU on the server side instead.
- the pipeline still preserves the baseline fallback path unless `--strict` is used.
