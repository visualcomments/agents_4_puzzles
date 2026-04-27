# Megaminx JSON runtime enforcement

This patch fixes a remaining runtime gap where some Megaminx prompt variants (notably `structured`, `heuristic_boosted`, and `improved`) could still elicit legacy unstructured code answers even though the prompt files themselves had already been migrated to the `code_response.v2` envelope.

## Root cause

The prompt files were updated, but the runtime stack still depended mainly on instruction-following:

- the coder/fixer prompts placed the strict JSON contract late in a long prompt
- OpenAI/OpenAI-compatible backends were not asked for JSON mode / JSON schema output
- `pipeline_cli.py` still defaulted `AGENTLAB_G4F_STOP_AT_PYTHON_FENCE=1`, which biased the runtime toward the legacy fenced-code pathway

## Fixes

1. Front-load the JSON contract in coder and fixer prompts via `## OUTPUT CONTRACT (HIGHEST PRIORITY)`.
2. Add shared detection helper `prompt_requests_code_json_envelope(...)` in `llm_code_contract.py`.
3. Enable `response_format` automatically for code-envelope prompts:
   - `json_schema` for supported GPT-4o-family models
   - `json_object` fallback for other OpenAI-compatible models
4. Disable `AGENTLAB_G4F_STOP_AT_PYTHON_FENCE` by default for prompt bundles that request the JSON code envelope.
5. Add regression tests for Megaminx runtime enforcement.
