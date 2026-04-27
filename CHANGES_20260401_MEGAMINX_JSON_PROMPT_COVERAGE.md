# Megaminx prompt coverage for JSON code envelopes

Date: 2026-04-01

## What changed

Completed the migration of **all** CayleyPy Megaminx prompt bundles to the shared `code_response.v2` JSON output contract.

Updated files:

- `competitions/cayley-py-megaminx/prompts/custom_prompts_dataset_adapted.json`
- `competitions/cayley-py-megaminx/prompts/custom_prompts_heuristic_boosted.json`
- `competitions/cayley-py-megaminx/prompts/custom_prompts_improved.json`
- `competitions/cayley-py-megaminx/prompts/custom_prompts_master_hybrid.json`
- `competitions/cayley-py-megaminx/prompts/custom_prompts_regular.json`
- `competitions/cayley-py-megaminx/prompts/custom_prompts_structured.json`
- `competitions/cayley-py-megaminx/prompts/custom_prompts_template.json`
- `competitions/cayley-py-megaminx/prompts/user_prompt.txt`
- `competitions/cayley-py-megaminx/prompts/user_prompt_dataset_adapted.txt`
- `competitions/cayley-py-megaminx/prompts/user_prompt_heuristic_boosted.txt`
- `competitions/cayley-py-megaminx/prompts/user_prompt_improved.txt`
- `competitions/cayley-py-megaminx/prompts/user_prompt_master_hybrid.txt`
- `competitions/cayley-py-megaminx/prompts/user_prompt_regular.txt`
- `competitions/cayley-py-megaminx/prompts/user_prompt_structured.txt`
- `competitions/cayley-py-megaminx/prompts/README.md`
- `competitions/cayley-py-megaminx/prompts/STRUCTURED_PROMPT_PACKAGES.md`

## Why

Previously, Megaminx prompt variants still instructed the model to answer with a single fenced Python block. The extractor had already been hardened to prefer a strict JSON envelope, but these prompts were still biasing the model toward legacy formatting.

Now the entire Megaminx prompt family uses the same structured contract as the rest of the hardened pipeline, which improves extraction precision and reduces contamination from surrounding prose.

## Compatibility

The pipeline still keeps legacy fenced-code and raw-Python fallback parsing for providers that ignore the new contract, but the primary/expected path is now the JSON envelope for every Megaminx prompt variant.
