# JSON code envelope hardening (2026-04-01)

## What changed

- Added a shared parser/contract module: `llm_code_contract.py`.
- Switched the main coder/fixer contract from markdown ` ```python ` fences to a strict JSON envelope.
- Updated the main AgentLaboratory permutation pipeline to extract code from the JSON envelope first.
- Kept legacy fenced/raw-Python extraction as a fallback for provider drift and backward compatibility.
- Updated `llm-puzzles/CallLLM.py` self-check and extraction path to use the same shared contract.
- Updated default/custom prompts to require the new machine-readable envelope.
- Added regression tests for direct JSON envelopes and fenced JSON envelopes with surrounding prose.

## Canonical response shape

```json
{
  "version": "code_response.v2",
  "artifact_type": "python_module",
  "language": "python",
  "filename": "solve_module.py",
  "code": "...full python file..."
}
```

## Architecture

1. **Single source of truth for the contract**  
   `llm_code_contract.py` now owns:
   - prompt contract text
   - JSON example payload
   - structured envelope parsing
   - legacy fenced/raw extraction fallback
   - optional code cleanup helpers

2. **Structured-first extraction**  
   All main codegen paths now attempt extraction in this order:
   - strict JSON envelope
   - fenced JSON envelope
   - balanced JSON object embedded in prose
   - legacy fenced Python
   - legacy raw Python fallback

3. **Backward compatibility preserved**  
   Existing providers that still emit markdown/code fences can continue to work, but compliant providers can now be parsed deterministically with no dependence on fence placement or surrounding explanation text.

## Why this is stronger

- The parser no longer depends on markdown fence correctness.
- Code is carried in a dedicated JSON field rather than mixed with prose.
- Prompts and extractors now share one canonical contract.
- The same contract is used in both the main pipeline and the lightweight self-check path.
