# 2026-04-01 — Safe extraction for syntactically valid code with docstring-only bodies

## Problem

A correct Python module returned by the LLM could be rejected before validation because the shared extractor always tried to strip comments and docstrings from candidate code.

That is usually harmless, but it can break otherwise valid Python when a module/class/function body relies on a docstring as its only statement, for example:

```python
class SolverError(Exception):
    """Raised when the solver contract is violated."""
```

After aggressive stripping, this became:

```python
class SolverError(Exception):
```

which is invalid syntax. The pipeline would then fail compile/validation even though the model had produced correct code.

## Fix

Implemented a shared helper in `llm_code_contract.py` that only keeps the stripped version when it still parses successfully.

Behavior now:

- Prefer a stripped candidate only when the stripped candidate still parses.
- Otherwise keep the original extracted Python exactly as returned by the LLM.
- Apply this rule consistently to:
  - JSON code envelopes
  - fenced code blocks
  - raw Python candidate extraction

## Files changed

- `llm_code_contract.py`
- `tests_test_codegen_pipeline.py`
- `tests_test_low_ram_optimizations.py`

## New regression coverage

Added tests proving that valid code with a docstring-only class body still compiles after extraction in both:

- the main permutation pipeline extractor
- the legacy `CallLLM` extraction path
