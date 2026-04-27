# Fix: auto-heal generated solvers that call `json.loads(sys.argv)`

## Problem

In the notebook run, the first generated Megaminx solver failed validation with:

- `TypeError: the JSON object must be str, bytes or bytearray, not list`
- failing line: `vec = json.loads(sys.argv)`

The validator passes the puzzle vector as a JSON string in `sys.argv[1]`, so parsing the whole `sys.argv` list is invalid.

## Root cause

The LLM occasionally emits a syntactically valid but semantically wrong CLI entrypoint:

```python
vec = json.loads(sys.argv)
```

Because this code still compiles, the defect was only caught during runtime validation, which meant the pipeline could burn multiple fixer iterations on a trivial, recurring pattern.

## Fix

Added a deterministic normalization step in `AgentLaboratory/perm_pipeline/run_perm_pipeline.py` inside `_sanitize_candidate_python(...)`.

The sanitizer now rewrites these specific bad CLI JSON-loading patterns before compile/validation:

- `json.loads(sys.argv)` -> `json.loads(sys.argv[1])`
- `json.loads(argv)` -> `json.loads(argv[1])`

The rewrite is AST-based, so it only touches real Python calls instead of blindly replacing text in comments or strings.

## Why this is the right fix

- fixes the exact notebook failure deterministically;
- addresses the source of the failure in the generation pipeline, not just one generated file;
- reduces wasted fixer iterations on a known low-value error class;
- preserves the existing solver contract and validator flow.

## Validation

Added regression tests in `tests_test_codegen_pipeline.py` covering both:

- `json.loads(sys.argv)`
- `json.loads(argv)`
