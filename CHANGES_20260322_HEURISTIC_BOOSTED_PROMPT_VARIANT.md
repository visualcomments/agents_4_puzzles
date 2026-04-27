# Changes: heuristic_boosted Megaminx prompt variant

Added a new CLI-selectable prompt variant for `cayley-py-megaminx`:

- `--prompt-variant heuristic_boosted`

What was added:

- `competitions/cayley-py-megaminx/prompts/user_prompt_heuristic_boosted.txt`
- `competitions/cayley-py-megaminx/prompts/custom_prompts_heuristic_boosted.json`
- `competitions/cayley-py-megaminx/HEURISTIC_BOOSTED_PROMPT_NOTES.md`
- `tests_test_megaminx_prompt_variant_heuristic_boosted.py`

What was changed:

- `pipeline_cli.py` now accepts `heuristic_boosted` in `--prompt-variant` for both `generate-solver` and `run`.
- Megaminx prompt documentation and examples were updated.

Heuristics emphasized by the new prompt bundle:

- official commutation graph derived from `puzzle_info.json`
- order-5 face reduction
- multiple deterministic commuting-order policies
- per-row best-of-fixed-candidates selection
- stronger exact short-word effect tables
- bidirectional bounded-window local rewrites
