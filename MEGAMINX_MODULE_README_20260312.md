# Megaminx module added

This repository now contains a real offline competition bundle for `cayley-py-megaminx`:
- official `data/puzzle_info.json`
- official `data/test.csv`
- official `data/sample_submission.csv`
- a validator that checks generator legality and whether the path reaches `central_state`
- a baseline solver that replays the official sample path for any known test state
- prompts tailored to the actual Megaminx rules and data format
- submission helpers and regression tests

## Prompt modes

The original Megaminx prompt mixed two different workflows:
- a **regular coder flow**, where the model should write `solve_module.py` independently; and
- a **baseline patcher flow**, where a known solver may be shown and minimally edited.

That mismatch was problematic because the main coder stage did not actually receive the baseline source code, while the prompt still told the model to improve it.

This repo now includes:
- `prompts/user_prompt.txt` — the default **regular** Megaminx prompt for writing the solver from scratch;
- `prompts/user_prompt_baseline_patch.txt` — the preserved legacy prompt for explicit baseline-improvement experiments;
- `prompts/custom_prompts_regular.json` — regular system prompts that do not assume a hidden baseline exists.

## From-scratch generation

To force AgentLaboratory to avoid baseline-code injection during generation, use:

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_regular.py \
  --prompt-file competitions/cayley-py-megaminx/prompts/user_prompt.txt \
  --custom-prompts competitions/cayley-py-megaminx/prompts/custom_prompts_regular.json \
  --from-scratch \
  --strict
```

`--from-scratch` disables the baseline-patcher branch inside AgentLaboratory, so the model writes code independently instead of being asked to patch a hidden baseline. Pair it with `--strict` if you also want to avoid the final offline fallback to the checked-in baseline solver.
