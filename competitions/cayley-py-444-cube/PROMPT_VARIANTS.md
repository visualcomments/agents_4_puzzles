# 444-cube prompt variants

This directory contains the Megaminx prompt-sweep prompt families adapted for `cayley-py-444-cube`.

The important 444-specific changes are:

- target competition slug: `cayley-py-444-cube`;
- state vector length: 96;
- submission schema: `initial_state_id,path`;
- legal moves: exactly the `generators` keys in `data/puzzle_info.json`;
- optimizer assumptions must be derived from official permutations, not from Megaminx order-5 face turns;
- every candidate path must be replay-validated to `central_state` before promotion.

The sweep runner discovers every `user_prompt_<variant>.txt` / `custom_prompts_<variant>.json` pair in this folder when `--variants all` is used.
