# Megaminx self-improving prompt architecture

## Why this layer was added

The repository already had two useful improvement loops:

1. **inner AgentLaboratory loop** — planner/coder/fixer refine one candidate solver;
2. **outer `--keep-improving` loop** — repeated validated solver generation rounds.

But the outer loop previously reused nearly the same prompt bundle on every round. That means the code evolved, while the framing of the next attempt stayed mostly static.

For Megaminx this is a real bottleneck, because local word compression often improves only after the model is explicitly told to move to a *different* deterministic optimization family rather than keep polishing the same idea.

## New architecture

A new competition-specific module was added:

- `competitions/cayley-py-megaminx/prompt_self_improver.py`

It sits **between** the outer improvement loop and the AgentLaboratory generation loop.

### Data flow per round

1. The current **best validated solver** becomes the round baseline.
2. The self-improver inspects that solver code and extracts a compact structural snapshot:
   - exact lookup first or not,
   - exact short-word atlas or not,
   - bounded local DP or not,
   - multi-policy sweep or not,
   - bidirectional rewrite or not,
   - macro mining or not,
   - candidate-bank scoring or not,
   - key tuning constants.
3. The self-improver also reads recent round history:
   - accepted rounds,
   - validated-but-not-selected rounds,
   - outright failures.
4. From that information it synthesizes a **new round-specific prompt bundle**:
   - updated `user_prompt.txt`,
   - updated `custom_prompts.json`,
   - metadata sidecar with selected architecture deltas.
5. AgentLaboratory runs with that synthesized bundle and the current best solver injected as baseline.
6. The candidate is validated and scored.
7. If accepted, it becomes the new baseline for the next round.
8. The next round receives a prompt that is now conditioned on the new best solver and prior failures.

## Design principle

The new layer does **not** try to make the solver “search forever” inside one generation.
Instead it implements an **open-ended outer evolutionary loop**:

- each round is finite,
- each round is validated,
- each round has explicit architectural deltas,
- the sequence of rounds can be extended arbitrarily.

That gives the “infinite self-improvement” behavior in practice while keeping every individual step bounded and competition-safe.

## What makes the prompts materially different each round

The self-improver does not simply append “improve more”.
It forces each round to target concrete Megaminx-specific deltas such as:

- multi-policy commuting-order sweeps,
- bidirectional bounded local rewrites,
- stronger exact short-effect atlases,
- small-support commutator/conjugate mining,
- deterministic candidate-bank scoring.

It also rotates away from directives used in recent rounds, so the next prompt is nudged toward **alternative algorithmic families**, not just constant retuning.

## Safety / competition invariants

The synthesized prompts explicitly preserve:

- exact lookup first,
- legal official move names only,
- deterministic replay,
- standard-library-only implementation,
- bounded local optimization only,
- no generic whole-state Megaminx search.

## Files emitted during runs

When enabled, the pipeline writes:

- `<solver_stem>_prompt_rounds/round_XXXX_user_prompt.txt`
- `<solver_stem>_prompt_rounds/round_XXXX_custom_prompts.json`
- `<solver_stem>_prompt_rounds/round_XXXX_meta.json`
- `<solver_stem>_prompt_evolution.json`

These files make the prompt evolution auditable and reproducible.

## CLI flag

Enable the layer with:

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --prompt-variant master_hybrid \
  --keep-improving \
  --improvement-rounds 8 \
  --self-improve-prompts \
  --output competitions/cayley-py-megaminx/submissions/submission.csv
```

The same flag is available for `generate-solver`.
