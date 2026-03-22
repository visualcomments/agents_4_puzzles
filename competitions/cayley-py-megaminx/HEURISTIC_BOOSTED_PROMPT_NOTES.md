# Heuristic-boosted Megaminx prompt variant

This prompt bundle is designed to steer the planner / coder / fixer pipeline toward the best competition-safe heuristics we identified while improving bundled Megaminx submissions:

- derive the official commutation graph from `puzzle_info.json`;
- exploit order-5 face reduction aggressively;
- normalize commuting blocks under multiple deterministic admissible face orders rather than one fixed order;
- evaluate a tiny fixed bank of such orderings per row and keep the shortest valid candidate;
- strengthen exact short-word effect tables while preserving the shortest representative per effect;
- run bidirectional bounded-window local rewrite passes with small fixed limits.

CLI example:

```bash
python pipeline_cli.py generate-solver   --competition cayley-py-megaminx   --out generated/solve_megaminx_heuristic_boosted.py   --prompt-variant heuristic_boosted
```
