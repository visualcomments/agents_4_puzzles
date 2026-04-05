# Megaminx hybrid changes summary

Добавлено:

- `competitions/cayley-py-megaminx/megaminx_cayleypy_llm_hybrid_solver.py`
  - комплексный orchestration script;
  - объединяет deterministic candidates, search_v3 и optional LLM submissions;
  - делает row-wise fusion и финальный post-refine.
- `competitions/cayley-py-megaminx/MEGAMINX_CAYLEYPY_LLM_HYBRID_RUNBOOK.md`
- `tp/cp/`
  - локально вендоренная копия CayleyPy;
  - добавлен stub `kagglehub.py` для offline import.

Обновлено:

- `competitions/cayley-py-megaminx/cayley_adapter.py`
  - корректная работа с локальным CayleyPy;
  - target-centered CayleyGraph;
  - exact BFS/MITM pass;
  - beam search fallback.
- `competitions/cayley-py-megaminx/search_improver_v3.py`
- `competitions/cayley-py-megaminx/build_optimized_assets.py`
  - поправлен локальный импорт `solve_module.py`, чтобы не подхватывался корневой `solve_module.py`.

Smoke tests:

- `megaminx_cayleypy_llm_hybrid_solver.py --skip-post-refine`:
  - локальная валидация 1001/1001;
  - score `414305`.
- прямой test вызова `MegaminxSearchAdapter.search(...)` на состоянии в 1 ход от цели:
  - backend: `cayleypy`;
  - exact hit длины `1`.
