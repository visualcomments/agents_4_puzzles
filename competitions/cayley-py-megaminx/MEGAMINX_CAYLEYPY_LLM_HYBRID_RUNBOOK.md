# Megaminx CayleyPy + LLM hybrid strategy

Новый отдельный вариант решения находится здесь:

```text
competitions/cayley-py-megaminx/megaminx_cayleypy_llm_hybrid_solver.py
```

## Что делает стратегия

Архитектура состоит из 4 слоёв:

1. **Deterministic anchors**
   - `megaminx_best_tested_solver.py`
   - `optimized_submission.csv`
   - опционально `search_improver_v3.py`
2. **LLM candidates**
   - умеет подхватывать уже готовые LLM submission-файлы;
   - умеет запускать новые через `pipeline_cli.py run --prompt-variant ...`.
3. **Ensemble / fusion**
   - для каждой строки берёт кратчайший локально валидный путь.
4. **CayleyPy-backed post-refinement**
   - поверх fused submission прогоняется `search_improver_v3.improve_submission_rows(...)`;
   - внутри используется обновлённый `cayley_adapter.py` с:
     - exact BFS neighborhood hit,
     - beam search на target-centered CayleyGraph,
     - internal fallback.

## Вендоринг CayleyPy

В архив репозитория добавлена локальная копия библиотеки:

```text
third_party/cayleypy-main/
```

Это позволяет запускать гибридный режим даже без отдельного `pip install cayleypy`, если зависимости окружения уже есть.

## Быстрый запуск без LLM-генерации

```bash
python competitions/cayley-py-megaminx/megaminx_cayleypy_llm_hybrid_solver.py \
  --run-search-v3 \
  --out competitions/cayley-py-megaminx/submissions/submission_cayleypy_llm_hybrid.csv
```

Этот режим:
- построит fresh best-tested candidate,
- подхватит bundled candidates,
- прогонит search_v3,
- соберёт fused submission,
- сделает post-refine.

## Полный гибрид с LLM-кандидатами

```bash
python competitions/cayley-py-megaminx/megaminx_cayleypy_llm_hybrid_solver.py \
  --generate-llm \
  --llm-variants structured,heuristic_boosted,master_hybrid \
  --agent-models "planner=command-a;coder=command-a;fixer=command-a" \
  --keep-improving-llm \
  --llm-improvement-rounds 8 \
  --run-search-v3 \
  --out competitions/cayley-py-megaminx/submissions/submission_cayleypy_llm_hybrid.csv
```

## Отправка на Kaggle

После локальной валидации можно добавить:

```bash
  --submit \
  --message "cayleypy+llm hybrid validated" \
  --kaggle-json ~/.kaggle/kaggle.json
```

## Полезные артефакты

По умолчанию сохраняются:

- итоговый CSV;
- `submission_cayleypy_llm_hybrid.stats.json`;
- `submission_cayleypy_llm_hybrid.profiles.json`.

В stats лежат:
- список использованных candidate submissions,
- fusion winner counts,
- итог локальной валидации,
- результат post-refinement.
