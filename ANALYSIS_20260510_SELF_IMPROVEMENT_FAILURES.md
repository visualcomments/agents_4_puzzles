# Анализ прогонов самоулучшающихся промптов от 2026-05-09

## Входы

Проанализированы:

- `agents_4_puzzles-main (1).zip`
- `self_improvement_run_20260509_143226.zip`
- `self_improvement_run_20260509_230905.zip`

Целевая задача: `competitions/cayley-py-megaminx`.

## Что показали результаты

Все финальные submission-файлы из обоих прогонов имеют тот же локальный score, что и baseline:

| run | scenario | rows | score | max row len | sha256 |
| --- | --- | ---: | ---: | ---: | --- |
| 20260509_143226 | basic_strict_self_improvement | 1001 | 414305 | 839 | `07ae02aea834898a...` |
| 20260509_143226 | advanced_failure_aware_self_improvement | 1001 | 414305 | 839 | `07ae02aea834898a...` |
| 20260509_143226 | baseline_for_submit | 1001 | 414305 | 839 | `07ae02aea834898a...` |
| 20260509_230905 | basic_strict_self_improvement | 1001 | 414305 | 839 | `07ae02aea834898a...` |
| 20260509_230905 | advanced_failure_aware_self_improvement | 1001 | 414305 | 839 | `07ae02aea834898a...` |
| 20260509_230905 | baseline_for_submit | 1001 | 414305 | 839 | `07ae02aea834898a...` |
| repo baseline | `.score_megaminx_best_tested_solver.csv` | 1001 | 414305 | 839 | `828b3ba37ee2b65...` |

Байтовый hash baseline отличается из-за представления файла, но каноническое сравнение `(initial_state_id, path)` показывает: все submission-файлы прогонов path-wise идентичны baseline. Детали сохранены в `SELF_IMPROVEMENT_EVIDENCE_SUMMARY_20260510.json`.

`self_improvement_summary.json` в обоих архивах показывает `baseline_score = 414305`, `eligible_candidates = []`, `min_improvement = 1`. Поэтому auto-submit корректно не выбрал кандидата.

## Почему не получилось превзойти baseline

1. **Preflight модели проверял только conversational ping.**  
   В обоих архивах `logs/model_preflight.json` проверял prompt `ping`, а `r1-1776` отвечала объяснением ping. Это не доказывает способность модели вернуть строгий JSON/Python-кандидат. В первом прогоне basic-ветка потратила три раунда на `CalledProcessError`, потому что `run_perm_pipeline.py` не получил валидный Python-файл.

2. **Промпт-эволюция порождала baseline replay вместо новой row-level оптимизации.**  
   Во втором basic-прогоне и во всех advanced-раундах кандидаты валидировались, но каждый round имел `score = 414305`, `accepted = false`, `rejection_reasons = ['no_novelty_identical_submission', 'no_per_row_improvement']`. Guard работал правильно: он не пропустил submission без улучшенной строки.

3. **Baseline уже является сильным lookup-first solver.**  
   `competitions/cayley-py-megaminx/megaminx_best_tested_solver.py` в первую очередь возвращает заранее найденный path из lookup/submission-артефактов. Минимальная обёртка вокруг такого solver почти неизбежно даёт тот же CSV, если в промпте нет жёсткого требования сделать guarded post-lookup optimization или заменить хотя бы одну hard row exact-valid путём.

4. **В архивы прогонов не попадали самые полезные артефакты отладки.**  
   В `run_log` были ссылки на `generated/..._candidate_archive`, prompt-round файлы и кандидатные solver-файлы, но сами эти директории отсутствовали в run ZIP. Это затрудняло postmortem: было видно, что кандидаты не улучшили baseline, но не всегда сохранялись raw responses/extracted code/fixer reports.

5. **`--max-iters=100000` усиливал стоимость неправильной модели.**  
   При модели, которая не выдаёт Python, такой лимит превращает fail-fast ситуацию в долгий цикл fixer/codegen без реального прогресса.

6. **В adaptive row routing была ошибка масштаба.**  
   `search_improver_v3.py` создавал `RowFeatures` без `total_rows=top_k`. В результате `difficulty_percentile()` использовал default `100`; на full/top-k=1001 строках хвостовые строки получали искажённый percentile, что портило распределение effort по hard rows.

## Что изменено в репозитории

### `pipeline_cli.py`

- `check-g4f-models` получил флаги:
  - `--require-code-envelope`
  - `--full-detail`
- При `--require-code-envelope` probe теперь требует извлекаемый Python-код с `def solve` и `sorted_array` через существующий `code_contract.extract_python_candidate`.
- Probe больше не режет внутренний `detail` до 80 символов; для console output используется `preview`, а в JSON сохраняется полный ответ при необходимости.
- Discovery моделей теперь сохраняет registry/fallback порядок после de-duplication вместо жёсткой preference-сортировки, чтобы не проталкивать conversational-only модели до проверки code capability.

### `scripts/run_self_improvement_scenarios_kaggle_inline.py`

- Добавлен строгий code-envelope preflight поверх старого ping-preflight.
- Runner пишет `logs/model_preflight_code.json`.
- Если есть модели, прошедшие code preflight, runner автоматически использует только их как `effective_models`.
- Если ни одна модель не прошла code preflight, runner fail-fast, если явно не заданы `--allow-non-code-models` или `--skip-code-preflight`.
- Default `--max-iters` снижен с `100000` до `12`.
- Добавлен сбор missing artifacts в директорию каждого scenario run: prompt rounds, candidate archives, candidate solvers, best submission, baseline snapshot, prompt evolution JSON. Это делает будущие ZIP-архивы пригодными для диагностики без доступа к исходной рабочей директории.

### `competitions/cayley-py-megaminx/prompt_self_improver.py`

- Добавлен блок `Post-lookup novelty requirement`.
- Planner/coder/fixer prompts теперь явно запрещают выдавать pure `optimized_submission` wrapper как улучшение.
- Для breakthrough после повторных `no_novelty/no_per_row_improvement` prompt переводится в `lane_hard_row_micro`: выбрать 1 hard row, сделать bounded segment optimization, exact replay, rollback on no improvement, вывести trace/delta.

### `competitions/cayley-py-megaminx/search_improver_v3.py`

- Исправлен `RowFeatures(..., total_rows=top_k)`, чтобы percentile routing соответствовал реальному числу строк в прогоне.

## Рекомендованный следующий basic-прогон

Basic-сценарий должен оставаться консервативным: не переписывать весь solver, а получить одну exact-valid per-row дельту поверх lookup baseline.

```bash
python scripts/run_self_improvement_scenarios_kaggle_inline.py \
  --scenario basic \
  --competition cayley-py-megaminx \
  --models '<comma-separated candidate models>' \
  --rounds-basic 5 \
  --max-iters 12 \
  --auto-submit-kaggle \
  --submit-min-improvement 1
```

Ключевые условия:

- не использовать `--skip-code-preflight` для реального прогона;
- не использовать `--allow-non-code-models`, если цель не диагностика;
- считать успехом только `score < 414305` и `improved_rows >= 1`;
- first target: hard-row micro improvement, exact replay, rollback if candidate length is not shorter.

## Рекомендованный breakthrough/advanced-прогон

Advanced-сценарий должен быть не «ещё одной переформулировкой baseline», а портфелем точечных exact-valid lane-кандидатов:

```bash
python scripts/run_self_improvement_scenarios_kaggle_inline.py \
  --scenario advanced \
  --competition cayley-py-megaminx \
  --models '<comma-separated candidate models>' \
  --rounds-advanced 8 \
  --max-iters 12 \
  --auto-submit-kaggle \
  --submit-min-improvement 1
```

Рекомендуемая структура advanced lane:

1. выбрать 10-30 самых длинных или historically hard rows;
2. для каждой строки держать incumbent lookup path как oracle/fallback;
3. запускать bounded deterministic search only around segments, not whole-solver paraphrase;
4. сохранять trace: target row, baseline_len, candidate_len, replay_ok, accepted/rejected reason;
5. делать portfolio fusion только из exact-valid shorter paths;
6. если два round подряд дали no-novelty, запретить broad refactor и перейти к single-row micro-target.

## Локальная проверка после правок

Выполнено:

```bash
python -m py_compile pipeline_cli.py \
  scripts/run_self_improvement_scenarios_kaggle_inline.py \
  competitions/cayley-py-megaminx/prompt_self_improver.py \
  competitions/cayley-py-megaminx/search_improver_v3.py \
  AgentLaboratory/perm_pipeline/run_perm_pipeline.py

python -m pytest -q \
  tests_test_g4f_model_check.py \
  tests_test_megaminx_prompt_self_improver.py \
  tests_test_megaminx_search_v3.py
```

Результат: `16 passed`.

## Важное ограничение

Эти правки не являются доказанным новым leaderboard improvement. Они исправляют причины, по которым текущие прогоны застряли на baseline, и делают следующий прогон способным отсеивать не-code-capable модели, сохранять диагностические артефакты и принуждать кандидатов к настоящей post-lookup row-level дельте.
