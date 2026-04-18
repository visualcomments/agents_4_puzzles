# Megaminx Breakthrough Stack — пошаговое руководство под ключ

## Для чего этот гайд

Этот документ — не обзор всех prompt-variant'ов, а **пошаговая инструкция по полной настройке самых сильных сценариев** в текущем Megaminx-стеке:

1. **exact_score_population** — prompt/code evolution с отбором только по реальной метрике.
2. **portfolio_orchestrated** — row-wise best-of merge нескольких lane'ов.
3. **hard_row_routed** — усиленный поиск только по hardest rows.
4. **external_adapter_lane** + **wrapper-backed manifests** — подключение внешних candidate generators.
5. **score_guarded** — безопасный fallback и anti-regression слой.

Идея гайда: поднять воспроизводимый контур, который максимизирует шанс реального снижения move-count, а не просто улучшает текст промпта.

---

## 0. Что вы в итоге получите

После выполнения всех шагов у вас будет готовый pipeline:

- локально воспроизводимый;
- с prompt/code self-improvement;
- с exact validator и row-level scoring;
- с shadow-dev / shadow-holdout splits;
- с portfolio fusion;
- с routed-search для hardest rows;
- с внешними solver lanes через manifests и wrappers;
- с готовыми точками для дальнейшего масштабирования.

---

## 1. Какие сценарии считаются основными

### Основной production stack

Используйте в качестве основы именно эту связку:

- `score_guarded`
- `exact_score_population`
- `portfolio_orchestrated`
- `hard_row_routed`
- `external_adapter_lane`

### Что делать с классическими prompt-variant'ами

Оставляйте их как внутренние candidate lanes:

- `master_hybrid`
- `heuristic_boosted`
- `dataset_adapted`
- `algorithmic_population`

### Что не использовать как основную ставку

Не делайте основной production-стратегией:

- `regular`
- чистый prompt-only refinement без exact-score gating
- одиночный lane без portfolio merge

---

## 2. Архитектура целевого контура

Итоговая схема должна быть такой:

1. **Offline optimizer** собирает и улучшает базовые assets.
2. **Prompt/code generators** создают нескольких кандидатов.
3. **Validator** отсеивает всё невалидное.
4. **Exact scorer** считает row-level quality.
5. **Portfolio orchestrator** выбирает лучший путь по каждой строке.
6. **Hard-row routed search** тратит дорогой compute только на worst rows.
7. **External adapter lane** добавляет внешние candidate generators.
8. Только после этого собирается новый `optimized_submission.csv` и `optimized_lookup.json`.

---

## 3. Что должно быть в репозитории

Проверьте, что у вас есть следующие файлы и модули:

### Базовый Megaminx bundle

- `competitions/cayley-py-megaminx/data/test.csv`
- `competitions/cayley-py-megaminx/data/sample_submission.csv`
- `competitions/cayley-py-megaminx/data/puzzle_info.json`
- `competitions/cayley-py-megaminx/validate_solve_output.py`
- `competitions/cayley-py-megaminx/build_optimized_assets.py`

### Новые breakthrough-модули

- `competitions/cayley-py-megaminx/row_scoreboard.py`
- `competitions/cayley-py-megaminx/portfolio_orchestrator.py`
- `competitions/cayley-py-megaminx/hard_row_routed_search.py`
- `competitions/cayley-py-megaminx/prompt_population_runner.py`
- `competitions/cayley-py-megaminx/shadow_splits.json`
- `competitions/cayley-py-megaminx/external_adapter_lane.py`

### External adapter слой

- `competitions/cayley-py-megaminx/external_solver_adapters/README.md`
- `competitions/cayley-py-megaminx/external_solver_adapters/manifest_*.example.json`
- `competitions/cayley-py-megaminx/external_solver_adapters/wrappers/wrapper_common.py`
- `run_megaminxolver_wrapper.py`
- `run_llminxsolver_wrapper.py`
- `run_solve_twisty_puzzles_wrapper.py`
- `run_megaminx_simulator_astar_wrapper.py`

### Prompt bundles

- `score_guarded`
- `exact_score_population`
- `portfolio_orchestrated`
- `hard_row_routed`
- плюс classic lanes: `master_hybrid`, `heuristic_boosted`, `dataset_adapted`, `algorithmic_population`

---

## 4. Пошаговая настройка с нуля

## Шаг 1. Подготовьте окружение

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-full.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-full.txt
```

### Минимальная проверка

```bash
python -V
pip -V
pytest --version
```

---

## Шаг 2. Поставьте CayleyPy

Если вы используете CayleyPy-backed search lanes:

```bash
pip install git+https://github.com/cayleypy/cayleypy.git
```

Проверьте импорт:

```bash
python -c "import cayleypy; print('cayleypy ok')"
```

---

## Шаг 3. Проверьте, что pipeline видит Megaminx

```bash
python pipeline_cli.py list-pipelines
python pipeline_cli.py show-pipeline --competition cayley-py-megaminx
```

Ожидаемый результат:

- competition зарегистрирован;
- видны baseline solver и validator;
- prompt routing работает.

---

## Шаг 4. Прогоните обязательный smoke-check

```bash
pytest -q
```

Если тестов слишком много и вы хотите быстрый sanity check, начните с Megaminx-набора:

```bash
pytest -q -k megaminx
```

---

## Шаг 5. Пересоберите базовые assets

Сначала надо построить чистую отправную точку.

```bash
cd competitions/cayley-py-megaminx
python build_optimized_assets.py
cd ../..
```

После этого проверьте, что появились или обновились:

- `optimized_submission.csv`
- `optimized_lookup.json`
- `optimized_stats.json`

---

## 5. Настройка exact-score foundation

## Шаг 6. Постройте row-level scoreboard

Смысл этого шага: перестать смотреть только на суммарный score и перейти к пониманию, **какие именно строки самые дорогие**.

Если в вашем модуле уже есть CLI, запускайте его. Базовый смысл такой:

```bash
python competitions/cayley-py-megaminx/row_scoreboard.py \
  --submission competitions/cayley-py-megaminx/data/optimized_submission.csv \
  --output competitions/cayley-py-megaminx/data/row_scoreboard.json
```

После запуска у вас должен появиться профиль вида:

- `row_id`
- `path_len`
- `rank`
- `bucket`
- `is_hard_row`

Если файл не генерируется, сначала убедитесь, что `optimized_submission.csv` действительно существует.

---

## Шаг 7. Проверьте shadow splits

`shadow_splits.json` должен разбивать bundle хотя бы на:

- `shadow_dev`
- `shadow_holdout`
- опционально `public_full`

Проверьте, что:

- строки не пересекаются;
- `shadow_dev` содержит действительно трудные строки;
- `shadow_holdout` не слишком маленький.

Если нужен быстрый sanity check, откройте файл и убедитесь, что там есть непустые массивы row ids.

---

## 6. Настройка prompt/code self-improvement

## Шаг 8. Поднимите безопасный anti-regression слой: `score_guarded`

Это не финальный breakthrough-режим, а обязательный стабилизатор.

### Запуск

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --prompt-variant score_guarded \
  --keep-improving \
  --self-improve-prompts \
  --improvement-rounds 3 \
  --output generated/solve_module_score_guarded.py
```

### Что проверить

- файл solver сгенерировался;
- история prompt evolution записалась;
- нет деградации по validator pass;
- не сломан JSON contract.

### Зачем это нужно

`score_guarded` нужен как безопасная стартовая оболочка перед более агрессивными exact-score/population режимами.

---

## Шаг 9. Переключитесь на `exact_score_population`

Это уже главный режим для prompt/code evolution.

### Запуск

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --prompt-variant exact_score_population \
  --keep-improving \
  --self-improve-prompts \
  --improvement-rounds 5 \
  --output generated/solve_module_exact_score_population.py
```

### Правильный режим

Используйте его **не как одиночный one-shot prompt**, а как population candidate source.

### После генерации обязательно проверить

```bash
python -m py_compile generated/solve_module_exact_score_population.py
```

И затем validator:

```bash
python competitions/cayley-py-megaminx/validate_solve_output.py \
  --solver generated/solve_module_exact_score_population.py
```

### Критерий принятия

Принятие новой версии должно происходить только если одновременно выполнено всё:

- compile success;
- validator success;
- no JSON contract regressions;
- better exact score на `shadow_dev`;
- zero critical regressions на `shadow_holdout`.

---

## 7. Настройка portfolio orchestration

## Шаг 10. Соберите внутренние candidate lanes

Перед portfolio merge у вас должно быть несколько реальных кандидатов.

Минимальный рекомендуемый набор:

- `master_hybrid`
- `heuristic_boosted`
- `dataset_adapted`
- `algorithmic_population`
- `score_guarded`
- `exact_score_population`

### Пример генерации нескольких запусков

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --prompt-variant master_hybrid \
  --output competitions/cayley-py-megaminx/submissions/submission_master_hybrid.csv

python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --prompt-variant heuristic_boosted \
  --output competitions/cayley-py-megaminx/submissions/submission_heuristic_boosted.csv

python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --prompt-variant dataset_adapted \
  --output competitions/cayley-py-megaminx/submissions/submission_dataset_adapted.csv
```

---

## Шаг 11. Запустите `portfolio_orchestrated`

Идея: выбрать лучший путь по **каждой строке**, а не лучший общий lane вручную.

### Пример запуска

```bash
python competitions/cayley-py-megaminx/portfolio_orchestrator.py \
  --inputs \
    competitions/cayley-py-megaminx/submissions/submission_master_hybrid.csv \
    competitions/cayley-py-megaminx/submissions/submission_heuristic_boosted.csv \
    competitions/cayley-py-megaminx/submissions/submission_dataset_adapted.csv \
    competitions/cayley-py-megaminx/submissions/submission_exact_score_population.csv \
  --shadow-splits competitions/cayley-py-megaminx/shadow_splits.json \
  --output competitions/cayley-py-megaminx/submissions/submission_portfolio.csv
```

### Что должен делать orchestrator

- валидировать все входы;
- отбрасывать невалидные строки;
- считать row-level score;
- выбирать shortest valid path per row;
- фиксировать lineage и происхождение победителя по каждой строке.

### Что проверить после запуска

- появился `submission_portfolio.csv`;
- есть сводка по rows improved;
- видно, какой lane выиграл по каждой строке.

---

## 8. Настройка hard-row routed search

## Шаг 12. Выделите hardest rows

На этом этапе вам нужен row scoreboard из шага 6.

Цель: составить подмножество строк, на которых:

- path length максимален;
- ожидаемый выигрыш самый большой;
- локальные rewrite'ы уже исчерпаны.

Практически начните с top-50 или top-100 hardest rows.

---

## Шаг 13. Запустите routed-search только на hardest rows

```bash
python competitions/cayley-py-megaminx/hard_row_routed_search.py \
  --row-scoreboard competitions/cayley-py-megaminx/data/row_scoreboard.json \
  --shadow-splits competitions/cayley-py-megaminx/shadow_splits.json \
  --top-k 50 \
  --output competitions/cayley-py-megaminx/submissions/submission_hard_row_routed.csv
```

### Что должно происходить внутри

- строки сортируются по сложности;
- дорогой search budget тратится только на hardest shard;
- для каждого hard row пробуются разные policy configs;
- результаты снова проходят через validator и exact-score gating.

### Как понять, что режим работает правильно

Смотрите не на общий объём compute, а на:

- `saved_moves_per_cpu_hour`
- `rows_improved`
- `median_saved_moves_on_hard_rows`

Если routed-search не даёт выигрыша на top-50 hardest rows, его не стоит масштабировать на весь bundle.

---

## 9. Подключение внешних candidate generators

## Шаг 14. Подготовьте каталог `external/`

Рядом с проектом создайте каталог, в котором будут лежать внешние checkout'ы:

```text
external/
  MegaminXolver/
  llminxsolver/
  solve-twisty-puzzles/
  Megaminx-Simulator-AStar/
```

Названия могут немного отличаться, но лучше придерживаться стабильной структуры.

---

## Шаг 15. Используйте wrapper-backed manifests

Не редактируйте ранний шаблонный слой вручную, а стартуйте от готовых manifests:

- `manifest_megaminxolver.example.json`
- `manifest_llminxsolver.example.json`
- `manifest_solve_twisty_puzzles.example.json`
- `manifest_megaminx_simulator_astar.example.json`

### Что сделать

1. Скопируйте нужный `.example.json` в рабочий manifest.
2. Пропишите путь к checkout внешнего репозитория.
3. Убедитесь, что wrapper действительно может его запустить.

Пример:

```bash
cp competitions/cayley-py-megaminx/external_solver_adapters/manifest_megaminxolver.example.json \
   competitions/cayley-py-megaminx/external_solver_adapters/manifest_megaminxolver.json
```

---

## Шаг 16. Проверьте wrapper отдельно

Каждый wrapper должен уметь:

- autodiscovery entrypoint'ов;
- fallback на готовые `submission.csv` / `*.jsonl` / `*candidates*.csv`;
- нормализовать результат в Kaggle-style CSV `initial_state_id,path`.

Прогоняйте wrapper отдельно до интеграции в orchestrator.

Общий шаблон:

```bash
python competitions/cayley-py-megaminx/external_solver_adapters/wrappers/run_megaminxolver_wrapper.py \
  --manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_megaminxolver.json \
  --output /tmp/megaminxolver_submission.csv
```

После запуска проверьте, что CSV:

- существует;
- имеет заголовок `initial_state_id,path`;
- не пустой.

---

## Шаг 17. Подключите `external_adapter_lane`

Теперь уже можно пускать внешний lane через официальный validator и move normalization.

```bash
python competitions/cayley-py-megaminx/external_adapter_lane.py \
  --external-manifest competitions/cayley-py-megaminx/external_solver_adapters/manifest_megaminxolver.json \
  --fallback-submission competitions/cayley-py-megaminx/data/optimized_submission.csv \
  --output competitions/cayley-py-megaminx/submissions/submission_external_megaminxolver.csv
```

### Что происходит внутри

- wrapper создаёт внешний candidate CSV;
- adapter нормализует внешнюю нотацию в официальные ходы;
- строки проходят replay-validation по `puzzle_info.json`;
- invalid / missing rows заменяются fallback submission;
- результат становится полноценным lane для portfolio merge.

---

## Шаг 18. Добавьте внешние lanes в portfolio

Как только external lanes отдают валидный CSV, их нужно добавлять в `portfolio_orchestrated` на равных правах с внутренними lanes.

Пример:

```bash
python competitions/cayley-py-megaminx/portfolio_orchestrator.py \
  --inputs \
    competitions/cayley-py-megaminx/submissions/submission_master_hybrid.csv \
    competitions/cayley-py-megaminx/submissions/submission_exact_score_population.csv \
    competitions/cayley-py-megaminx/submissions/submission_external_megaminxolver.csv \
    competitions/cayley-py-megaminx/submissions/submission_external_llminxsolver.csv \
  --shadow-splits competitions/cayley-py-megaminx/shadow_splits.json \
  --output competitions/cayley-py-megaminx/submissions/submission_portfolio_with_external.csv
```

---

## 10. Настройка через ноутбук

Если вы используете обновлённый ноутбук, выставляйте параметры так.

### Режим точного population search

```python
PROMPT_VARIANT = "exact_score_population"
BREAKTHROUGH_SCENARIO = "exact_score_population"
SHADOW_SPLITS_PATH = "competitions/cayley-py-megaminx/shadow_splits.json"
```

### Режим portfolio orchestration

```python
PROMPT_VARIANT = "portfolio_orchestrated"
BREAKTHROUGH_SCENARIO = "portfolio_orchestrated"
BREAKTHROUGH_TOP_K = 5
```

### Режим hard-row routing

```python
PROMPT_VARIANT = "hard_row_routed"
BREAKTHROUGH_SCENARIO = "hard_row_routed"
```

### Режим external adapters

```python
BREAKTHROUGH_SCENARIO = "external_adapter_lane"
EXTERNAL_MANIFESTS = [
    "competitions/cayley-py-megaminx/external_solver_adapters/manifest_megaminxolver.json",
    "competitions/cayley-py-megaminx/external_solver_adapters/manifest_llminxsolver.json",
]
EXTERNAL_FALLBACK_SUBMISSION = "competitions/cayley-py-megaminx/data/optimized_submission.csv"
EXTERNAL_OUT_DIR = "competitions/cayley-py-megaminx/submissions/external"
```

---

## 11. Рекомендуемый полный запуск под ключ

Ниже — рекомендуемый порядок реального production-like прогона.

## Фаза A. Основа

1. Установить зависимости.
2. Установить CayleyPy.
3. Прогнать тесты.
4. Пересобрать базовые assets.
5. Построить row scoreboard.
6. Проверить shadow splits.

## Фаза B. Внутренние lanes

7. Прогнать `master_hybrid`.
8. Прогнать `heuristic_boosted`.
9. Прогнать `dataset_adapted`.
10. Прогнать `score_guarded`.
11. Прогнать `exact_score_population`.

## Фаза C. Hard rows

12. Выделить top hardest rows.
13. Запустить `hard_row_routed` на hardest shard.

## Фаза D. Внешние lanes

14. Подготовить checkout'ы внешних репозиториев.
15. Настроить manifests.
16. Отдельно проверить wrappers.
17. Прогнать `external_adapter_lane`.

## Фаза E. Merge и отбор

18. Запустить `portfolio_orchestrated` на всех внутренних и внешних lanes.
19. Проверить final score и row improvements.
20. Если результат лучше incumbent — только тогда продвигать assets.

---

## 12. Жёсткие правила приёмки результатов

Новая версия должна приниматься только если одновременно выполняется всё:

- solver компилируется;
- validator проходит;
- JSON envelope не сломан;
- на `shadow_dev` score лучше incumbent;
- на `shadow_holdout` нет критических regressions;
- при portfolio merge сохранены только shortest valid paths;
- внешние lanes не тащат невалидные ходы;
- сохраняется provenance каждой улучшенной строки.

---

## 13. Что делать, если результатов нет

### Если `exact_score_population` не даёт выигрыша

Проверьте:

- не принимает ли self-improver кандидаты по старому rubric вместо exact metric;
- достаточно ли сильный `shadow_dev`;
- не слишком ли мало раундов;
- не завышен ли threshold принятия.

### Если `hard_row_routed` не даёт выигрыша

Проверьте:

- правильно ли выделены hardest rows;
- не слишком ли маленький `top-k`;
- не исчерпана ли pure-local-rewrite стратегия;
- не пора ли усиливать external lanes.

### Если `external_adapter_lane` ничего не улучшает

Проверьте:

- действительно ли внешний solver отдаёт короткие пути;
- корректно ли нормализуются ходы;
- не заменяются ли почти все строки fallback submission'ом;
- не сломан ли wrapper.

---

## 14. Когда переключаться на ещё более агрессивную стратегию

Переходите в приоритет `external_adapter_lane` + full portfolio, если:

- после запуска `portfolio_orchestrated` + `hard_row_routed` score всё ещё далёк от целевого;
- внутренние lanes повторяют один и тот же family of local rewrites;
- `saved_moves_per_cpu_hour` на hardest rows резко падает;
- `rows_improved` почти перестал расти.

---

## 15. Минимальный checklist перед большим прогоном

Перед полноценным ночным/длинным прогоном проверьте:

- [ ] зависимости установлены;
- [ ] CayleyPy импортируется;
- [ ] тесты проходят;
- [ ] assets пересобраны;
- [ ] `shadow_splits.json` валиден;
- [ ] row scoreboard построен;
- [ ] `score_guarded` запускается;
- [ ] `exact_score_population` проходит validator;
- [ ] `portfolio_orchestrated` собирает итоговый CSV;
- [ ] `hard_row_routed` работает хотя бы на top-10;
- [ ] внешний wrapper хотя бы один раз отдал валидный CSV;
- [ ] `external_adapter_lane` корректно подставляет fallback на invalid rows.

---

## 16. Итоговая рекомендуемая конфигурация

Если нужен один рекомендуемый практический стек, запускайте так:

### Prompt/code layer

- основной evolving variant: `exact_score_population`
- защитный fallback: `score_guarded`

### Search/orchestration layer

- merge: `portfolio_orchestrated`
- targeted expensive search: `hard_row_routed`

### External candidates

- `external_adapter_lane`
- wrapper-backed manifests
- минимум два внешних lane'а

### Acceptance rule

- только exact metric
- только через validator
- только без regressions на holdout

---

## 17. Что сохранять после каждого большого прогона

Обязательно сохраняйте:

- итоговый `submission.csv`
- row-level scoreboard
- список improved rows
- lineage победителей по строкам
- used prompt variants
- used external manifests
- used notebook parameters
- build logs
- validator logs
- summary по `saved_moves_per_cpu_hour`

Без этого невозможно понять, какой именно компонент дал прогресс.

---

## 18. Практический финальный совет

Если цель — не просто "получить рабочий run", а **максимизировать шанс прорывного снижения move-count**, не пытайтесь дожать всё одним супер-промптом.

Рабочая стратегия такая:

1. построить **exact-score acceptance loop**;
2. собрать **несколько внутренних lanes**;
3. добавить **hard-row routed search**;
4. подключить **внешние candidate generators**;
5. объединять всё через **portfolio_orchestrated**;
6. принимать изменения только по **реальной метрике и validator'у**.

Именно такой стек лучше всего подходит для реального движения к очень низкому move-count в вашем Megaminx-репозитории.
