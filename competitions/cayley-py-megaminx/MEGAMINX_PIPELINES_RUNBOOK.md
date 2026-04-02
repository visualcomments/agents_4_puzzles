# Megaminx pipelines: подробная инструкция по запуску

Этот файл — практический runbook по всем основным сценариям запуска `cayley-py-megaminx` в этом репозитории.

Он покрывает:
- оффлайн baseline без LLM;
- генерацию солвера через AgentLaboratory;
- все prompt variants;
- отдельный запуск лучшего протестированного солвера;
- пересборку оффлайн-ассетов;
- запуск `v3` post-optimizer;
- валидацию solver-файла;
- примеры для `regular` режима, где модель пишет код **с нуля**, без baseline.

## 1. Где находится Megaminx-пайплайн

Основная директория соревнования:

```bash
competitions/cayley-py-megaminx/
```

Ключевые файлы:

```text
competitions/cayley-py-megaminx/
  data/
    test.csv
    sample_submission.csv
    puzzle_info.json
    optimized_lookup.json
    optimized_stats.json
  submissions/
    optimized_submission.csv
  solve_module.py
  megaminx_best_tested_solver.py
  build_optimized_assets.py
  search_improver_v3.py
  search_policy_v3.py
  cayley_adapter.py
  validate_solve_output.py
  prompts/
```

## 2. Подготовка окружения

Минимальный вариант:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-min.txt
```

Если нужен более полный стек AgentLaboratory/LLM-запусков:

```bash
pip install -r requirements.txt
```

### Опционально: CayleyPy backend

`v3` умеет работать без `cayleypy`, через внутренний fallback backend. Но если вы хотите прогонять именно нативный beam/MITM backend CayleyPy, ставьте библиотеку отдельно.

Рекомендуемый вариант:

```bash
pip install git+https://github.com/cayleypy/cayleypy
```

Более простой, но потенциально менее свежий вариант:

```bash
pip install cayleypy
```

## 3. Быстрая проверка, что пайплайн доступен

Список всех пайплайнов:

```bash
python pipeline_cli.py list-pipelines
```

Проверка, что Megaminx зарегистрирован:

```bash
python pipeline_cli.py list-pipelines | grep cayley-py-megaminx
```

## 4. Самый простой запуск: baseline без LLM

Это самый безопасный и быстрый сценарий. Он просто копирует baseline solver из registry и строит submission без генерации нового кода моделью.

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission.csv \
  --no-llm
```

Что произойдёт:
- будет использован baseline из `pipeline_registry.py`;
- для `cayley-py-megaminx` baseline сейчас указывает на `megaminx_best_tested_solver.py`;
- на выходе появится готовый `submission.csv`.

Вариант с явным путём к puzzles CSV:

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --puzzles competitions/cayley-py-megaminx/data/test.csv \
  --output competitions/cayley-py-megaminx/submissions/submission.csv \
  --no-llm
```

## 5. Сгенерировать solver-файл без запуска на всём датасете

Если нужен просто новый solver-файл в `generated/`, используйте `generate-solver`.

### 5.1. Скопировать baseline без LLM

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_baseline.py \
  --no-llm
```

## 6. Генерация solver через LLM: все prompt variants

Ниже — все рабочие варианты prompt bundles для Megaminx.

### Важно про `regular`

`regular` — это **from-scratch режим**.
Он не должен использовать baseline-код в prompt и предназначен для ситуации, когда модель пишет решение с нуля.

### 6.1. `regular` — from scratch, без baseline

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_regular.py \
  --prompt-variant regular
```

Вариант с дополнительными улучшательными раундами:

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_regular.py \
  --prompt-variant regular \
  --keep-improving \
  --improvement-rounds 3
```

### 6.2. `improved`

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_improved.py \
  --prompt-variant improved
```

### 6.3. `dataset_adapted`

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_dataset_adapted.py \
  --prompt-variant dataset_adapted
```

### 6.4. `structured`

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_structured.py \
  --prompt-variant structured
```

### 6.5. `heuristic_boosted`

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_heuristic_boosted.py \
  --prompt-variant heuristic_boosted
```

### 6.6. `master_hybrid`

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_master_hybrid.py \
  --prompt-variant master_hybrid
```

## 7. Генерация solver с указанием моделей

Пример с удалёнными моделями/бэкендами:

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_hybrid.py \
  --prompt-variant heuristic_boosted \
  --models "gpt-4o-mini,claude-3-5-sonnet"
```

Пример для локального Transformers backend:

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_local.py \
  --prompt-variant regular \
  --models "local:Qwen/Qwen2.5-0.5B-Instruct"
```

## 8. Запуск пайплайна с prompt variant сразу до submission

Если нужен не только solver-файл, но и полный прогон пайплайна до `submission.csv`:

### 8.1. `regular` с нуля

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission_regular.csv \
  --prompt-variant regular
```

### 8.2. `heuristic_boosted`

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission_heuristic_boosted.csv \
  --prompt-variant heuristic_boosted
```

### 8.3. `master_hybrid`

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission_master_hybrid.csv \
  --prompt-variant master_hybrid
```

## 9. Использовать свой prompt file или custom prompts JSON

Если хотите полностью переопределить bundle:

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_custom.py \
  --prompt-file competitions/cayley-py-megaminx/prompts/user_prompt_regular.txt \
  --custom-prompts competitions/cayley-py-megaminx/prompts/custom_prompts_regular.json
```

Примечание:
- `--prompt-file` и `--custom-prompts` имеют приоритет над `--prompt-variant`.

## 10. Лучший протестированный solver: прямой запуск

Если не нужен весь `pipeline_cli`, можно запускать лучший протестированный solver напрямую.

### 10.1. Показать, какой lookup-источник он использует

```bash
python competitions/cayley-py-megaminx/megaminx_best_tested_solver.py --print-source
```

### 10.2. Решить одно состояние в JSON-формате

```bash
python competitions/cayley-py-megaminx/megaminx_best_tested_solver.py \
  --state-json "[0,1,2,3,4,5]"
```

### 10.3. Решить одно состояние в CSV-формате

```bash
python competitions/cayley-py-megaminx/megaminx_best_tested_solver.py \
  --state-csv "0,1,2,3,4,5"
```

### 10.4. Построить полный submission напрямую

```bash
python competitions/cayley-py-megaminx/megaminx_best_tested_solver.py \
  --build-submission competitions/cayley-py-megaminx/submissions/submission_best_tested.csv
```

## 11. Пересобрать детерминированные ассеты оффлайн

Этот шаг нужен, если вы хотите заново собрать `optimized_lookup.json`, `optimized_stats.json` и `optimized_submission.csv`.

```bash
cd competitions/cayley-py-megaminx
python build_optimized_assets.py
```

## 12. Пересборка ассетов + v3 search в одном запуске

### 12.1. Базовый вариант

```bash
cd competitions/cayley-py-megaminx
python build_optimized_assets.py \
  --search-version v3 \
  --search-top-k 150
```

### 12.2. Более агрессивный вариант

```bash
cd competitions/cayley-py-megaminx
python build_optimized_assets.py \
  --search-version v3 \
  --search-top-k 300 \
  --search-light-min-path-len 560 \
  --search-aggressive-min-path-len 700 \
  --search-force-aggressive-top-n 24 \
  --search-min-improvement 2 \
  --search-light-time-budget-per-row 0.25 \
  --search-aggressive-time-budget-per-row 0.75 \
  --search-light-beam-width 96 \
  --search-aggressive-beam-width 192 \
  --search-light-max-steps 8 \
  --search-aggressive-max-steps 12 \
  --search-light-history-depth 0 \
  --search-aggressive-history-depth 2 \
  --search-light-mitm-depth 2 \
  --search-aggressive-mitm-depth 3 \
  --search-light-window-lengths 14,18,22 \
  --search-aggressive-window-lengths 18,24,30,36 \
  --search-light-window-samples 8 \
  --search-aggressive-window-samples 14 \
  --search-light-beam-mode simple \
  --search-aggressive-beam-mode advanced
```

### 12.3. Явно отключить CayleyPy backend

```bash
cd competitions/cayley-py-megaminx
python build_optimized_assets.py \
  --search-version v3 \
  --search-top-k 150 \
  --search-disable-cayleypy
```

## 13. Отдельный запуск `search_improver_v3.py`

### 13.1. Базовый пример

```bash
cd competitions/cayley-py-megaminx
python search_improver_v3.py \
  --submission submissions/optimized_submission.csv \
  --out submissions/submission_search_improved_v3.csv \
  --stats-out submissions/submission_search_improved_v3.stats.json \
  --profile-out submissions/submission_search_improved_v3.profiles.json \
  --top-k 150
```

### 13.2. Прогон top-300

```bash
cd competitions/cayley-py-megaminx
python search_improver_v3.py \
  --submission submissions/optimized_submission.csv \
  --out submissions/submission_search_improved_v3_top300.csv \
  --stats-out submissions/submission_search_improved_v3_top300.stats.json \
  --profile-out submissions/submission_search_improved_v3_top300.profiles.json \
  --top-k 300
```

### 13.3. Принудительно выключить CayleyPy и оставить только fallback backend

```bash
cd competitions/cayley-py-megaminx
python search_improver_v3.py \
  --submission submissions/optimized_submission.csv \
  --out submissions/submission_search_improved_v3.csv \
  --stats-out submissions/submission_search_improved_v3.stats.json \
  --profile-out submissions/submission_search_improved_v3.profiles.json \
  --top-k 150 \
  --disable-cayleypy
```

### 13.4. Управление tier thresholds

```bash
cd competitions/cayley-py-megaminx
python search_improver_v3.py \
  --submission submissions/optimized_submission.csv \
  --out submissions/submission_search_improved_v3.csv \
  --stats-out submissions/submission_search_improved_v3.stats.json \
  --profile-out submissions/submission_search_improved_v3.profiles.json \
  --top-k 300 \
  --light-min-path-len 560 \
  --aggressive-min-path-len 700 \
  --force-aggressive-top-n 24
```

### 13.5. Управление budgets и beam-параметрами

```bash
cd competitions/cayley-py-megaminx
python search_improver_v3.py \
  --submission submissions/optimized_submission.csv \
  --out submissions/submission_search_improved_v3.csv \
  --stats-out submissions/submission_search_improved_v3.stats.json \
  --profile-out submissions/submission_search_improved_v3.profiles.json \
  --top-k 150 \
  --min-improvement 2 \
  --light-time-budget-per-row 0.25 \
  --aggressive-time-budget-per-row 0.75 \
  --light-beam-width 96 \
  --aggressive-beam-width 192 \
  --light-max-steps 8 \
  --aggressive-max-steps 12 \
  --light-history-depth 0 \
  --aggressive-history-depth 2 \
  --light-mitm-depth 2 \
  --aggressive-mitm-depth 3 \
  --light-window-lengths 14,18,22 \
  --aggressive-window-lengths 18,24,30,36 \
  --light-window-samples 8 \
  --aggressive-window-samples 14 \
  --light-beam-mode simple \
  --aggressive-beam-mode advanced
```

## 14. Проверка и валидация solver-файла

Если вы сгенерировали новый solver и хотите проверить его на одном состоянии:

```bash
python competitions/cayley-py-megaminx/validate_solve_output.py \
  --solver generated/solve_megaminx_regular.py \
  --vector "[0,1,2,3,4,5]"
```

Практический вариант: взять реальное первое состояние из `test.csv` и тут же провалидировать solver.

```bash
python - <<'PY'
import csv, json
from pathlib import Path
p = Path('competitions/cayley-py-megaminx/data/test.csv')
with p.open(newline='', encoding='utf-8') as f:
    row = next(csv.DictReader(f))
print(json.dumps(row['initial_state']))
PY
```

Потом подставить это значение в команду validator.

## 15. Рекомендуемые сценарии по задачам

### Сценарий A: нужен просто лучший стабильный baseline

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission.csv \
  --no-llm
```

### Сценарий B: нужен solver, который модель пишет с нуля

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_regular.py \
  --prompt-variant regular
```

### Сценарий C: нужен baseline-backed prompt bundle

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_heuristic_boosted.py \
  --prompt-variant heuristic_boosted
```

### Сценарий D: нужен оффлайн deterministic rebuild

```bash
cd competitions/cayley-py-megaminx
python build_optimized_assets.py
```

### Сценарий E: нужен оффлайн search pass по длинным строкам

```bash
cd competitions/cayley-py-megaminx
python search_improver_v3.py \
  --submission submissions/optimized_submission.csv \
  --out submissions/submission_search_improved_v3.csv \
  --stats-out submissions/submission_search_improved_v3.stats.json \
  --profile-out submissions/submission_search_improved_v3.profiles.json \
  --top-k 150
```

## 16. Частые ошибки

### Ошибка: `regular` всё ещё использует baseline

Ожидаемое поведение сейчас другое:
- `regular` — from scratch;
- baseline-backed остаются `improved`, `dataset_adapted`, `structured`, `heuristic_boosted`, `master_hybrid`.

### Ошибка: `cayleypy_enabled=false` в stats

Это значит, что `v3` отработал через fallback backend, а не через нативный CayleyPy.
Проверьте, что библиотека установлена и импортируется:

```bash
python - <<'PY'
import cayleypy
print('ok', cayleypy.__file__)
PY
```

### Ошибка: solver не проходит validator

Проверьте три вещи:
- solver печатает **JSON**, а не обычный текст;
- в JSON есть поля `moves` и `sorted_array`;
- `moves` состоят только из легальных ходов из `puzzle_info.json`.

## 17. Что использовать по умолчанию

Если вы не хотите долго выбирать, безопасные дефолты такие:

### Лучший baseline

```bash
python pipeline_cli.py run \
  --competition cayley-py-megaminx \
  --output competitions/cayley-py-megaminx/submissions/submission.csv \
  --no-llm
```

### Лучший baseline-backed prompt bundle

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_heuristic_boosted.py \
  --prompt-variant heuristic_boosted
```

### Настоящий from-scratch запуск

```bash
python pipeline_cli.py generate-solver \
  --competition cayley-py-megaminx \
  --out generated/solve_megaminx_regular.py \
  --prompt-variant regular
```

### Offline search поверх лучших детерминированных ассетов

```bash
cd competitions/cayley-py-megaminx
python build_optimized_assets.py --search-version v3 --search-top-k 150
```


## GUI

- Локальный Gradio GUI: `python megaminx_gui_app.py`
- Colab/Jupyter GUI notebook: `colab/agents_4_puzzles_megaminx_gui.ipynb`
- GUI поддерживает выбор пайплайнов, probe всех/выбранных g4f-моделей, выбор single/selected/working model mode и Kaggle submit.
