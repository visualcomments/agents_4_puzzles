# RUN_MEGAMINX_MAX

Этот файл описывает, как запускать `run_megaminx_max.sh` для Megaminx с кастомным `solve.py`.

## Что делает скрипт

Скрипт запускает 4 фазы:

1. **Build baseline CSV** из вашего `solve.py` через `pipeline_cli.py build-submission`.
2. **Генерация LLM candidates** через `pipeline_cli.py run` для нескольких prompt variants.
3. **Fusion + search_v3** через `competitions/cayley-py-megaminx/megaminx_cayleypy_llm_hybrid_solver.py`.
4. **Финальный ultra-polish** через `colab/megaminx_full_colab_runner_lowmem.py`.

Именно такой порядок сейчас является безопасным для custom `.py` baseline:
сначала `solve.py` превращается в явный `submission.csv`, потом этот CSV участвует в fusion и финальном refine.

---

## Что нужно перед запуском

Минимум:

- репозиторий уже распакован
- в корне репозитория лежит ваш `solve.py`
- установлен Python 3

Если нужен сабмит в Kaggle:

- рядом есть `kaggle.json`
- установлен `kaggle` CLI
- credentials валидны

---

## Быстрый старт

Из корня репозитория:

```bash
chmod +x ./run_megaminx_max.sh
./run_megaminx_max.sh \
  --solver "$PWD/solve.py" \
  --kaggle-json "$PWD/kaggle.json" \
  --submit
```

Если сабмит не нужен:

```bash
chmod +x ./run_megaminx_max.sh
./run_megaminx_max.sh --solver "$PWD/solve.py"
```

---

## Все основные параметры

### Обязательные на практике

- `--solver PATH` — путь к вашему custom baseline `solve.py`

### Часто используемые

- `--repo-root PATH` — корень репозитория
- `--out-dir PATH` — каталог для промежуточных CSV/JSON
- `--python-bin BIN` — например `python3`
- `--models LIST` — список моделей для `pipeline_cli.py run`
- `--agent-models MAP` — явный маппинг моделей для planner/coder/fixer
- `--llm-variants LIST` — варианты prompt bundle через запятую
- `--run-name NAME` — имя финального low-mem прогона
- `--kaggle-json PATH` — путь к `kaggle.json`
- `--submit` — отправить финальный CSV в Kaggle
- `--no-submit` — не отправлять

### Отладочные

- `--skip-llm` — пропустить генерацию LLM candidates
- `--skip-fusion` — пропустить fusion/search_v3
- `--skip-polish` — пропустить финальный ultra-polish

---

## Параметры по умолчанию внутри скрипта

По умолчанию скрипт использует:

- `MODELS="r1-1776"`
- `AGENT_MODELS="planner=r1-1776;coder=r1-1776;fixer=r1-1776"`
- `LLM_VARIANTS="improved,structured,dataset_adapted,heuristic_boosted,master_hybrid"`
- `RUN_NAME="megaminx_ultra_polish"`

### Сильные параметры fusion/search_v3

- `--search-v3-top-k 240`
- `--light-min-path-len 540`
- `--aggressive-min-path-len 660`
- `--force-aggressive-top-n 32`
- `--light-time-budget-per-row 0.35`
- `--aggressive-time-budget-per-row 1.00`
- `--light-beam-width 128`
- `--aggressive-beam-width 224`

### Сильные параметры final polish

- `--chunk-size 12`
- `--max-passes 3`
- `--profile-mode full`
- `--min-improvement 1`
- `--light-min-path-len 520`
- `--aggressive-min-path-len 640`
- `--force-aggressive-top-n 32`
- `--light-time-budget-per-row 0.40`
- `--aggressive-time-budget-per-row 1.20`
- `--light-beam-width 128`
- `--aggressive-beam-width 256`

Это не «самые быстрые» параметры. Это набор под более тяжёлый и более качественный прогон.

---

## Рекомендованные сценарии запуска

### 1. Полный max-run

```bash
./run_megaminx_max.sh \
  --solver "$PWD/solve.py" \
  --kaggle-json "$PWD/kaggle.json" \
  --submit
```

### 2. Полный max-run без Kaggle submit

```bash
./run_megaminx_max.sh \
  --solver "$PWD/solve.py"
```

### 3. Проверить только baseline + polish без LLM

```bash
./run_megaminx_max.sh \
  --solver "$PWD/solve.py" \
  --skip-llm \
  --skip-fusion
```

### 4. Построить baseline и fusion, но не запускать final polish

```bash
./run_megaminx_max.sh \
  --solver "$PWD/solve.py" \
  --skip-polish
```

---

## Куда сохраняются результаты

### Промежуточные файлы

По умолчанию:

```text
competitions/cayley-py-megaminx/submissions/max_run/
```

Там появятся:

- `submission_custom_baseline.csv`
- `submission_llm_improved.csv`
- `submission_llm_structured.csv`
- `submission_llm_dataset_adapted.csv`
- `submission_llm_heuristic_boosted.csv`
- `submission_llm_master_hybrid.csv`
- `submission_fused_max.csv`
- `submission_fused_max.stats.json`
- `submission_fused_max.profiles.json`

### Финальный polished результат

По умолчанию:

```text
colab_runs/megaminx_ultra_polish/submission_final.csv
```

Если задан другой `--run-name`, путь будет:

```text
colab_runs/<run-name>/submission_final.csv
```

---

## Чем этот запуск отличается от старой команды

### Было раньше

Вы использовали wrapper-команду с такими аргументами:

- `--baseline-script ...`
- `--baseline-script-interpreter python3`
- `--baseline-script-output-csv ...`
- `--baseline-script-args ...`

### Теперь

Основная безопасная схема такая:

1. `solve.py` сначала превращается в baseline CSV.
2. Этот baseline CSV явно участвует в fusion как candidate.
3. Финальный low-mem refine идёт уже от лучшего CSV, а не от неявно закэшированного Python-модуля.

Именно это устраняет ситуацию, когда разные прогоны визуально стартуют «с одного и того же результата».

---

## Что важно помнить

### 1. Ваш `solve.py` должен быть совместим с текущим пайплайном

Ожидается baseline solver-модуль, который корректно обрабатывается `pipeline_cli.py build-submission`.

### 2. `--submit` делает сабмит только на финальной фазе

Промежуточные LLM candidate CSV и fused CSV автоматически не отправляются.

### 3. Если хотите быстро проверить только подхват custom `.py`

Запускайте так:

```bash
./run_megaminx_max.sh \
  --solver "$PWD/solve.py" \
  --skip-llm \
  --skip-fusion
```

Это самый короткий способ убедиться, что ваш `solve.py` теперь реально используется.

---

## Типичные проблемы

### Ошибка: `Не найден solver`

Проверьте путь:

```bash
ls -l "$PWD/solve.py"
```

### Ошибка: `Не найден kaggle.json`

Либо передайте правильный путь:

```bash
--kaggle-json "$PWD/kaggle.json"
```

либо запускайте без `--submit`.

### Ошибка на LLM-фазе

Можно сначала проверить deterministic путь:

```bash
./run_megaminx_max.sh --solver "$PWD/solve.py" --skip-llm --skip-fusion
```

Если deterministic часть проходит, проблема почти наверняка в модели, g4f/backend или сетевом окружении.

---

## Минимальная команда, которую стоит запомнить

```bash
./run_megaminx_max.sh \
  --solver "$PWD/solve.py" \
  --kaggle-json "$PWD/kaggle.json" \
  --submit
```

Если нужен только локальный результат:

```bash
./run_megaminx_max.sh --solver "$PWD/solve.py"
```
