#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
run_megaminx_max.sh

Максимальный пайплайн для Megaminx:
  1) build custom baseline CSV из solve.py
  2) generate LLM candidates
  3) fuse candidates + search_v3
  4) final ultra-polish low-mem refine

Использование:
  ./run_megaminx_max.sh [options]

Опции:
  --repo-root PATH           Корень репозитория. По умолчанию: директория скрипта.
  --solver PATH              Путь к custom baseline solve.py.
  --out-dir PATH             Каталог для промежуточных CSV и JSON.
  --python-bin BIN           Python интерпретатор. По умолчанию: python3.
  --models LIST              Список моделей для pipeline_cli.py run.
  --agent-models MAP         Маппинг planner/coder/fixer моделей.
  --llm-variants LIST        Список вариантов через запятую.
  --kaggle-json PATH         Путь к kaggle.json.
  --run-name NAME            Имя финального low-mem прогона.
  --submit                   Отправить финальный результат в Kaggle.
  --no-submit                Не отправлять в Kaggle.
  --skip-llm                 Пропустить генерацию LLM кандидатов.
  --skip-fusion              Пропустить fusion/search_v3 и взять только baseline CSV.
  --skip-polish              Пропустить финальный low-mem polish.
  --help                     Показать эту справку.

Пример:
  ./run_megaminx_max.sh \
    --solver "$PWD/solve.py" \
    --kaggle-json "$PWD/kaggle.json" \
    --submit
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

ROOT_DIR_DEFAULT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$ROOT_DIR_DEFAULT"
SOLVER="$ROOT_DIR_DEFAULT/solve.py"
OUT_DIR="$ROOT_DIR_DEFAULT/competitions/cayley-py-megaminx/submissions/max_run"
PYTHON_BIN="python3"
MODELS="r1-1776"
AGENT_MODELS="planner=r1-1776;coder=r1-1776;fixer=r1-1776"
LLM_VARIANTS="improved,structured,dataset_adapted,heuristic_boosted,master_hybrid"
KAGGLE_JSON="${KAGGLE_JSON:-}"
RUN_NAME="megaminx_ultra_polish"
SUBMIT=0
SKIP_LLM=0
SKIP_FUSION=0
SKIP_POLISH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --solver)
      SOLVER="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --agent-models)
      AGENT_MODELS="$2"
      shift 2
      ;;
    --llm-variants)
      LLM_VARIANTS="$2"
      shift 2
      ;;
    --kaggle-json)
      KAGGLE_JSON="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --submit)
      SUBMIT=1
      shift
      ;;
    --no-submit)
      SUBMIT=0
      shift
      ;;
    --skip-llm)
      SKIP_LLM=1
      shift
      ;;
    --skip-fusion)
      SKIP_FUSION=1
      shift
      ;;
    --skip-polish)
      SKIP_POLISH=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Неизвестный аргумент: $1"
      ;;
  esac
done

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
SOLVER="$(python3 - <<'PY' "$SOLVER"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
OUT_DIR="$(python3 - <<'PY' "$OUT_DIR"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

PIPELINE_CLI="$REPO_ROOT/pipeline_cli.py"
HYBRID_SOLVER="$REPO_ROOT/competitions/cayley-py-megaminx/megaminx_cayleypy_llm_hybrid_solver.py"
LOWMEM_RUNNER="$REPO_ROOT/colab/megaminx_full_colab_runner_lowmem.py"
TEST_CSV="$REPO_ROOT/competitions/cayley-py-megaminx/data/test.csv"

[[ -f "$PIPELINE_CLI" ]] || die "Не найден pipeline_cli.py: $PIPELINE_CLI"
[[ -f "$HYBRID_SOLVER" ]] || die "Не найден hybrid solver: $HYBRID_SOLVER"
[[ -f "$LOWMEM_RUNNER" ]] || die "Не найден lowmem runner: $LOWMEM_RUNNER"
[[ -f "$SOLVER" ]] || die "Не найден solver: $SOLVER"
[[ -f "$TEST_CSV" ]] || die "Не найден test.csv: $TEST_CSV"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Не найден python интерпретатор: $PYTHON_BIN"

if [[ "$SUBMIT" -eq 1 ]]; then
  [[ -n "$KAGGLE_JSON" ]] || die "Для --submit нужен --kaggle-json"
  [[ -f "$KAGGLE_JSON" ]] || die "Не найден kaggle.json: $KAGGLE_JSON"
fi

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

BASELINE_CSV="$OUT_DIR/submission_custom_baseline.csv"
FUSED_CSV="$OUT_DIR/submission_fused_max.csv"
FUSED_STATS="$OUT_DIR/submission_fused_max.stats.json"
FUSED_PROFILES="$OUT_DIR/submission_fused_max.profiles.json"
FINAL_CSV="$REPO_ROOT/colab_runs/$RUN_NAME/submission_final.csv"

log "Шаг 1/4. Строю baseline CSV из custom solve.py"
"$PYTHON_BIN" "$PIPELINE_CLI" build-submission \
  --competition cayley-py-megaminx \
  --solver "$SOLVER" \
  --output "$BASELINE_CSV" \
  --no-progress \
  --schema-check

CANDIDATE_ARGS=(--candidate "$BASELINE_CSV")

if [[ "$SKIP_LLM" -eq 0 ]]; then
  log "Шаг 2/4. Генерирую LLM candidates"
  IFS=',' read -r -a VARIANTS <<< "$LLM_VARIANTS"
  for variant in "${VARIANTS[@]}"; do
    variant="${variant// /}"
    [[ -n "$variant" ]] || continue
    out_csv="$OUT_DIR/submission_llm_${variant}.csv"
    log "  -> variant=$variant"
    "$PYTHON_BIN" "$PIPELINE_CLI" run \
      --competition cayley-py-megaminx \
      --output "$out_csv" \
      --prompt-variant "$variant" \
      --baseline "$SOLVER" \
      --allow-baseline \
      --models "$MODELS" \
      --agent-models "$AGENT_MODELS" \
      --g4f-async \
      --g4f-request-timeout 120 \
      --g4f-recovery-rounds 2 \
      --g4f-recovery-max-iters 2 \
      --g4f-recovery-sleep 1.5 \
      --keep-improving \
      --improvement-rounds 4 \
      --schema-check
    CANDIDATE_ARGS+=(--candidate "$out_csv")
  done
else
  log "Шаг 2/4 пропущен: --skip-llm"
fi

INPUT_FOR_POLISH="$BASELINE_CSV"

if [[ "$SKIP_FUSION" -eq 0 ]]; then
  log "Шаг 3/4. Fuse candidates + search_v3 + strong deterministic refine"
  HYBRID_CMD=(
    "$PYTHON_BIN" "$HYBRID_SOLVER"
    "${CANDIDATE_ARGS[@]}"
    --run-search-v3
    --search-v3-top-k 240
    --out "$FUSED_CSV"
    --stats-out "$FUSED_STATS"
    --profiles-out "$FUSED_PROFILES"
    --light-min-path-len 540
    --aggressive-min-path-len 660
    --force-aggressive-top-n 32
    --light-time-budget-per-row 0.35
    --aggressive-time-budget-per-row 1.00
    --light-beam-width 128
    --aggressive-beam-width 224
    --light-max-steps 10
    --aggressive-max-steps 12
    --light-history-depth 1
    --aggressive-history-depth 2
    --light-mitm-depth 2
    --aggressive-mitm-depth 4
    --light-window-lengths 14,18,22,26
    --aggressive-window-lengths 18,24,30,36
    --light-window-samples 10
    --aggressive-window-samples 16
    --light-beam-mode simple
    --aggressive-beam-mode advanced
  )
  "${HYBRID_CMD[@]}"
  INPUT_FOR_POLISH="$FUSED_CSV"
else
  log "Шаг 3/4 пропущен: --skip-fusion"
fi

if [[ "$SKIP_POLISH" -eq 0 ]]; then
  log "Шаг 4/4. Финальный ultra-polish low-mem refine"
  POLISH_CMD=(
    "$PYTHON_BIN" "$LOWMEM_RUNNER"
    --repo-root "$REPO_ROOT"
    --baseline "$INPUT_FOR_POLISH"
    --run-name "$RUN_NAME"
    --chunk-size 12
    --max-passes 3
    --profile-mode full
    --min-improvement 1
    --light-min-path-len 520
    --aggressive-min-path-len 640
    --force-aggressive-top-n 32
    --light-time-budget-per-row 0.40
    --aggressive-time-budget-per-row 1.20
    --light-beam-width 128
    --aggressive-beam-width 256
    --light-max-steps 10
    --aggressive-max-steps 14
    --light-history-depth 1
    --aggressive-history-depth 2
    --light-mitm-depth 2
    --aggressive-mitm-depth 4
    --light-window-lengths 14,18,22,26
    --aggressive-window-lengths 18,24,30,36,42
    --light-window-samples 10
    --aggressive-window-samples 18
    --light-beam-mode advanced
    --aggressive-beam-mode advanced
  )
  if [[ "$SUBMIT" -eq 1 ]]; then
    export KAGGLE_CONFIG_DIR="$(mktemp -d)"
    cp "$KAGGLE_JSON" "$KAGGLE_CONFIG_DIR/kaggle.json"
    chmod 600 "$KAGGLE_CONFIG_DIR/kaggle.json"
    POLISH_CMD+=(--submit --submit-message "megaminx ultra polish")
  fi
  "${POLISH_CMD[@]}"
else
  log "Шаг 4/4 пропущен: --skip-polish"
fi

log "Готово"
printf 'Baseline CSV: %s\n' "$BASELINE_CSV"
if [[ -f "$FUSED_CSV" ]]; then
  printf 'Fused CSV:    %s\n' "$FUSED_CSV"
fi
if [[ -f "$FINAL_CSV" ]]; then
  printf 'Final CSV:    %s\n' "$FINAL_CSV"
else
  printf 'Final CSV:    (не создан; вероятно, использован --skip-polish)\n'
fi
