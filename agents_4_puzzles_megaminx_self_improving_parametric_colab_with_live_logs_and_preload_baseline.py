# %% [markdown]
# # agents_4_puzzles — Megaminx self-improving Colab runner
# 
# Этот notebook рассчитан на **уже пропатченный** репозиторий с поддержкой `--self-improve-prompts`.
# 
# Что умеет:
# - распаковать **готовый patched archive** или клонировать репозиторий;
# - установить зависимости из `requirements-full.txt`;
# - дать форму-параметры для `g4f` / AgentLaboratory;
# - **заранее** подгрузить `kaggle.json` **до основного прогона**;
# - **заранее** подгрузить свой кастомный baseline `solve.py` и передать его через `--baseline`;
# - собрать и запустить `pipeline_cli.py run` с нужными флагами;
# - сохранить **полный вывод** запуска в лог-файл;
# - показывать **последние 30 строк логов каждую секунду** прямо в notebook;
# - отдельно прогнать `check-g4f-models`;
# - сделать `kaggle-preflight` и submit.
# 
# По умолчанию параметры предзаполнены для strongest-safe варианта `score_guarded` и без встроенных секретов.

# %%
# @title 1. Источник репозитория и рабочая папка
import os
import shutil
import subprocess
from pathlib import Path

SOURCE_MODE = "patched_archive"  # @param ["patched_archive", "github_clone"]
PATCHED_ARCHIVE_PATH = "/content/agents_4_puzzles-main_megaminx_score_guarded_patch.zip"  # @param {type:"string"}
GIT_REPO_URL = "https://github.com/visualcomments/agents_4_puzzles.git"  # @param {type:"string"}
GIT_BRANCH = "main"  # @param {type:"string"}
WORKDIR = "/content/work_agents_4_puzzles"  # @param {type:"string"}
REPO_SUBDIR_NAME = "agents_4_puzzles-main"  # @param {type:"string"}

workdir = Path(WORKDIR)
if workdir.exists():
    shutil.rmtree(workdir)
workdir.mkdir(parents=True, exist_ok=True)

if SOURCE_MODE == "patched_archive":
    archive_path = Path(PATCHED_ARCHIVE_PATH)
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Не найден архив {archive_path}. Загрузите patched archive в Colab или переключите SOURCE_MODE='github_clone'."
        )
    subprocess.run(["unzip", "-q", str(archive_path), "-d", str(workdir)], check=True)
    repo_dir = workdir / REPO_SUBDIR_NAME
else:
    subprocess.run(["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO_URL, str(workdir / "repo")], check=True)
    repo_dir = workdir / "repo"

if not repo_dir.exists():
    raise FileNotFoundError(f"Не найдена рабочая папка репозитория: {repo_dir}")

os.chdir(repo_dir)
print("repo_dir =", repo_dir)
print("cwd =", Path.cwd())
print("pipeline_cli exists =", (repo_dir / "pipeline_cli.py").exists())

# %%
# @title 2. Предзагрузка Kaggle credentials и кастомного baseline solve.py (до основного прогона)
import os
import json
import shutil
import importlib.util
from pathlib import Path

UPLOAD_KAGGLE_JSON_FROM_BROWSER = False  # @param {type:"boolean"}
KAGGLE_JSON_UPLOAD_PATH = ""  # @param {type:"string"}
KAGGLE_JSON_INLINE = ""  # @param {type:"string"}

UPLOAD_CUSTOM_BASELINE_FROM_BROWSER = False  # @param {type:"boolean"}
CUSTOM_BASELINE_UPLOAD_PATH = ""  # @param {type:"string"}
CUSTOM_BASELINE_RUNTIME_PATH = ""  # @param {type:"string"}
CUSTOM_BASELINE_TARGET_NAME = "custom_baseline_solve.py"  # @param {type:"string"}

repo_dir = Path.cwd()
user_uploads_dir = repo_dir / "_user_uploads"
baseline_upload_dir = user_uploads_dir / "baselines"
user_uploads_dir.mkdir(parents=True, exist_ok=True)
baseline_upload_dir.mkdir(parents=True, exist_ok=True)

def _in_colab():
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

def _upload_via_browser(prompt_name: str, target_dir: Path) -> Path:
    if not _in_colab():
        raise RuntimeError(f"Browser upload for {prompt_name} is available only inside Google Colab.")
    from google.colab import files  # type: ignore

    print(f"Upload {prompt_name} ...")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError(f"No file uploaded for {prompt_name}.")
    first_name, first_payload = next(iter(uploaded.items()))
    dst = target_dir / Path(first_name).name
    dst.write_bytes(first_payload)
    print(f"Uploaded {prompt_name} -> {dst}")
    return dst

def _resolve_existing_path(raw: str) -> Path:
    candidate = Path(os.path.expanduser(raw)).resolve()
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate

# --- Kaggle credentials ---
home_kaggle_dir = Path.home() / ".kaggle"
home_kaggle_dir.mkdir(parents=True, exist_ok=True)
home_kaggle_json = home_kaggle_dir / "kaggle.json"
os.environ["KAGGLE_CONFIG_DIR"] = str(home_kaggle_dir)

resolved_kaggle_source = None
if KAGGLE_JSON_INLINE.strip():
    payload = json.loads(KAGGLE_JSON_INLINE)
    home_kaggle_json.write_text(json.dumps(payload), encoding="utf-8")
    resolved_kaggle_source = "inline-json"
elif KAGGLE_JSON_UPLOAD_PATH.strip():
    src = _resolve_existing_path(KAGGLE_JSON_UPLOAD_PATH.strip())
    shutil.copyfile(src, home_kaggle_json)
    resolved_kaggle_source = str(src)
elif UPLOAD_KAGGLE_JSON_FROM_BROWSER:
    src = _upload_via_browser("kaggle.json", user_uploads_dir)
    shutil.copyfile(src, home_kaggle_json)
    resolved_kaggle_source = str(src)

if home_kaggle_json.exists():
    os.chmod(home_kaggle_json, 0o600)

# --- Custom baseline solve.py ---
RESOLVED_CUSTOM_BASELINE_PATH = ""
resolved_baseline_source = None

if CUSTOM_BASELINE_RUNTIME_PATH.strip():
    src = _resolve_existing_path(CUSTOM_BASELINE_RUNTIME_PATH.strip())
    resolved_baseline_source = str(src)
elif CUSTOM_BASELINE_UPLOAD_PATH.strip():
    src = _resolve_existing_path(CUSTOM_BASELINE_UPLOAD_PATH.strip())
    resolved_baseline_source = str(src)
elif UPLOAD_CUSTOM_BASELINE_FROM_BROWSER:
    src = _upload_via_browser("custom baseline solve.py", baseline_upload_dir)
    resolved_baseline_source = str(src)
else:
    src = None

def _baseline_has_callable_solve(path: Path) -> bool:
    try:
        spec = importlib.util.spec_from_file_location(f"custom_baseline_{abs(hash(str(path)))}", path)
        if spec is None or spec.loader is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        solve = getattr(module, "solve", None)
        return callable(solve)
    except Exception:
        return False

if src is not None:
    if src.suffix.lower() != ".py":
        raise ValueError(f"Custom baseline must be a .py file, got: {src}")
    baseline_target = baseline_upload_dir / CUSTOM_BASELINE_TARGET_NAME
    shutil.copyfile(src, baseline_target)
    if _baseline_has_callable_solve(baseline_target):
        RESOLVED_CUSTOM_BASELINE_PATH = str(baseline_target.resolve())
    else:
        print(f"[baseline-warning] Uploaded custom baseline has no callable solve(vec): {baseline_target}")
        print("[baseline-warning] Falling back to the repository default baseline.")
        RESOLVED_CUSTOM_BASELINE_PATH = ""

print("kaggle_json_exists =", home_kaggle_json.exists(), home_kaggle_json)
print("kaggle_json_source =", resolved_kaggle_source)
print("custom_baseline =", RESOLVED_CUSTOM_BASELINE_PATH or "<not set>")
print("custom_baseline_source =", resolved_baseline_source)

# %%
# @title 3. Установка зависимостей
import subprocess
from pathlib import Path

repo_dir = Path.cwd()
subprocess.run(["python3", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], check=True)
subprocess.run(["python3", "-m", "pip", "install", "-r", str(repo_dir / "requirements-full.txt")], check=True)
print("requirements-full.txt installed")

# %%
# @title 4. Параметры запуска Megaminx
COMPETITION = "cayley-py-megaminx"  # @param {type:"string"}
PROMPT_VARIANT = "score_guarded"  # @param ["regular", "improved", "dataset_adapted", "structured", "heuristic_boosted", "master_hybrid", "score_guarded"]
OUTPUT_PATH = "competitions/cayley-py-megaminx/submissions/submission_best.csv"  # @param {type:"string"}

# Модели
USE_AGENT_MODELS = True  # @param {type:"boolean"}
MODELS = "gpt-4"  # @param {type:"string"}
AGENT_MODELS = "planner=gpt-4;coder=gpt-4;fixer=gpt-4"  # @param {type:"string"}
PLANNER_MODELS = ""  # @param {type:"string"}
CODER_MODELS = ""  # @param {type:"string"}
FIXER_MODELS = ""  # @param {type:"string"}

# Search / refinement
SEARCH_MODE = "hybrid"  # @param ["classic", "hybrid"]
PLAN_BEAM_WIDTH = 16  # @param {type:"integer"}
FRONTIER_WIDTH = 32  # @param {type:"integer"}
ARCHIVE_SIZE = 48  # @param {type:"integer"}
REFINE_ROUNDS = 100  # @param {type:"integer"}
MAX_ITERS = 100000  # @param {type:"integer"}

# g4f / generation
G4F_ASYNC = True  # @param {type:"boolean"}
G4F_REQUEST_TIMEOUT = 120  # @param {type:"integer"}
G4F_STOP_AT_PYTHON_FENCE = True  # @param {type:"boolean"}
MAX_RESPONSE_CHARS = 0  # @param {type:"integer"}
PRINT_GENERATION = True  # @param {type:"boolean"}
PRINT_GENERATION_MAX_CHARS = 16000  # @param {type:"integer"}

# Improvement
KEEP_IMPROVING = True  # @param {type:"boolean"}
IMPROVEMENT_ROUNDS = 128  # @param {type:"integer"}
SELF_IMPROVE_PROMPTS = True  # @param {type:"boolean"}

# Baseline override
BASELINE_PATH_OVERRIDE = ""  # @param {type:"string"}
REQUIRE_KAGGLE_JSON_BEFORE_RUN = False  # @param {type:"boolean"}

# Submit
SUBMIT = False  # @param {type:"boolean"}
SUBMIT_VIA = "auto"  # @param ["auto", "kaggle", "none"]
SUBMIT_COMPETITION = "cayley-py-megaminx"  # @param {type:"string"}
MESSAGE = "megaminx-score-guarded"  # @param {type:"string"}
KAGGLE_JSON_PATH = "~/.kaggle/kaggle.json"  # @param {type:"string"}

# Оценочный датасет / шард
PUZZLES_CSV = ""  # @param {type:"string"}
VECTOR_COL_OVERRIDE = ""  # @param {type:"string"}
MAX_ROWS = 0  # @param {type:"integer"}
WRITE_RUN_CONFIG_JSON = True  # @param {type:"boolean"}
RUN_CONFIG_PATH = "logs/megaminx_run_config.json"  # @param {type:"string"}

# Live logs
ENABLE_LIVE_LOG_TAIL = True  # @param {type:"boolean"}
LIVE_LOG_PATH = "logs/megaminx_live_run.log"  # @param {type:"string"}
TAIL_LINES = 30  # @param {type:"integer"}
TAIL_REFRESH_SECONDS = 1.0  # @param {type:"number"}
CLEAR_OUTPUT_EACH_REFRESH = True  # @param {type:"boolean"}

print("Parameters loaded")

# %%
# @title 5. Проверка доступных g4f-моделей (опционально)
import subprocess
subprocess.run(["python3", "pipeline_cli.py", "check-g4f-models", "--list-only"], check=False)

# %%
# @title 6. Сборка команды run
import os
from pathlib import Path

repo_dir = Path.cwd()

def add_flag(args, flag, value=None, allow_empty=False):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            args.append(flag)
        return
    s = str(value)
    if s == "" and not allow_empty:
        return
    args.extend([flag, s])

def ensure_kaggle_json_ready():
    expected = Path(os.path.expanduser(KAGGLE_JSON_PATH.strip() or "~/.kaggle/kaggle.json"))
    if not expected.exists():
        raise FileNotFoundError(
            f"Kaggle credentials file not found: {expected}. "
            "Сначала выполните ячейку предзагрузки kaggle.json."
        )
    return expected

if REQUIRE_KAGGLE_JSON_BEFORE_RUN or SUBMIT or SUBMIT_VIA in {"auto", "kaggle"}:
    try:
        ensure_kaggle_json_ready()
        print("Kaggle credentials are ready.")
    except FileNotFoundError as exc:
        print(f"[kaggle-warning] {exc}")

baseline_override = ""
if BASELINE_PATH_OVERRIDE.strip():
    baseline_override = str(Path(os.path.expanduser(BASELINE_PATH_OVERRIDE.strip())).resolve())
elif globals().get("RESOLVED_CUSTOM_BASELINE_PATH"):
    baseline_override = RESOLVED_CUSTOM_BASELINE_PATH

cmd = [
    "python3", "-u", "pipeline_cli.py", "run",
    "--competition", COMPETITION,
    "--prompt-variant", PROMPT_VARIANT,
    "--output", OUTPUT_PATH,
    "--search-mode", SEARCH_MODE,
    "--plan-beam-width", str(PLAN_BEAM_WIDTH),
    "--frontier-width", str(FRONTIER_WIDTH),
    "--archive-size", str(ARCHIVE_SIZE),
    "--refine-rounds", str(REFINE_ROUNDS),
    "--max-iters", str(MAX_ITERS),
    "--g4f-request-timeout", str(G4F_REQUEST_TIMEOUT),
    "--max-response-chars", str(MAX_RESPONSE_CHARS),
    "--print-generation-max-chars", str(PRINT_GENERATION_MAX_CHARS),
    "--improvement-rounds", str(IMPROVEMENT_ROUNDS),
    "--submit-via", SUBMIT_VIA,
    "--submit-competition", SUBMIT_COMPETITION or COMPETITION,
    "--message", MESSAGE,
    "--kaggle-json", os.path.expanduser(KAGGLE_JSON_PATH),
]

if USE_AGENT_MODELS and AGENT_MODELS.strip():
    add_flag(cmd, "--agent-models", AGENT_MODELS.strip())
elif MODELS.strip():
    add_flag(cmd, "--models", MODELS.strip())

add_flag(cmd, "--planner-models", PLANNER_MODELS.strip())
add_flag(cmd, "--coder-models", CODER_MODELS.strip())
add_flag(cmd, "--fixer-models", FIXER_MODELS.strip())
add_flag(cmd, "--puzzles", PUZZLES_CSV.strip())
add_flag(cmd, "--vector-col", VECTOR_COL_OVERRIDE.strip())
if int(MAX_ROWS) > 0:
    add_flag(cmd, "--max-rows", int(MAX_ROWS))

add_flag(cmd, "--g4f-async", G4F_ASYNC)
add_flag(cmd, "--g4f-stop-at-python-fence", G4F_STOP_AT_PYTHON_FENCE)
add_flag(cmd, "--print-generation", PRINT_GENERATION)
add_flag(cmd, "--keep-improving", KEEP_IMPROVING)
add_flag(cmd, "--self-improve-prompts", SELF_IMPROVE_PROMPTS)
add_flag(cmd, "--submit", SUBMIT)
add_flag(cmd, "--baseline", baseline_override)

RUN_CMD = cmd
RUN_CONFIG = {
    "competition": COMPETITION,
    "prompt_variant": PROMPT_VARIANT,
    "output_path": OUTPUT_PATH,
    "models": {
        "use_agent_models": USE_AGENT_MODELS,
        "models": MODELS,
        "agent_models": AGENT_MODELS,
        "planner_models": PLANNER_MODELS,
        "coder_models": CODER_MODELS,
        "fixer_models": FIXER_MODELS,
    },
    "search": {
        "mode": SEARCH_MODE,
        "plan_beam_width": PLAN_BEAM_WIDTH,
        "frontier_width": FRONTIER_WIDTH,
        "archive_size": ARCHIVE_SIZE,
        "refine_rounds": REFINE_ROUNDS,
        "max_iters": MAX_ITERS,
    },
    "improvement": {
        "keep_improving": KEEP_IMPROVING,
        "improvement_rounds": IMPROVEMENT_ROUNDS,
        "self_improve_prompts": SELF_IMPROVE_PROMPTS,
    },
    "score": {
        "puzzles_csv": PUZZLES_CSV,
        "vector_col_override": VECTOR_COL_OVERRIDE,
        "max_rows": MAX_ROWS,
    },
    "baseline_override": baseline_override,
}
if WRITE_RUN_CONFIG_JSON:
    cfg_path = Path(RUN_CONFIG_PATH)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(RUN_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
    print("run_config_path =", cfg_path)
print("RUN_CMD:")
print(" ".join(RUN_CMD))
print("baseline_override =", baseline_override or "<default repo baseline>")

# %%
# @title 7. Live logging helper
import os
import time
import threading
from collections import deque
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

try:
    from IPython.display import clear_output
except Exception:
    clear_output = None

def stream_command_with_live_tail(cmd, cwd=None, env=None, log_path="run.log", tail_lines=30, refresh_seconds=1.0, clear_screen=True):
    log_path = Path(log_path)
    if cwd is not None:
        cwd = str(cwd)
        log_path = log_path if log_path.is_absolute() else Path(cwd) / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")

    last_lines = deque(maxlen=max(1, int(tail_lines)))
    lock = threading.Lock()
    state = {"lines": 0, "started": time.time()}

    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        proc = Popen(
            cmd,
            cwd=cwd,
            env=merged_env,
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        def reader():
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                logf.write(raw_line)
                with lock:
                    last_lines.append(line)
                    state["lines"] += 1

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        while proc.poll() is None:
            time.sleep(max(0.1, float(refresh_seconds)))
            with lock:
                snapshot = list(last_lines)
                n_lines = state["lines"]
                started = state["started"]
            if clear_screen and clear_output is not None:
                clear_output(wait=True)
            print(f"log_path: {log_path}")
            print(f"elapsed_sec: {time.time() - started:.1f}")
            print(f"captured_lines: {n_lines}")
            print("=" * 100)
            if snapshot:
                print("\n".join(snapshot))
            else:
                print("<waiting for log output>")

        t.join(timeout=5)
        with lock:
            snapshot = list(last_lines)
            n_lines = state["lines"]
            started = state["started"]

        if clear_screen and clear_output is not None:
            clear_output(wait=True)
        print(f"log_path: {log_path}")
        print(f"elapsed_sec: {time.time() - started:.1f}")
        print(f"captured_lines: {n_lines}")
        print("=" * 100)
        print("\n".join(snapshot) if snapshot else "<empty log>")

        return proc.returncode, log_path

# %%
# @title 8. Kaggle preflight (рекомендуется перед live submit)
import os
import subprocess
from pathlib import Path

expected_kaggle_json = Path(os.path.expanduser(KAGGLE_JSON_PATH.strip() or "~/.kaggle/kaggle.json"))
if not expected_kaggle_json.exists():
    raise FileNotFoundError(
        f"Не найден {expected_kaggle_json}. "
        "Сначала выполните ячейку предзагрузки kaggle.json."
    )

preflight = [
    "python3", "pipeline_cli.py", "kaggle-preflight",
    "--competition", SUBMIT_COMPETITION or COMPETITION,
    "--submit-via", SUBMIT_VIA,
]
if KAGGLE_JSON_PATH.strip():
    preflight.extend(["--kaggle-json", os.path.expanduser(KAGGLE_JSON_PATH.strip())])

print(" ".join(preflight))
subprocess.run(preflight, check=False)

# %%
# @title 9. Запуск сценария c live-логом
import os
import subprocess
from pathlib import Path

if "RUN_CMD" not in globals():
    raise RuntimeError("Сначала выполните ячейку 'Сборка команды run'")

if REQUIRE_KAGGLE_JSON_BEFORE_RUN:
    expected_kaggle_json = Path(os.path.expanduser(KAGGLE_JSON_PATH.strip() or "~/.kaggle/kaggle.json"))
    if not expected_kaggle_json.exists():
        raise FileNotFoundError(
            f"Не найден {expected_kaggle_json}. "
            "Сначала выполните ячейку предзагрузки kaggle.json."
        )

print("Running:")
print(" ".join(RUN_CMD))

repo_dir = Path.cwd()
if ENABLE_LIVE_LOG_TAIL:
    exit_code, actual_log_path = stream_command_with_live_tail(
        RUN_CMD,
        cwd=repo_dir,
        env={"PYTHONUNBUFFERED": "1"},
        log_path=LIVE_LOG_PATH,
        tail_lines=TAIL_LINES,
        refresh_seconds=TAIL_REFRESH_SECONDS,
        clear_screen=CLEAR_OUTPUT_EACH_REFRESH,
    )
    print("exit_code =", exit_code)
    print("actual_log_path =", actual_log_path)
else:
    subprocess.run(RUN_CMD, check=False)
    actual_log_path = None

out_path = Path(OUTPUT_PATH)
print("output exists =", out_path.exists(), out_path)
if out_path.exists():
    print("output size =", out_path.stat().st_size)

# %%
# @title 10. Показать последние 30 строк уже сохранённого лога (повторный просмотр)
from pathlib import Path

existing_log_path = Path(LIVE_LOG_PATH)
if not existing_log_path.is_absolute():
    existing_log_path = Path.cwd() / existing_log_path

if not existing_log_path.exists():
    raise FileNotFoundError(existing_log_path)

lines = existing_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
print("Log path:", existing_log_path)
print("Total log lines:", len(lines))
print("=" * 100)
print("\n".join(lines[-30:]) if lines else "<empty log>")

# %%
# @title 11. Отдельный submit через официальный kaggle CLI (опционально)
import subprocess
from pathlib import Path

expected_kaggle_json = Path(os.path.expanduser(KAGGLE_JSON_PATH.strip() or "~/.kaggle/kaggle.json"))
if not expected_kaggle_json.exists():
    raise FileNotFoundError(
        f"Не найден {expected_kaggle_json}. "
        "Сначала выполните ячейку предзагрузки kaggle.json."
    )

out_path = Path(OUTPUT_PATH)
if not out_path.exists():
    raise FileNotFoundError(f"Сначала получите submission file: {out_path}")

submit_cmd = [
    "kaggle", "competitions", "submit",
    SUBMIT_COMPETITION or COMPETITION,
    "-f", str(out_path),
    "-m", MESSAGE or "megaminx self improving run",
]
print(" ".join(submit_cmd))
subprocess.run(submit_cmd, check=False)

# %%
# @title 12. Скачать submission, лог-файл и кастомный baseline (Colab)
from pathlib import Path

out_path = Path(OUTPUT_PATH)
log_path = Path(LIVE_LOG_PATH)
if not log_path.is_absolute():
    log_path = Path.cwd() / log_path

baseline_path = Path(RESOLVED_CUSTOM_BASELINE_PATH) if globals().get("RESOLVED_CUSTOM_BASELINE_PATH") else None

try:
    from google.colab import files
    if out_path.exists():
        files.download(str(out_path))
    else:
        print("Submission file not found:", out_path)
    if log_path.exists():
        files.download(str(log_path))
    else:
        print("Log file not found:", log_path)
    if baseline_path and baseline_path.exists():
        files.download(str(baseline_path))
except Exception as exc:
    print("Автоскачивание недоступно вне Colab:", exc)
    print("submission:", out_path.resolve())
    print("log:", log_path.resolve())
    if baseline_path:
        print("baseline:", baseline_path.resolve())
