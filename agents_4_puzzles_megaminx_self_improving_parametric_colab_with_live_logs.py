# %% [markdown]
# agents_4_puzzles — Megaminx self-improving Colab runner

Этот notebook рассчитан на **уже пропатченный** репозиторий с поддержкой `--self-improve-prompts`.

Что умеет:
- распаковать **готовый patched archive** или клонировать репозиторий;
- установить зависимости из `requirements-full.txt`;
- дать форму-параметры для `g4f`/AgentLaboratory;
- собрать и запустить `pipeline_cli.py run` с нужными флагами;
- сохранить **полный вывод** запуска в лог-файл;
- показывать **последние 30 строк логов каждую секунду** прямо в notebook;
- отдельно прогнать `check-g4f-models`;
- настроить `kaggle.json`, сделать preflight и submit.

По умолчанию параметры предзаполнены по вашему длинному примеру команды.

# %%
# @title 1. Источник репозитория и рабочая папка
import os
import shutil
import subprocess
from pathlib import Path

SOURCE_MODE = "patched_archive"  # @param ["patched_archive", "github_clone"]
PATCHED_ARCHIVE_PATH = "/content/agents_4_puzzles-main_megaminx_self_improving_patched_with_parametric_colab_and_live_logs.zip"  # @param {type:"string"}
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
    subprocess.run(["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_REPO_URL, str(workdir / 'repo')], check=True)
    repo_dir = workdir / 'repo'

if not repo_dir.exists():
    raise FileNotFoundError(f"Не найдена рабочая папка репозитория: {repo_dir}")

os.chdir(repo_dir)
print("repo_dir =", repo_dir)
print("cwd =", Path.cwd())
print("pipeline_cli exists =", (repo_dir / 'pipeline_cli.py').exists())

# %%
# @title 2. Установка зависимостей
import subprocess
from pathlib import Path

repo_dir = Path.cwd()
subprocess.run(["python3", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], check=True)
subprocess.run(["python3", "-m", "pip", "install", "-r", str(repo_dir / 'requirements-full.txt')], check=True)
print("requirements-full.txt installed")

# %%
# @title 3. Параметры запуска Megaminx
COMPETITION = "cayley-py-megaminx"  # @param {type:"string"}
PROMPT_VARIANT = "regular"  # @param ["regular", "improved", "dataset_adapted", "structured", "heuristic_boosted", "master_hybrid"]
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

# g4f / generation behaviour
G4F_ASYNC = True  # @param {type:"boolean"}
G4F_REQUEST_TIMEOUT = 120  # @param {type:"integer"}
G4F_STOP_AT_PYTHON_FENCE = True  # @param {type:"boolean"}
MAX_RESPONSE_CHARS = 0  # @param {type:"integer"}
PRINT_GENERATION = True  # @param {type:"boolean"}
PRINT_GENERATION_MAX_CHARS = 16000  # @param {type:"integer"}

# Improvement loop
KEEP_IMPROVING = True  # @param {type:"boolean"}
IMPROVEMENT_ROUNDS = 50000  # @param {type:"integer"}
SELF_IMPROVE_PROMPTS = True  # @param {type:"boolean"}

# Kaggle submit inside run
SUBMIT = True  # @param {type:"boolean"}
SUBMIT_VIA = "auto"  # @param ["auto", "api", "cli"]
SUBMIT_COMPETITION = "cayley-py-megaminx"  # @param {type:"string"}
MESSAGE = "bobik"  # @param {type:"string"}
KAGGLE_JSON_PATH = "~/.kaggle/kaggle.json"  # @param {type:"string"}

# Optional extras
PUZZLES_PATH = ""  # @param {type:"string"}
RUN_LOG_PATH = ""  # @param {type:"string"}
NO_PROGRESS = False  # @param {type:"boolean"}
SCHEMA_CHECK = False  # @param {type:"boolean"}
NO_SCHEMA_CHECK_IDS = False  # @param {type:"boolean"}

# Live notebook logging
ENABLE_LIVE_LOG_TAIL = True  # @param {type:"boolean"}
LIVE_LOG_PATH = "logs/megaminx_live_run.log"  # @param {type:"string"}
TAIL_LINES = 30  # @param {type:"integer"}
TAIL_REFRESH_SECONDS = 1.0  # @param {type:"number"}
CLEAR_OUTPUT_EACH_REFRESH = True  # @param {type:"boolean"}

print("Parameters loaded")

# %%
# @title 4. Настройка Kaggle credentials (опционально)
import os
import json
from pathlib import Path

# Вариант A: вставить JSON целиком строкой.
KAGGLE_JSON_INLINE = ""  # @param {type:"string"}

# Вариант B: заранее загрузить kaggle.json в Colab Files и указать путь ниже.
KAGGLE_JSON_UPLOAD_PATH = ""  # @param {type:"string"}

home_kaggle_dir = Path.home() / ".kaggle"
home_kaggle_dir.mkdir(parents=True, exist_ok=True)
home_kaggle_json = home_kaggle_dir / "kaggle.json"

written = False
if KAGGLE_JSON_INLINE.strip():
    payload = json.loads(KAGGLE_JSON_INLINE)
    home_kaggle_json.write_text(json.dumps(payload), encoding="utf-8")
    written = True
elif KAGGLE_JSON_UPLOAD_PATH.strip():
    src = Path(os.path.expanduser(KAGGLE_JSON_UPLOAD_PATH)).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Не найден uploaded kaggle.json: {src}")
    home_kaggle_json.write_bytes(src.read_bytes())
    written = True

if written:
    os.chmod(home_kaggle_json, 0o600)
    print("kaggle.json prepared at", home_kaggle_json)
else:
    print("Credentials not changed. If SUBMIT=True, убедитесь, что ~/.kaggle/kaggle.json уже существует.")

# %%
# @title 5. Проверка доступных g4f-моделей (опционально)
import subprocess
subprocess.run(["python3", "pipeline_cli.py", "check-g4f-models", "--list-only"], check=False)

# %%
# @title 6. Сборка команды run
import os
import shlex
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
]

if USE_AGENT_MODELS and AGENT_MODELS.strip():
    cmd.extend(["--agent-models", AGENT_MODELS.strip()])
elif MODELS.strip():
    cmd.extend(["--models", MODELS.strip()])

add_flag(cmd, "--planner-models", PLANNER_MODELS.strip())
add_flag(cmd, "--coder-models", CODER_MODELS.strip())
add_flag(cmd, "--fixer-models", FIXER_MODELS.strip())
add_flag(cmd, "--puzzles", PUZZLES_PATH.strip())
add_flag(cmd, "--run-log", RUN_LOG_PATH.strip())
add_flag(cmd, "--g4f-async", G4F_ASYNC)
add_flag(cmd, "--g4f-stop-at-python-fence", G4F_STOP_AT_PYTHON_FENCE)
add_flag(cmd, "--print-generation", PRINT_GENERATION)
add_flag(cmd, "--keep-improving", KEEP_IMPROVING)
add_flag(cmd, "--self-improve-prompts", SELF_IMPROVE_PROMPTS)
add_flag(cmd, "--no-progress", NO_PROGRESS)
add_flag(cmd, "--schema-check", SCHEMA_CHECK)
add_flag(cmd, "--no-schema-check-ids", NO_SCHEMA_CHECK_IDS)

if SUBMIT:
    cmd.append("--submit")
    cmd.extend(["--submit-via", SUBMIT_VIA])
    if SUBMIT_COMPETITION.strip():
        cmd.extend(["--submit-competition", SUBMIT_COMPETITION.strip()])
    if MESSAGE.strip():
        cmd.extend(["--message", MESSAGE.strip()])
    if KAGGLE_JSON_PATH.strip():
        cmd.extend(["--kaggle-json", os.path.expanduser(KAGGLE_JSON_PATH.strip())])

cmd_display = " \\\n  ".join(shlex.quote(x) for x in cmd)
print(cmd_display)
RUN_CMD = cmd

# %%
# @title 7. Live logging helper
import os
import sys
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
            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    with lock:
                        last_lines.append(line)
                        state["lines"] += 1
                    logf.write(raw_line)
                    logf.flush()
            finally:
                try:
                    if proc.stdout is not None:
                        proc.stdout.close()
                except Exception:
                    pass

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        def render(final=False):
            with lock:
                snapshot = list(last_lines)
                total_lines = state["lines"]
            elapsed = time.time() - state["started"]
            status = proc.poll()
            if clear_screen and clear_output is not None:
                clear_output(wait=True)
            print("Live tail monitor")
            print("Command:")
            print(" ".join(cmd))
            print(f"Log file: {log_path}")
            print(f"Elapsed: {elapsed:.1f}s | Captured lines: {total_lines} | Return code: {status}")
            print("=" * 100)
            print(f"Last {tail_lines} lines:")
            if snapshot:
                print("\n".join(snapshot))
            else:
                print("<log is empty yet>")
            if final:
                print("=" * 100)
                print("Process finished")

        try:
            while proc.poll() is None or thread.is_alive():
                render(final=False)
                time.sleep(max(0.2, float(refresh_seconds)))
            thread.join(timeout=2.0)
            render(final=True)
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            thread.join(timeout=2.0)
            render(final=True)
            raise

    return proc.returncode, log_path

print("stream_command_with_live_tail ready")

# %%
# @title 8. Kaggle preflight (рекомендуется перед live submit)
import os
import subprocess

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

if 'RUN_CMD' not in globals():
    raise RuntimeError("Сначала выполните ячейку 'Сборка команды run'")

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
import os
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
# @title 12. Скачать submission и лог-файл (Colab)
from pathlib import Path

out_path = Path(OUTPUT_PATH)
log_path = Path(LIVE_LOG_PATH)
if not log_path.is_absolute():
    log_path = Path.cwd() / log_path

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
except Exception as exc:
    print("Автоскачивание недоступно вне Colab:", exc)
    print("submission:", out_path.resolve())
    print("log:", log_path.resolve())
