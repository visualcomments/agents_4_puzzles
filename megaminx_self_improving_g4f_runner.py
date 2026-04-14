# %% [markdown]
# agents_4_puzzles — Megaminx self-improving Colab runner

Этот notebook рассчитан на **уже пропатченный** репозиторий с поддержкой `--self-improve-prompts`.

Что умеет:
- распаковать **готовый patched archive** или клонировать репозиторий;
- установить зависимости из `requirements-full.txt`;
- дать форму-параметры для `g4f`/AgentLaboratory;
- собрать и запустить `pipeline_cli.py run` с нужными флагами;
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
PATCHED_ARCHIVE_PATH = "/content/agents_4_puzzles-main_megaminx_self_improving_patched_with_parametric_colab.zip"  # @param {type:"string"}
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
import os
import subprocess
from pathlib import Path

repo_dir = Path.cwd()
subprocess.run(["python3", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], check=True)
subprocess.run(["python3", "-m", "pip", "install", "-r", str(repo_dir / 'requirements-full.txt')], check=True)
print("requirements-full.txt installed")

# %%
# @title 3. Параметры запуска Megaminx
from pathlib import Path

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
    "python3", "pipeline_cli.py", "run",
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
# @title 7. Kaggle preflight (рекомендуется перед live submit)
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
# @title 8. Запуск сценария
import subprocess
from pathlib import Path

if 'RUN_CMD' not in globals():
    raise RuntimeError("Сначала выполните ячейку 'Сборка команды run'")

print("Running:")
print(" ".join(RUN_CMD))
subprocess.run(RUN_CMD, check=False)

out_path = Path(OUTPUT_PATH)
print("output exists =", out_path.exists(), out_path)
if out_path.exists():
    print("output size =", out_path.stat().st_size)

# %%
# @title 9. Отдельный submit через официальный kaggle CLI (опционально)
import os
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
# @title 10. Скачать submission файл (Colab)
from pathlib import Path

out_path = Path(OUTPUT_PATH)
if not out_path.exists():
    raise FileNotFoundError(out_path)

try:
    from google.colab import files
    files.download(str(out_path))
except Exception as exc:
    print("Автоскачивание недоступно вне Colab:", exc)
    print("Файл лежит здесь:", out_path.resolve())
