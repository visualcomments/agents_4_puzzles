#!/usr/bin/env python3
"""
Standalone launcher for agents_4_puzzles CayleyPy Pancake prompt sweep.

This script was generated from the Colab notebook:
agents_4_puzzles_pancake_full_prompt_sweep_kaggle_feedback_FIXED_single_inline_kaggle_json.ipynb

What it does:
1. prepares a workspace;
2. clones or unpacks the repository;
3. installs repository dependencies and extra packages;
4. configures LLM/g4f and Kaggle credentials from safe CLI arguments/env vars;
5. writes the CayleyPy Pancake prompt-sweep runner into colab/;
6. starts the full pipeline with live logs;
7. prints the latest summary and optionally adds notebook-style analytics.

Security note:
- The original notebook had inline Kaggle credentials.
- They are intentionally NOT embedded here.
- Pass credentials via --kaggle-json-path, --kaggle-json-inline,
  --kaggle-username/--kaggle-key, or environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Optional


def get_script_dir() -> Path:
    """Return the directory where this launcher script is located."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def default_workspace_dir() -> str:
    """Use a writable local default instead of Colab-only /content."""
    env_value = os.environ.get("PANCAKE_WORKDIR")
    if env_value:
        return str(Path(env_value).expanduser())
    return str(get_script_dir() / "pancake_prompt_sweep_workspace")


def default_archive_path() -> str:
    """Default zip path next to this script; used only for local_zip_path/upload_zip modes."""
    env_value = os.environ.get("PANCAKE_ARCHIVE_PATH")
    if env_value:
        return str(Path(env_value).expanduser())
    return str(get_script_dir() / "agents_4_puzzles_pancake_prompt_sweep_pipeline_FULL.zip")


def ensure_writable_dir(path: Path, label: str) -> Path:
    """Create a directory and raise an actionable error if the location is not writable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        fallback = (Path.cwd() / "pancake_prompt_sweep_workspace").resolve()
        raise PermissionError(
            f"Cannot create {label}: {path}. This location is not writable for the current user. "
            f"Run the script with --workdir {fallback} or choose another writable directory."
        ) from exc
    except OSError as exc:
        fallback = (Path.cwd() / "pancake_prompt_sweep_workspace").resolve()
        raise OSError(
            f"Cannot create {label}: {path}. Original error: {exc}. "
            f"Run the script with --workdir {fallback} or choose another writable directory."
        ) from exc
    return path


EMBEDDED_RUNNER_SOURCE = '#!/usr/bin/env python3\n"""CayleyPy Pancake prompt sweep with Kaggle-score feedback.\n\nThis runner sits above pipeline_cli.py and is intentionally standard-library only.\nIt tests all CayleyPy Pancake prompt bundles, records full logs/metrics, submits each\nsuccessful submission to Kaggle when enabled, reads the public score from Kaggle\nsubmission history, and uses score movement to adjust the next prompt strategy.\nIt also supports ordered g4f model fallback lists and writes analytics artifacts.\n\nThe runner does not store Kaggle credentials. Configure Kaggle in Colab via\n~/.kaggle/kaggle.json, KAGGLE_USERNAME/KAGGLE_KEY, or --kaggle-json.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport csv\nimport json\nimport math\nimport os\nimport re\nimport shutil\nimport subprocess\nimport sys\nimport time\nimport traceback\nimport zipfile\nfrom dataclasses import asdict, dataclass, field\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple\n\nPROMPT_VARIANTS: List[str] = [\n    "regular",\n    "improved",\n    "dataset_adapted",\n    "structured",\n    "heuristic_boosted",\n    "master_hybrid",\n    "neighbour_model_hybrid",\n    "score_guarded",\n    "algorithmic_population",\n    "portfolio_orchestrated",\n    "hard_row_routed",\n    "exact_score_population",\n]\n\nBREAKTHROUGH_VARIANTS: List[str] = [\n    "exact_score_population",\n    "hard_row_routed",\n    "portfolio_orchestrated",\n    "algorithmic_population",\n    "master_hybrid",\n    "score_guarded",\n]\n\nSAFE_VARIANTS: List[str] = [\n    "score_guarded",\n    "heuristic_boosted",\n    "dataset_adapted",\n    "neighbour_model_hybrid",\n]\n\n\ndef split_model_list(value: Optional[str]) -> List[str]:\n    """Parse comma/semicolon/newline separated model lists while keeping single values intact."""\n    if value is None:\n        return []\n    text = str(value).strip()\n    if not text:\n        return []\n    parts = [p.strip() for p in re.split(r"[;,\\n]+", text) if p.strip()]\n    return parts or [text]\n\n\ndef resolve_model_fallbacks(args: argparse.Namespace) -> List[str]:\n    """Return ordered model fallbacks.\n\n    If --model-fallbacks is passed it wins; otherwise --models itself can be a\n    comma/semicolon/newline separated list. The runner passes exactly one model\n    to pipeline_cli.py per attempt, so a failing g4f model does not block the\n    entire prompt variant.\n    """\n    models = split_model_list(getattr(args, "model_fallbacks", None))\n    if not models:\n        models = split_model_list(getattr(args, "models", None))\n    if not models:\n        models = ["g4f:gpt-4o-mini"]\n    seen: set[str] = set()\n    out: List[str] = []\n    for m in models:\n        if m not in seen:\n            out.append(m)\n            seen.add(m)\n    return out\n\n\ndef looks_like_g4f_fallback(models: Sequence[str]) -> bool:\n    return len(models) > 1 and all(str(m).startswith("g4f:") or str(m).startswith("g4f/") for m in models)\n\n\nMODEL_FAILURE_PATTERNS: Tuple[str, ...] = (\n    "ratelimiterror",\n    "rate limit",\n    "missingautherror",\n    "credentials required",\n    "requires api key",\n    "api key required",\n    "need api key",\n    "unauthorized",\n    "forbidden",\n    "provider is not working",\n    "providernotworkingerror",\n    "no working providers",\n    "all providers failed",\n    "retryprovider failed",\n    "timed out",\n    "timeout",\n    "readtimeout",\n    "connecttimeout",\n    "remote protocol error",\n    "connection error",\n    "bad gateway",\n    "service unavailable",\n    "cloudflare",\n    "captcha",\n    "blocked",\n    "empty response",\n    "did not return a python file",\n    "offline fallback",\n    "offline-baseline",\n    "falling back to baseline",\n    "baseline fallback",\n)\n\n\n# Fallback artifacts must remain failed attempts even when the lower layer exits with rc=0\n# and creates a CSV. Keep these markers specific so a legitimate solver may still use\n# the word "fallback" internally for safe in-algorithm rollback logic.\nFALLBACK_ARTIFACT_PATTERNS: Tuple[str, ...] = (\n    "sample_submission fallback",\n    "sample submission fallback",\n    "using bundled sample submission",\n    "copied sample_submission",\n    "copied sample submission",\n    "fallback to sample_submission",\n    "fallback to sample submission",\n    "known-good offline baseline",\n    "known-good baseline",\n    "reference baseline fallback",\n    "offline baseline fallback",\n    "llm generation failed",\n    "llm_generation_failed",\n    "generation failed; using baseline",\n    "using fallback baseline",\n    "emergency fallback",\n)\n\n\ndef _read_text_if_exists(path: Optional[Path], max_chars: int = 200000) -> str:\n    if path is None or not path.exists():\n        return ""\n    try:\n        data = path.read_text(encoding="utf-8", errors="replace")\n    except TypeError:\n        data = path.read_text(encoding="utf-8")\n    except Exception:\n        return ""\n    if len(data) > max_chars:\n        return data[-max_chars:]\n    return data\n\n\ndef _collect_model_failure_text(run_payload: Dict[str, Any], stdout_log: Optional[Path]) -> str:\n    parts: List[str] = []\n    if run_payload:\n        try:\n            parts.append(json.dumps(run_payload, ensure_ascii=False, default=str))\n        except Exception:\n            parts.append(str(run_payload))\n    parts.append(_read_text_if_exists(stdout_log))\n    return "\\n".join(p for p in parts if p)\n\n\ndef detect_model_failure(run_payload: Dict[str, Any], stdout_log: Optional[Path]) -> Tuple[bool, List[str]]:\n    """Return True when subprocess output shows that the selected g4f model did not really work.\n\n    pipeline_cli.py can exit with rc=0 and write a CSV even when the selected g4f model\n    fails and the lower layer falls back to a known-good offline baseline. For model\n    sweeps this is a failed model attempt, so the next g4f model should be tried.\n    """\n    text = _collect_model_failure_text(run_payload, stdout_log).lower()\n    markers = [p for p in MODEL_FAILURE_PATTERNS if p in text]\n    if markers:\n        return True, markers\n\n    stages = run_payload.get("stages") if isinstance(run_payload, dict) else None\n    if isinstance(stages, dict):\n        for stage_name, stage_payload in stages.items():\n            if "generate" in str(stage_name).lower() and isinstance(stage_payload, dict):\n                status = str(stage_payload.get("status") or stage_payload.get("ok") or "").lower()\n                if status in {"false", "failed", "error"}:\n                    return True, [f"stage:{stage_name}:{status}"]\n                err = str(stage_payload.get("error") or stage_payload.get("exception") or "").lower()\n                if any(p in err for p in MODEL_FAILURE_PATTERNS):\n                    return True, [p for p in MODEL_FAILURE_PATTERNS if p in err]\n    return False, []\n\n\ndef _safe_int_metric(value: Any) -> Optional[int]:\n    try:\n        if value is None:\n            return None\n        return int(value)\n    except Exception:\n        return None\n\n\ndef classify_generated_attempt_success(\n    *,\n    rc: int,\n    csv_exists: bool,\n    solver_path: Optional[Path],\n    model_failure: bool,\n    model_failure_markers: List[str],\n    submission_stats: Dict[str, Any],\n    run_payload: Dict[str, Any],\n    stdout_log: Optional[Path],\n) -> Tuple[bool, List[str]]:\n    """Classify a prompt/model attempt using artifact quality, not just process exit code.\n\n    A successful prompt must generate an importable solver artifact and a non-empty\n    submission without UNSOLVED/blank rows. Any provider, baseline, or sample-submission\n    fallback remains a failed attempt and is copied to failed_scripts.\n    """\n    reasons: List[str] = []\n    if rc != 0:\n        reasons.append(f"nonzero_returncode:{rc}")\n    if not csv_exists:\n        reasons.append("missing_output_csv")\n    if model_failure:\n        joined = ";".join(model_failure_markers) if model_failure_markers else "unknown"\n        reasons.append(f"model_failure_or_lower_layer_fallback:{joined}")\n    if solver_path is None or not solver_path.exists():\n        reasons.append("missing_generated_solver")\n\n    rows = _safe_int_metric(submission_stats.get("row_count"))\n    empty_rows = _safe_int_metric(submission_stats.get("empty_rows")) or 0\n    unsolved_rows = _safe_int_metric(submission_stats.get("unsolved_rows")) or 0\n    if rows is None or rows <= 0:\n        reasons.append("empty_or_unreadable_submission")\n    if empty_rows > 0:\n        reasons.append(f"submission_has_empty_rows:{empty_rows}")\n    if unsolved_rows > 0:\n        reasons.append(f"submission_has_unsolved_rows:{unsolved_rows}")\n    if submission_stats.get("csv_error"):\n        reasons.append(f"submission_csv_error:{submission_stats.get(\'csv_error\')}")\n\n    artifact_text = _collect_model_failure_text(run_payload, stdout_log).lower()\n    artifact_markers = [p for p in FALLBACK_ARTIFACT_PATTERNS if p in artifact_text]\n    if artifact_markers:\n        reasons.append("fallback_artifact_markers:" + ";".join(artifact_markers))\n\n    return (len(reasons) == 0), reasons\n\n\ndef utc_now() -> str:\n    return datetime.now(timezone.utc).isoformat()\n\n\ndef slugify(text: str, max_len: int = 80) -> str:\n    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())\n    out = re.sub(r"_+", "_", out).strip("_")\n    return (out or "item")[:max_len]\n\n\ndef parse_float(value: Any) -> Optional[float]:\n    if value is None:\n        return None\n    text = str(value).strip()\n    if not text or text.lower() in {"none", "nan", "null", "pending", "error"}:\n        return None\n    text = text.replace(",", "")\n    try:\n        val = float(text)\n    except Exception:\n        return None\n    if math.isnan(val) or math.isinf(val):\n        return None\n    return val\n\n\ndef read_json(path: Path) -> Optional[Any]:\n    try:\n        return json.loads(path.read_text(encoding="utf-8"))\n    except Exception:\n        return None\n\n\ndef normalize_run_payload(payload: Any) -> Dict[str, Any]:\n    """Return a dict-shaped run payload even when pipeline_cli writes a JSON list.\n\n    Some versions of pipeline_cli.py write run_log.json as a list of stage/event\n    records instead of a single summary dict. The sweep runner expects dict.get(),\n    so normalize the shape and keep the original list under raw_run_log_events.\n    """\n    if isinstance(payload, dict):\n        return payload\n    if isinstance(payload, list):\n        normalized: Dict[str, Any] = {\n            "raw_run_log_type": "list",\n            "raw_run_log_len": len(payload),\n            "raw_run_log_events": payload,\n        }\n        for item in payload:\n            if not isinstance(item, dict):\n                continue\n            # Keep useful top-level summary fields if any event contains them.\n            for key in ("solver", "output", "submission", "status", "ok", "competition", "prompt_variant"):\n                if key in item and key not in normalized:\n                    normalized[key] = item.get(key)\n            stages = normalized.setdefault("stages", {})\n            stage_name = item.get("stage") or item.get("name") or item.get("step")\n            if stage_name and isinstance(stages, dict):\n                stages[str(stage_name)] = item\n        return normalized\n    if payload is None:\n        return {}\n    return {"raw_run_log_type": type(payload).__name__, "raw_run_log_value": str(payload)}\n\n\ndef write_json(path: Path, payload: Any) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")\n\n\ndef append_jsonl(path: Path, payload: Dict[str, Any]) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    with path.open("a", encoding="utf-8") as f:\n        f.write(json.dumps(payload, ensure_ascii=False) + "\\n")\n\n\ndef file_sha256(path: Path) -> Optional[str]:\n    try:\n        import hashlib\n        h = hashlib.sha256()\n        with path.open("rb") as f:\n            for chunk in iter(lambda: f.read(1024 * 1024), b""):\n                h.update(chunk)\n        return h.hexdigest()\n    except Exception:\n        return None\n\n\ndef light_file_stats(path: Optional[Path]) -> Dict[str, Any]:\n    if path is None or not path.exists():\n        return {"exists": False}\n    return {\n        "exists": True,\n        "path": str(path),\n        "bytes": path.stat().st_size,\n        "sha256": file_sha256(path),\n        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),\n    }\n\n\ndef csv_submission_stats(path: Optional[Path]) -> Dict[str, Any]:\n    stats: Dict[str, Any] = light_file_stats(path)\n    if path is None or not path.exists():\n        return stats\n    row_count = 0\n    move_col = None\n    total_tokens = 0\n    max_tokens = 0\n    empty_rows = 0\n    unsolved_rows = 0\n    sample_rows: List[Dict[str, str]] = []\n    try:\n        with path.open("r", encoding="utf-8", newline="") as f:\n            reader = csv.DictReader(f)\n            fieldnames = reader.fieldnames or []\n            for candidate in ("moves", "solution", "move_sequence", "answer", "path"):\n                if candidate in fieldnames:\n                    move_col = candidate\n                    break\n            if move_col is None and len(fieldnames) >= 2:\n                move_col = fieldnames[-1]\n            for row in reader:\n                row_count += 1\n                if len(sample_rows) < 3:\n                    sample_rows.append({k: str(v)[:200] for k, v in row.items()})\n                moves = str(row.get(move_col, "") if move_col else "")\n                if not moves.strip():\n                    empty_rows += 1\n                if moves.strip().upper() == "UNSOLVED":\n                    unsolved_rows += 1\n                toks = [tok for tok in re.split(r"[\\s,;]+", moves.strip()) if tok]\n                total_tokens += len(toks)\n                max_tokens = max(max_tokens, len(toks))\n        stats.update(\n            {\n                "row_count": row_count,\n                "columns": fieldnames,\n                "move_column": move_col,\n                "total_move_tokens": total_tokens,\n                "mean_move_tokens": (total_tokens / row_count) if row_count else None,\n                "max_move_tokens": max_tokens,\n                "empty_rows": empty_rows,\n                "unsolved_rows": unsolved_rows,\n                "sample_rows": sample_rows,\n            }\n        )\n    except Exception as exc:\n        stats["csv_error"] = f"{type(exc).__name__}: {exc}"\n    return stats\n\n\ndef validate_pancake_submission_replay(repo_dir: Path, submission_csv: Path, log_path: Path) -> Dict[str, Any]:\n    """Replay-validate every generated CayleyPy Pancake submission row against official data."""\n    validator = repo_dir / "competitions" / "cayleypy-pancake" / "validate_submission_csv.py"\n    payload: Dict[str, Any] = {\n        "ok": False,\n        "validator": str(validator),\n        "submission_csv": str(submission_csv),\n        "log": str(log_path),\n    }\n    log_path.parent.mkdir(parents=True, exist_ok=True)\n    if not validator.exists():\n        payload["error"] = "missing_validate_submission_csv.py"\n        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")\n        return payload\n    timeout_s = int(float(os.getenv("PANCAKE_SUBMISSION_VALIDATOR_TIMEOUT_S", "180")))\n    cmd = [sys.executable, str(validator), "--submission", str(submission_csv)]\n    try:\n        proc = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True, timeout=timeout_s)\n        log_path.write_text(\n            "# cmd=" + " ".join(cmd) + "\\n"\n            + "# returncode=" + str(proc.returncode) + "\\n\\n"\n            + (proc.stdout or "")\n            + ("\\n" + proc.stderr if proc.stderr else ""),\n            encoding="utf-8",\n        )\n        payload["returncode"] = int(proc.returncode or 0)\n        payload["stdout"] = (proc.stdout or "")[-4000:]\n        payload["stderr"] = (proc.stderr or "")[-4000:]\n        payload["ok"] = proc.returncode == 0\n        if proc.returncode == 0:\n            try:\n                payload["stats"] = json.loads((proc.stdout or "{}").strip().splitlines()[-1])\n            except Exception:\n                payload["stats"] = {}\n        else:\n            payload["error"] = payload.get("stderr") or payload.get("stdout") or "replay_validation_failed"\n    except subprocess.TimeoutExpired as exc:\n        payload["error"] = f"timeout_after_{timeout_s}s"\n        log_path.write_text(\n            "# TIMEOUT\\n"\n            + (exc.stdout or "")\n            + ("\\n" + exc.stderr if exc.stderr else ""),\n            encoding="utf-8",\n        )\n    except Exception as exc:\n        payload["error"] = f"{type(exc).__name__}: {exc}"\n        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")\n    return payload\n\n\ndef copy_if_exists(src: Optional[Path], dst_dir: Path, prefix: str = "") -> Optional[Path]:\n    if src is None or not src.exists():\n        return None\n    dst_dir.mkdir(parents=True, exist_ok=True)\n    dst = dst_dir / f"{prefix}{src.name}"\n    try:\n        shutil.copy2(src, dst)\n        return dst\n    except Exception:\n        return None\n\n\ndef run_streaming(\n    cmd: Sequence[str],\n    *,\n    cwd: Path,\n    log_path: Path,\n    env: Optional[Dict[str, str]] = None,\n    timeout: Optional[int] = None,\n) -> Tuple[int, float]:\n    log_path.parent.mkdir(parents=True, exist_ok=True)\n    started = time.time()\n    merged_env = os.environ.copy()\n    if env:\n        merged_env.update({k: str(v) for k, v in env.items() if v is not None})\n    with log_path.open("w", encoding="utf-8") as log:\n        log.write(f"# started_utc={utc_now()}\\n")\n        log.write("# cwd=" + str(cwd) + "\\n")\n        log.write("# cmd=" + " ".join(map(str, cmd)) + "\\n\\n")\n        log.flush()\n        try:\n            proc = subprocess.Popen(\n                list(cmd),\n                cwd=str(cwd),\n                env=merged_env,\n                stdout=subprocess.PIPE,\n                stderr=subprocess.STDOUT,\n                text=True,\n                bufsize=1,\n            )\n            assert proc.stdout is not None\n            while True:\n                line = proc.stdout.readline()\n                if line:\n                    print(line, end="")\n                    log.write(line)\n                    log.flush()\n                if proc.poll() is not None:\n                    rest = proc.stdout.read()\n                    if rest:\n                        print(rest, end="")\n                        log.write(rest)\n                    break\n                if timeout and (time.time() - started) > timeout:\n                    proc.kill()\n                    log.write(f"\\n# TIMEOUT after {timeout}s\\n")\n                    return 124, time.time() - started\n            rc = int(proc.returncode or 0)\n            log.write(f"\\n# finished_utc={utc_now()} returncode={rc}\\n")\n            return rc, time.time() - started\n        except Exception:\n            log.write("\\n# EXCEPTION\\n")\n            log.write(traceback.format_exc())\n            return 125, time.time() - started\n\n\n@dataclass\nclass PromptBundle:\n    variant: str\n    prompt_file: Optional[str] = None\n    custom_prompts_file: Optional[str] = None\n    planner_variant_files: List[str] = field(default_factory=list)\n    exists: bool = True\n\n\ndef discover_prompt_bundles(repo_dir: Path, explicit_variants: Optional[List[str]] = None) -> List[PromptBundle]:\n    prompt_dir = repo_dir / "competitions" / "cayleypy-pancake" / "prompts"\n    variants = list(explicit_variants or PROMPT_VARIANTS)\n    if explicit_variants is None and prompt_dir.exists():\n        for p in sorted(prompt_dir.glob("user_prompt_*.txt")):\n            v = p.stem.replace("user_prompt_", "")\n            if v not in variants:\n                variants.append(v)\n        for p in sorted(prompt_dir.glob("custom_prompts_*.json")):\n            v = p.stem.replace("custom_prompts_", "")\n            if v not in variants and v != "template":\n                variants.append(v)\n    bundles: List[PromptBundle] = []\n    for v in variants:\n        if v == "regular":\n            prompt = prompt_dir / "user_prompt_regular.txt"\n            if not prompt.exists():\n                prompt = prompt_dir / "user_prompt.txt"\n            custom = prompt_dir / "custom_prompts_regular.json"\n            if not custom.exists():\n                custom = prompt_dir / "custom_prompts_template.json"\n        else:\n            prompt = prompt_dir / f"user_prompt_{v}.txt"\n            custom = prompt_dir / f"custom_prompts_{v}.json"\n        planners = sorted(str(p) for p in prompt_dir.glob("planner_variant_*.json"))\n        bundles.append(\n            PromptBundle(\n                variant=v,\n                prompt_file=str(prompt) if prompt.exists() else None,\n                custom_prompts_file=str(custom) if custom.exists() else None,\n                planner_variant_files=planners,\n                exists=bool((prompt.exists() or custom.exists()) and prompt_dir.exists()),\n            )\n        )\n    return bundles\n\n\ndef parse_kaggle_csv(text: str) -> List[Dict[str, str]]:\n    lines = [line for line in (text or "").splitlines() if line.strip()]\n    if not lines:\n        return []\n    header_idx = 0\n    for i, line in enumerate(lines):\n        low = line.lower()\n        if ("score" in low and "status" in low) or ("description" in low and "date" in low):\n            header_idx = i\n            break\n    csv_text = "\\n".join(lines[header_idx:])\n    try:\n        return list(csv.DictReader(csv_text.splitlines()))\n    except Exception:\n        return []\n\n\ndef fetch_kaggle_submissions(\n    *,\n    repo_dir: Path,\n    competition: str,\n    log_path: Path,\n    kaggle_json: Optional[Path] = None,\n) -> Tuple[List[Dict[str, str]], int]:\n    env = os.environ.copy()\n    if kaggle_json:\n        # Kaggle CLI primarily reads ~/.kaggle/kaggle.json; this is a fallback for API helpers.\n        env["KAGGLE_CONFIG_DIR"] = str(kaggle_json.parent)\n    cmd = ["kaggle", "competitions", "submissions", competition, "-v", "-q"]\n    log_path.parent.mkdir(parents=True, exist_ok=True)\n    proc = subprocess.run(cmd, cwd=str(repo_dir), env=env, text=True, capture_output=True)\n    log_path.write_text((proc.stdout or "") + ("\\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")\n    return parse_kaggle_csv((proc.stdout or "") + "\\n" + (proc.stderr or "")), int(proc.returncode or 0)\n\n\ndef extract_score(row: Dict[str, str]) -> Optional[float]:\n    for key in row.keys():\n        low = key.lower().replace("_", "")\n        if low in {"publicscore", "score"} or ("public" in low and "score" in low):\n            val = parse_float(row.get(key))\n            if val is not None:\n                return val\n    for key in row.keys():\n        if "score" in key.lower():\n            val = parse_float(row.get(key))\n            if val is not None:\n                return val\n    return None\n\n\ndef find_matching_submission(rows: List[Dict[str, str]], message: str, file_name: str) -> Optional[Dict[str, str]]:\n    if not rows:\n        return None\n    msg = message.strip()\n    for row in rows:\n        blob = " ".join(str(v) for v in row.values())\n        if msg and msg in blob:\n            return row\n    for row in rows:\n        blob = " ".join(str(v) for v in row.values())\n        if file_name and file_name in blob:\n            return row\n    return rows[0]\n\n\ndef submit_to_kaggle(\n    *,\n    repo_dir: Path,\n    competition: str,\n    submission_csv: Path,\n    message: str,\n    submit_via: str,\n    kaggle_json: Optional[Path],\n    log_path: Path,\n) -> Tuple[Dict[str, Any], int, float]:\n    env = os.environ.copy()\n    if kaggle_json:\n        env["KAGGLE_CONFIG_DIR"] = str(kaggle_json.parent)\n    # Always submit the already generated CSV directly. Do not call pipeline_cli.py\n    # here, because that would rebuild/overwrite the successful candidate output.\n    # The --submit-via option is kept for notebook compatibility, but this outer\n    # score-feedback runner relies on Kaggle CLI so that the exact generated CSV\n    # is the file that reaches Kaggle.\n    cmd = ["kaggle", "competitions", "submit", competition, "-f", str(submission_csv), "-m", message]\n    rc, seconds = run_streaming(cmd, cwd=repo_dir, log_path=log_path, env=env)\n    return {"submitted": rc == 0, "returncode": rc, "message": message, "log": str(log_path)}, rc, seconds\n\n\ndef wait_for_kaggle_score(\n    *,\n    repo_dir: Path,\n    competition: str,\n    message: str,\n    file_name: str,\n    kaggle_json: Optional[Path],\n    log_dir: Path,\n    polls: int,\n    sleep_seconds: float,\n) -> Dict[str, Any]:\n    best: Dict[str, Any] = {"score": None, "row": None, "polls": 0, "status": "not_polled"}\n    for idx in range(max(0, polls)):\n        rows, rc = fetch_kaggle_submissions(\n            repo_dir=repo_dir,\n            competition=competition,\n            kaggle_json=kaggle_json,\n            log_path=log_dir / f"kaggle_submissions_poll_{idx+1:02d}.csvlog",\n        )\n        row = find_matching_submission(rows, message=message, file_name=file_name)\n        score = extract_score(row) if row else None\n        best = {"score": score, "row": row, "polls": idx + 1, "returncode": rc, "status": "score_ready" if score is not None else "pending"}\n        if score is not None:\n            return best\n        if idx < polls - 1:\n            time.sleep(max(0.0, sleep_seconds))\n    return best\n\n\n@dataclass\nclass StrategyState:\n    direction: str = "min"\n    best_score: Optional[float] = None\n    best_variant: Optional[str] = None\n    best_round: Optional[int] = None\n    no_score_change_streak: int = 0\n    success_streak: int = 0\n    failure_streak: int = 0\n    breakthrough_pressure: int = 0\n    decisions: List[Dict[str, Any]] = field(default_factory=list)\n\n    def better(self, score: Optional[float]) -> bool:\n        if score is None:\n            return False\n        if self.best_score is None:\n            return True\n        if self.direction == "max":\n            return score > self.best_score\n        return score < self.best_score\n\n    def update(self, *, round_idx: int, variant: str, ok: bool, score: Optional[float]) -> Dict[str, Any]:\n        prev = self.best_score\n        improved = ok and self.better(score)\n        score_changed = score is not None and prev is not None and abs(score - prev) > 1e-12\n        if improved:\n            self.best_score = score\n            self.best_variant = variant\n            self.best_round = round_idx\n            self.success_streak += 1\n            self.failure_streak = 0\n            # User-requested behavior: score movement should trigger more breakthrough attempts.\n            self.breakthrough_pressure = max(self.breakthrough_pressure, 2)\n            self.no_score_change_streak = 0\n            mode = "score_improved_promote_then_breakthrough"\n        elif ok:\n            self.success_streak += 1\n            self.failure_streak = 0\n            self.no_score_change_streak += 1\n            self.breakthrough_pressure += 1\n            mode = "validated_but_score_flat_use_more_breakthrough"\n        else:\n            self.failure_streak += 1\n            self.success_streak = 0\n            # When scripts fail, back off into safer guard-heavy prompts before trying breakthroughs again.\n            self.breakthrough_pressure = max(0, self.breakthrough_pressure - 1)\n            mode = "script_failed_backoff_to_guarded"\n        decision = {\n            "round": round_idx,\n            "variant": variant,\n            "ok": ok,\n            "score": score,\n            "previous_best_score": prev,\n            "best_score": self.best_score,\n            "best_variant": self.best_variant,\n            "score_changed": score_changed,\n            "improved": improved,\n            "mode": mode,\n            "breakthrough_pressure": self.breakthrough_pressure,\n            "no_score_change_streak": self.no_score_change_streak,\n        }\n        self.decisions.append(decision)\n        return decision\n\n\ndef reorder_queue(queue: List[str], state: StrategyState) -> List[str]:\n    if not queue:\n        return queue\n    if state.failure_streak >= 2:\n        priority = SAFE_VARIANTS\n    elif state.breakthrough_pressure > 0 or state.no_score_change_streak > 0:\n        priority = BREAKTHROUGH_VARIANTS\n    else:\n        return queue\n    prioritized = [v for v in priority if v in queue]\n    rest = [v for v in queue if v not in prioritized]\n    return prioritized + rest\n\n\ndef build_run_command(\n    args: argparse.Namespace,\n    variant: str,\n    output_csv: Path,\n    run_log: Path,\n    state: StrategyState,\n    model_override: Optional[str] = None,\n) -> List[str]:\n    cmd = [\n        sys.executable,\n        "pipeline_cli.py",\n        "run",\n        "--competition",\n        args.competition,\n        "--output",\n        str(output_csv),\n        "--prompt-variant",\n        variant,\n        "--models",\n        model_override or args.models,\n        "--run-log",\n        str(run_log),\n        "--schema-check",\n    ]\n    if args.puzzles:\n        cmd.extend(["--puzzles", args.puzzles])\n    if args.agent_models:\n        cmd.extend(["--agent-models", args.agent_models])\n    if args.planner_models:\n        cmd.extend(["--planner-models", args.planner_models])\n    if args.coder_models:\n        cmd.extend(["--coder-models", args.coder_models])\n    if args.fixer_models:\n        cmd.extend(["--fixer-models", args.fixer_models])\n    cmd.extend(["--search-mode", args.search_mode])\n    cmd.extend(["--plan-beam-width", str(args.plan_beam_width)])\n    cmd.extend(["--frontier-width", str(args.frontier_width)])\n    cmd.extend(["--archive-size", str(args.archive_size)])\n    cmd.extend(["--refine-rounds", str(args.refine_rounds)])\n    max_iters = args.max_iters\n    improvement_rounds = args.improvement_rounds\n    if state.breakthrough_pressure > 0 and variant in BREAKTHROUGH_VARIANTS:\n        # Breakthrough mode: spend a bit more search budget after a score movement or plateau.\n        max_iters = max(max_iters, args.breakthrough_max_iters)\n        improvement_rounds = max(improvement_rounds, args.breakthrough_improvement_rounds)\n        cmd.extend(["--plan-beam-width", str(max(args.plan_beam_width, args.breakthrough_plan_beam_width))])\n        cmd.extend(["--frontier-width", str(max(args.frontier_width, args.breakthrough_frontier_width))])\n    cmd.extend(["--max-iters", str(max_iters)])\n    if args.keep_improving:\n        cmd.append("--keep-improving")\n        cmd.extend(["--improvement-rounds", str(improvement_rounds)])\n    if args.self_improve_prompts:\n        cmd.append("--self-improve-prompts")\n    if args.allow_baseline:\n        cmd.append("--allow-baseline")\n    if args.no_llm:\n        cmd.append("--no-llm")\n    if args.max_rows:\n        cmd.extend(["--max-rows", str(args.max_rows)])\n    if args.baseline:\n        cmd.extend(["--baseline", args.baseline])\n    if args.g4f_async == "on":\n        cmd.append("--g4f-async")\n    elif args.g4f_async == "off":\n        cmd.append("--no-g4f-async")\n    if args.g4f_recovery_rounds is not None:\n        cmd.extend(["--g4f-recovery-rounds", str(args.g4f_recovery_rounds)])\n    if args.baseline_patch_max_iters is not None:\n        cmd.extend(["--baseline-patch-max-iters", str(args.baseline_patch_max_iters)])\n    if args.g4f_recovery_max_iters is not None:\n        cmd.extend(["--g4f-recovery-max-iters", str(args.g4f_recovery_max_iters)])\n    if args.g4f_recovery_sleep is not None:\n        cmd.extend(["--g4f-recovery-sleep", str(args.g4f_recovery_sleep)])\n    if args.print_generation:\n        cmd.append("--print-generation")\n    if args.print_generation_max_chars is not None:\n        cmd.extend(["--print-generation-max-chars", str(args.print_generation_max_chars)])\n    if args.max_response_chars is not None:\n        cmd.extend(["--max-response-chars", str(args.max_response_chars)])\n    if args.g4f_request_timeout is not None:\n        cmd.extend(["--g4f-request-timeout", str(args.g4f_request_timeout)])\n    if args.g4f_stop_at_python_fence == "on":\n        cmd.append("--g4f-stop-at-python-fence")\n    elif args.g4f_stop_at_python_fence == "off":\n        cmd.append("--no-g4f-stop-at-python-fence")\n    if looks_like_g4f_fallback(resolve_model_fallbacks(args)):\n        # Let AgentLaboratory/g4f try alternate providers inside a selected model, while\n        # this runner still performs model-level fallback between model names.\n        os.environ.setdefault("AGENTLAB_G4F_PROVIDER_ALLOW_AUTO_FALLBACK", "1")\n    if args.no_progress:\n        cmd.append("--no-progress")\n    return cmd\n\n\ndef flatten_metrics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n    flat: List[Dict[str, Any]] = []\n    for r in records:\n        sub = r.get("submission_stats") or {}\n        run_log = r.get("run_log_summary") or {}\n        decision = r.get("strategy_decision") or {}\n        flat.append(\n            {\n                "round": r.get("round"),\n                "variant": r.get("variant"),\n                "model": r.get("model"),\n                "model_attempt_count": len(r.get("model_attempts") or []),\n                "attempted_models": ";".join([str(a.get("model")) for a in (r.get("model_attempts") or [])]),\n                "ok": r.get("ok"),\n                "returncode": r.get("returncode"),\n                "runtime_seconds": r.get("runtime_seconds"),\n                "solver_path": r.get("solver_path"),\n                "output_csv": r.get("output_csv"),\n                "csv_exists": sub.get("exists"),\n                "model_failure_detected": any(bool(a.get("model_failure_detected")) for a in (r.get("model_attempts") or [])),\n                "model_failure_markers": ";".join(sorted({str(marker) for a in (r.get("model_attempts") or []) for marker in str(a.get("model_failure_markers") or "").split(";") if marker})),\n                "row_count": sub.get("row_count"),\n                "total_move_tokens": sub.get("total_move_tokens"),\n                "mean_move_tokens": sub.get("mean_move_tokens"),\n                "max_move_tokens": sub.get("max_move_tokens"),\n                "empty_rows": sub.get("empty_rows"),\n                "unsolved_rows": sub.get("unsolved_rows"),\n                "kaggle_submitted": (r.get("kaggle") or {}).get("submitted"),\n                "kaggle_score": (r.get("kaggle_score") or {}).get("score"),\n                "score_improved": decision.get("improved"),\n                "strategy_mode": decision.get("mode"),\n                "best_score_after_round": decision.get("best_score"),\n                "stage_seconds_generate": ((run_log.get("stages") or {}).get("generate_solver") or {}).get("seconds"),\n                "stage_seconds_build": ((run_log.get("stages") or {}).get("build_submission") or {}).get("seconds"),\n                "stage_seconds_validate": ((run_log.get("stages") or {}).get("validate_solver") or {}).get("seconds"),\n            }\n        )\n    return flat\n\n\ndef write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    keys: List[str] = []\n    for row in rows:\n        for key in row.keys():\n            if key not in keys:\n                keys.append(key)\n    with path.open("w", encoding="utf-8", newline="") as f:\n        writer = csv.DictWriter(f, fieldnames=keys)\n        writer.writeheader()\n        for row in rows:\n            writer.writerow({k: row.get(k) for k in keys})\n\n\n\ndef _group_rows(rows: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:\n    groups: Dict[Any, List[Dict[str, Any]]] = {}\n    for row in rows:\n        groups.setdefault(row.get(key), []).append(row)\n    return groups\n\n\ndef _mean(values: Iterable[Any]) -> Optional[float]:\n    nums = [parse_float(v) for v in values]\n    nums = [v for v in nums if v is not None]\n    return sum(nums) / len(nums) if nums else None\n\n\ndef _best(values: Iterable[Any], direction: str) -> Optional[float]:\n    nums = [parse_float(v) for v in values]\n    nums = [v for v in nums if v is not None]\n    if not nums:\n        return None\n    return max(nums) if direction == "max" else min(nums)\n\n\ndef _walk_prompt_signals(obj: Any, prefix: str = "") -> List[Dict[str, Any]]:\n    """Best-effort extraction of self-improving prompt traces from arbitrary run_log JSON."""\n    signals: List[Dict[str, Any]] = []\n    interesting = ("prompt", "improv", "refine", "self", "feedback", "strategy", "breakthrough")\n    if isinstance(obj, dict):\n        for k, v in obj.items():\n            path = f"{prefix}.{k}" if prefix else str(k)\n            lk = str(k).lower()\n            if any(tok in lk for tok in interesting):\n                preview = json.dumps(v, ensure_ascii=False)[:2000] if not isinstance(v, str) else v[:2000]\n                signals.append({"path": path, "key": str(k), "preview": preview})\n            if isinstance(v, (dict, list)):\n                signals.extend(_walk_prompt_signals(v, path))\n    elif isinstance(obj, list):\n        for i, v in enumerate(obj[:200]):\n            path = f"{prefix}[{i}]"\n            if isinstance(v, (dict, list)):\n                signals.extend(_walk_prompt_signals(v, path))\n            elif isinstance(v, str) and any(tok in v.lower() for tok in interesting):\n                signals.append({"path": path, "key": "list_item", "preview": v[:2000]})\n    return signals\n\n\ndef _grep_log_signals(path: Optional[str], limit: int = 80) -> List[Dict[str, Any]]:\n    if not path:\n        return []\n    p = Path(path)\n    if not p.exists():\n        return []\n    out: List[Dict[str, Any]] = []\n    needles = ("self-improve", "self improve", "improvement", "refine", "prompt", "breakthrough", "score")\n    try:\n        with p.open("r", encoding="utf-8", errors="replace") as f:\n            for line_no, line in enumerate(f, start=1):\n                low = line.lower()\n                if any(n in low for n in needles):\n                    out.append({"line": line_no, "text": line.rstrip()[:1200]})\n                    if len(out) >= limit:\n                        break\n    except Exception as exc:\n        out.append({"line": None, "text": f"log_read_error: {type(exc).__name__}: {exc}"})\n    return out\n\n\ndef generate_analytics(run_dir: Path, records: List[Dict[str, Any]], state: StrategyState) -> Dict[str, Any]:\n    """Create analysis artifacts that are included in the auto-downloaded zip."""\n    analytics_dir = run_dir / "analytics"\n    analytics_dir.mkdir(parents=True, exist_ok=True)\n    flat = flatten_metrics(records)\n    write_csv(analytics_dir / "score_timeline.csv", flat)\n\n    # Prompt-level summary\n    prompt_rows: List[Dict[str, Any]] = []\n    for variant, rows in _group_rows(flat, "variant").items():\n        prompt_rows.append({\n            "variant": variant,\n            "runs": len(rows),\n            "successes": sum(1 for r in rows if str(r.get("ok")).lower() == "true"),\n            "failures": sum(1 for r in rows if str(r.get("ok")).lower() != "true"),\n            "success_rate": (sum(1 for r in rows if str(r.get("ok")).lower() == "true") / len(rows)) if rows else None,\n            "best_kaggle_score": _best([r.get("kaggle_score") for r in rows], state.direction),\n            "mean_runtime_seconds": _mean([r.get("runtime_seconds") for r in rows]),\n            "mean_move_tokens": _mean([r.get("mean_move_tokens") for r in rows]),\n            "score_improvements": sum(1 for r in rows if str(r.get("score_improved")).lower() == "true"),\n        })\n    write_csv(analytics_dir / "prompt_variant_summary.csv", prompt_rows)\n\n    # Model fallback analytics\n    attempts: List[Dict[str, Any]] = []\n    for r in records:\n        for a in r.get("model_attempts") or []:\n            attempts.append({\n                "round": r.get("round"),\n                "variant": r.get("variant"),\n                "model": a.get("model"),\n                "attempt_index": a.get("attempt_index"),\n                "ok": a.get("ok"),\n                "returncode": a.get("returncode"),\n                "runtime_seconds": a.get("runtime_seconds"),\n                "output_csv": a.get("output_csv"),\n                "csv_exists": a.get("csv_exists"),\n                "model_failure_detected": a.get("model_failure_detected"),\n                "model_failure_markers": a.get("model_failure_markers"),\n                "stdout_log": a.get("stdout_log"),\n                "run_log": a.get("run_log"),\n            })\n    write_csv(analytics_dir / "model_fallback_attempts.csv", attempts)\n    model_rows: List[Dict[str, Any]] = []\n    for model, rows in _group_rows(attempts, "model").items():\n        model_rows.append({\n            "model": model,\n            "attempts": len(rows),\n            "successes": sum(1 for r in rows if str(r.get("ok")).lower() == "true"),\n            "failures": sum(1 for r in rows if str(r.get("ok")).lower() != "true"),\n            "success_rate": (sum(1 for r in rows if str(r.get("ok")).lower() == "true") / len(rows)) if rows else None,\n            "mean_runtime_seconds": _mean([r.get("runtime_seconds") for r in rows]),\n        })\n    write_csv(analytics_dir / "model_fallback_summary.csv", model_rows)\n\n    # Strategy and self-improvement traces\n    strategy_rows = list(state.decisions)\n    write_csv(analytics_dir / "strategy_transitions.csv", strategy_rows)\n    prompt_signal_rows: List[Dict[str, Any]] = []\n    lineage_rows: List[Dict[str, Any]] = []\n    for r in records:\n        signals = _walk_prompt_signals(r.get("run_log_summary") or {})\n        log_signals = _grep_log_signals(r.get("stdout_log"))\n        decision = r.get("strategy_decision") or {}\n        lineage_rows.append({\n            "round": r.get("round"),\n            "variant": r.get("variant"),\n            "model": r.get("model"),\n            "ok": r.get("ok"),\n            "kaggle_score": (r.get("kaggle_score") or {}).get("score"),\n            "score_improved": decision.get("improved"),\n            "strategy_mode": decision.get("mode"),\n            "breakthrough_pressure": decision.get("breakthrough_pressure"),\n            "prompt_json_signal_count": len(signals),\n            "prompt_log_signal_count": len(log_signals),\n        })\n        for sig in signals:\n            prompt_signal_rows.append({\n                "round": r.get("round"),\n                "variant": r.get("variant"),\n                "model": r.get("model"),\n                "source": "run_log_json",\n                "path": sig.get("path"),\n                "key": sig.get("key"),\n                "preview": sig.get("preview"),\n            })\n        for sig in log_signals:\n            prompt_signal_rows.append({\n                "round": r.get("round"),\n                "variant": r.get("variant"),\n                "model": r.get("model"),\n                "source": "stdout_log",\n                "path": sig.get("line"),\n                "key": "log_line",\n                "preview": sig.get("text"),\n            })\n    write_csv(analytics_dir / "self_improving_prompt_lineage.csv", lineage_rows)\n    write_csv(analytics_dir / "self_improving_prompt_signals.csv", prompt_signal_rows)\n    with (analytics_dir / "self_improving_prompt_signals.jsonl").open("w", encoding="utf-8") as f:\n        for row in prompt_signal_rows:\n            f.write(json.dumps(row, ensure_ascii=False) + "\\n")\n\n    # Successful vs failed scripts manifest\n    script_rows: List[Dict[str, Any]] = []\n    for r in records:\n        script_rows.append({\n            "round": r.get("round"),\n            "variant": r.get("variant"),\n            "model": r.get("model"),\n            "ok": r.get("ok"),\n            "solver_path": r.get("solver_path"),\n            "copied_solver": r.get("copied_solver"),\n            "returncode": r.get("returncode"),\n            "kaggle_score": (r.get("kaggle_score") or {}).get("score"),\n        })\n    write_csv(analytics_dir / "successful_vs_failed_scripts.csv", script_rows)\n\n    best = state.best_score\n    report = []\n    report.append("# CayleyPy Pancake prompt sweep analytics\\n")\n    report.append(f"Generated UTC: {utc_now()}\\n")\n    report.append(f"Score direction: `{state.direction}`; best score: `{best}`; best variant: `{state.best_variant}`; best round: `{state.best_round}`.\\n")\n    report.append("## Model fallback\\n")\n    if attempts:\n        report.append("The runner tried models sequentially per prompt variant and stopped after the first successful CSV. See `model_fallback_attempts.csv` and `model_fallback_summary.csv`.\\n")\n    else:\n        report.append("No model fallback attempts were recorded.\\n")\n    report.append("## Self-improving prompt dynamics\\n")\n    report.append("Use `self_improving_prompt_lineage.csv` as the timeline: it joins round, prompt variant, selected model, Kaggle score, strategy mode, breakthrough pressure, and counts of extracted prompt-improvement signals.\\n")\n    report.append("Use `self_improving_prompt_signals.csv` / `.jsonl` for the extracted prompt/refinement/feedback snippets from run logs. Extraction is best-effort because pipeline logs can vary by prompt variant.\\n")\n    report.append("## Strategy interpretation\\n")\n    report.append("`score_improved_promote_then_breakthrough` means a Kaggle-score movement promoted the lineage but also increased breakthrough pressure. `validated_but_score_flat_use_more_breakthrough` means the generated code passed local checks but did not improve the score, so more aggressive prompt families should be tried. `script_failed_backoff_to_guarded` means the runner backs off into safer prompts.\\n")\n    report.append("## Key files\\n")\n    report.append("- `prompt_variant_summary.csv` — aggregate metrics per prompt family.\\n")\n    report.append("- `score_timeline.csv` — one row per completed round.\\n")\n    report.append("- `strategy_transitions.csv` — score-driven strategy decisions.\\n")\n    report.append("- `model_fallback_summary.csv` — reliability of each g4f fallback model.\\n")\n    report.append("- `successful_vs_failed_scripts.csv` — manifest of solver scripts and outcomes.\\n")\n    report_path = analytics_dir / "analysis_report.md"\n    report_path.write_text("\\n".join(report), encoding="utf-8")\n    return {\n        "analytics_dir": str(analytics_dir),\n        "analysis_report": str(report_path),\n        "prompt_signal_count": len(prompt_signal_rows),\n        "model_attempt_count": len(attempts),\n    }\n\n\ndef package_results(run_dir: Path, output_zip: Path) -> None:\n    output_zip.parent.mkdir(parents=True, exist_ok=True)\n    if output_zip.exists():\n        output_zip.unlink()\n    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:\n        for path in sorted(run_dir.rglob("*")):\n            if path.is_file():\n                z.write(path, path.relative_to(run_dir.parent))\n\n\ndef main(argv: Optional[Sequence[str]] = None) -> int:\n    parser = argparse.ArgumentParser(description="Run every CayleyPy Pancake prompt bundle and use Kaggle score feedback for strategy.")\n    parser.add_argument("--repo-dir", default=".")\n    parser.add_argument("--competition", default="cayleypy-pancake")\n    parser.add_argument("--kaggle-competition", default="CayleyPy-pancake")\n    parser.add_argument("--puzzles", default=None)\n    parser.add_argument("--models", default="g4f:gpt-4o-mini", help="One model or comma/semicolon/newline-separated fallback list. Example: g4f:gpt-4o-mini,g4f:gpt-4o")\n    parser.add_argument("--model-fallbacks", default=None, help="Explicit ordered model fallback list. Overrides --models when provided.")\n    parser.add_argument("--agent-models", default=None)\n    parser.add_argument("--planner-models", default=None)\n    parser.add_argument("--coder-models", default=None)\n    parser.add_argument("--fixer-models", default=None)\n    parser.add_argument("--variants", default="all", help="Comma-separated variants or \'all\'.")\n    parser.add_argument("--max-total-runs", type=int, default=None)\n    parser.add_argument("--score-direction", choices=["min", "max"], default="min")\n    parser.add_argument("--run-name", default=None)\n    parser.add_argument("--output-root", default="runs/pancake_prompt_sweep")\n    parser.add_argument("--search-mode", choices=["classic", "hybrid"], default="hybrid")\n    parser.add_argument("--plan-beam-width", type=int, default=3)\n    parser.add_argument("--frontier-width", type=int, default=6)\n    parser.add_argument("--archive-size", type=int, default=8)\n    parser.add_argument("--refine-rounds", type=int, default=1)\n    parser.add_argument("--max-iters", type=int, default=8000)\n    parser.add_argument("--breakthrough-max-iters", type=int, default=20000)\n    parser.add_argument("--breakthrough-plan-beam-width", type=int, default=5)\n    parser.add_argument("--breakthrough-frontier-width", type=int, default=10)\n    parser.add_argument("--keep-improving", action="store_true")\n    parser.add_argument("--improvement-rounds", type=int, default=3)\n    parser.add_argument("--breakthrough-improvement-rounds", type=int, default=6)\n    parser.add_argument("--self-improve-prompts", action="store_true")\n    parser.add_argument("--allow-baseline", action="store_true")\n    parser.add_argument("--baseline", default=None)\n    parser.add_argument("--no-llm", action="store_true")\n    parser.add_argument("--max-rows", type=int, default=None)\n    parser.add_argument("--run-timeout", type=int, default=None)\n    parser.add_argument("--no-progress", action="store_true")\n    parser.add_argument("--g4f-async", choices=["auto", "on", "off"], default="auto")\n    parser.add_argument("--g4f-recovery-rounds", type=int, default=None)\n    parser.add_argument("--baseline-patch-max-iters", type=int, default=None)\n    parser.add_argument("--g4f-recovery-max-iters", type=int, default=None)\n    parser.add_argument("--g4f-recovery-sleep", type=float, default=None)\n    parser.add_argument("--print-generation", action="store_true")\n    parser.add_argument("--print-generation-max-chars", type=int, default=None)\n    parser.add_argument("--max-response-chars", type=int, default=None)\n    parser.add_argument("--g4f-request-timeout", type=float, default=None)\n    parser.add_argument("--g4f-stop-at-python-fence", choices=["auto", "on", "off"], default="auto")\n    parser.add_argument("--submit", action="store_true", help="Submit every successful prompt output to Kaggle.")\n    parser.add_argument("--require-submit-success", action=argparse.BooleanOptionalAction, default=True, help="When --submit is enabled, classify a round as successful only if Kaggle upload is confirmed.")\n    parser.add_argument("--submit-via", choices=["cli", "api", "auto"], default="cli")\n    parser.add_argument("--kaggle-json", default=None)\n    parser.add_argument("--score-polls", type=int, default=6)\n    parser.add_argument("--score-poll-sleep", type=float, default=30.0)\n    parser.add_argument("--message-prefix", default="CayleyPy Pancake prompt sweep")\n    parser.add_argument("--dry-run", action="store_true")\n    args = parser.parse_args(argv)\n\n    repo_dir = Path(args.repo_dir).resolve()\n    if not (repo_dir / "pipeline_cli.py").exists():\n        raise SystemExit(f"pipeline_cli.py not found under {repo_dir}")\n\n    if args.variants.strip().lower() == "all":\n        explicit = None\n    else:\n        explicit = [v.strip() for v in args.variants.split(",") if v.strip()]\n    bundles = discover_prompt_bundles(repo_dir, explicit)\n    queue = [b.variant for b in bundles if b.exists]\n    if not queue:\n        raise SystemExit("No CayleyPy Pancake prompt bundles were found.")\n    if args.max_total_runs is not None:\n        queue = queue[: max(0, int(args.max_total_runs))]\n\n    model_fallbacks = resolve_model_fallbacks(args)\n\n    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")\n    run_dir = (repo_dir / args.output_root / slugify(run_name)).resolve()\n    logs_dir = run_dir / "logs"\n    outputs_dir = run_dir / "submissions"\n    success_dir = run_dir / "successful_scripts"\n    failed_dir = run_dir / "failed_scripts"\n    prompt_dir = run_dir / "prompt_manifests"\n    run_dir.mkdir(parents=True, exist_ok=True)\n    for _dir in (logs_dir, outputs_dir, success_dir, failed_dir, prompt_dir, run_dir / "run_logs"):\n        _dir.mkdir(parents=True, exist_ok=True)\n\n    write_json(prompt_dir / "prompt_bundles.json", [asdict(b) for b in bundles])\n    write_csv(prompt_dir / "prompt_bundles.csv", [asdict(b) for b in bundles])\n\n    state = StrategyState(direction=args.score_direction)\n    records: List[Dict[str, Any]] = []\n    kaggle_json = Path(args.kaggle_json).expanduser().resolve() if args.kaggle_json else None\n\n    print(f"[sweep] repo_dir={repo_dir}")\n    print(f"[sweep] run_dir={run_dir}")\n    print(f"[sweep] variants={queue}")\n    print(f"[sweep] submit={args.submit} kaggle_competition={args.kaggle_competition}")\n    print(f"[sweep] model_fallbacks={model_fallbacks}")\n\n    round_idx = 0\n    while queue:\n        queue = reorder_queue(queue, state)\n        variant = queue.pop(0)\n        round_idx += 1\n        round_slug = f"round_{round_idx:03d}_{slugify(variant)}"\n        print(f"\\n===== {round_slug} =====")\n        print("[strategy]", state.decisions[-1] if state.decisions else {"mode": "initial_sweep"})\n\n        model_attempts: List[Dict[str, Any]] = []\n        successful_attempt: Optional[Dict[str, Any]] = None\n        for model_idx, model_name in enumerate(model_fallbacks, start=1):\n            model_slug = slugify(model_name, max_len=48)\n            attempt_slug = f"{round_slug}__model_{model_idx:02d}_{model_slug}"\n            output_csv = outputs_dir / f"{attempt_slug}.csv"\n            run_log = run_dir / "run_logs" / f"{attempt_slug}.run_log.json"\n            stdout_log = logs_dir / f"{attempt_slug}.stdout.log"\n            cmd = build_run_command(args, variant, output_csv, run_log, state, model_override=model_name)\n            print(f"[model-fallback] attempt {model_idx}/{len(model_fallbacks)} model={model_name}")\n            print("[cmd]", " ".join(cmd))\n            if args.dry_run:\n                rc, seconds = 0, 0.0\n                stdout_log.write_text("DRY RUN\\n" + " ".join(cmd), encoding="utf-8")\n            else:\n                rc, seconds = run_streaming(cmd, cwd=repo_dir, log_path=stdout_log, timeout=args.run_timeout)\n            run_payload = normalize_run_payload(read_json(run_log))\n            solver_path = Path(run_payload.get("solver", "")) if run_payload.get("solver") else None\n            csv_exists = output_csv.exists()\n            model_failure, model_failure_markers = detect_model_failure(run_payload, stdout_log)\n            attempt_submission_stats = csv_submission_stats(output_csv if csv_exists else None)\n            # A CSV created by offline/baseline/sample-submission fallback is useful as an\n            # artifact, but it is not proof that the selected model generated a valid solver.\n            # Keep those attempts failed and continue to the next fallback model.\n            attempt_ok, attempt_failure_reasons = classify_generated_attempt_success(\n                rc=rc,\n                csv_exists=csv_exists,\n                solver_path=solver_path,\n                model_failure=model_failure,\n                model_failure_markers=model_failure_markers,\n                submission_stats=attempt_submission_stats,\n                run_payload=run_payload,\n                stdout_log=stdout_log,\n            )\n            attempt_replay_report: Dict[str, Any] = {"skipped": True}\n            if attempt_ok and args.competition == "cayleypy-pancake" and output_csv.exists():\n                replay_log = logs_dir / f"{attempt_slug}.replay_validation.log"\n                attempt_replay_report = validate_pancake_submission_replay(repo_dir, output_csv, replay_log)\n                if not attempt_replay_report.get("ok"):\n                    attempt_ok = False\n                    attempt_failure_reasons.append("pancake_submission_replay_validation_failed")\n            attempt = {\n                "attempt_index": model_idx,\n                "model": model_name,\n                "ok": attempt_ok,\n                "returncode": rc,\n                "runtime_seconds": seconds,\n                "cmd": cmd,\n                "stdout_log": str(stdout_log),\n                "run_log": str(run_log),\n                "run_log_summary": run_payload,\n                "output_csv": str(output_csv),\n                "csv_exists": csv_exists,\n                "model_failure_detected": model_failure,\n                "model_failure_markers": ";".join(model_failure_markers),\n                "attempt_failure_reasons": attempt_failure_reasons,\n                "attempt_submission_stats": attempt_submission_stats,\n                "replay_validation": attempt_replay_report,\n                "solver_path": str(solver_path) if solver_path else None,\n            }\n            model_attempts.append(attempt)\n            attempt_dest = success_dir if attempt_ok else failed_dir\n            copy_if_exists(solver_path, attempt_dest, prefix=f"{attempt_slug}__")\n            copy_if_exists(stdout_log, attempt_dest, prefix=f"{attempt_slug}__")\n            copy_if_exists(run_log, attempt_dest, prefix=f"{attempt_slug}__")\n            copy_if_exists(output_csv if output_csv.exists() else None, attempt_dest, prefix=f"{attempt_slug}__")\n            if attempt_ok:\n                successful_attempt = attempt\n                print(f"[model-fallback] success with model={model_name}; remaining fallback models skipped for this prompt variant")\n                break\n            if model_failure:\n                print(f"[model-fallback] model={model_name} produced only fallback/provider failure markers={model_failure_markers}; trying next fallback if available")\n            else:\n                print(f"[model-fallback] failed with model={model_name}; reasons={attempt_failure_reasons}; trying next fallback if available")\n\n        chosen_attempt = successful_attempt or (model_attempts[-1] if model_attempts else {})\n        output_csv = Path(chosen_attempt.get("output_csv", outputs_dir / f"{round_slug}.csv"))\n        run_log = Path(chosen_attempt.get("run_log", run_dir / "run_logs" / f"{round_slug}.run_log.json"))\n        stdout_log = Path(chosen_attempt.get("stdout_log", logs_dir / f"{round_slug}.stdout.log"))\n        run_payload = normalize_run_payload(chosen_attempt.get("run_log_summary"))\n        solver_path = Path(chosen_attempt.get("solver_path", "")) if chosen_attempt.get("solver_path") else None\n        copied_solver = None\n        for pattern in success_dir.glob(f"{round_slug}__model_*__*" if successful_attempt else "__never__"):\n            if solver_path and pattern.name.endswith(solver_path.name):\n                copied_solver = pattern\n                break\n        rc = int(chosen_attempt.get("returncode", 1) or 0)\n        seconds = sum(float(a.get("runtime_seconds") or 0.0) for a in model_attempts)\n        ok = successful_attempt is not None\n\n        submission_stats = csv_submission_stats(output_csv if output_csv.exists() else None)\n        kaggle_report: Dict[str, Any] = {"submitted": False, "skipped": True}\n        kaggle_score: Dict[str, Any] = {"score": None, "status": "not_submitted"}\n        submit_required_failure: Optional[str] = None\n        if ok and args.submit:\n            message = f"{args.message_prefix} | {round_slug} | {datetime.now(timezone.utc).strftime(\'%Y-%m-%d %H:%M:%S UTC\')}"\n            submit_log = logs_dir / f"{round_slug}.kaggle_submit.log"\n            kaggle_report, submit_rc, submit_seconds = submit_to_kaggle(\n                repo_dir=repo_dir,\n                competition=args.kaggle_competition,\n                submission_csv=output_csv,\n                message=message,\n                submit_via=args.submit_via,\n                kaggle_json=kaggle_json,\n                log_path=submit_log,\n            )\n            kaggle_report.update({"seconds": submit_seconds, "returncode": submit_rc})\n            if kaggle_report.get("submitted"):\n                kaggle_score = wait_for_kaggle_score(\n                    repo_dir=repo_dir,\n                    competition=args.kaggle_competition,\n                    message=message,\n                    file_name=output_csv.name,\n                    kaggle_json=kaggle_json,\n                    log_dir=logs_dir / f"{round_slug}_score_polls",\n                    polls=args.score_polls,\n                    sleep_seconds=args.score_poll_sleep,\n                )\n        if ok and args.submit and args.require_submit_success and not kaggle_report.get("submitted"):\n            submit_required_failure = "kaggle_submit_required_but_not_confirmed"\n            ok = False\n            copy_if_exists(solver_path, failed_dir, prefix=f"{round_slug}__submit_failed__")\n            copy_if_exists(stdout_log, failed_dir, prefix=f"{round_slug}__submit_failed__")\n            copy_if_exists(run_log, failed_dir, prefix=f"{round_slug}__submit_failed__")\n            copy_if_exists(output_csv if output_csv.exists() else None, failed_dir, prefix=f"{round_slug}__submit_failed__")\n            print("[kaggle] submit was required but was not confirmed; classifying this round as failed", flush=True)\n        score = kaggle_score.get("score") if isinstance(kaggle_score, dict) else None\n        decision = state.update(round_idx=round_idx, variant=variant, ok=ok, score=parse_float(score))\n        if state.breakthrough_pressure > 0:\n            state.breakthrough_pressure -= 1\n\n        record: Dict[str, Any] = {\n            "round": round_idx,\n            "variant": variant,\n            "started_utc": utc_now(),\n            "ok": ok,\n            "returncode": rc,\n            "runtime_seconds": seconds,\n            "model": chosen_attempt.get("model"),\n            "model_attempts": model_attempts,\n            "cmd": chosen_attempt.get("cmd"),\n            "stdout_log": str(stdout_log),\n            "run_log": str(run_log),\n            "run_log_summary": run_payload,\n            "output_csv": str(output_csv),\n            "submission_stats": submission_stats,\n            "solver_path": str(solver_path) if solver_path else None,\n            "copied_solver": str(copied_solver) if copied_solver else None,\n            "kaggle": kaggle_report,\n            "kaggle_score": kaggle_score,\n            "submit_required_failure": submit_required_failure,\n            "strategy_decision": decision,\n            "remaining_queue_after_decision": reorder_queue(list(queue), state),\n        }\n        records.append(record)\n        append_jsonl(run_dir / "run_index.jsonl", record)\n        append_jsonl(run_dir / "strategy_history.jsonl", decision)\n        write_json(run_dir / "strategy_state.json", asdict(state))\n        write_json(run_dir / "latest_record.json", record)\n        write_csv(run_dir / "per_round_metrics.csv", flatten_metrics(records))\n        queue = reorder_queue(queue, state)\n\n    summary = {\n        "created_utc": utc_now(),\n        "repo_dir": str(repo_dir),\n        "run_dir": str(run_dir),\n        "tested_variants": [r["variant"] for r in records],\n        "model_fallbacks": model_fallbacks,\n        "success_count": sum(1 for r in records if r.get("ok")),\n        "failure_count": sum(1 for r in records if not r.get("ok")),\n        "submit_required": bool(args.submit and args.require_submit_success),\n        "submitted_success_count": sum(1 for r in records if isinstance(r.get("kaggle"), dict) and r["kaggle"].get("submitted")),\n        "best_score": state.best_score,\n        "best_variant": state.best_variant,\n        "best_round": state.best_round,\n        "score_direction": state.direction,\n        "all_records_jsonl": str(run_dir / "run_index.jsonl"),\n        "metrics_csv": str(run_dir / "per_round_metrics.csv"),\n        "strategy_history_jsonl": str(run_dir / "strategy_history.jsonl"),\n        "successful_scripts_dir": str(success_dir),\n        "failed_scripts_dir": str(failed_dir),\n    }\n    analytics_report = generate_analytics(run_dir, records, state)\n    summary["analytics"] = analytics_report\n    write_json(run_dir / "summary.json", summary)\n    output_zip = run_dir.with_suffix(".zip")\n    package_results(run_dir, output_zip)\n    print("\\n===== SUMMARY =====")\n    print(json.dumps(summary, ensure_ascii=False, indent=2))\n    print(f"[sweep] packaged: {output_zip}")\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value!r}")


def run_cmd(cmd: Iterable[str], *, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd_list = [str(x) for x in cmd]
    print("[cmd]", " ".join(shlex.quote(x) for x in cmd_list), flush=True)
    return subprocess.run(cmd_list, cwd=str(cwd) if cwd else None, check=check, text=True)


def stream_cmd(cmd: Iterable[str], *, cwd: Path) -> int:
    cmd_list = [str(x) for x in cmd]
    print("Command:")
    print(" ".join(shlex.quote(x) for x in cmd_list), flush=True)
    process = subprocess.Popen(
        cmd_list,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    ret = process.wait()
    print("\nreturncode =", ret, flush=True)
    return ret


def is_placeholder(value: str) -> bool:
    value = (value or "").strip()
    return (
        not value
        or value.startswith("PASTE_")
        or value in {"your_kaggle_username", "your_kaggle_api_key"}
    )


def write_kaggle_json(payload: Any, *, source_label: str, kaggle_config_dir: Optional[str] = None) -> Path:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TypeError("kaggle.json payload must be a JSON object with username and key")

    username = str(payload.get("username", "")).strip()
    key = str(payload.get("key", "")).strip()
    if is_placeholder(username) or is_placeholder(key):
        raise ValueError(
            "Kaggle credentials are placeholders. Pass the whole kaggle.json via "
            "--kaggle-json-inline, use --kaggle-json-path, or use --kaggle-username/--kaggle-key."
        )

    kaggle_dir = Path(kaggle_config_dir or os.environ.get("KAGGLE_CONFIG_DIR") or (Path.home() / ".kaggle")).expanduser()
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"
    normalized = {"username": username, "key": key}
    kaggle_json_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    try:
        kaggle_json_path.chmod(0o600)
    except OSError:
        pass

    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    print(f"Kaggle credentials configured from {source_label}: {kaggle_json_path}", flush=True)
    return kaggle_json_path


def configure_llm_env(args: argparse.Namespace) -> None:
    values = {
        "OPENAI_API_KEY": args.openai_api_key,
        "OPENROUTER_API_KEY": args.openrouter_api_key,
        "GEMINI_API_KEY": args.gemini_api_key,
        "ANTHROPIC_API_KEY": args.anthropic_api_key,
        "GROQ_API_KEY": args.groq_api_key,
        "TOGETHER_API_KEY": args.together_api_key,
        "G4F_API_KEY": args.g4f_api_key,
        "G4F_PROVIDER": args.g4f_provider,
        "G4F_PROVIDER_LIST": args.g4f_provider_list,
    }
    for key, value in values.items():
        if str(value or "").strip():
            os.environ[key] = str(value).strip()

    os.environ["G4F_ALLOW_AUTO_FALLBACK"] = "1" if args.g4f_allow_auto_fallback else "0"

    set_keys = [
        key
        for key in values
        if os.environ.get(key) and (key.endswith("API_KEY") or key in {"G4F_PROVIDER", "G4F_PROVIDER_LIST"})
    ]
    print("LLM environment prepared. Set keys:", set_keys, flush=True)


def configure_kaggle(args: argparse.Namespace) -> Optional[Path]:
    mode = args.kaggle_credential_mode
    kaggle_json_path: Optional[Path] = None

    if mode == "none":
        print("Kaggle credentials skipped. Use --no-submit-to-kaggle unless credentials already exist in runtime.", flush=True)
    elif mode == "env":
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        key = os.environ.get("KAGGLE_KEY", "").strip()
        if username and key:
            kaggle_json_path = write_kaggle_json(
                {"username": username, "key": key},
                source_label="environment variables KAGGLE_USERNAME/KAGGLE_KEY",
                kaggle_config_dir=args.kaggle_config_dir,
            )
        else:
            existing = Path(args.kaggle_config_dir or os.environ.get("KAGGLE_CONFIG_DIR") or (Path.home() / ".kaggle")).expanduser() / "kaggle.json"
            if existing.exists():
                os.environ["KAGGLE_CONFIG_DIR"] = str(existing.parent)
                kaggle_json_path = existing
                print(f"Kaggle credentials found at existing path: {existing}", flush=True)
            else:
                print("Kaggle env credentials were not found. Submit will fail unless runner can authenticate another way.", flush=True)
    elif mode == "inline_json":
        if not args.kaggle_json_inline.strip():
            raise ValueError("--kaggle-json-inline is required when --kaggle-credential-mode=inline_json")
        kaggle_json_path = write_kaggle_json(
            args.kaggle_json_inline,
            source_label="--kaggle-json-inline",
            kaggle_config_dir=args.kaggle_config_dir,
        )
    elif mode == "existing_path":
        path = Path(args.kaggle_json_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        kaggle_json_path = write_kaggle_json(
            path.read_text(encoding="utf-8"),
            source_label=str(path),
            kaggle_config_dir=args.kaggle_config_dir,
        )
    elif mode == "username_key":
        if is_placeholder(args.kaggle_username) or is_placeholder(args.kaggle_key):
            raise ValueError("--kaggle-username and --kaggle-key are required for --kaggle-credential-mode=username_key")
        kaggle_json_path = write_kaggle_json(
            {"username": args.kaggle_username, "key": args.kaggle_key},
            source_label="--kaggle-username/--kaggle-key",
            kaggle_config_dir=args.kaggle_config_dir,
        )
    else:
        raise ValueError(f"Unknown Kaggle credential mode: {mode}")

    if kaggle_json_path and args.kaggle_smoke_check:
        run_cmd(["kaggle", "--version"], check=False)

    return kaggle_json_path


def prepare_workspace(args: argparse.Namespace) -> Path:
    if args.repo_dir:
        repo_dir = Path(args.repo_dir).expanduser().resolve()
        if not (repo_dir / "pipeline_cli.py").exists():
            raise FileNotFoundError(f"pipeline_cli.py not found in --repo-dir: {repo_dir}")
        print("repo_dir =", repo_dir, flush=True)
        return repo_dir

    workdir = Path(args.workdir).expanduser().resolve()
    if workdir.exists() and args.force_clean:
        shutil.rmtree(workdir)
    ensure_writable_dir(workdir, "workspace directory")

    if args.source_mode == "git_clone":
        target_dir = workdir / args.clone_dir_name
        if target_dir.exists() and args.force_clean:
            shutil.rmtree(target_dir)
        clone_cmd = ["git", "clone"]
        if args.git_depth and args.git_depth > 0:
            clone_cmd += ["--depth", str(args.git_depth)]
        if args.git_branch:
            clone_cmd += ["--branch", args.git_branch]
        clone_cmd += [args.git_repo_url, str(target_dir)]
        run_cmd(clone_cmd)
    elif args.source_mode in {"local_zip_path", "drive_zip", "upload_zip"}:
        archive = Path(args.archive_path).expanduser().resolve()
        if not archive.exists():
            raise FileNotFoundError(
                f"Archive not found: {archive}. For a standalone .py script, --source-mode=upload_zip also expects --archive-path."
            )
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(workdir)
        print("archive =", archive, flush=True)
    else:
        raise ValueError(args.source_mode)

    candidates = sorted(workdir.rglob("pipeline_cli.py"))
    if not candidates:
        raise FileNotFoundError("pipeline_cli.py not found after extraction/clone")
    repo_dir = candidates[0].parent.resolve()
    print("repo_dir =", repo_dir, flush=True)
    print("Python =", sys.executable, flush=True)
    return repo_dir


def install_dependencies(args: argparse.Namespace, repo_dir: Path) -> None:
    if not args.install_deps:
        print("Dependency installation skipped (--no-install-deps).", flush=True)
        return

    if args.upgrade_pip_tools:
        run_cmd([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    req_paths = []
    if args.requirements_file:
        req_paths.append(repo_dir / args.requirements_file)
    req_paths.extend([repo_dir / "requirements-full.txt", repo_dir / "requirements.txt"])

    installed_req = False
    for req in req_paths:
        if req.exists():
            run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req)])
            installed_req = True
            break

    if not installed_req:
        print("No requirements file found; only extra packages will be installed.", flush=True)

    extra_packages = [p for p in str(args.extra_packages or "").split() if p]
    if extra_packages:
        run_cmd([sys.executable, "-m", "pip", "install", *extra_packages])

    print("Dependencies installed", flush=True)


def write_runner(repo_dir: Path, *, overwrite: bool = True) -> Path:
    runner_path = repo_dir / "colab" / "pancake_prompt_sweep_kaggle_feedback.py"
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    if runner_path.exists() and not overwrite:
        print(f"Runner already exists and will not be overwritten: {runner_path}", flush=True)
    else:
        runner_path.write_text(EMBEDDED_RUNNER_SOURCE, encoding="utf-8")
        try:
            runner_path.chmod(0o755)
        except OSError:
            pass
        print(f"Runner written: {runner_path}", flush=True)
    return runner_path


def append_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None and str(value) != "":
        cmd.extend([flag, str(value)])


def build_runner_cmd(args: argparse.Namespace, repo_dir: Path, kaggle_json_path: Optional[Path]) -> list[str]:
    cmd = [
        sys.executable,
        "colab/pancake_prompt_sweep_kaggle_feedback.py",
        "--repo-dir", str(repo_dir),
        "--competition", args.competition,
        "--kaggle-competition", args.kaggle_competition,
        "--variants", args.variants,
        "--models", args.models,
        "--score-direction", args.score_direction,
        "--run-name", args.run_name,
        "--max-iters", str(args.max_iters),
        "--breakthrough-max-iters", str(args.breakthrough_max_iters),
        "--plan-beam-width", str(args.plan_beam_width),
        "--frontier-width", str(args.frontier_width),
        "--breakthrough-plan-beam-width", str(args.breakthrough_plan_beam_width),
        "--breakthrough-frontier-width", str(args.breakthrough_frontier_width),
        "--improvement-rounds", str(args.improvement_rounds),
        "--breakthrough-improvement-rounds", str(args.breakthrough_improvement_rounds),
        "--score-polls", str(args.score_polls),
        "--score-poll-sleep", str(args.score_poll_sleep),
        "--submit-via", args.submit_via,
        "--output-root", args.output_root,
        "--search-mode", args.search_mode,
        "--archive-size", str(args.archive_size),
        "--refine-rounds", str(args.refine_rounds),
        "--g4f-async", args.g4f_async,
        "--g4f-stop-at-python-fence", args.g4f_stop_at_python_fence,
        "--message-prefix", args.message_prefix,
    ]

    append_arg(cmd, "--puzzles", args.puzzles)
    append_arg(cmd, "--max-total-runs", args.max_total_runs if args.max_total_runs and args.max_total_runs > 0 else None)
    append_arg(cmd, "--model-fallbacks", args.model_fallbacks.strip() if args.model_fallbacks else None)
    append_arg(cmd, "--agent-models", args.agent_models.strip() if args.agent_models else None)
    append_arg(cmd, "--planner-models", args.planner_models.strip() if args.planner_models else None)
    append_arg(cmd, "--coder-models", args.coder_models.strip() if args.coder_models else None)
    append_arg(cmd, "--fixer-models", args.fixer_models.strip() if args.fixer_models else None)
    append_arg(cmd, "--baseline", args.baseline)
    append_arg(cmd, "--max-rows", args.max_rows if args.max_rows and args.max_rows > 0 else None)
    append_arg(cmd, "--run-timeout", args.run_timeout)
    append_arg(cmd, "--g4f-recovery-rounds", args.g4f_recovery_rounds)
    append_arg(cmd, "--baseline-patch-max-iters", args.baseline_patch_max_iters)
    append_arg(cmd, "--g4f-recovery-max-iters", args.g4f_recovery_max_iters)
    append_arg(cmd, "--g4f-recovery-sleep", args.g4f_recovery_sleep)
    append_arg(cmd, "--print-generation-max-chars", args.print_generation_max_chars)
    append_arg(cmd, "--max-response-chars", args.max_response_chars)
    append_arg(cmd, "--g4f-request-timeout", args.g4f_request_timeout)

    if args.keep_improving:
        cmd.append("--keep-improving")
    if args.self_improve_prompts:
        cmd.append("--self-improve-prompts")
    if args.submit_to_kaggle:
        cmd.append("--submit")
    if args.no_llm:
        cmd.append("--no-llm")
    if args.allow_baseline:
        cmd.append("--allow-baseline")
    if args.no_progress:
        cmd.append("--no-progress")
    if args.print_generation:
        cmd.append("--print-generation")
    if args.dry_run:
        cmd.append("--dry-run")

    if kaggle_json_path:
        cmd += ["--kaggle-json", str(kaggle_json_path)]

    return cmd


def latest_run_dir(repo_dir: Path, output_root: str) -> Optional[Path]:
    root = Path(output_root)
    if not root.is_absolute():
        root = repo_dir / root
    if not root.exists():
        return None
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0] if run_dirs else None


def show_summary(repo_dir: Path, output_root: str) -> None:
    latest = latest_run_dir(repo_dir, output_root)
    if not latest:
        print("No run directory found for summary.", flush=True)
        return
    print("latest_run_dir =", latest, flush=True)

    summary_path = latest / "summary.json"
    metrics_path = latest / "per_round_metrics.csv"
    strategy_path = latest / "strategy_history.jsonl"

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        except Exception as exc:
            print(f"Could not read summary.json: {type(exc).__name__}: {exc}", flush=True)

    if metrics_path.exists():
        print("metrics_csv =", metrics_path, flush=True)

    if strategy_path.exists():
        print("\nStrategy history tail:", flush=True)
        lines = strategy_path.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
        for line in lines:
            print(line, flush=True)

    zip_path = latest.with_suffix(".zip")
    if zip_path.exists():
        print("zip_path =", zip_path, flush=True)


def make_extra_analytics(repo_dir: Path, output_root: str) -> None:
    latest = latest_run_dir(repo_dir, output_root)
    if not latest:
        print("No run directory found for extra analytics.", flush=True)
        return

    analytics_dir = latest / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    print("analytics_dir =", analytics_dir, flush=True)

    try:
        import pandas as pd
    except Exception as exc:
        print(f"Extra analytics skipped: pandas is unavailable ({type(exc).__name__}: {exc})", flush=True)
        return

    metrics_path = latest / "per_round_metrics.csv"
    strategy_path = latest / "strategy_history.jsonl"
    lineage_path = analytics_dir / "self_improving_prompt_lineage.csv"
    model_attempts_path = analytics_dir / "model_fallback_attempts.csv"
    prompt_summary_path = analytics_dir / "prompt_variant_summary.csv"
    addendum_path = analytics_dir / "notebook_analytics_addendum.md"

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    prompt_summary = pd.read_csv(prompt_summary_path) if prompt_summary_path.exists() else pd.DataFrame()
    lineage = pd.read_csv(lineage_path) if lineage_path.exists() else pd.DataFrame()
    model_attempts = pd.read_csv(model_attempts_path) if model_attempts_path.exists() else pd.DataFrame()

    try:
        import matplotlib.pyplot as plt

        if not metrics.empty and "round" in metrics.columns and "kaggle_score" in metrics.columns and metrics["kaggle_score"].notna().any():
            ax = metrics.plot(x="round", y="kaggle_score", marker="o", title="Kaggle score by round")
            ax.set_xlabel("Round")
            ax.set_ylabel("Kaggle score")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(analytics_dir / "kaggle_score_timeline.png", dpi=150)
            plt.close(fig)

        if not prompt_summary.empty and "variant" in prompt_summary.columns and "success_rate" in prompt_summary.columns:
            ax = prompt_summary.sort_values("success_rate", ascending=False).plot.bar(
                x="variant", y="success_rate", title="Prompt variant success rate"
            )
            ax.set_xlabel("Prompt variant")
            ax.set_ylabel("Success rate")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(analytics_dir / "prompt_variant_success_rate.png", dpi=150)
            plt.close(fig)

        if not model_attempts.empty and "model" in model_attempts.columns and "ok" in model_attempts.columns:
            tmp = model_attempts.copy()
            tmp["ok_bool"] = tmp["ok"].astype(str).str.lower().eq("true")
            model_summary = tmp.groupby("model", dropna=False).agg(
                attempts=("ok_bool", "size"), successes=("ok_bool", "sum")
            ).reset_index()
            model_summary["success_rate"] = model_summary["successes"] / model_summary["attempts"]
            model_summary.to_csv(analytics_dir / "notebook_model_success_summary.csv", index=False)
            ax = model_summary.sort_values("success_rate", ascending=False).plot.bar(
                x="model", y="success_rate", title="Fallback model success rate"
            )
            ax.set_xlabel("Model")
            ax.set_ylabel("Success rate")
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(analytics_dir / "fallback_model_success_rate.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        print("Plot generation skipped:", type(exc).__name__, exc, flush=True)

    summary_lines = ["# Notebook analytics addendum", "", f"Run directory: `{latest}`"]
    if not metrics.empty:
        summary_lines.append(f"Total completed prompt rounds: **{len(metrics)}**.")
        if "ok" in metrics.columns:
            ok_count = metrics["ok"].astype(str).str.lower().eq("true").sum()
            summary_lines.append(f"Successful rounds: **{ok_count}**, failed rounds: **{len(metrics) - ok_count}**.")
        if "kaggle_score" in metrics.columns and metrics["kaggle_score"].notna().any():
            best_score = metrics["kaggle_score"].dropna().min()
            best_rows = metrics[metrics["kaggle_score"] == best_score]
            best_variant = best_rows.iloc[0].get("variant", "unknown")
            summary_lines.append(f"Best observed Kaggle score: **{best_score}** from variant `{best_variant}`.")
    if not prompt_summary.empty:
        summary_lines.append("")
        summary_lines.append(f"Prompt variants summarized: **{len(prompt_summary)}**.")
    if not lineage.empty:
        summary_lines.append(f"Self-improving prompt lineage rows: **{len(lineage)}**.")
    if not model_attempts.empty:
        summary_lines.append(f"Fallback model attempts: **{len(model_attempts)}**.")
    if strategy_path.exists():
        summary_lines.append("")
        summary_lines.append("## Strategy history tail")
        for line in strategy_path.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]:
            summary_lines.append(f"- `{line}`")

    addendum_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("extra_analytics_addendum =", addendum_path, flush=True)


def copy_zip_if_requested(args: argparse.Namespace, repo_dir: Path) -> None:
    if not args.copy_zip_to:
        return
    latest = latest_run_dir(repo_dir, args.output_root)
    if not latest:
        print("No latest run directory found; cannot copy zip.", flush=True)
        return
    zip_path = latest.with_suffix(".zip")
    if not zip_path.exists():
        print(f"Zip not found: {zip_path}", flush=True)
        return
    target = Path(args.copy_zip_to).expanduser()
    if target.is_dir() or str(args.copy_zip_to).endswith(os.sep):
        target.mkdir(parents=True, exist_ok=True)
        target = target / zip_path.name
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, target)
    print("copied_zip_path =", target.resolve(), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clone/unpack agents_4_puzzles, install dependencies, configure credentials, and run CayleyPy Pancake prompt sweep."
    )

    workspace = parser.add_argument_group("Workspace / repository source")
    workspace.add_argument("--source-mode", choices=["git_clone", "local_zip_path", "drive_zip", "upload_zip"], default="git_clone")
    workspace.add_argument("--archive-path", default=default_archive_path(), help="Path to repository zip for local_zip_path/drive_zip/upload_zip modes. Defaults to a zip next to this script or PANCAKE_ARCHIVE_PATH.")
    workspace.add_argument("--git-repo-url", default="https://github.com/visualcomments/agents_4_puzzles.git")
    workspace.add_argument("--git-branch", default="main")
    workspace.add_argument("--git-depth", type=int, default=1)
    workspace.add_argument("--clone-dir-name", default="agents_4_puzzles-main")
    workspace.add_argument("--workdir", default=default_workspace_dir(), help="Writable workspace directory. Defaults to ./pancake_prompt_sweep_workspace next to this script or PANCAKE_WORKDIR.")
    workspace.add_argument("--repo-dir", default="", help="Use an already prepared repository directory instead of cloning/unpacking.")
    workspace.add_argument("--force-clean", action=argparse.BooleanOptionalAction, default=True)

    deps = parser.add_argument_group("Dependency installation")
    deps.add_argument("--install-deps", action=argparse.BooleanOptionalAction, default=True)
    deps.add_argument("--upgrade-pip-tools", action=argparse.BooleanOptionalAction, default=True)
    deps.add_argument("--requirements-file", default="", help="Optional requirements file relative to repo root.")
    deps.add_argument("--extra-packages", default="kaggle pandas g4f")

    llm = parser.add_argument_group("LLM / g4f / API credentials")
    llm.add_argument("--openai-api-key", default="")
    llm.add_argument("--openrouter-api-key", default="")
    llm.add_argument("--gemini-api-key", default="")
    llm.add_argument("--anthropic-api-key", default="")
    llm.add_argument("--groq-api-key", default="")
    llm.add_argument("--together-api-key", default="")
    llm.add_argument("--g4f-api-key", default="")
    llm.add_argument("--g4f-provider", default="")
    llm.add_argument("--g4f-provider-list", default="")
    llm.add_argument("--g4f-allow-auto-fallback", action=argparse.BooleanOptionalAction, default=True)

    kaggle = parser.add_argument_group("Kaggle credentials")
    kaggle.add_argument("--kaggle-credential-mode", choices=["none", "env", "inline_json", "existing_path", "username_key"], default="env")
    kaggle.add_argument("--kaggle-json-inline", default="")
    kaggle.add_argument("--kaggle-json-path", default="")
    kaggle.add_argument("--kaggle-username", default="")
    kaggle.add_argument("--kaggle-key", default="")
    kaggle.add_argument("--kaggle-config-dir", default="")
    kaggle.add_argument("--kaggle-smoke-check", action=argparse.BooleanOptionalAction, default=True)

    main = parser.add_argument_group("Main pipeline behavior")
    main.add_argument("--competition", default="cayleypy-pancake")
    main.add_argument("--kaggle-competition", default="cayleypy-pancake")
    main.add_argument("--puzzles", default=None)
    main.add_argument("--variants", default="all")
    main.add_argument("--max-total-runs", type=int, default=12)
    main.add_argument("--run-name", default="pancake_full_prompt_sweep")
    main.add_argument("--submit-to-kaggle", action=argparse.BooleanOptionalAction, default=True)
    main.add_argument("--score-direction", choices=["min", "max"], default="min")
    main.add_argument("--output-root", default="runs/pancake_prompt_sweep")
    main.add_argument("--dry-run", action="store_true")

    models = parser.add_argument_group("LLM/model behavior")
    models.add_argument("--models", default="g4f:r1-1776,g4f:command-a,g4f:gemini-2.5-flash,g4f:command-r,g4f:gemini-2.0-flash")
    models.add_argument("--model-fallbacks", default="")
    models.add_argument("--agent-models", default="")
    models.add_argument("--planner-models", default="")
    models.add_argument("--coder-models", default="")
    models.add_argument("--fixer-models", default="")
    models.add_argument("--no-llm", action="store_true")

    budget = parser.add_argument_group("Search budget / prompt improvement")
    budget.add_argument("--keep-improving", action=argparse.BooleanOptionalAction, default=True)
    budget.add_argument("--self-improve-prompts", action=argparse.BooleanOptionalAction, default=True)
    budget.add_argument("--improvement-rounds", type=int, default=3)
    budget.add_argument("--breakthrough-improvement-rounds", type=int, default=6)
    budget.add_argument("--max-iters", type=int, default=8000)
    budget.add_argument("--breakthrough-max-iters", type=int, default=20000)
    budget.add_argument("--plan-beam-width", type=int, default=3)
    budget.add_argument("--frontier-width", type=int, default=6)
    budget.add_argument("--breakthrough-plan-beam-width", type=int, default=5)
    budget.add_argument("--breakthrough-frontier-width", type=int, default=10)
    budget.add_argument("--max-rows", type=int, default=0)
    budget.add_argument("--search-mode", choices=["classic", "hybrid"], default="hybrid")
    budget.add_argument("--archive-size", type=int, default=8)
    budget.add_argument("--refine-rounds", type=int, default=1)
    budget.add_argument("--allow-baseline", action="store_true")
    budget.add_argument("--baseline", default="competitions/cayleypy-pancake/baselines/pancake_baseline_solver.py")
    budget.add_argument("--run-timeout", type=int, default=None)
    budget.add_argument("--no-progress", action="store_true")

    g4f = parser.add_argument_group("g4f/recovery runner options")
    g4f.add_argument("--g4f-async", choices=["auto", "on", "off"], default="auto")
    g4f.add_argument("--g4f-recovery-rounds", type=int, default=None)
    g4f.add_argument("--baseline-patch-max-iters", type=int, default=None)
    g4f.add_argument("--g4f-recovery-max-iters", type=int, default=None)
    g4f.add_argument("--g4f-recovery-sleep", type=float, default=None)
    g4f.add_argument("--print-generation", action="store_true")
    g4f.add_argument("--print-generation-max-chars", type=int, default=None)
    g4f.add_argument("--max-response-chars", type=int, default=None)
    g4f.add_argument("--g4f-request-timeout", type=float, default=None)
    g4f.add_argument("--g4f-stop-at-python-fence", choices=["auto", "on", "off"], default="auto")

    scoring = parser.add_argument_group("Kaggle score polling / submit")
    scoring.add_argument("--score-polls", type=int, default=8)
    scoring.add_argument("--score-poll-sleep", type=float, default=45)
    scoring.add_argument("--submit-via", choices=["cli", "api", "auto"], default="cli")
    scoring.add_argument("--message-prefix", default="CayleyPy Pancake prompt sweep")

    post = parser.add_argument_group("Post-run output")
    post.add_argument("--overwrite-runner", action=argparse.BooleanOptionalAction, default=True)
    post.add_argument("--show-summary", action=argparse.BooleanOptionalAction, default=True)
    post.add_argument("--make-extra-analytics", action=argparse.BooleanOptionalAction, default=True)
    post.add_argument("--copy-zip-to", default="", help="Optional file path or directory where the resulting zip should be copied.")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_dir = prepare_workspace(args)
    install_dependencies(args, repo_dir)
    configure_llm_env(args)
    kaggle_json_path = configure_kaggle(args)
    write_runner(repo_dir, overwrite=args.overwrite_runner)

    cmd = build_runner_cmd(args, repo_dir, kaggle_json_path)
    ret = stream_cmd(cmd, cwd=repo_dir)
    if ret != 0:
        raise RuntimeError(f"Prompt sweep failed with returncode={ret}")

    if args.make_extra_analytics:
        make_extra_analytics(repo_dir, args.output_root)

    if args.show_summary:
        show_summary(repo_dir, args.output_root)

    copy_zip_if_requested(args, repo_dir)
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
