#!/usr/bin/env python3
"""Christophers Jewel prompt sweep with Kaggle-score feedback.

This runner sits above pipeline_cli.py and is intentionally standard-library only.
It tests all Christophers Jewel prompt bundles, records full logs/metrics, submits each
successful submission to Kaggle when enabled, reads the public score from Kaggle
submission history, and uses score movement to adjust the next prompt strategy.
It also supports ordered g4f model fallback lists and writes analytics artifacts.

The runner does not store Kaggle credentials. Configure Kaggle in Colab via
~/.kaggle/kaggle.json, KAGGLE_USERNAME/KAGGLE_KEY, or --kaggle-json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROMPT_VARIANTS: List[str] = [
    "regular",
    "improved",
    "dataset_adapted",
    "structured",
    "heuristic_boosted",
    "master_hybrid",
    "neighbour_model_hybrid",
    "score_guarded",
    "algorithmic_population",
    "portfolio_orchestrated",
    "hard_row_routed",
    "exact_score_population",
]

BREAKTHROUGH_VARIANTS: List[str] = [
    "exact_score_population",
    "hard_row_routed",
    "portfolio_orchestrated",
    "algorithmic_population",
    "master_hybrid",
    "score_guarded",
]

SAFE_VARIANTS: List[str] = [
    "score_guarded",
    "heuristic_boosted",
    "dataset_adapted",
    "neighbour_model_hybrid",
]


def split_model_list(value: Optional[str]) -> List[str]:
    """Parse comma/semicolon/newline separated model lists while keeping single values intact."""
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[;,\n]+", text) if p.strip()]
    return parts or [text]


def resolve_model_fallbacks(args: argparse.Namespace) -> List[str]:
    """Return ordered model fallbacks.

    If --model-fallbacks is passed it wins; otherwise --models itself can be a
    comma/semicolon/newline separated list. The runner passes exactly one model
    to pipeline_cli.py per attempt, so a failing g4f model does not block the
    entire prompt variant.
    """
    models = split_model_list(getattr(args, "model_fallbacks", None))
    if not models:
        models = split_model_list(getattr(args, "models", None))
    if not models:
        models = ["g4f:gpt-4o-mini"]
    seen: set[str] = set()
    out: List[str] = []
    for m in models:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def looks_like_g4f_fallback(models: Sequence[str]) -> bool:
    return len(models) > 1 and all(str(m).startswith("g4f:") or str(m).startswith("g4f/") for m in models)


MODEL_FAILURE_PATTERNS: Tuple[str, ...] = (
    "ratelimiterror",
    "rate limit",
    "missingautherror",
    "credentials required",
    "requires api key",
    "api key required",
    "need api key",
    "unauthorized",
    "forbidden",
    "provider is not working",
    "providernotworkingerror",
    "no working providers",
    "all providers failed",
    "retryprovider failed",
    "timed out",
    "timeout",
    "readtimeout",
    "connecttimeout",
    "remote protocol error",
    "connection error",
    "bad gateway",
    "service unavailable",
    "cloudflare",
    "captcha",
    "blocked",
    "empty response",
    "did not return a python file",
    "offline fallback",
    "offline-baseline",
    "falling back to baseline",
    "baseline fallback",
)


# Fallback artifacts must remain failed attempts even when the lower layer exits with rc=0
# and creates a CSV. Keep these markers specific so a legitimate solver may still use
# the word "fallback" internally for safe in-algorithm rollback logic.
FALLBACK_ARTIFACT_PATTERNS: Tuple[str, ...] = (
    "sample_submission fallback",
    "sample submission fallback",
    "using bundled sample submission",
    "copied sample_submission",
    "copied sample submission",
    "fallback to sample_submission",
    "fallback to sample submission",
    "known-good offline baseline",
    "known-good baseline",
    "reference baseline fallback",
    "offline baseline fallback",
    "llm generation failed",
    "llm_generation_failed",
    "generation failed; using baseline",
    "using fallback baseline",
    "emergency fallback",
)


def _read_text_if_exists(path: Optional[Path], max_chars: int = 200000) -> str:
    if path is None or not path.exists():
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except TypeError:
        data = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(data) > max_chars:
        return data[-max_chars:]
    return data


def _collect_model_failure_text(run_payload: Dict[str, Any], stdout_log: Optional[Path]) -> str:
    parts: List[str] = []
    if run_payload:
        try:
            parts.append(json.dumps(run_payload, ensure_ascii=False, default=str))
        except Exception:
            parts.append(str(run_payload))
    parts.append(_read_text_if_exists(stdout_log))
    return "\n".join(p for p in parts if p)


def detect_model_failure(run_payload: Dict[str, Any], stdout_log: Optional[Path]) -> Tuple[bool, List[str]]:
    """Return True when subprocess output shows that the selected g4f model did not really work.

    pipeline_cli.py can exit with rc=0 and write a CSV even when the selected g4f model
    fails and the lower layer falls back to a known-good offline baseline. For model
    sweeps this is a failed model attempt, so the next g4f model should be tried.
    """
    text = _collect_model_failure_text(run_payload, stdout_log).lower()
    markers = [p for p in MODEL_FAILURE_PATTERNS if p in text]
    if markers:
        return True, markers

    stages = run_payload.get("stages") if isinstance(run_payload, dict) else None
    if isinstance(stages, dict):
        for stage_name, stage_payload in stages.items():
            if "generate" in str(stage_name).lower() and isinstance(stage_payload, dict):
                status = str(stage_payload.get("status") or stage_payload.get("ok") or "").lower()
                if status in {"false", "failed", "error"}:
                    return True, [f"stage:{stage_name}:{status}"]
                err = str(stage_payload.get("error") or stage_payload.get("exception") or "").lower()
                if any(p in err for p in MODEL_FAILURE_PATTERNS):
                    return True, [p for p in MODEL_FAILURE_PATTERNS if p in err]
    return False, []


def _safe_int_metric(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def classify_generated_attempt_success(
    *,
    rc: int,
    csv_exists: bool,
    solver_path: Optional[Path],
    model_failure: bool,
    model_failure_markers: List[str],
    submission_stats: Dict[str, Any],
    run_payload: Dict[str, Any],
    stdout_log: Optional[Path],
) -> Tuple[bool, List[str]]:
    """Classify a prompt/model attempt using artifact quality, not just process exit code.

    A successful prompt must generate an importable solver artifact and a non-empty
    submission without UNSOLVED/blank rows. Any provider, baseline, or sample-submission
    fallback remains a failed attempt and is copied to failed_scripts.
    """
    reasons: List[str] = []
    if rc != 0:
        reasons.append(f"nonzero_returncode:{rc}")
    if not csv_exists:
        reasons.append("missing_output_csv")
    if model_failure:
        joined = ";".join(model_failure_markers) if model_failure_markers else "unknown"
        reasons.append(f"model_failure_or_lower_layer_fallback:{joined}")
    if solver_path is None or not solver_path.exists():
        reasons.append("missing_generated_solver")

    rows = _safe_int_metric(submission_stats.get("row_count"))
    empty_rows = _safe_int_metric(submission_stats.get("empty_rows")) or 0
    unsolved_rows = _safe_int_metric(submission_stats.get("unsolved_rows")) or 0
    if rows is None or rows <= 0:
        reasons.append("empty_or_unreadable_submission")
    if empty_rows > 0:
        reasons.append(f"submission_has_empty_rows:{empty_rows}")
    if unsolved_rows > 0:
        reasons.append(f"submission_has_unsolved_rows:{unsolved_rows}")
    if submission_stats.get("csv_error"):
        reasons.append(f"submission_csv_error:{submission_stats.get('csv_error')}")

    artifact_text = _collect_model_failure_text(run_payload, stdout_log).lower()
    artifact_markers = [p for p in FALLBACK_ARTIFACT_PATTERNS if p in artifact_text]
    if artifact_markers:
        reasons.append("fallback_artifact_markers:" + ";".join(artifact_markers))

    return (len(reasons) == 0), reasons


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str, max_len: int = 80) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return (out or "item")[:max_len]


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "pending", "error"}:
        return None
    text = text.replace(",", "")
    try:
        val = float(text)
    except Exception:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_run_payload(payload: Any) -> Dict[str, Any]:
    """Return a dict-shaped run payload even when pipeline_cli writes a JSON list.

    Some versions of pipeline_cli.py write run_log.json as a list of stage/event
    records instead of a single summary dict. The sweep runner expects dict.get(),
    so normalize the shape and keep the original list under raw_run_log_events.
    """
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        normalized: Dict[str, Any] = {
            "raw_run_log_type": "list",
            "raw_run_log_len": len(payload),
            "raw_run_log_events": payload,
        }
        for item in payload:
            if not isinstance(item, dict):
                continue
            # Keep useful top-level summary fields if any event contains them.
            for key in ("solver", "output", "submission", "status", "ok", "competition", "prompt_variant"):
                if key in item and key not in normalized:
                    normalized[key] = item.get(key)
            stages = normalized.setdefault("stages", {})
            stage_name = item.get("stage") or item.get("name") or item.get("step")
            if stage_name and isinstance(stages, dict):
                stages[str(stage_name)] = item
        return normalized
    if payload is None:
        return {}
    return {"raw_run_log_type": type(payload).__name__, "raw_run_log_value": str(payload)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> Optional[str]:
    try:
        import hashlib
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def light_file_stats(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False}
    return {
        "exists": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def csv_submission_stats(path: Optional[Path]) -> Dict[str, Any]:
    stats: Dict[str, Any] = light_file_stats(path)
    if path is None or not path.exists():
        return stats
    row_count = 0
    move_col = None
    total_tokens = 0
    max_tokens = 0
    empty_rows = 0
    unsolved_rows = 0
    sample_rows: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for candidate in ("moves", "solution", "move_sequence", "answer", "path"):
                if candidate in fieldnames:
                    move_col = candidate
                    break
            if move_col is None and len(fieldnames) >= 2:
                move_col = fieldnames[-1]
            for row in reader:
                row_count += 1
                if len(sample_rows) < 3:
                    sample_rows.append({k: str(v)[:200] for k, v in row.items()})
                moves = str(row.get(move_col, "") if move_col else "")
                if not moves.strip():
                    empty_rows += 1
                if moves.strip().upper() == "UNSOLVED":
                    unsolved_rows += 1
                toks = [tok for tok in re.split(r"[\s,;.]+", moves.strip()) if tok]
                total_tokens += len(toks)
                max_tokens = max(max_tokens, len(toks))
        stats.update(
            {
                "row_count": row_count,
                "columns": fieldnames,
                "move_column": move_col,
                "total_move_tokens": total_tokens,
                "mean_move_tokens": (total_tokens / row_count) if row_count else None,
                "max_move_tokens": max_tokens,
                "empty_rows": empty_rows,
                "unsolved_rows": unsolved_rows,
                "sample_rows": sample_rows,
            }
        )
    except Exception as exc:
        stats["csv_error"] = f"{type(exc).__name__}: {exc}"
    return stats


def validate_jewel_submission_replay(repo_dir: Path, submission_csv: Path, log_path: Path) -> Dict[str, Any]:
    """Replay-validate every generated Christophers Jewel submission row against official data."""
    validator = repo_dir / "competitions" / "cayleypy-christophers-jewel" / "validate_submission_csv.py"
    payload: Dict[str, Any] = {
        "ok": False,
        "validator": str(validator),
        "submission_csv": str(submission_csv),
        "log": str(log_path),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not validator.exists():
        payload["error"] = "missing_validate_submission_csv.py"
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    timeout_s = int(float(os.getenv("JEWEL_SUBMISSION_VALIDATOR_TIMEOUT_S", "180")))
    cmd = [sys.executable, str(validator), "--submission", str(submission_csv)]
    try:
        proc = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True, timeout=timeout_s)
        log_path.write_text(
            "# cmd=" + " ".join(cmd) + "\n"
            + "# returncode=" + str(proc.returncode) + "\n\n"
            + (proc.stdout or "")
            + ("\n" + proc.stderr if proc.stderr else ""),
            encoding="utf-8",
        )
        payload["returncode"] = int(proc.returncode or 0)
        payload["stdout"] = (proc.stdout or "")[-4000:]
        payload["stderr"] = (proc.stderr or "")[-4000:]
        payload["ok"] = proc.returncode == 0
        if proc.returncode == 0:
            try:
                payload["stats"] = json.loads((proc.stdout or "{}").strip().splitlines()[-1])
            except Exception:
                payload["stats"] = {}
        else:
            payload["error"] = payload.get("stderr") or payload.get("stdout") or "replay_validation_failed"
    except subprocess.TimeoutExpired as exc:
        payload["error"] = f"timeout_after_{timeout_s}s"
        log_path.write_text(
            "# TIMEOUT\n"
            + (exc.stdout or "")
            + ("\n" + exc.stderr if exc.stderr else ""),
            encoding="utf-8",
        )
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def copy_if_exists(src: Optional[Path], dst_dir: Path, prefix: str = "") -> Optional[Path]:
    if src is None or not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{prefix}{src.name}"
    try:
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def run_streaming(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> Tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items() if v is not None})
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# started_utc={utc_now()}\n")
        log.write("# cwd=" + str(cwd) + "\n")
        log.write("# cmd=" + " ".join(map(str, cmd)) + "\n\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                list(cmd),
                cwd=str(cwd),
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            while True:
                line = proc.stdout.readline()
                if line:
                    print(line, end="")
                    log.write(line)
                    log.flush()
                if proc.poll() is not None:
                    rest = proc.stdout.read()
                    if rest:
                        print(rest, end="")
                        log.write(rest)
                    break
                if timeout and (time.time() - started) > timeout:
                    proc.kill()
                    log.write(f"\n# TIMEOUT after {timeout}s\n")
                    return 124, time.time() - started
            rc = int(proc.returncode or 0)
            log.write(f"\n# finished_utc={utc_now()} returncode={rc}\n")
            return rc, time.time() - started
        except Exception:
            log.write("\n# EXCEPTION\n")
            log.write(traceback.format_exc())
            return 125, time.time() - started


@dataclass
class PromptBundle:
    variant: str
    prompt_file: Optional[str] = None
    custom_prompts_file: Optional[str] = None
    planner_variant_files: List[str] = field(default_factory=list)
    exists: bool = True


def discover_prompt_bundles(repo_dir: Path, explicit_variants: Optional[List[str]] = None) -> List[PromptBundle]:
    prompt_dir = repo_dir / "competitions" / "cayleypy-christophers-jewel" / "prompts"
    variants = list(explicit_variants or PROMPT_VARIANTS)
    if explicit_variants is None and prompt_dir.exists():
        for p in sorted(prompt_dir.glob("user_prompt_*.txt")):
            v = p.stem.replace("user_prompt_", "")
            if v not in variants:
                variants.append(v)
        for p in sorted(prompt_dir.glob("custom_prompts_*.json")):
            v = p.stem.replace("custom_prompts_", "")
            if v not in variants and v != "template":
                variants.append(v)
    bundles: List[PromptBundle] = []
    for v in variants:
        if v == "regular":
            prompt = prompt_dir / "user_prompt_regular.txt"
            if not prompt.exists():
                prompt = prompt_dir / "user_prompt.txt"
            custom = prompt_dir / "custom_prompts_regular.json"
            if not custom.exists():
                custom = prompt_dir / "custom_prompts_template.json"
        else:
            prompt = prompt_dir / f"user_prompt_{v}.txt"
            custom = prompt_dir / f"custom_prompts_{v}.json"
        planners = sorted(str(p) for p in prompt_dir.glob("planner_variant_*.json"))
        bundles.append(
            PromptBundle(
                variant=v,
                prompt_file=str(prompt) if prompt.exists() else None,
                custom_prompts_file=str(custom) if custom.exists() else None,
                planner_variant_files=planners,
                exists=bool((prompt.exists() or custom.exists()) and prompt_dir.exists()),
            )
        )
    return bundles


def parse_kaggle_csv(text: str) -> List[Dict[str, str]]:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return []
    header_idx = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if ("score" in low and "status" in low) or ("description" in low and "date" in low):
            header_idx = i
            break
    csv_text = "\n".join(lines[header_idx:])
    try:
        return list(csv.DictReader(csv_text.splitlines()))
    except Exception:
        return []


def fetch_kaggle_submissions(
    *,
    repo_dir: Path,
    competition: str,
    log_path: Path,
    kaggle_json: Optional[Path] = None,
) -> Tuple[List[Dict[str, str]], int]:
    env = os.environ.copy()
    if kaggle_json:
        # Kaggle CLI primarily reads ~/.kaggle/kaggle.json; this is a fallback for API helpers.
        env["KAGGLE_CONFIG_DIR"] = str(kaggle_json.parent)
    cmd = ["kaggle", "competitions", "submissions", competition, "-v", "-q"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(repo_dir), env=env, text=True, capture_output=True)
    log_path.write_text((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    return parse_kaggle_csv((proc.stdout or "") + "\n" + (proc.stderr or "")), int(proc.returncode or 0)


def extract_score(row: Dict[str, str]) -> Optional[float]:
    for key in row.keys():
        low = key.lower().replace("_", "")
        if low in {"publicscore", "score"} or ("public" in low and "score" in low):
            val = parse_float(row.get(key))
            if val is not None:
                return val
    for key in row.keys():
        if "score" in key.lower():
            val = parse_float(row.get(key))
            if val is not None:
                return val
    return None


def find_matching_submission(rows: List[Dict[str, str]], message: str, file_name: str) -> Optional[Dict[str, str]]:
    if not rows:
        return None
    msg = message.strip()
    for row in rows:
        blob = " ".join(str(v) for v in row.values())
        if msg and msg in blob:
            return row
    for row in rows:
        blob = " ".join(str(v) for v in row.values())
        if file_name and file_name in blob:
            return row
    return rows[0]


def submit_to_kaggle(
    *,
    repo_dir: Path,
    competition: str,
    submission_csv: Path,
    message: str,
    submit_via: str,
    kaggle_json: Optional[Path],
    log_path: Path,
) -> Tuple[Dict[str, Any], int, float]:
    env = os.environ.copy()
    if kaggle_json:
        env["KAGGLE_CONFIG_DIR"] = str(kaggle_json.parent)
    # Always submit the already generated CSV directly. Do not call pipeline_cli.py
    # here, because that would rebuild/overwrite the successful candidate output.
    # The --submit-via option is kept for notebook compatibility, but this outer
    # score-feedback runner relies on Kaggle CLI so that the exact generated CSV
    # is the file that reaches Kaggle.
    cmd = ["kaggle", "competitions", "submit", competition, "-f", str(submission_csv), "-m", message]
    rc, seconds = run_streaming(cmd, cwd=repo_dir, log_path=log_path, env=env)
    return {"submitted": rc == 0, "returncode": rc, "message": message, "log": str(log_path)}, rc, seconds


def wait_for_kaggle_score(
    *,
    repo_dir: Path,
    competition: str,
    message: str,
    file_name: str,
    kaggle_json: Optional[Path],
    log_dir: Path,
    polls: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    best: Dict[str, Any] = {"score": None, "row": None, "polls": 0, "status": "not_polled"}
    for idx in range(max(0, polls)):
        rows, rc = fetch_kaggle_submissions(
            repo_dir=repo_dir,
            competition=competition,
            kaggle_json=kaggle_json,
            log_path=log_dir / f"kaggle_submissions_poll_{idx+1:02d}.csvlog",
        )
        row = find_matching_submission(rows, message=message, file_name=file_name)
        score = extract_score(row) if row else None
        best = {"score": score, "row": row, "polls": idx + 1, "returncode": rc, "status": "score_ready" if score is not None else "pending"}
        if score is not None:
            return best
        if idx < polls - 1:
            time.sleep(max(0.0, sleep_seconds))
    return best


@dataclass
class StrategyState:
    direction: str = "min"
    best_score: Optional[float] = None
    best_variant: Optional[str] = None
    best_round: Optional[int] = None
    no_score_change_streak: int = 0
    success_streak: int = 0
    failure_streak: int = 0
    breakthrough_pressure: int = 0
    decisions: List[Dict[str, Any]] = field(default_factory=list)

    def better(self, score: Optional[float]) -> bool:
        if score is None:
            return False
        if self.best_score is None:
            return True
        if self.direction == "max":
            return score > self.best_score
        return score < self.best_score

    def update(self, *, round_idx: int, variant: str, ok: bool, score: Optional[float]) -> Dict[str, Any]:
        prev = self.best_score
        improved = ok and self.better(score)
        score_changed = score is not None and prev is not None and abs(score - prev) > 1e-12
        if improved:
            self.best_score = score
            self.best_variant = variant
            self.best_round = round_idx
            self.success_streak += 1
            self.failure_streak = 0
            # User-requested behavior: score movement should trigger more breakthrough attempts.
            self.breakthrough_pressure = max(self.breakthrough_pressure, 2)
            self.no_score_change_streak = 0
            mode = "score_improved_promote_then_breakthrough"
        elif ok:
            self.success_streak += 1
            self.failure_streak = 0
            self.no_score_change_streak += 1
            self.breakthrough_pressure += 1
            mode = "validated_but_score_flat_use_more_breakthrough"
        else:
            self.failure_streak += 1
            self.success_streak = 0
            # When scripts fail, back off into safer guard-heavy prompts before trying breakthroughs again.
            self.breakthrough_pressure = max(0, self.breakthrough_pressure - 1)
            mode = "script_failed_backoff_to_guarded"
        decision = {
            "round": round_idx,
            "variant": variant,
            "ok": ok,
            "score": score,
            "previous_best_score": prev,
            "best_score": self.best_score,
            "best_variant": self.best_variant,
            "score_changed": score_changed,
            "improved": improved,
            "mode": mode,
            "breakthrough_pressure": self.breakthrough_pressure,
            "no_score_change_streak": self.no_score_change_streak,
        }
        self.decisions.append(decision)
        return decision


def reorder_queue(queue: List[str], state: StrategyState) -> List[str]:
    if not queue:
        return queue
    if state.failure_streak >= 2:
        priority = SAFE_VARIANTS
    elif state.breakthrough_pressure > 0 or state.no_score_change_streak > 0:
        priority = BREAKTHROUGH_VARIANTS
    else:
        return queue
    prioritized = [v for v in priority if v in queue]
    rest = [v for v in queue if v not in prioritized]
    return prioritized + rest


def build_run_command(
    args: argparse.Namespace,
    variant: str,
    output_csv: Path,
    run_log: Path,
    state: StrategyState,
    model_override: Optional[str] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        "pipeline_cli.py",
        "run",
        "--competition",
        args.competition,
        "--output",
        str(output_csv),
        "--prompt-variant",
        variant,
        "--models",
        model_override or args.models,
        "--run-log",
        str(run_log),
        "--schema-check",
    ]
    if args.puzzles:
        cmd.extend(["--puzzles", args.puzzles])
    if args.agent_models:
        cmd.extend(["--agent-models", args.agent_models])
    if args.planner_models:
        cmd.extend(["--planner-models", args.planner_models])
    if args.coder_models:
        cmd.extend(["--coder-models", args.coder_models])
    if args.fixer_models:
        cmd.extend(["--fixer-models", args.fixer_models])
    cmd.extend(["--search-mode", args.search_mode])
    cmd.extend(["--plan-beam-width", str(args.plan_beam_width)])
    cmd.extend(["--frontier-width", str(args.frontier_width)])
    cmd.extend(["--archive-size", str(args.archive_size)])
    cmd.extend(["--refine-rounds", str(args.refine_rounds)])
    max_iters = args.max_iters
    improvement_rounds = args.improvement_rounds
    if state.breakthrough_pressure > 0 and variant in BREAKTHROUGH_VARIANTS:
        # Breakthrough mode: spend a bit more search budget after a score movement or plateau.
        max_iters = max(max_iters, args.breakthrough_max_iters)
        improvement_rounds = max(improvement_rounds, args.breakthrough_improvement_rounds)
        cmd.extend(["--plan-beam-width", str(max(args.plan_beam_width, args.breakthrough_plan_beam_width))])
        cmd.extend(["--frontier-width", str(max(args.frontier_width, args.breakthrough_frontier_width))])
    cmd.extend(["--max-iters", str(max_iters)])
    if args.keep_improving:
        cmd.append("--keep-improving")
        cmd.extend(["--improvement-rounds", str(improvement_rounds)])
    if args.self_improve_prompts:
        cmd.append("--self-improve-prompts")
    if args.allow_baseline:
        cmd.append("--allow-baseline")
    if args.no_llm:
        cmd.append("--no-llm")
    if args.max_rows:
        cmd.extend(["--max-rows", str(args.max_rows)])
    if args.baseline:
        cmd.extend(["--baseline", args.baseline])
    if args.g4f_async == "on":
        cmd.append("--g4f-async")
    elif args.g4f_async == "off":
        cmd.append("--no-g4f-async")
    if args.g4f_recovery_rounds is not None:
        cmd.extend(["--g4f-recovery-rounds", str(args.g4f_recovery_rounds)])
    if args.baseline_patch_max_iters is not None:
        cmd.extend(["--baseline-patch-max-iters", str(args.baseline_patch_max_iters)])
    if args.g4f_recovery_max_iters is not None:
        cmd.extend(["--g4f-recovery-max-iters", str(args.g4f_recovery_max_iters)])
    if args.g4f_recovery_sleep is not None:
        cmd.extend(["--g4f-recovery-sleep", str(args.g4f_recovery_sleep)])
    if args.print_generation:
        cmd.append("--print-generation")
    if args.print_generation_max_chars is not None:
        cmd.extend(["--print-generation-max-chars", str(args.print_generation_max_chars)])
    if args.max_response_chars is not None:
        cmd.extend(["--max-response-chars", str(args.max_response_chars)])
    if args.g4f_request_timeout is not None:
        cmd.extend(["--g4f-request-timeout", str(args.g4f_request_timeout)])
    if args.g4f_stop_at_python_fence == "on":
        cmd.append("--g4f-stop-at-python-fence")
    elif args.g4f_stop_at_python_fence == "off":
        cmd.append("--no-g4f-stop-at-python-fence")
    if looks_like_g4f_fallback(resolve_model_fallbacks(args)):
        # Let AgentLaboratory/g4f try alternate providers inside a selected model, while
        # this runner still performs model-level fallback between model names.
        os.environ.setdefault("AGENTLAB_G4F_PROVIDER_ALLOW_AUTO_FALLBACK", "1")
    if args.no_progress:
        cmd.append("--no-progress")
    return cmd


def flatten_metrics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for r in records:
        sub = r.get("submission_stats") or {}
        run_log = r.get("run_log_summary") or {}
        decision = r.get("strategy_decision") or {}
        flat.append(
            {
                "round": r.get("round"),
                "variant": r.get("variant"),
                "model": r.get("model"),
                "model_attempt_count": len(r.get("model_attempts") or []),
                "attempted_models": ";".join([str(a.get("model")) for a in (r.get("model_attempts") or [])]),
                "ok": r.get("ok"),
                "returncode": r.get("returncode"),
                "runtime_seconds": r.get("runtime_seconds"),
                "solver_path": r.get("solver_path"),
                "output_csv": r.get("output_csv"),
                "csv_exists": sub.get("exists"),
                "model_failure_detected": any(bool(a.get("model_failure_detected")) for a in (r.get("model_attempts") or [])),
                "model_failure_markers": ";".join(sorted({str(marker) for a in (r.get("model_attempts") or []) for marker in str(a.get("model_failure_markers") or "").split(";") if marker})),
                "row_count": sub.get("row_count"),
                "total_move_tokens": sub.get("total_move_tokens"),
                "mean_move_tokens": sub.get("mean_move_tokens"),
                "max_move_tokens": sub.get("max_move_tokens"),
                "empty_rows": sub.get("empty_rows"),
                "unsolved_rows": sub.get("unsolved_rows"),
                "kaggle_submitted": (r.get("kaggle") or {}).get("submitted"),
                "kaggle_score": (r.get("kaggle_score") or {}).get("score"),
                "score_improved": decision.get("improved"),
                "strategy_mode": decision.get("mode"),
                "best_score_after_round": decision.get("best_score"),
                "stage_seconds_generate": ((run_log.get("stages") or {}).get("generate_solver") or {}).get("seconds"),
                "stage_seconds_build": ((run_log.get("stages") or {}).get("build_submission") or {}).get("seconds"),
                "stage_seconds_validate": ((run_log.get("stages") or {}).get("validate_solver") or {}).get("seconds"),
            }
        )
    return flat


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})



def _group_rows(rows: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(key), []).append(row)
    return groups


def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [parse_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    return sum(nums) / len(nums) if nums else None


def _best(values: Iterable[Any], direction: str) -> Optional[float]:
    nums = [parse_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None
    return max(nums) if direction == "max" else min(nums)


def _walk_prompt_signals(obj: Any, prefix: str = "") -> List[Dict[str, Any]]:
    """Best-effort extraction of self-improving prompt traces from arbitrary run_log JSON."""
    signals: List[Dict[str, Any]] = []
    interesting = ("prompt", "improv", "refine", "self", "feedback", "strategy", "breakthrough")
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            lk = str(k).lower()
            if any(tok in lk for tok in interesting):
                preview = json.dumps(v, ensure_ascii=False)[:2000] if not isinstance(v, str) else v[:2000]
                signals.append({"path": path, "key": str(k), "preview": preview})
            if isinstance(v, (dict, list)):
                signals.extend(_walk_prompt_signals(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):
            path = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                signals.extend(_walk_prompt_signals(v, path))
            elif isinstance(v, str) and any(tok in v.lower() for tok in interesting):
                signals.append({"path": path, "key": "list_item", "preview": v[:2000]})
    return signals


def _grep_log_signals(path: Optional[str], limit: int = 80) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    needles = ("self-improve", "self improve", "improvement", "refine", "prompt", "breakthrough", "score")
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                low = line.lower()
                if any(n in low for n in needles):
                    out.append({"line": line_no, "text": line.rstrip()[:1200]})
                    if len(out) >= limit:
                        break
    except Exception as exc:
        out.append({"line": None, "text": f"log_read_error: {type(exc).__name__}: {exc}"})
    return out


def generate_analytics(run_dir: Path, records: List[Dict[str, Any]], state: StrategyState) -> Dict[str, Any]:
    """Create analysis artifacts that are included in the auto-downloaded zip."""
    analytics_dir = run_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    flat = flatten_metrics(records)
    write_csv(analytics_dir / "score_timeline.csv", flat)

    # Prompt-level summary
    prompt_rows: List[Dict[str, Any]] = []
    for variant, rows in _group_rows(flat, "variant").items():
        prompt_rows.append({
            "variant": variant,
            "runs": len(rows),
            "successes": sum(1 for r in rows if str(r.get("ok")).lower() == "true"),
            "failures": sum(1 for r in rows if str(r.get("ok")).lower() != "true"),
            "success_rate": (sum(1 for r in rows if str(r.get("ok")).lower() == "true") / len(rows)) if rows else None,
            "best_kaggle_score": _best([r.get("kaggle_score") for r in rows], state.direction),
            "mean_runtime_seconds": _mean([r.get("runtime_seconds") for r in rows]),
            "mean_move_tokens": _mean([r.get("mean_move_tokens") for r in rows]),
            "score_improvements": sum(1 for r in rows if str(r.get("score_improved")).lower() == "true"),
        })
    write_csv(analytics_dir / "prompt_variant_summary.csv", prompt_rows)

    # Model fallback analytics
    attempts: List[Dict[str, Any]] = []
    for r in records:
        for a in r.get("model_attempts") or []:
            attempts.append({
                "round": r.get("round"),
                "variant": r.get("variant"),
                "model": a.get("model"),
                "attempt_index": a.get("attempt_index"),
                "ok": a.get("ok"),
                "returncode": a.get("returncode"),
                "runtime_seconds": a.get("runtime_seconds"),
                "output_csv": a.get("output_csv"),
                "csv_exists": a.get("csv_exists"),
                "model_failure_detected": a.get("model_failure_detected"),
                "model_failure_markers": a.get("model_failure_markers"),
                "stdout_log": a.get("stdout_log"),
                "run_log": a.get("run_log"),
            })
    write_csv(analytics_dir / "model_fallback_attempts.csv", attempts)
    model_rows: List[Dict[str, Any]] = []
    for model, rows in _group_rows(attempts, "model").items():
        model_rows.append({
            "model": model,
            "attempts": len(rows),
            "successes": sum(1 for r in rows if str(r.get("ok")).lower() == "true"),
            "failures": sum(1 for r in rows if str(r.get("ok")).lower() != "true"),
            "success_rate": (sum(1 for r in rows if str(r.get("ok")).lower() == "true") / len(rows)) if rows else None,
            "mean_runtime_seconds": _mean([r.get("runtime_seconds") for r in rows]),
        })
    write_csv(analytics_dir / "model_fallback_summary.csv", model_rows)

    # Strategy and self-improvement traces
    strategy_rows = list(state.decisions)
    write_csv(analytics_dir / "strategy_transitions.csv", strategy_rows)
    prompt_signal_rows: List[Dict[str, Any]] = []
    lineage_rows: List[Dict[str, Any]] = []
    for r in records:
        signals = _walk_prompt_signals(r.get("run_log_summary") or {})
        log_signals = _grep_log_signals(r.get("stdout_log"))
        decision = r.get("strategy_decision") or {}
        lineage_rows.append({
            "round": r.get("round"),
            "variant": r.get("variant"),
            "model": r.get("model"),
            "ok": r.get("ok"),
            "kaggle_score": (r.get("kaggle_score") or {}).get("score"),
            "score_improved": decision.get("improved"),
            "strategy_mode": decision.get("mode"),
            "breakthrough_pressure": decision.get("breakthrough_pressure"),
            "prompt_json_signal_count": len(signals),
            "prompt_log_signal_count": len(log_signals),
        })
        for sig in signals:
            prompt_signal_rows.append({
                "round": r.get("round"),
                "variant": r.get("variant"),
                "model": r.get("model"),
                "source": "run_log_json",
                "path": sig.get("path"),
                "key": sig.get("key"),
                "preview": sig.get("preview"),
            })
        for sig in log_signals:
            prompt_signal_rows.append({
                "round": r.get("round"),
                "variant": r.get("variant"),
                "model": r.get("model"),
                "source": "stdout_log",
                "path": sig.get("line"),
                "key": "log_line",
                "preview": sig.get("text"),
            })
    write_csv(analytics_dir / "self_improving_prompt_lineage.csv", lineage_rows)
    write_csv(analytics_dir / "self_improving_prompt_signals.csv", prompt_signal_rows)
    with (analytics_dir / "self_improving_prompt_signals.jsonl").open("w", encoding="utf-8") as f:
        for row in prompt_signal_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Successful vs failed scripts manifest
    script_rows: List[Dict[str, Any]] = []
    for r in records:
        script_rows.append({
            "round": r.get("round"),
            "variant": r.get("variant"),
            "model": r.get("model"),
            "ok": r.get("ok"),
            "solver_path": r.get("solver_path"),
            "copied_solver": r.get("copied_solver"),
            "returncode": r.get("returncode"),
            "kaggle_score": (r.get("kaggle_score") or {}).get("score"),
        })
    write_csv(analytics_dir / "successful_vs_failed_scripts.csv", script_rows)

    best = state.best_score
    report = []
    report.append("# Christophers Jewel prompt sweep analytics\n")
    report.append(f"Generated UTC: {utc_now()}\n")
    report.append(f"Score direction: `{state.direction}`; best score: `{best}`; best variant: `{state.best_variant}`; best round: `{state.best_round}`.\n")
    report.append("## Model fallback\n")
    if attempts:
        report.append("The runner tried models sequentially per prompt variant and stopped after the first successful CSV. See `model_fallback_attempts.csv` and `model_fallback_summary.csv`.\n")
    else:
        report.append("No model fallback attempts were recorded.\n")
    report.append("## Self-improving prompt dynamics\n")
    report.append("Use `self_improving_prompt_lineage.csv` as the timeline: it joins round, prompt variant, selected model, Kaggle score, strategy mode, breakthrough pressure, and counts of extracted prompt-improvement signals.\n")
    report.append("Use `self_improving_prompt_signals.csv` / `.jsonl` for the extracted prompt/refinement/feedback snippets from run logs. Extraction is best-effort because pipeline logs can vary by prompt variant.\n")
    report.append("## Strategy interpretation\n")
    report.append("`score_improved_promote_then_breakthrough` means a Kaggle-score movement promoted the lineage but also increased breakthrough pressure. `validated_but_score_flat_use_more_breakthrough` means the generated code passed local checks but did not improve the score, so more aggressive prompt families should be tried. `script_failed_backoff_to_guarded` means the runner backs off into safer prompts.\n")
    report.append("## Key files\n")
    report.append("- `prompt_variant_summary.csv` — aggregate metrics per prompt family.\n")
    report.append("- `score_timeline.csv` — one row per completed round.\n")
    report.append("- `strategy_transitions.csv` — score-driven strategy decisions.\n")
    report.append("- `model_fallback_summary.csv` — reliability of each g4f fallback model.\n")
    report.append("- `successful_vs_failed_scripts.csv` — manifest of solver scripts and outcomes.\n")
    report_path = analytics_dir / "analysis_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return {
        "analytics_dir": str(analytics_dir),
        "analysis_report": str(report_path),
        "prompt_signal_count": len(prompt_signal_rows),
        "model_attempt_count": len(attempts),
    }


def package_results(run_dir: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(run_dir.parent))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run every Christophers Jewel prompt bundle and use Kaggle score feedback for strategy.")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--competition", default="cayleypy-christophers-jewel")
    parser.add_argument("--kaggle-competition", default="cayleypy-christophers-jewel")
    parser.add_argument("--puzzles", default=None)
    parser.add_argument("--models", default="g4f:gpt-4o-mini", help="One model or comma/semicolon/newline-separated fallback list. Example: g4f:gpt-4o-mini,g4f:gpt-4o")
    parser.add_argument("--model-fallbacks", default=None, help="Explicit ordered model fallback list. Overrides --models when provided.")
    parser.add_argument("--agent-models", default=None)
    parser.add_argument("--planner-models", default=None)
    parser.add_argument("--coder-models", default=None)
    parser.add_argument("--fixer-models", default=None)
    parser.add_argument("--variants", default="all", help="Comma-separated variants or 'all'.")
    parser.add_argument("--max-total-runs", type=int, default=None)
    parser.add_argument("--score-direction", choices=["min", "max"], default="min")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="runs/jewel_prompt_sweep")
    parser.add_argument("--search-mode", choices=["classic", "hybrid"], default="hybrid")
    parser.add_argument("--plan-beam-width", type=int, default=3)
    parser.add_argument("--frontier-width", type=int, default=6)
    parser.add_argument("--archive-size", type=int, default=8)
    parser.add_argument("--refine-rounds", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=8000)
    parser.add_argument("--breakthrough-max-iters", type=int, default=20000)
    parser.add_argument("--breakthrough-plan-beam-width", type=int, default=5)
    parser.add_argument("--breakthrough-frontier-width", type=int, default=10)
    parser.add_argument("--keep-improving", action="store_true")
    parser.add_argument("--improvement-rounds", type=int, default=3)
    parser.add_argument("--breakthrough-improvement-rounds", type=int, default=6)
    parser.add_argument("--self-improve-prompts", action="store_true")
    parser.add_argument("--allow-baseline", action="store_true")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--run-timeout", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--g4f-async", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--g4f-recovery-rounds", type=int, default=None)
    parser.add_argument("--baseline-patch-max-iters", type=int, default=None)
    parser.add_argument("--g4f-recovery-max-iters", type=int, default=None)
    parser.add_argument("--g4f-recovery-sleep", type=float, default=None)
    parser.add_argument("--print-generation", action="store_true")
    parser.add_argument("--print-generation-max-chars", type=int, default=None)
    parser.add_argument("--max-response-chars", type=int, default=None)
    parser.add_argument("--g4f-request-timeout", type=float, default=None)
    parser.add_argument("--g4f-stop-at-python-fence", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--submit", action="store_true", help="Submit every successful prompt output to Kaggle.")
    parser.add_argument("--require-submit-success", action=argparse.BooleanOptionalAction, default=True, help="When --submit is enabled, classify a round as successful only if Kaggle upload is confirmed.")
    parser.add_argument("--submit-via", choices=["cli", "api", "auto"], default="cli")
    parser.add_argument("--kaggle-json", default=None)
    parser.add_argument("--score-polls", type=int, default=6)
    parser.add_argument("--score-poll-sleep", type=float, default=30.0)
    parser.add_argument("--message-prefix", default="Christophers Jewel prompt sweep")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_dir = Path(args.repo_dir).resolve()
    if not (repo_dir / "pipeline_cli.py").exists():
        raise SystemExit(f"pipeline_cli.py not found under {repo_dir}")

    if args.variants.strip().lower() == "all":
        explicit = None
    else:
        explicit = [v.strip() for v in args.variants.split(",") if v.strip()]
    bundles = discover_prompt_bundles(repo_dir, explicit)
    queue = [b.variant for b in bundles if b.exists]
    if not queue:
        raise SystemExit("No Christophers Jewel prompt bundles were found.")
    if args.max_total_runs is not None:
        queue = queue[: max(0, int(args.max_total_runs))]

    model_fallbacks = resolve_model_fallbacks(args)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (repo_dir / args.output_root / slugify(run_name)).resolve()
    logs_dir = run_dir / "logs"
    outputs_dir = run_dir / "submissions"
    success_dir = run_dir / "successful_scripts"
    failed_dir = run_dir / "failed_scripts"
    prompt_dir = run_dir / "prompt_manifests"
    run_dir.mkdir(parents=True, exist_ok=True)
    for _dir in (logs_dir, outputs_dir, success_dir, failed_dir, prompt_dir, run_dir / "run_logs"):
        _dir.mkdir(parents=True, exist_ok=True)

    write_json(prompt_dir / "prompt_bundles.json", [asdict(b) for b in bundles])
    write_csv(prompt_dir / "prompt_bundles.csv", [asdict(b) for b in bundles])

    state = StrategyState(direction=args.score_direction)
    records: List[Dict[str, Any]] = []
    kaggle_json = Path(args.kaggle_json).expanduser().resolve() if args.kaggle_json else None

    print(f"[sweep] repo_dir={repo_dir}")
    print(f"[sweep] run_dir={run_dir}")
    print(f"[sweep] variants={queue}")
    print(f"[sweep] submit={args.submit} kaggle_competition={args.kaggle_competition}")
    print(f"[sweep] model_fallbacks={model_fallbacks}")

    round_idx = 0
    while queue:
        queue = reorder_queue(queue, state)
        variant = queue.pop(0)
        round_idx += 1
        round_slug = f"round_{round_idx:03d}_{slugify(variant)}"
        print(f"\n===== {round_slug} =====")
        print("[strategy]", state.decisions[-1] if state.decisions else {"mode": "initial_sweep"})

        model_attempts: List[Dict[str, Any]] = []
        successful_attempt: Optional[Dict[str, Any]] = None
        for model_idx, model_name in enumerate(model_fallbacks, start=1):
            model_slug = slugify(model_name, max_len=48)
            attempt_slug = f"{round_slug}__model_{model_idx:02d}_{model_slug}"
            output_csv = outputs_dir / f"{attempt_slug}.csv"
            run_log = run_dir / "run_logs" / f"{attempt_slug}.run_log.json"
            stdout_log = logs_dir / f"{attempt_slug}.stdout.log"
            cmd = build_run_command(args, variant, output_csv, run_log, state, model_override=model_name)
            print(f"[model-fallback] attempt {model_idx}/{len(model_fallbacks)} model={model_name}")
            print("[cmd]", " ".join(cmd))
            if args.dry_run:
                rc, seconds = 0, 0.0
                stdout_log.write_text("DRY RUN\n" + " ".join(cmd), encoding="utf-8")
            else:
                rc, seconds = run_streaming(cmd, cwd=repo_dir, log_path=stdout_log, timeout=args.run_timeout)
            run_payload = normalize_run_payload(read_json(run_log))
            solver_path = Path(run_payload.get("solver", "")) if run_payload.get("solver") else None
            csv_exists = output_csv.exists()
            model_failure, model_failure_markers = detect_model_failure(run_payload, stdout_log)
            attempt_submission_stats = csv_submission_stats(output_csv if csv_exists else None)
            # A CSV created by offline/baseline/sample-submission fallback is useful as an
            # artifact, but it is not proof that the selected model generated a valid solver.
            # Keep those attempts failed and continue to the next fallback model.
            attempt_ok, attempt_failure_reasons = classify_generated_attempt_success(
                rc=rc,
                csv_exists=csv_exists,
                solver_path=solver_path,
                model_failure=model_failure,
                model_failure_markers=model_failure_markers,
                submission_stats=attempt_submission_stats,
                run_payload=run_payload,
                stdout_log=stdout_log,
            )
            attempt_replay_report: Dict[str, Any] = {"skipped": True}
            if attempt_ok and args.competition == "cayleypy-christophers-jewel" and output_csv.exists():
                replay_log = logs_dir / f"{attempt_slug}.replay_validation.log"
                attempt_replay_report = validate_jewel_submission_replay(repo_dir, output_csv, replay_log)
                if not attempt_replay_report.get("ok"):
                    attempt_ok = False
                    attempt_failure_reasons.append("jewel_submission_replay_validation_failed")
            attempt = {
                "attempt_index": model_idx,
                "model": model_name,
                "ok": attempt_ok,
                "returncode": rc,
                "runtime_seconds": seconds,
                "cmd": cmd,
                "stdout_log": str(stdout_log),
                "run_log": str(run_log),
                "run_log_summary": run_payload,
                "output_csv": str(output_csv),
                "csv_exists": csv_exists,
                "model_failure_detected": model_failure,
                "model_failure_markers": ";".join(model_failure_markers),
                "attempt_failure_reasons": attempt_failure_reasons,
                "attempt_submission_stats": attempt_submission_stats,
                "replay_validation": attempt_replay_report,
                "solver_path": str(solver_path) if solver_path else None,
            }
            model_attempts.append(attempt)
            attempt_dest = success_dir if attempt_ok else failed_dir
            copy_if_exists(solver_path, attempt_dest, prefix=f"{attempt_slug}__")
            copy_if_exists(stdout_log, attempt_dest, prefix=f"{attempt_slug}__")
            copy_if_exists(run_log, attempt_dest, prefix=f"{attempt_slug}__")
            copy_if_exists(output_csv if output_csv.exists() else None, attempt_dest, prefix=f"{attempt_slug}__")
            if attempt_ok:
                successful_attempt = attempt
                print(f"[model-fallback] success with model={model_name}; remaining fallback models skipped for this prompt variant")
                break
            if model_failure:
                print(f"[model-fallback] model={model_name} produced only fallback/provider failure markers={model_failure_markers}; trying next fallback if available")
            else:
                print(f"[model-fallback] failed with model={model_name}; reasons={attempt_failure_reasons}; trying next fallback if available")

        chosen_attempt = successful_attempt or (model_attempts[-1] if model_attempts else {})
        output_csv = Path(chosen_attempt.get("output_csv", outputs_dir / f"{round_slug}.csv"))
        run_log = Path(chosen_attempt.get("run_log", run_dir / "run_logs" / f"{round_slug}.run_log.json"))
        stdout_log = Path(chosen_attempt.get("stdout_log", logs_dir / f"{round_slug}.stdout.log"))
        run_payload = normalize_run_payload(chosen_attempt.get("run_log_summary"))
        solver_path = Path(chosen_attempt.get("solver_path", "")) if chosen_attempt.get("solver_path") else None
        copied_solver = None
        for pattern in success_dir.glob(f"{round_slug}__model_*__*" if successful_attempt else "__never__"):
            if solver_path and pattern.name.endswith(solver_path.name):
                copied_solver = pattern
                break
        rc = int(chosen_attempt.get("returncode", 1) or 0)
        seconds = sum(float(a.get("runtime_seconds") or 0.0) for a in model_attempts)
        ok = successful_attempt is not None

        submission_stats = csv_submission_stats(output_csv if output_csv.exists() else None)
        kaggle_report: Dict[str, Any] = {"submitted": False, "skipped": True}
        kaggle_score: Dict[str, Any] = {"score": None, "status": "not_submitted"}
        submit_required_failure: Optional[str] = None
        if ok and args.submit:
            message = f"{args.message_prefix} | {round_slug} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            submit_log = logs_dir / f"{round_slug}.kaggle_submit.log"
            kaggle_report, submit_rc, submit_seconds = submit_to_kaggle(
                repo_dir=repo_dir,
                competition=args.kaggle_competition,
                submission_csv=output_csv,
                message=message,
                submit_via=args.submit_via,
                kaggle_json=kaggle_json,
                log_path=submit_log,
            )
            kaggle_report.update({"seconds": submit_seconds, "returncode": submit_rc})
            if kaggle_report.get("submitted"):
                kaggle_score = wait_for_kaggle_score(
                    repo_dir=repo_dir,
                    competition=args.kaggle_competition,
                    message=message,
                    file_name=output_csv.name,
                    kaggle_json=kaggle_json,
                    log_dir=logs_dir / f"{round_slug}_score_polls",
                    polls=args.score_polls,
                    sleep_seconds=args.score_poll_sleep,
                )
        if ok and args.submit and args.require_submit_success and not kaggle_report.get("submitted"):
            submit_required_failure = "kaggle_submit_required_but_not_confirmed"
            ok = False
            copy_if_exists(solver_path, failed_dir, prefix=f"{round_slug}__submit_failed__")
            copy_if_exists(stdout_log, failed_dir, prefix=f"{round_slug}__submit_failed__")
            copy_if_exists(run_log, failed_dir, prefix=f"{round_slug}__submit_failed__")
            copy_if_exists(output_csv if output_csv.exists() else None, failed_dir, prefix=f"{round_slug}__submit_failed__")
            print("[kaggle] submit was required but was not confirmed; classifying this round as failed", flush=True)
        score = kaggle_score.get("score") if isinstance(kaggle_score, dict) else None
        decision = state.update(round_idx=round_idx, variant=variant, ok=ok, score=parse_float(score))
        if state.breakthrough_pressure > 0:
            state.breakthrough_pressure -= 1

        record: Dict[str, Any] = {
            "round": round_idx,
            "variant": variant,
            "started_utc": utc_now(),
            "ok": ok,
            "returncode": rc,
            "runtime_seconds": seconds,
            "model": chosen_attempt.get("model"),
            "model_attempts": model_attempts,
            "cmd": chosen_attempt.get("cmd"),
            "stdout_log": str(stdout_log),
            "run_log": str(run_log),
            "run_log_summary": run_payload,
            "output_csv": str(output_csv),
            "submission_stats": submission_stats,
            "solver_path": str(solver_path) if solver_path else None,
            "copied_solver": str(copied_solver) if copied_solver else None,
            "kaggle": kaggle_report,
            "kaggle_score": kaggle_score,
            "submit_required_failure": submit_required_failure,
            "strategy_decision": decision,
            "remaining_queue_after_decision": reorder_queue(list(queue), state),
        }
        records.append(record)
        append_jsonl(run_dir / "run_index.jsonl", record)
        append_jsonl(run_dir / "strategy_history.jsonl", decision)
        write_json(run_dir / "strategy_state.json", asdict(state))
        write_json(run_dir / "latest_record.json", record)
        write_csv(run_dir / "per_round_metrics.csv", flatten_metrics(records))
        queue = reorder_queue(queue, state)

    summary = {
        "created_utc": utc_now(),
        "repo_dir": str(repo_dir),
        "run_dir": str(run_dir),
        "tested_variants": [r["variant"] for r in records],
        "model_fallbacks": model_fallbacks,
        "success_count": sum(1 for r in records if r.get("ok")),
        "failure_count": sum(1 for r in records if not r.get("ok")),
        "submit_required": bool(args.submit and args.require_submit_success),
        "submitted_success_count": sum(1 for r in records if isinstance(r.get("kaggle"), dict) and r["kaggle"].get("submitted")),
        "best_score": state.best_score,
        "best_variant": state.best_variant,
        "best_round": state.best_round,
        "score_direction": state.direction,
        "all_records_jsonl": str(run_dir / "run_index.jsonl"),
        "metrics_csv": str(run_dir / "per_round_metrics.csv"),
        "strategy_history_jsonl": str(run_dir / "strategy_history.jsonl"),
        "successful_scripts_dir": str(success_dir),
        "failed_scripts_dir": str(failed_dir),
    }
    analytics_report = generate_analytics(run_dir, records, state)
    summary["analytics"] = analytics_report
    write_json(run_dir / "summary.json", summary)
    output_zip = run_dir.with_suffix(".zip")
    package_results(run_dir, output_zip)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[sweep] packaged: {output_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
