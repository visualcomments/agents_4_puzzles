#!/usr/bin/env python3
"""Strict Megaminx prompt sweep runner.

This wrapper auto-installs https://github.com/visualcomments/agents_4_puzzles
when it is launched outside a local repo checkout, then runs pipeline_cli.py with
the self-improvement guards added in this archive:
- provider/Kaggle preflight before submit runs;
- no fallback/sample/offline baseline counted as success;
- no promotion for identical solver/submission fingerprints;
- per-row delta artifacts for every locally scored improvement round;
- Kaggle submit required to succeed when --submit is used.

Example from an empty directory:
    python megaminx_guarded_sweep.py \
      --install-dir ./agents_4_puzzles \

Example inside an existing repo:
    python megaminx_guarded_sweep.py \
      --competition cayley-py-megaminx \
      --variants strict_self_improvement,score_guarded \
      --models g4f:gpt-4o-mini \
      --output-root runs/guarded_sweep \
      --no-submit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
DEFAULT_REPO_URL = "https://github.com/visualcomments/agents_4_puzzles.git"
DEFAULT_INSTALL_DIR = "agents_4_puzzles"


FALLBACK_MARKERS = (
    "sample_submission fallback",
    "sample submission fallback",
    "offline fallback",
    "offline-baseline",
    "known-good baseline",
    "known-good offline baseline",
    "using fallback baseline",
    "generation failed; using baseline",
    "credentials required",
    "provider is not working",
    "no working providers",
    "all providers failed",
    "timed out",
    "timeout",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return out or "item"


def read_text(path: Optional[Path], max_chars: int = 300000) -> str:
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:] if len(text) > max_chars else text


def normalize_run_report(raw: Any) -> dict[str, Any]:
    """Normalize pipeline_cli run-log payloads to a single dict.

    pipeline_cli.py may write run_log.json either as a single JSON object or
    as a JSON list of event/report objects. The guarded sweep runner expects
    dict-like access, so list payloads are folded into one report while keeping
    solver/output paths, stages, and Kaggle submit diagnostics.
    """
    if isinstance(raw, dict):
        return raw

    if isinstance(raw, list):
        dicts = [x for x in raw if isinstance(x, dict)]
        if not dicts:
            return {
                "status": "error",
                "error": {
                    "type": "RunLogShapeError",
                    "message": "run_log.json was a list but contained no dict events",
                },
                "_run_log_shape": "list",
                "_run_log_event_count": len(raw),
            }

        primary = next(
            (
                d for d in dicts
                if any(k in d for k in ("solver", "output_csv", "stages", "files", "kaggle_submit"))
            ),
            dicts[-1],
        )
        out = dict(primary)

        for event in dicts:
            for key, value in event.items():
                if value in (None, "", [], {}):
                    continue
                if key not in out or out.get(key) in (None, "", [], {}):
                    out[key] = value
                elif isinstance(out.get(key), dict) and isinstance(value, dict):
                    merged = dict(out[key])
                    merged.update(value)
                    out[key] = merged

        out["_run_log_shape"] = "list"
        out["_run_log_event_count"] = len(raw)
        return out

    return {
        "status": "error",
        "error": {
            "type": "RunLogShapeError",
            "message": f"run_log.json had unsupported top-level type: {type(raw).__name__}",
        },
        "_run_log_shape": type(raw).__name__,
    }


def load_run_report(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return normalize_run_report(raw)
    except Exception as exc:
        return {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}


def report_path(report: dict[str, Any], key: str) -> Optional[Path]:
    value = report.get(key)
    if not value:
        files = report.get("files")
        if isinstance(files, dict):
            item = files.get(key)
            if isinstance(item, dict):
                value = item.get("path")
            elif isinstance(item, str):
                value = item
    if not value:
        return None
    try:
        return Path(str(value))
    except Exception:
        return None


def sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_moves(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text.upper() == "UNSOLVED":
        return []
    if "." in text:
        return [p for p in text.split(".") if p]
    return [p for p in text.split() if p]


def csv_stats(path: Optional[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "exists": bool(path and path.exists()),
        "row_count": 0,
        "empty_rows": 0,
        "unsolved_rows": 0,
        "total_move_tokens": 0,
        "mean_move_tokens": None,
        "max_move_tokens": 0,
        "digest": None,
    }
    if path is None or not path.exists():
        return out
    try:
        h = hashlib.sha256()
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                raise ValueError("missing CSV header")
            id_col = fieldnames[0]
            move_col = next((c for c in ("path", "moves", "solution") if c in fieldnames), None)
            if move_col is None:
                move_col = next((c for c in fieldnames if c.strip().lower() not in {"id", "initial_state_id"}), None)
            if move_col is None:
                raise ValueError("could not infer moves column")
            lens: list[int] = []
            for row in reader:
                rid = str(row.get(id_col, "")).strip()
                path_value = str(row.get(move_col, "")).strip()
                h.update(rid.encode("utf-8")); h.update(b"\x1f"); h.update(path_value.encode("utf-8")); h.update(b"\n")
                out["row_count"] += 1
                if not path_value:
                    out["empty_rows"] += 1
                if path_value.upper() == "UNSOLVED":
                    out["unsolved_rows"] += 1
                n = len(split_moves(path_value))
                lens.append(n)
                out["total_move_tokens"] += n
            out["digest"] = h.hexdigest()
            if lens:
                out["mean_move_tokens"] = out["total_move_tokens"] / len(lens)
                out["max_move_tokens"] = max(lens)
            out["id_column"] = id_col
            out["move_column"] = move_col
    except Exception as exc:
        out["csv_error"] = f"{type(exc).__name__}: {exc}"
    return out


def flatten_improvement(report: dict[str, Any]) -> dict[str, Any]:
    improvement = (((report.get("stages") or {}).get("generate_solver") or {}).get("improvement") or {})
    history = improvement.get("history") if isinstance(improvement, dict) else []
    if not isinstance(history, list):
        history = []
    return {
        "best_score": improvement.get("best_score"),
        "best_round": improvement.get("best_round"),
        "best_solver_sha256": improvement.get("best_solver_sha256"),
        "best_submission_digest": improvement.get("best_submission_digest"),
        "history": history,
        "accepted_rounds": [h.get("round") for h in history if h.get("accepted") is True],
        "rejected_rounds": [h.get("round") for h in history if h.get("accepted") is not True],
        "failure_kinds": [h.get("failure_kind") for h in history if h.get("failure_kind")],
    }


def classify_attempt(*, rc: int, stdout: str, run_report: dict[str, Any], submission: Optional[Path], solver: Optional[Path], require_submit: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if rc != 0:
        reasons.append(f"nonzero_returncode:{rc}")
    status = str(run_report.get("status") or "").lower()
    if status and status != "ok":
        reasons.append(f"run_status:{status}")
    text = (stdout + "\n" + json.dumps(run_report, ensure_ascii=False, default=str)).lower()
    markers = [m for m in FALLBACK_MARKERS if m in text]
    if markers:
        reasons.append("fallback_or_provider_markers:" + ";".join(sorted(set(markers))))
    if solver is None or not solver.exists():
        reasons.append("missing_solver")
    stats = csv_stats(submission)
    if not stats.get("exists"):
        reasons.append("missing_submission")
    if stats.get("csv_error"):
        reasons.append("submission_csv_error:" + str(stats.get("csv_error")))
    if int(stats.get("row_count") or 0) <= 0:
        reasons.append("empty_submission")
    if int(stats.get("empty_rows") or 0) > 0:
        reasons.append(f"empty_rows:{stats.get('empty_rows')}")
    if int(stats.get("unsolved_rows") or 0) > 0:
        reasons.append(f"unsolved_rows:{stats.get('unsolved_rows')}")
    imp = flatten_improvement(run_report)
    if imp["history"] and not imp["accepted_rounds"]:
        reasons.append("no_accepted_improvement_round")
    if any(k in {"no_novelty_identical_solver", "no_novelty_identical_submission", "no_per_row_improvement"} for k in imp["failure_kinds"]):
        reasons.append("no_novelty_or_no_per_row_improvement")
    submit_report = run_report.get("kaggle_submit") if isinstance(run_report.get("kaggle_submit"), dict) else {}
    if require_submit and not bool(submit_report.get("submitted") or submit_report.get("status")):
        reasons.append("kaggle_submit_required_but_not_confirmed")
    return len(reasons) == 0, reasons


def has_python_module(module_name: str) -> bool:
    """Return True when module_name can be imported by the current Python."""
    try:
        probe = subprocess.run(
            [PYTHON, "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
        return probe.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[setup] warning: import probe timed out for module {module_name}")
        return False


def run_setup_cmd(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    print("[setup] " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"setup command failed with rc={proc.returncode}: {' '.join(cmd)}")


def looks_like_repo_root(path: Path) -> bool:
    return (path / "pipeline_cli.py").exists() and (path / "competitions").exists()


def git_clone_or_update(*, repo_url: str, install_dir: Path, branch: str, force_reinstall: bool, update_repo: bool) -> Path:
    install_dir = install_dir.expanduser().resolve()
    if force_reinstall and install_dir.exists():
        print(f"[setup] removing existing install-dir: {install_dir}")
        shutil.rmtree(install_dir)
    if not install_dir.exists():
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([repo_url, str(install_dir)])
        run_setup_cmd(cmd)
    elif looks_like_repo_root(install_dir):
        if update_repo and (install_dir / ".git").exists():
            run_setup_cmd(["git", "pull", "--ff-only"], cwd=install_dir)
    else:
        raise SystemExit(
            f"install-dir exists but is not an agents_4_puzzles repo: {install_dir}. "
            "Pass --force-reinstall or choose another --install-dir."
        )
    if not looks_like_repo_root(install_dir):
        raise SystemExit(f"cloned directory does not look like agents_4_puzzles repo: {install_dir}")
    return install_dir


def maybe_install_dependencies(args: argparse.Namespace, repo: Path) -> None:
    mode = str(getattr(args, "install_deps", "auto") or "auto")
    if mode == "none":
        return
    if mode == "full":
        req = repo / "requirements-full.txt"
        if req.exists():
            run_setup_cmd([PYTHON, "-m", "pip", "install", "-r", str(req)], cwd=repo)
        else:
            run_setup_cmd([PYTHON, "-m", "pip", "install", "kaggle", "g4f"], cwd=repo)
        return
    # auto mode: install only packages that this run likely needs and that are missing.
    packages: list[str] = []
    models = parse_csv_list(str(getattr(args, "models", "")))
    wants_g4f = any(m.lower().startswith("g4f:") for m in models)
    wants_kaggle = bool(getattr(args, "submit", False) or getattr(args, "kaggle_json", ""))
    if wants_kaggle and not has_python_module("kaggle"):
        packages.append("kaggle")
    if wants_g4f and not has_python_module("g4f") and not (repo / "gpt4free").exists():
        packages.append("g4f")
    if packages:
        run_setup_cmd([PYTHON, "-m", "pip", "install", *packages], cwd=repo)


def copy_local_guarded_runner(repo: Path) -> None:
    """Copy this runner into the cloned repo for reproducibility, if needed."""
    src = Path(__file__).resolve()
    dst = repo / src.name
    try:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            print(f"[setup] copied runner to {dst}")
    except Exception as exc:
        print(f"[setup] warning: could not copy runner into repo: {exc}")


def ensure_repository(args: argparse.Namespace) -> Path:
    """Resolve or clone the agents_4_puzzles repo before running the sweep."""
    explicit_repo = str(getattr(args, "repo_dir", "") or "").strip()
    if explicit_repo:
        repo = Path(explicit_repo).expanduser().resolve()
        if not looks_like_repo_root(repo):
            raise SystemExit(f"--repo-dir does not look like agents_4_puzzles repo: {repo}")
        maybe_install_dependencies(args, repo)
        return repo

    # If the script is already inside a repo checkout, prefer that checkout.
    if looks_like_repo_root(REPO_ROOT):
        repo = REPO_ROOT.resolve()
        maybe_install_dependencies(args, repo)
        return repo

    repo = git_clone_or_update(
        repo_url=str(getattr(args, "repo_url", DEFAULT_REPO_URL) or DEFAULT_REPO_URL),
        install_dir=Path(str(getattr(args, "install_dir", DEFAULT_INSTALL_DIR) or DEFAULT_INSTALL_DIR)),
        branch=str(getattr(args, "branch", "") or ""),
        force_reinstall=bool(getattr(args, "force_reinstall", False)),
        update_repo=bool(getattr(args, "update_repo", True)),
    )
    maybe_install_dependencies(args, repo)
    copy_local_guarded_runner(repo)
    return repo


def run_cmd(cmd: list[str], *, cwd: Path, log_path: Path, timeout: Optional[int]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        start = time.time()
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            if timeout and time.time() - start > timeout:
                proc.kill()
                log.write(f"\n[guarded-sweep] killed after timeout={timeout}s\n")
                return 124
        return proc.wait()


def parse_csv_list(value: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;\n]+", value or "") if p.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run strict guarded Megaminx prompt sweep")
    p.add_argument("--repo-dir", default="", help="Existing agents_4_puzzles repo. If omitted, use current repo or auto-clone --repo-url into --install-dir.")
    p.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Git URL to clone when no local repo is found.")
    p.add_argument("--install-dir", default=DEFAULT_INSTALL_DIR, help="Where to clone the repo when --repo-dir is omitted and the script is not inside a repo.")
    p.add_argument("--branch", default="", help="Optional branch/tag to clone.")
    p.add_argument("--force-reinstall", action="store_true", help="Delete --install-dir and clone it again.")
    p.add_argument("--update-repo", action=argparse.BooleanOptionalAction, default=True, help="Run git pull --ff-only for an existing git checkout in --install-dir.")
    p.add_argument("--install-deps", choices=["auto", "none", "full"], default="auto", help="Dependency install mode. auto installs only missing kaggle/g4f when needed.")
    p.add_argument("--competition", default="cayley-py-megaminx")
    p.add_argument("--variants", default="strict_self_improvement,score_guarded,hard_row_routed,exact_score_population")
    p.add_argument("--models", default="g4f:gpt-4o-mini")
    p.add_argument("--output-root", default="runs/guarded_prompt_sweep")
    p.add_argument("--run-name", default="megaminx_guarded_sweep")
    p.add_argument("--improvement-rounds", type=int, default=3)
    p.add_argument("--max-iters", type=int, default=8000)
    p.add_argument("--max-runs", type=int, default=0, help="0 means all variant/model combinations")
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--submit", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--submit-via", choices=["auto", "api", "cli"], default="cli")
    p.add_argument("--message-prefix", default="megaminx guarded self-improvement")
    p.add_argument("--kaggle-json", default="")
    p.add_argument("--puzzles", default="")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--no-llm", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo = ensure_repository(args)
    run_dir = (repo / args.output_root / args.run_name).resolve()
    logs = run_dir / "logs"
    subs = run_dir / "submissions"
    success_dir = run_dir / "successful_scripts"
    failed_dir = run_dir / "failed_scripts"
    for d in (logs, subs, success_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    variants = parse_csv_list(args.variants)
    models = parse_csv_list(args.models)
    records: list[dict[str, Any]] = []

    if args.submit:
        preflight_cmd = [PYTHON, "pipeline_cli.py", "kaggle-preflight", "--competition", args.competition, "--submit-via", args.submit_via]
        if args.kaggle_json:
            preflight_cmd.extend(["--kaggle-json", args.kaggle_json])
        rc = run_cmd(preflight_cmd, cwd=repo, log_path=logs / "kaggle_preflight.log", timeout=args.timeout or None)
        if rc != 0:
            raise SystemExit("Kaggle preflight failed; not starting guarded submit sweep")

    idx = 0
    for variant in variants:
        for model in models:
            idx += 1
            if args.max_runs and idx > args.max_runs:
                break
            slug = f"round_{idx:03d}_{slugify(variant)}__{slugify(model)}"
            output_csv = subs / f"{slug}.csv"
            run_log = logs / f"{slug}.run_log.json"
            stdout_log = logs / f"{slug}.stdout.log"
            cmd = [
                PYTHON, "pipeline_cli.py", "run",
                "--competition", args.competition,
                "--prompt-variant", variant,
                "--models", model,
                "--output", str(output_csv),
                "--run-log", str(run_log),
                "--keep-improving",
                "--self-improve-prompts",
                "--reject-identical-candidates",
                "--write-per-row-delta",
                "--improvement-rounds", str(args.improvement_rounds),
                "--max-iters", str(args.max_iters),
                "--require-submit-success",
                "--submit-via", args.submit_via,
                "--message", f"{args.message_prefix} | {slug} | {utc_now()}",
                "--no-progress",
            ]
            if args.no_llm:
                cmd.append("--no-llm")
            if args.puzzles:
                cmd.extend(["--puzzles", args.puzzles])
            if args.max_rows:
                cmd.extend(["--max-rows", str(args.max_rows)])
            if args.submit:
                cmd.append("--submit")
            if args.kaggle_json:
                cmd.extend(["--kaggle-json", args.kaggle_json])
            started = time.time()
            rc = run_cmd(cmd, cwd=repo, log_path=stdout_log, timeout=args.timeout or None)
            seconds = time.time() - started
            report: dict[str, Any] = {}
            if run_log.exists():
                report = load_run_report(run_log)
            solver_path = report_path(report, "solver")
            ok, reasons = classify_attempt(
                rc=rc,
                stdout=read_text(stdout_log),
                run_report=report,
                submission=output_csv,
                solver=solver_path,
                require_submit=bool(args.submit),
            )
            dest = success_dir if ok else failed_dir
            for path in (solver_path, output_csv, run_log, stdout_log):
                if path and path.exists():
                    shutil.copy2(path, dest / f"{slug}__{path.name}")
            record = {
                "round": idx,
                "variant": variant,
                "model": model,
                "ok": ok,
                "failure_reasons": reasons,
                "returncode": rc,
                "seconds": seconds,
                "output_csv": str(output_csv),
                "solver": str(solver_path) if solver_path else None,
                "solver_sha256": sha256_file(solver_path),
                "submission_stats": csv_stats(output_csv),
                "improvement": flatten_improvement(report),
                "run_log": str(run_log),
                "stdout_log": str(stdout_log),
            }
            records.append(record)
            with (run_dir / "run_index.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[guarded-sweep] {slug}: ok={ok} reasons={reasons}")
        if args.max_runs and idx >= args.max_runs:
            break

    summary = {
        "created_utc": utc_now(),
        "repo_dir": str(repo),
        "run_dir": str(run_dir),
        "success_count": sum(1 for r in records if r.get("ok")),
        "failure_count": sum(1 for r in records if not r.get("ok")),
        "records": records,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = run_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(run_dir), "zip", root_dir=str(run_dir))
    print(json.dumps({"summary": str(run_dir / "summary.json"), "zip": str(zip_path)}, ensure_ascii=False, indent=2))
    return 0 if summary["success_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
