#!/usr/bin/env python3
"""Run Megaminx self-improvement scenarios from a clean GitHub clone.

This runner intentionally does NOT apply any patch or delta archive.
It assumes all self-improvement changes are already committed to the target
repository/branch.

Workflow:
1) git clone https://github.com/visualcomments/agents_4_puzzles
2) install dependencies
3) validate that the improved scenario files are present in the clone
4) optionally probe configured LLM/g4f models
5) run baseline, basic strict self-improvement, and/or advanced failure-aware
   self-improvement scenarios
6) optionally write inline kaggle.json credentials to a private runtime config dir
7) optionally auto-submit the best successful improved submission to Kaggle
8) write a compact JSON summary while preserving full logs/artifacts

Designed for Google Colab and local Linux/macOS shells.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_REPO_URL = "https://github.com/visualcomments/agents_4_puzzles"
DEFAULT_COMPETITION = "cayley-py-megaminx"
BASIC_VARIANT = "strict_self_improvement"
ADVANCED_VARIANT = "failure_aware_self_improvement"

EXPECTED_IMPROVED_FILES = [
    "pipeline_cli.py",
    "AgentLaboratory/perm_pipeline/run_perm_pipeline.py",
    "competitions/cayley-py-megaminx/prompt_self_improver.py",
    "competitions/cayley-py-megaminx/failure_aware_self_improvement/row_profile_memory.py",
    "competitions/cayley-py-megaminx/self_improvement_scenarios.py",
]


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title: str) -> None:
    print("\n" + "=" * 88, flush=True)
    print(title, flush=True)
    print("=" * 88, flush=True)


def run_streaming(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    check: bool = True,
) -> int:
    """Run a command with stdout/stderr streamed to console and optional log file."""
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    if cwd:
        print(f"$ cd {cwd} && {printable}", flush=True)
    else:
        print(f"$ {printable}", flush=True)

    log_fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8", errors="replace")
        log_fh.write(f"$ {printable}\n")
        if cwd:
            log_fh.write(f"cwd={cwd}\n")
        log_fh.flush()

    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if log_fh:
            log_fh.write(line)
    rc = proc.wait()
    if log_fh:
        log_fh.write(f"\n[exit_code] {rc}\n")
        log_fh.close()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, list(cmd))
    return int(rc)


def clone_repo(repo_url: str, repo_dir: Path, branch: Optional[str], force_reclone: bool) -> None:
    if repo_dir.exists() and force_reclone:
        shutil.rmtree(repo_dir)
    if repo_dir.exists():
        print(f"[repo] using existing checkout: {repo_dir}", flush=True)
        return

    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [repo_url, str(repo_dir)]
    run_streaming(cmd)


def install_dependencies(repo_dir: Path, mode: str, log_dir: Path) -> None:
    if mode == "none":
        print("[install] skipped", flush=True)
        return

    print_section(f"Installing dependencies: mode={mode}")
    run_streaming([sys.executable, "-m", "pip", "install", "-U", "pip"], cwd=repo_dir, log_path=log_dir / "pip_upgrade.log")

    requirements: List[Path] = []
    if mode in {"min", "llm", "local", "full"}:
        requirements.append(repo_dir / "requirements-min.txt")
    if mode in {"llm", "local", "full"}:
        requirements.append(repo_dir / "AgentLaboratory" / "requirements-llm.txt")
    if mode in {"local", "full"}:
        requirements.append(repo_dir / "AgentLaboratory" / "requirements-local-llm.txt")
    if mode == "full":
        requirements.append(repo_dir / "requirements-full.txt")

    for req in requirements:
        if req.exists():
            run_streaming(
                [sys.executable, "-m", "pip", "install", "-r", str(req)],
                cwd=repo_dir,
                log_path=log_dir / f"pip_{req.name}.log",
            )
        else:
            print(f"[install] missing requirements file, skipped: {req}", flush=True)


def require_files(repo_dir: Path, rel_paths: Iterable[str], *, strict: bool) -> List[str]:
    missing = [rel for rel in rel_paths if not (repo_dir / rel).exists()]
    if missing and strict:
        details = "\n".join(f"  - {x}" for x in missing)
        raise FileNotFoundError(
            "The cloned repository/branch does not contain the expected improved-scenario files.\n"
            "No patch archive is applied by this runner. Push the changes to GitHub or pass --branch with the updated branch.\n"
            f"Missing files:\n{details}"
        )
    if missing:
        print("[validate] missing expected improved files, continuing because --skip-improved-file-check was set:", flush=True)
        for rel in missing:
            print(f"  - {rel}", flush=True)
    return missing


def validate_python_syntax(repo_dir: Path, log_dir: Path, *, skip_improved_file_check: bool) -> Dict[str, Any]:
    print_section("Validating repository files")
    missing = require_files(repo_dir, EXPECTED_IMPROVED_FILES, strict=not skip_improved_file_check)
    py_targets = [repo_dir / rel for rel in EXPECTED_IMPROVED_FILES if (repo_dir / rel).exists() and rel.endswith(".py")]
    if not py_targets:
        raise FileNotFoundError("No expected Python files were found to compile")
    run_streaming([sys.executable, "-m", "py_compile", *[str(p) for p in py_targets]], cwd=repo_dir, log_path=log_dir / "py_compile.log")
    return {"missing_expected_files": missing, "compiled_files": [str(p.relative_to(repo_dir)) for p in py_targets]}


def maybe_probe_models(repo_dir: Path, models: str, log_dir: Path, skip: bool) -> Dict[str, Any]:
    if skip:
        print("[preflight] model probe skipped", flush=True)
        return {"skipped": True}

    print_section("Provider/model preflight")
    log_file = log_dir / "model_preflight.log"
    cmd = [
        sys.executable,
        "pipeline_cli.py",
        "check-g4f-models",
        "--models",
        models,
        "--timeout",
        "20",
        "--json",
    ]
    rc = run_streaming(cmd, cwd=repo_dir, log_path=log_file, check=False)
    payload: Dict[str, Any] = {"skipped": False, "exit_code": rc, "log": str(log_file)}

    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            parsed = json.loads(text[first : last + 1])
            payload["json"] = parsed
            (log_dir / "model_preflight.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        payload["parse_error"] = repr(exc)
    return payload


def moves_len(value: str) -> int:
    value = str(value or "").strip()
    if not value:
        return 0
    if value.upper() == "UNSOLVED":
        return 10**9
    if "." in value:
        return len([x for x in value.split(".") if x])
    return len([x for x in value.replace(",", " ").split() if x])


def score_submission(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {"exists": False, "path": str(csv_path)}

    total = 0
    row_count = 0
    max_len = 0
    col_name = None
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for candidate in ("path", "moves", "solution"):
            if candidate in fields:
                col_name = candidate
                break
        if col_name is None:
            return {"exists": True, "path": str(csv_path), "error": f"no path/moves/solution column in {fields}"}
        for row in reader:
            n = moves_len(row.get(col_name, ""))
            total += n
            max_len = max(max_len, n)
            row_count += 1
    return {
        "exists": True,
        "path": str(csv_path),
        "column": col_name,
        "rows": row_count,
        "total_move_tokens": total,
        "max_row_tokens": max_len,
    }


def get_submission_score_value(result: Dict[str, Any]) -> Optional[int]:
    """Return integer total score/move count from a scenario result, when available."""
    score = result.get("score") or {}
    if not isinstance(score, dict) or not score.get("exists"):
        return None
    value = score.get("total_move_tokens")
    if isinstance(value, int) and value < 10**9:
        return value
    return None


def install_kaggle_cli(repo_dir: Path, log_dir: Path) -> None:
    """Install/update the official Kaggle CLI package for competition submissions."""
    print_section("Installing Kaggle CLI")
    run_streaming(
        [sys.executable, "-m", "pip", "install", "-U", "kaggle"],
        cwd=repo_dir,
        log_path=log_dir / "pip_kaggle.log",
    )


def setup_kaggle_credentials(
    *,
    kaggle_json_inline: Optional[str],
    kaggle_config_dir: Path,
) -> Dict[str, Any]:
    """Create KAGGLE_CONFIG_DIR/kaggle.json from an inline JSON string.

    The secret itself is intentionally never returned or printed.
    """
    raw = (kaggle_json_inline or "").strip()
    if not raw:
        raise ValueError(
            "Kaggle auto-submit is enabled, but no kaggle.json was provided. "
            "Set KAGGLE_JSON_INLINE or pass --kaggle-json-inline with the JSON contents."
        )

    try:
        payload = json.loads(raw)
        if isinstance(payload, str):
            payload = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("kaggle.json inline value is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("kaggle.json inline value must decode to a JSON object")
    username = str(payload.get("username") or "").strip()
    key = str(payload.get("key") or "").strip()
    if not username or not key:
        raise ValueError('kaggle.json must contain non-empty "username" and "key" fields')

    kaggle_config_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_config_dir / "kaggle.json"
    kaggle_json_path.write_text(
        json.dumps({"username": username, "key": key}, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        os.chmod(kaggle_config_dir, 0o700)
        os.chmod(kaggle_json_path, 0o600)
    except PermissionError:
        # Windows or locked-down filesystems may not support chmod fully.
        pass

    return {
        "configured": True,
        "config_dir": str(kaggle_config_dir),
        "kaggle_json_path": str(kaggle_json_path),
        "username": username,
        "key_present": True,
    }


def kaggle_command_prefix() -> List[str]:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    return [sys.executable, "-m", "kaggle"]


def select_kaggle_submission_candidates(
    results: Sequence[Dict[str, Any]],
    *,
    min_improvement: int,
    allow_without_baseline: bool,
    submit_all_successful: bool,
) -> Dict[str, Any]:
    """Pick improved non-baseline submissions to send to Kaggle.

    Lower total_move_tokens is better for this Megaminx solver. By default, a
    non-baseline scenario is eligible only when it beats the best baseline seen
    in this runner by at least min_improvement.
    """
    baseline_scores: List[int] = []
    for result in results:
        scenario = str(result.get("scenario") or "")
        score = get_submission_score_value(result)
        if score is not None and scenario.startswith("baseline"):
            baseline_scores.append(score)
    baseline_best = min(baseline_scores) if baseline_scores else None

    candidates: List[Dict[str, Any]] = []
    for result in results:
        scenario = str(result.get("scenario") or "")
        if scenario.startswith("baseline"):
            continue
        if int(result.get("exit_code") or 0) != 0:
            continue
        score = get_submission_score_value(result)
        output_csv = Path(str(result.get("output_csv") or ""))
        if score is None or not output_csv.exists():
            continue
        improvement = None if baseline_best is None else baseline_best - score
        eligible = False
        reason = ""
        if baseline_best is not None:
            eligible = improvement is not None and improvement >= min_improvement
            reason = (
                f"score={score}, baseline={baseline_best}, improvement={improvement}, "
                f"required>={min_improvement}"
            )
        elif allow_without_baseline:
            eligible = True
            reason = "no baseline score available; allowed by --submit-without-baseline-improvement"
        else:
            reason = "no baseline score available and baseline-improvement gate is enabled"
        if eligible:
            candidates.append(
                {
                    "scenario": scenario,
                    "score": score,
                    "baseline_score": baseline_best,
                    "improvement": improvement,
                    "output_csv": str(output_csv),
                    "selection_reason": reason,
                }
            )

    candidates.sort(key=lambda x: (int(x["score"]), str(x["scenario"])))
    selected = candidates if submit_all_successful else candidates[:1]
    return {
        "baseline_score": baseline_best,
        "eligible_candidates": candidates,
        "selected_candidates": selected,
        "min_improvement": min_improvement,
        "allow_without_baseline": allow_without_baseline,
        "submit_all_successful": submit_all_successful,
    }


def submit_to_kaggle(
    *,
    repo_dir: Path,
    log_dir: Path,
    env: Dict[str, str],
    competition: str,
    submission_csv: Path,
    message: str,
    label: str,
    dry_run: bool,
) -> Dict[str, Any]:
    """Submit one CSV to Kaggle using the official CLI."""
    log_path = log_dir / f"kaggle_submit_{label}.log"
    cmd = kaggle_command_prefix() + [
        "competitions",
        "submit",
        competition,
        "-f",
        str(submission_csv),
        "-m",
        message,
    ]
    if dry_run:
        printable = " ".join(shlex.quote(str(x)) for x in cmd)
        log_path.write_text(f"[dry-run] {printable}\n", encoding="utf-8")
        print(f"[kaggle dry-run] {printable}", flush=True)
        return {
            "submitted": False,
            "dry_run": True,
            "competition": competition,
            "submission_csv": str(submission_csv),
            "message": message,
            "log": str(log_path),
        }

    rc = run_streaming(cmd, cwd=repo_dir, env=env, log_path=log_path, check=False)
    return {
        "submitted": rc == 0,
        "dry_run": False,
        "exit_code": rc,
        "competition": competition,
        "submission_csv": str(submission_csv),
        "message": message,
        "log": str(log_path),
    }


def auto_submit_successful_solutions(
    *,
    repo_dir: Path,
    run_root: Path,
    log_dir: Path,
    results: Sequence[Dict[str, Any]],
    kaggle_json_inline: Optional[str],
    kaggle_config_dir_arg: Optional[str],
    kaggle_competition: str,
    kaggle_submit_message: Optional[str],
    min_improvement: int,
    allow_without_baseline: bool,
    submit_all_successful: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    """Install Kaggle CLI, configure inline credentials, and submit selected outputs."""
    print_section("Kaggle auto-submit gate")
    selection = select_kaggle_submission_candidates(
        results,
        min_improvement=min_improvement,
        allow_without_baseline=allow_without_baseline,
        submit_all_successful=submit_all_successful,
    )
    selected = list(selection.get("selected_candidates") or [])
    payload: Dict[str, Any] = {
        "enabled": True,
        "competition": kaggle_competition,
        "selection": selection,
        "submissions": [],
    }
    if not selected:
        payload["skipped_reason"] = "no eligible improved successful submission found"
        print("[kaggle] no eligible improved candidate; nothing to submit", flush=True)
        return payload

    install_kaggle_cli(repo_dir, log_dir)
    config_dir = Path(kaggle_config_dir_arg).expanduser().resolve() if kaggle_config_dir_arg else run_root / ".kaggle"
    cred_meta = setup_kaggle_credentials(
        kaggle_json_inline=kaggle_json_inline,
        kaggle_config_dir=config_dir,
    )
    payload["credentials"] = cred_meta

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(config_dir)

    for index, candidate in enumerate(selected, start=1):
        scenario = str(candidate["scenario"])
        score = int(candidate["score"])
        improvement = candidate.get("improvement")
        label = f"{index:02d}_{scenario}".replace("/", "_").replace(" ", "_")
        if kaggle_submit_message:
            message = kaggle_submit_message.format(
                scenario=scenario,
                score=score,
                improvement=improvement,
                timestamp=now_stamp(),
            )
        else:
            if improvement is None:
                message = f"agents_4_puzzles {scenario} score={score} {now_stamp()}"
            else:
                message = f"agents_4_puzzles {scenario} score={score} improvement={improvement} {now_stamp()}"
        payload["submissions"].append(
            submit_to_kaggle(
                repo_dir=repo_dir,
                log_dir=log_dir,
                env=env,
                competition=kaggle_competition,
                submission_csv=Path(str(candidate["output_csv"])),
                message=message,
                label=label,
                dry_run=dry_run,
            )
        )
    return payload


def run_pipeline_scenario(
    *,
    repo_dir: Path,
    run_root: Path,
    log_dir: Path,
    competition: str,
    scenario_name: str,
    prompt_variant: Optional[str],
    models: str,
    rounds: int,
    max_iters: int,
    max_rows: Optional[int],
    extra_args: Sequence[str],
    no_llm: bool = False,
) -> Dict[str, Any]:
    print_section(f"Running scenario: {scenario_name}")
    out_dir = run_root / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / f"{scenario_name}_submission.csv"
    run_log = out_dir / f"{scenario_name}_run_log.json"
    stdout_log = log_dir / f"{scenario_name}.stdout.log"

    cmd = [
        sys.executable,
        "pipeline_cli.py",
        "run",
        "--competition",
        competition,
        "--output",
        str(output_csv),
        "--run-log",
        str(run_log),
        "--no-progress",
        "--schema-check",
    ]

    if no_llm:
        cmd.append("--no-llm")
    else:
        cmd += [
            "--models",
            models,
            "--keep-improving",
            "--self-improve-prompts",
            "--reject-identical-candidates",
            "--write-per-row-delta",
            "--improvement-rounds",
            str(rounds),
            "--max-iters",
            str(max_iters),
            "--g4f-stop-at-python-fence",
        ]
        if prompt_variant:
            cmd += ["--prompt-variant", prompt_variant]

    if max_rows is not None and max_rows > 0:
        cmd += ["--max-rows", str(max_rows)]

    cmd += list(extra_args)

    rc = run_streaming(cmd, cwd=repo_dir, log_path=stdout_log, check=False)
    result: Dict[str, Any] = {
        "scenario": scenario_name,
        "exit_code": rc,
        "output_csv": str(output_csv),
        "run_log": str(run_log),
        "stdout_log": str(stdout_log),
        "score": score_submission(output_csv),
    }
    if run_log.exists():
        try:
            result["run_log_json"] = json.loads(run_log.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            result["run_log_parse_error"] = repr(exc)
    return result


def write_summary(summary_path: Path, summary: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print_section("Run summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"\n[summary] written: {summary_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clone agents_4_puzzles and run basic/advanced Megaminx self-improvement scenarios without applying patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    p.add_argument("--branch", default=None, help="Optional git branch/tag/commit-ish to clone. Use this if improvements are not on main.")
    p.add_argument("--workdir", default="/content/agents_4_puzzles_runs" if Path("/content").exists() else "./agents_4_puzzles_runs")
    p.add_argument("--repo-dir", default=None, help="Optional checkout directory; defaults to <workdir>/agents_4_puzzles")
    p.add_argument("--force-reclone", action="store_true")
    p.add_argument("--install-mode", choices=["none", "min", "llm", "local", "full"], default="llm")
    p.add_argument("--skip-model-preflight", action="store_true")
    p.add_argument("--skip-improved-file-check", action="store_true", help="Continue even if scenario marker files are missing")
    p.add_argument("--scenario", choices=["baseline", "basic", "advanced", "both", "all"], default="both")
    p.add_argument("--competition", default=DEFAULT_COMPETITION)
    p.add_argument("--models", default=os.environ.get("AGENTS_4_PUZZLES_MODELS", "gpt-4o-mini"))
    p.add_argument("--rounds-basic", type=int, default=3)
    p.add_argument("--rounds-advanced", type=int, default=6)
    p.add_argument("--max-iters", type=int, default=100000)
    p.add_argument("--max-rows", type=int, default=None, help="Use for smoke tests; omit for full dataset")
    p.add_argument("--extra-basic-arg", action="append", default=[], help="Extra argument for the basic run; repeat as needed")
    p.add_argument("--extra-advanced-arg", action="append", default=[], help="Extra argument for the advanced run; repeat as needed")
    p.add_argument("--auto-submit-kaggle", action="store_true", help="Submit the best successful improved submission to Kaggle after the run")
    p.add_argument("--kaggle-json-inline", default=os.environ.get("KAGGLE_JSON_INLINE"), help="Inline contents of legacy kaggle.json; prefer the KAGGLE_JSON_INLINE env var over shell history")
    p.add_argument("--kaggle-config-dir", default=None, help="Directory where kaggle.json will be written; defaults to <run_root>/.kaggle")
    p.add_argument("--kaggle-competition", default=os.environ.get("KAGGLE_COMPETITION", DEFAULT_COMPETITION), help="Kaggle competition slug for auto-submit")
    p.add_argument("--kaggle-submit-message", default=None, help="Optional submission message template; may use {scenario}, {score}, {improvement}, {timestamp}")
    p.add_argument("--kaggle-submit-dry-run", action="store_true", help="Exercise selection and credential setup but do not call kaggle competitions submit")
    p.add_argument("--submit-min-improvement", type=int, default=1, help="Minimum move-token improvement over baseline required for Kaggle auto-submit")
    p.add_argument("--submit-without-baseline-improvement", action="store_true", help="Allow Kaggle submit when no baseline score was measured in this runner")
    p.add_argument("--submit-all-successful", action="store_true", help="Submit every eligible improved scenario; default submits only the best one")
    p.add_argument("--allow-failures", action="store_true", help="Do not return non-zero if one scenario fails")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(args.repo_dir).expanduser().resolve() if args.repo_dir else workdir / "agents_4_puzzles"
    run_root = workdir / f"self_improvement_run_{now_stamp()}"
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "runner": "no_patch",
        "repo_url": args.repo_url,
        "branch": args.branch,
        "repo_dir": str(repo_dir),
        "workdir": str(workdir),
        "run_root": str(run_root),
        "competition": args.competition,
        "scenario": args.scenario,
        "models": args.models,
        "rounds_basic": args.rounds_basic,
        "rounds_advanced": args.rounds_advanced,
        "max_iters": args.max_iters,
        "max_rows": args.max_rows,
        "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "patch_applied": False,
        "kaggle_auto_submit_enabled": bool(args.auto_submit_kaggle),
        "kaggle_competition": args.kaggle_competition,
        "kaggle_submit_dry_run": bool(args.kaggle_submit_dry_run),
        "submit_min_improvement": args.submit_min_improvement,
        "submit_without_baseline_improvement": bool(args.submit_without_baseline_improvement),
        "submit_all_successful": bool(args.submit_all_successful),
    }

    try:
        print_section("Clone repository")
        clone_repo(args.repo_url, repo_dir, args.branch, args.force_reclone)

        install_dependencies(repo_dir, args.install_mode, log_dir)
        summary["validation"] = validate_python_syntax(
            repo_dir,
            log_dir,
            skip_improved_file_check=args.skip_improved_file_check,
        )
        summary["model_preflight"] = maybe_probe_models(repo_dir, args.models, log_dir, args.skip_model_preflight)

        results: List[Dict[str, Any]] = []
        if args.scenario in {"baseline", "all"}:
            results.append(
                run_pipeline_scenario(
                    repo_dir=repo_dir,
                    run_root=run_root,
                    log_dir=log_dir,
                    competition=args.competition,
                    scenario_name="baseline",
                    prompt_variant=None,
                    models=args.models,
                    rounds=0,
                    max_iters=args.max_iters,
                    max_rows=args.max_rows,
                    extra_args=[],
                    no_llm=True,
                )
            )

        if args.scenario in {"basic", "both", "all"}:
            results.append(
                run_pipeline_scenario(
                    repo_dir=repo_dir,
                    run_root=run_root,
                    log_dir=log_dir,
                    competition=args.competition,
                    scenario_name="basic_strict_self_improvement",
                    prompt_variant=BASIC_VARIANT,
                    models=args.models,
                    rounds=args.rounds_basic,
                    max_iters=args.max_iters,
                    max_rows=args.max_rows,
                    extra_args=args.extra_basic_arg,
                    no_llm=False,
                )
            )

        if args.scenario in {"advanced", "both", "all"}:
            advanced_extra = [
                "--search-mode",
                "hybrid",
                "--plan-beam-width",
                "4",
                "--frontier-width",
                "8",
                "--archive-size",
                "12",
                "--refine-rounds",
                "2",
            ] + list(args.extra_advanced_arg)
            results.append(
                run_pipeline_scenario(
                    repo_dir=repo_dir,
                    run_root=run_root,
                    log_dir=log_dir,
                    competition=args.competition,
                    scenario_name="advanced_failure_aware_self_improvement",
                    prompt_variant=ADVANCED_VARIANT,
                    models=args.models,
                    rounds=args.rounds_advanced,
                    max_iters=args.max_iters,
                    max_rows=args.max_rows,
                    extra_args=advanced_extra,
                    no_llm=False,
                )
            )

        if args.auto_submit_kaggle and not args.submit_without_baseline_improvement:
            has_baseline = any(str(r.get("scenario") or "").startswith("baseline") for r in results)
            if not has_baseline:
                print_section("Running baseline for Kaggle submit gate")
                results.append(
                    run_pipeline_scenario(
                        repo_dir=repo_dir,
                        run_root=run_root,
                        log_dir=log_dir,
                        competition=args.competition,
                        scenario_name="baseline_for_submit",
                        prompt_variant=None,
                        models=args.models,
                        rounds=0,
                        max_iters=args.max_iters,
                        max_rows=args.max_rows,
                        extra_args=[],
                        no_llm=True,
                    )
                )

        if args.auto_submit_kaggle:
            summary["kaggle_auto_submit"] = auto_submit_successful_solutions(
                repo_dir=repo_dir,
                run_root=run_root,
                log_dir=log_dir,
                results=results,
                kaggle_json_inline=args.kaggle_json_inline,
                kaggle_config_dir_arg=args.kaggle_config_dir,
                kaggle_competition=args.kaggle_competition,
                kaggle_submit_message=args.kaggle_submit_message,
                min_improvement=args.submit_min_improvement,
                allow_without_baseline=args.submit_without_baseline_improvement,
                submit_all_successful=args.submit_all_successful,
                dry_run=args.kaggle_submit_dry_run,
            )
        else:
            summary["kaggle_auto_submit"] = {"enabled": False}

        summary["results"] = results
        summary["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        summary_path = run_root / "self_improvement_summary.json"
        write_summary(summary_path, summary)

        failed = [r for r in results if r.get("exit_code") != 0]
        if failed and not args.allow_failures:
            print(f"[done] {len(failed)} scenario(s) failed; see logs under {log_dir}", file=sys.stderr, flush=True)
            return 2
        print(f"[done] artifacts are under: {run_root}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        summary["fatal_error"] = repr(exc)
        summary["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        write_summary(run_root / "self_improvement_summary.json", summary)
        print(f"[fatal] {exc}", file=sys.stderr, flush=True)
        if args.allow_failures:
            return 0
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
