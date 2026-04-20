#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_VARIANTS = [
    "regular",
    "improved",
    "dataset_adapted",
    "structured",
    "heuristic_boosted",
    "master_hybrid",
    "score_guarded",
    "algorithmic_population",
    "portfolio_orchestrated",
    "hard_row_routed",
    "exact_score_population",
]

# The repository-wide Megaminx prompt self-improver is competition-specific and can
# synthesize stronger round-specific bundles. We keep `regular` excluded because the
# repo explicitly documents it as true from-scratch mode with no baseline grounding.
AUTO_SELF_IMPROVE_VARIANTS = {
    "improved",
    "dataset_adapted",
    "structured",
    "heuristic_boosted",
    "master_hybrid",
    "score_guarded",
    "algorithmic_population",
    "portfolio_orchestrated",
    "hard_row_routed",
    "exact_score_population",
}

DEFAULT_SEARCH_SETTINGS = {
    "search_mode": "hybrid",
    "plan_beam_width": 16,
    "frontier_width": 32,
    "archive_size": 48,
    "refine_rounds": 100,
    "max_iters": 100000,
    "g4f_request_timeout": 120.0,
    "print_generation_max_chars": 16000,
}


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "item"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def shlex_join(cmd: list[str]) -> str:
    try:
        return subprocess.list2cmdline(cmd) if os.name == "nt" else __import__("shlex").join(cmd)
    except Exception:
        return " ".join(cmd)


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    ensure_parent(stdout_path)
    ensure_parent(stderr_path)
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        stdout_f.write(f"$ {shlex_join(cmd)}\n\n")
        stdout_f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            check=False,
        )
    end = time.time()
    return {
        "cmd": cmd,
        "cmd_text": shlex_join(cmd),
        "returncode": proc.returncode,
        "started_at": datetime.fromtimestamp(start).isoformat(),
        "finished_at": datetime.fromtimestamp(end).isoformat(),
        "elapsed_seconds": round(end - start, 3),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def copy_file_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists() or not src.is_file():
        return False
    ensure_parent(dst)
    shutil.copy2(src, dst)
    return True


def copy_tree_any(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        ensure_parent(dst)
        shutil.copy2(src, dst)


def zip_directory(root_dir: Path, archive_path: Path) -> Path:
    ensure_parent(archive_path)
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(root_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root_dir.parent))
    return archive_path


def parse_variants(raw: str | None) -> list[str]:
    if not raw or raw.strip().lower() == "all":
        return list(DEFAULT_PROMPT_VARIANTS)
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in DEFAULT_PROMPT_VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown prompt variants: {unknown}. Allowed: {DEFAULT_PROMPT_VARIANTS}")
    return variants


def resolve_prompt_bundle(repo_root: Path, variant: str) -> dict[str, str | None]:
    prompts_dir = repo_root / "competitions" / "cayley-py-megaminx" / "prompts"
    prompt_file = prompts_dir / f"user_prompt_{variant}.txt"
    if not prompt_file.exists():
        fallback = prompts_dir / "user_prompt.txt"
        prompt_file = fallback if fallback.exists() else prompt_file

    custom_prompts = prompts_dir / f"custom_prompts_{variant}.json"
    if not custom_prompts.exists():
        fallback = prompts_dir / "custom_prompts_template.json"
        custom_prompts = fallback if fallback.exists() else custom_prompts

    return {
        "prompt_file": str(prompt_file) if prompt_file.exists() else None,
        "custom_prompts": str(custom_prompts) if custom_prompts.exists() else None,
    }


def maybe_last_run_record(run_log_path: Path) -> dict[str, Any] | None:
    if not run_log_path.exists():
        return None
    payload = read_json(run_log_path)
    if isinstance(payload, list) and payload:
        last = payload[-1]
        return last if isinstance(last, dict) else None
    if isinstance(payload, dict):
        return payload
    return None


def extract_selected_solver(run_record: dict[str, Any] | None) -> Path | None:
    if not isinstance(run_record, dict):
        return None
    raw = run_record.get("solver")
    if isinstance(raw, str) and raw.strip():
        return Path(raw)
    return None


def collect_generated_artifacts(repo_root: Path, run_record: dict[str, Any] | None, dest_dir: Path) -> list[str]:
    collected: list[str] = []
    if not isinstance(run_record, dict):
        return collected

    solver_path = extract_selected_solver(run_record)
    if solver_path is None:
        return collected

    improvement = (
        run_record.get("stages", {})
        .get("generate_solver", {})
        .get("improvement")
        if isinstance(run_record.get("stages"), dict)
        else None
    )

    candidates: list[Path] = []
    if solver_path.exists():
        candidates.append(solver_path)
        for sibling in sorted(solver_path.parent.glob(f"{solver_path.stem}*")):
            candidates.append(sibling)

    if isinstance(improvement, dict):
        for item in improvement.get("history") or []:
            if isinstance(item, dict):
                path_val = item.get("path")
                if isinstance(path_val, str) and path_val.strip():
                    candidates.append(Path(path_val))

    seen: set[str] = set()
    for src in candidates:
        try:
            src = src.resolve()
        except Exception:
            src = Path(src)
        key = str(src)
        if key in seen or not src.exists():
            continue
        seen.add(key)
        try:
            rel = src.relative_to(repo_root)
        except Exception:
            rel = Path(src.name)
        dst = dest_dir / rel
        copy_tree_any(src, dst)
        collected.append(str(dst))

    return collected


def choose_self_improve(variant: str, policy: str) -> bool:
    if policy == "always":
        return True
    if policy == "never":
        return False
    return variant in AUTO_SELF_IMPROVE_VARIANTS


def build_main_run_command(
    python_exe: str,
    repo_root: Path,
    variant_dir: Path,
    *,
    model: str,
    variant: str,
    improvement_rounds: int,
    submit_message: str | None,
    args: argparse.Namespace,
) -> tuple[list[str], Path, Path]:
    output_csv = variant_dir / "submission" / "submission.csv"
    run_log = variant_dir / "run_log_main.json"
    cmd = [
        python_exe,
        "pipeline_cli.py",
        "run",
        "--competition",
        args.competition,
        "--prompt-variant",
        variant,
        "--output",
        str(output_csv),
        "--run-log",
        str(run_log),
        "--models",
        model,
        "--search-mode",
        args.search_mode,
        "--plan-beam-width",
        str(args.plan_beam_width),
        "--frontier-width",
        str(args.frontier_width),
        "--archive-size",
        str(args.archive_size),
        "--refine-rounds",
        str(args.refine_rounds),
        "--max-iters",
        str(args.max_iters),
        "--g4f-request-timeout",
        str(args.g4f_request_timeout),
        "--print-generation",
        "--print-generation-max-chars",
        str(args.print_generation_max_chars),
        "--max-response-chars",
        "0",
        "--keep-improving",
        "--improvement-rounds",
        str(improvement_rounds),
        "--g4f-async",
        "--g4f-stop-at-python-fence",
    ]
    if choose_self_improve(variant, args.self_improve_policy):
        cmd.append("--self-improve-prompts")
    if args.allow_baseline:
        cmd.append("--allow-baseline")
    if args.max_rows is not None:
        cmd.extend(["--max-rows", str(args.max_rows)])
    if args.vector_col:
        cmd.extend(["--vector-col", args.vector_col])
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
    if args.g4f_recovery_rounds is not None:
        cmd.extend(["--g4f-recovery-rounds", str(args.g4f_recovery_rounds)])
    if args.g4f_recovery_max_iters is not None:
        cmd.extend(["--g4f-recovery-max-iters", str(args.g4f_recovery_max_iters)])
    if args.g4f_recovery_sleep is not None:
        cmd.extend(["--g4f-recovery-sleep", str(args.g4f_recovery_sleep)])
    if args.worker_no_kill_process_group:
        cmd.append("--worker-no-kill-process-group")
    if submit_message:
        cmd.extend(["--message", submit_message])
    return cmd, output_csv, run_log


def build_final_submit_command(
    python_exe: str,
    *,
    args: argparse.Namespace,
    baseline_solver: Path,
    output_csv: Path,
    run_log: Path,
    message: str,
) -> list[str]:
    cmd = [
        python_exe,
        "pipeline_cli.py",
        "run",
        "--competition",
        args.competition,
        "--no-llm",
        "--baseline",
        str(baseline_solver),
        "--output",
        str(output_csv),
        "--run-log",
        str(run_log),
        "--submit",
        "--submit-via",
        args.submit_via,
        "--submit-competition",
        args.submit_competition or args.competition,
        "--message",
        message,
    ]
    if args.puzzles:
        cmd.extend(["--puzzles", args.puzzles])
    if args.kaggle_json:
        cmd.extend(["--kaggle-json", args.kaggle_json])
    if args.kaggle_config_dir:
        cmd.extend(["--kaggle-config-dir", args.kaggle_config_dir])
    return cmd


def build_detector_command(python_exe: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        python_exe,
        "pipeline_cli.py",
        "check-g4f-models",
        "--json",
        "--timeout",
        str(args.check_timeout),
        "--probe-mode",
        args.probe_mode,
        "--concurrency",
        str(args.check_concurrency),
    ]
    if args.provider:
        cmd.extend(["--provider", args.provider])
    if args.max_models is not None:
        cmd.extend(["--max-models", str(args.max_models)])
    if args.models:
        cmd.extend(["--models", args.models])
    return cmd


def read_detector_payload(stdout_path: Path) -> dict[str, Any]:
    text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    brace = text.find("{")
    if brace < 0:
        raise RuntimeError(f"Could not find JSON payload in {stdout_path}")
    return json.loads(text[brace:])


def make_run_root(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.output_root:
        root = Path(args.output_root)
        if not root.is_absolute():
            root = repo_root / root
        return root
    return repo_root / "megaminx_g4f_runs" / f"run_{make_timestamp()}"


def build_parser() -> argparse.ArgumentParser:
    repo_default = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Megaminx g4f matrix runner: detect working g4f models, run every prompt variant, collect artifacts, and build a zip archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Notes:
              * improvement-rounds default to 1000 to match the requested long-run matrix.
              * final Kaggle submit is intentionally separated from the long run to avoid per-round submit storms.
              * all stdout/stderr logs, run manifests, solver artifacts, and copied prompt bundles are archived.
            """
        ),
    )
    p.add_argument("--repo-root", default=str(repo_default), help="Repository root (default: inferred from this script).")
    p.add_argument("--competition", default="cayley-py-megaminx")
    p.add_argument("--puzzles", default=None, help="Optional explicit puzzles CSV path.")
    p.add_argument("--output-root", default=None, help="Directory where the run tree will be written.")
    p.add_argument("--archive-path", default=None, help="Optional explicit .zip output path. Defaults to <run_root>.zip")
    p.add_argument("--prompt-variants", default="all", help="Comma-separated variants or 'all'.")
    p.add_argument("--iterations", "--improvement-rounds", dest="improvement_rounds", type=int, default=1000, help="Outer improvement rounds per prompt variant.")
    p.add_argument("--models", default=None, help="Optional comma-separated g4f candidate list to validate before running. If omitted, auto-discovery is used.")
    p.add_argument("--max-models", type=int, default=None, help="Optional cap for g4f candidate discovery/probing.")
    p.add_argument("--provider", default=None, help="Optional G4F provider name.")
    p.add_argument("--check-timeout", type=float, default=12.0, help="Per-model timeout for check-g4f-models.")
    p.add_argument("--check-concurrency", type=int, default=5, help="Async probe concurrency for check-g4f-models.")
    p.add_argument("--probe-mode", choices=["pipeline", "async"], default="pipeline")
    p.add_argument("--search-mode", default=DEFAULT_SEARCH_SETTINGS["search_mode"], choices=["classic", "hybrid"])
    p.add_argument("--plan-beam-width", type=int, default=DEFAULT_SEARCH_SETTINGS["plan_beam_width"])
    p.add_argument("--frontier-width", type=int, default=DEFAULT_SEARCH_SETTINGS["frontier_width"])
    p.add_argument("--archive-size", type=int, default=DEFAULT_SEARCH_SETTINGS["archive_size"])
    p.add_argument("--refine-rounds", type=int, default=DEFAULT_SEARCH_SETTINGS["refine_rounds"])
    p.add_argument("--max-iters", type=int, default=DEFAULT_SEARCH_SETTINGS["max_iters"], help="Forwarded to pipeline_cli.py run / AgentLaboratory fixer iterations.")
    p.add_argument("--g4f-request-timeout", type=float, default=DEFAULT_SEARCH_SETTINGS["g4f_request_timeout"])
    p.add_argument("--print-generation-max-chars", type=int, default=DEFAULT_SEARCH_SETTINGS["print_generation_max_chars"])
    p.add_argument("--self-improve-policy", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--allow-baseline", action="store_true")
    p.add_argument("--agent-models", default=None)
    p.add_argument("--planner-models", default=None)
    p.add_argument("--coder-models", default=None)
    p.add_argument("--fixer-models", default=None)
    p.add_argument("--g4f-recovery-rounds", type=int, default=None)
    p.add_argument("--g4f-recovery-max-iters", type=int, default=None)
    p.add_argument("--g4f-recovery-sleep", type=float, default=None)
    p.add_argument("--worker-no-kill-process-group", action="store_true")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--vector-col", default=None)
    p.add_argument("--sleep-seconds-between-runs", type=float, default=0.0)
    p.add_argument("--continue-on-error", action="store_true", default=True, help="Continue to the next model/prompt pair when a run fails.")
    p.add_argument("--stop-on-error", dest="continue_on_error", action="store_false", help="Stop immediately on the first failed model/prompt pair.")
    p.add_argument("--submit-final", action="store_true", help="After each completed model/prompt run, submit only the final selected solver to Kaggle via a separate no-llm run.")
    p.add_argument("--submit-via", choices=["auto", "api", "cli"], default="auto")
    p.add_argument("--submit-competition", default=None)
    p.add_argument("--submission-message-template", default="megaminx {model} {variant} final selected solver")
    p.add_argument("--kaggle-json", default=None)
    p.add_argument("--kaggle-config-dir", default=None)
    p.add_argument("--python", dest="python_exe", default=sys.executable, help="Python executable used for pipeline_cli.py calls.")
    p.add_argument("--dry-run", action="store_true", help="Write manifests and commands, but do not execute the pipeline.")
    return p


def format_submission_message(template: str, *, model: str, variant: str) -> str:
    return template.format(model=model, variant=variant)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    run_root = make_run_root(args, repo_root).resolve()
    archive_path = Path(args.archive_path).resolve() if args.archive_path else Path(str(run_root) + ".zip")
    variants = parse_variants(args.prompt_variants)

    if args.submit_final and not args.kaggle_json:
        raise SystemExit("--submit-final requires --kaggle-json so the separate final submit run can authenticate.")

    run_root.mkdir(parents=True, exist_ok=True)
    meta_dir = run_root / "meta"
    logs_dir = run_root / "logs"
    meta_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    readme = textwrap.dedent(
        f"""
        Megaminx g4f matrix run
        =======================

        Created: {now_utc()}
        Repo: {repo_root}
        Competition: {args.competition}
        Prompt variants: {', '.join(variants)}
        Improvement rounds per variant: {args.improvement_rounds}
        Submit final selected solver: {args.submit_final}

        Layout:
          meta/         detector payloads, settings, overall summary
          <model>/<variant>/
            logs/       stdout/stderr for generation and optional final submit
            submission/ final CSVs
            prompts/    copied prompt bundle used for that run
            artifacts/  copied generated solver files, prompt rounds, prompt evolution
            summary.json
        """
    ).strip() + "\n"
    write_text(run_root / "README.txt", readme)

    settings_payload = {
        "created_at": now_utc(),
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "archive_path": str(archive_path),
        "argv": sys.argv,
        "parsed_args": vars(args),
        "variants": variants,
        "default_search_settings": DEFAULT_SEARCH_SETTINGS,
        "auto_self_improve_variants": sorted(AUTO_SELF_IMPROVE_VARIANTS),
    }
    write_json(meta_dir / "settings.json", settings_payload)

    detector_stdout = logs_dir / "check_g4f_models.stdout.log"
    detector_stderr = logs_dir / "check_g4f_models.stderr.log"
    detector_cmd = build_detector_command(args.python_exe, args)

    detector_result = {
        "cmd": detector_cmd,
        "cmd_text": shlex_join(detector_cmd),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        write_json(meta_dir / "g4f_detector_result.json", detector_result)
        working_models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
        detector_payload = {
            "working_models": working_models,
            "working_count": len(working_models),
            "checked_count": len(working_models),
            "source": "dry-run",
        }
    else:
        detector_result.update(run_command(detector_cmd, cwd=repo_root, stdout_path=detector_stdout, stderr_path=detector_stderr, env={**os.environ, "PYTHONUNBUFFERED": "1"}))
        write_json(meta_dir / "g4f_detector_result.json", detector_result)
        if detector_result["returncode"] != 0:
            raise SystemExit(f"check-g4f-models failed with return code {detector_result['returncode']}. See {detector_stdout} and {detector_stderr}")
        detector_payload = read_detector_payload(detector_stdout)
    write_json(meta_dir / "g4f_working_models.json", detector_payload)

    working_models = [str(item) for item in detector_payload.get("working_models") or [] if str(item).strip()]
    if not working_models:
        raise SystemExit("No working g4f models were found. Nothing to run.")

    overall_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for model in working_models:
        model_slug = slugify(model)
        for variant in variants:
            variant_slug = slugify(variant)
            variant_dir = run_root / model_slug / variant_slug
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "logs").mkdir(exist_ok=True)
            (variant_dir / "submission").mkdir(exist_ok=True)
            (variant_dir / "artifacts").mkdir(exist_ok=True)
            (variant_dir / "prompts").mkdir(exist_ok=True)

            prompt_bundle = resolve_prompt_bundle(repo_root, variant)
            if prompt_bundle.get("prompt_file"):
                copy_file_if_exists(Path(str(prompt_bundle["prompt_file"])), variant_dir / "prompts" / Path(str(prompt_bundle["prompt_file"])).name)
            if prompt_bundle.get("custom_prompts"):
                copy_file_if_exists(Path(str(prompt_bundle["custom_prompts"])), variant_dir / "prompts" / Path(str(prompt_bundle["custom_prompts"])).name)
            write_json(variant_dir / "prompt_bundle.json", prompt_bundle)

            message = format_submission_message(args.submission_message_template, model=model, variant=variant)
            main_cmd, output_csv, run_log_path = build_main_run_command(
                args.python_exe,
                repo_root,
                variant_dir,
                model=model,
                variant=variant,
                improvement_rounds=args.improvement_rounds,
                submit_message=message,
                args=args,
            )

            main_stdout = variant_dir / "logs" / "main_run.stdout.log"
            main_stderr = variant_dir / "logs" / "main_run.stderr.log"
            main_result = {
                "cmd": main_cmd,
                "cmd_text": shlex_join(main_cmd),
                "dry_run": args.dry_run,
            }
            main_run_record: dict[str, Any] | None = None
            collected_artifacts: list[str] = []
            submit_result: dict[str, Any] | None = None
            submit_run_record: dict[str, Any] | None = None
            selected_solver: Path | None = None

            try:
                if args.dry_run:
                    main_result["returncode"] = 0
                else:
                    main_result.update(run_command(main_cmd, cwd=repo_root, stdout_path=main_stdout, stderr_path=main_stderr, env={**os.environ, "PYTHONUNBUFFERED": "1"}))
                write_json(variant_dir / "main_run.command.json", main_result)

                if not args.dry_run and run_log_path.exists():
                    main_run_record = maybe_last_run_record(run_log_path)
                selected_solver = extract_selected_solver(main_run_record)
                if not args.dry_run:
                    collected_artifacts = collect_generated_artifacts(repo_root, main_run_record, variant_dir / "artifacts")

                if args.submit_final:
                    if selected_solver is None or not selected_solver.exists():
                        raise RuntimeError("Selected solver was not found after the main run, so final Kaggle submit cannot be executed.")
                    submit_output_csv = variant_dir / "submission" / "submission_final_submit.csv"
                    submit_run_log = variant_dir / "run_log_submit.json"
                    submit_cmd = build_final_submit_command(
                        args.python_exe,
                        args=args,
                        baseline_solver=selected_solver,
                        output_csv=submit_output_csv,
                        run_log=submit_run_log,
                        message=message,
                    )
                    submit_result = {
                        "cmd": submit_cmd,
                        "cmd_text": shlex_join(submit_cmd),
                        "dry_run": args.dry_run,
                    }
                    submit_stdout = variant_dir / "logs" / "submit_run.stdout.log"
                    submit_stderr = variant_dir / "logs" / "submit_run.stderr.log"
                    if args.dry_run:
                        submit_result["returncode"] = 0
                    else:
                        submit_result.update(run_command(submit_cmd, cwd=repo_root, stdout_path=submit_stdout, stderr_path=submit_stderr, env={**os.environ, "PYTHONUNBUFFERED": "1"}))
                        if submit_run_log.exists():
                            submit_run_record = maybe_last_run_record(submit_run_log)
                    write_json(variant_dir / "submit_run.command.json", submit_result)

                improvement = (
                    main_run_record.get("stages", {}).get("generate_solver", {}).get("improvement")
                    if isinstance(main_run_record, dict)
                    else None
                )
                summary = {
                    "model": model,
                    "model_slug": model_slug,
                    "variant": variant,
                    "variant_slug": variant_slug,
                    "prompt_bundle": prompt_bundle,
                    "main_result": main_result,
                    "selected_solver": str(selected_solver) if selected_solver else None,
                    "main_run_record": main_run_record,
                    "improvement_summary": {
                        "best_round": improvement.get("best_round") if isinstance(improvement, dict) else None,
                        "best_score": improvement.get("best_score") if isinstance(improvement, dict) else None,
                        "rounds_requested": improvement.get("rounds_requested") if isinstance(improvement, dict) else None,
                        "scoring_enabled": improvement.get("scoring_enabled") if isinstance(improvement, dict) else None,
                    },
                    "collected_artifacts": collected_artifacts,
                    "submit_result": submit_result,
                    "submit_run_record": submit_run_record,
                    "status": "ok" if int(main_result.get("returncode", 0)) == 0 and (submit_result is None or int(submit_result.get("returncode", 0)) == 0) else "error",
                    "created_at": now_utc(),
                }
                write_json(variant_dir / "summary.json", summary)
                overall_rows.append(summary)

                if summary["status"] != "ok":
                    failures.append({
                        "model": model,
                        "variant": variant,
                        "summary": str(variant_dir / "summary.json"),
                    })
                    if not args.continue_on_error:
                        raise RuntimeError(f"Run failed for model={model} variant={variant}")
            except Exception as exc:
                failure_summary = {
                    "model": model,
                    "variant": variant,
                    "status": "error",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "main_result": main_result,
                    "submit_result": submit_result,
                    "selected_solver": str(selected_solver) if selected_solver else None,
                    "created_at": now_utc(),
                }
                write_json(variant_dir / "summary.json", failure_summary)
                overall_rows.append(failure_summary)
                failures.append({
                    "model": model,
                    "variant": variant,
                    "summary": str(variant_dir / "summary.json"),
                })
                if not args.continue_on_error:
                    raise
            finally:
                if args.sleep_seconds_between_runs > 0:
                    time.sleep(args.sleep_seconds_between_runs)

    overall_summary = {
        "created_at": now_utc(),
        "run_root": str(run_root),
        "archive_path": str(archive_path),
        "working_models": working_models,
        "variants": variants,
        "total_pairs": len(working_models) * len(variants),
        "completed_pairs": len(overall_rows),
        "failure_count": len(failures),
        "failures": failures,
        "rows": overall_rows,
    }
    write_json(meta_dir / "overall_summary.json", overall_summary)

    if not args.dry_run:
        zip_directory(run_root, archive_path)
        write_text(meta_dir / "archive_path.txt", str(archive_path) + "\n")

    print(json.dumps({
        "run_root": str(run_root),
        "archive_path": None if args.dry_run else str(archive_path),
        "working_models": working_models,
        "variants": variants,
        "failure_count": len(failures),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
