import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pilgrim.parallel import gpu_ids_to_cli_args

LOG_DIR = PACKAGE_ROOT / "logs"

DEFAULT_VALUE_BEAM_EXPONENTS = [12, 13, 14, 15, 16, 17]
DEFAULT_Q_BEAM_EXPONENTS = [12, 13, 14, 15, 16, 17, 18, 19]


def sanitize_cli_token(token):
    text = str(token)
    if "/" in text or "\\" in text:
        return Path(text).name
    return text


def sanitize_command(command):
    return [sanitize_cli_token(token) if idx == 0 else str(token) for idx, token in enumerate(command)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and aggregate the fixed value-vs-Q beam-width benchmark."
    )
    parser.add_argument("--group_id", type=int, default=900, help="Puzzle group id.")
    parser.add_argument("--target_id", type=int, default=0, help="Target id.")
    parser.add_argument("--value_model_id", type=int, required=True, help="Value-model id.")
    parser.add_argument("--q_model_id", type=int, required=True, help="Q-model id.")
    parser.add_argument("--tests_num", type=int, default=30, help="Number of random scrambles.")
    parser.add_argument("--rnd_depth", type=int, default=1000, help="Random-walk scramble length.")
    parser.add_argument("--rnd_seed", type=int, default=42, help="Seed for scramble generation.")
    parser.add_argument("--search_seed", type=int, default=0, help="Seed for deterministic search hashing.")
    parser.add_argument("--eval_batch_size", type=int, default=2**14, help="eval_batch_size forwarded to test.py.")
    parser.add_argument("--num_attempts", type=int, default=2, help="Number of search restarts.")
    parser.add_argument("--num_steps", type=int, default=200, help="Maximum beam-search steps.")
    parser.add_argument("--gpu_ids", type=str, help="Comma-separated CUDA ids, e.g. '0,3,5,6'.")
    parser.add_argument(
        "--value_beam_exponents",
        type=int,
        nargs="+",
        default=DEFAULT_VALUE_BEAM_EXPONENTS,
        help="Beam-width exponents for the value model.",
    )
    parser.add_argument(
        "--q_beam_exponents",
        type=int,
        nargs="+",
        default=DEFAULT_Q_BEAM_EXPONENTS,
        help="Beam-width exponents for the Q model.",
    )
    parser.add_argument("--output_json", type=Path, help="Optional aggregated output JSON path.")
    parser.add_argument("--python", type=str, default=sys.executable or "python3", help="Python executable used to launch test.py.")
    parser.add_argument("--force", action="store_true", help="Re-run test.py even when the per-beam log already exists.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for single-GPU inference.")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead", help="torch.compile mode forwarded to test.py.")
    parser.add_argument(
        "--compile_skip_dynamic_cudagraphs",
        action="store_true",
        help="Disable dynamic-shape CUDAGraph capture in torch.compile runs.",
    )
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity forwarded to test.py.")
    return parser.parse_args()


def print_args(args):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] [compare_models] benchmark config:", flush=True)
    for key, value in sorted(vars(args).items()):
        print(f"  {key:<22} {value}", flush=True)


def resolve_model_info_path(log_dir, group_id, target_id, model_id):
    base = f"model_p{int(group_id):03d}-t{int(target_id):03d}"
    candidates = sorted(Path(log_dir).glob(f"{base}*_{int(model_id)}.json"))
    if not candidates:
        raise FileNotFoundError(f"metadata for model_id={model_id} not found under {log_dir}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple metadata files found for model_id={model_id}: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def load_model_info(log_dir, group_id, target_id, model_id):
    info_path = resolve_model_info_path(log_dir, group_id, target_id, model_id)
    with info_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), info_path


def powers_of_two(exponents):
    return [2 ** int(exponent) for exponent in exponents]


def dataset_label(rnd_depth, rnd_seed):
    return f"rnd-k{int(rnd_depth)}-s{int(rnd_seed)}"


def expected_log_path(log_dir, model_name, dataset_name, model_id, beam_size, compile_enabled=False):
    compile_suffix = "_compile" if compile_enabled else ""
    return Path(log_dir) / f"test_{model_name}-{dataset_name}_{int(model_id)}_best_B{int(beam_size)}{compile_suffix}.json"


def is_complete_log(path, expected_entries):
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return False
    return isinstance(data, list) and len(data) == expected_entries


def run_test_command(args, model_id, beam_size):
    command = [
        args.python,
        "test.py",
        "--group_id",
        str(args.group_id),
        "--target_id",
        str(args.target_id),
        "--rnd_depth",
        str(args.rnd_depth),
        "--rnd_seed",
        str(args.rnd_seed),
        "--search_seed",
        str(args.search_seed),
        "--model_id",
        str(model_id),
        "--best",
        "--B",
        str(beam_size),
        "--num_attempts",
        str(args.num_attempts),
        "--num_steps",
        str(args.num_steps),
        "--tests_num",
        str(args.tests_num),
        *gpu_ids_to_cli_args(args.gpu_ids),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--verbose",
        str(args.verbose),
        "--compile_mode",
        str(args.compile_mode),
    ]
    if args.compile:
        command.append("--compile")
    if args.compile_skip_dynamic_cudagraphs:
        command.append("--compile_skip_dynamic_cudagraphs")
    return command


def ensure_test_log(args, model_name, model_id, beam_size):
    log_path = expected_log_path(
        log_dir=LOG_DIR,
        model_name=model_name,
        dataset_name=dataset_label(args.rnd_depth, args.rnd_seed),
        model_id=model_id,
        beam_size=beam_size,
        compile_enabled=args.compile,
    )
    command = run_test_command(args, model_id=model_id, beam_size=beam_size)
    if args.force or not is_complete_log(log_path, args.tests_num):
        started_at = time.time()
        subprocess.run(command, cwd=PACKAGE_ROOT, check=True)
        elapsed = time.time() - started_at
        print(f"[compare_models] finished model_id={model_id}, B={beam_size} in {elapsed:.2f}s")
    else:
        print(f"[compare_models] reusing existing log for model_id={model_id}, B={beam_size}: {log_path}")

    if not is_complete_log(log_path, args.tests_num):
        raise RuntimeError(f"incomplete log after run for model_id={model_id}, B={beam_size}: {log_path}")
    return log_path, command


def safe_mean(values):
    return statistics.fmean(values) if values else None


def safe_median(values):
    return statistics.median(values) if values else None


def normalize_entries(entries, tests_num):
    by_test_num = {}
    for entry in entries:
        test_num = int(entry["test_num"])
        if test_num in by_test_num:
            raise RuntimeError(f"duplicate test_num={test_num} in test log")
        by_test_num[test_num] = entry

    ordered_test_nums = list(range(tests_num))
    missing = [test_num for test_num in ordered_test_nums if test_num not in by_test_num]
    if missing:
        raise RuntimeError(f"missing test ids in test log: {missing[:10]}")

    normalized = []
    for test_num in ordered_test_nums:
        entry = by_test_num[test_num]
        normalized.append(
            {
                "test_num": test_num,
                "solution_length": entry["solution_length"],
                "time_sec": entry["time"],
                "attempts": entry["attempts"],
                "solved": entry["solution_length"] is not None,
            }
        )
    return normalized


def summarize_run(entries):
    solved_lengths = [entry["solution_length"] for entry in entries if entry["solution_length"] is not None]
    all_times = [float(entry["time_sec"]) for entry in entries]
    solved_times = [float(entry["time_sec"]) for entry in entries if entry["solution_length"] is not None]
    return {
        "num_tests": len(entries),
        "solved_count": len(solved_lengths),
        "solved_rate": (len(solved_lengths) / len(entries)) if entries else None,
        "mean_solution_length": safe_mean(solved_lengths),
        "median_solution_length": safe_median(solved_lengths),
        "min_solution_length": min(solved_lengths) if solved_lengths else None,
        "max_solution_length": max(solved_lengths) if solved_lengths else None,
        "mean_time_sec": safe_mean(all_times),
        "median_time_sec": safe_median(all_times),
        "min_time_sec": min(all_times) if all_times else None,
        "max_time_sec": max(all_times) if all_times else None,
        "mean_time_sec_solved": safe_mean(solved_times),
        "median_time_sec_solved": safe_median(solved_times),
    }


def load_run_result(log_path, tests_num):
    with log_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    normalized = normalize_entries(entries, tests_num=tests_num)
    return {
        "summary": summarize_run(normalized),
        "raw": {
            "test_num": [entry["test_num"] for entry in normalized],
            "solution_length": [entry["solution_length"] for entry in normalized],
            "time_sec": [entry["time_sec"] for entry in normalized],
            "attempts": [entry["attempts"] for entry in normalized],
            "solved": [entry["solved"] for entry in normalized],
        },
    }


def pairwise_comparison(value_run, q_run):
    value_lengths = value_run["raw"]["solution_length"]
    q_lengths = q_run["raw"]["solution_length"]
    value_times = value_run["raw"]["time_sec"]
    q_times = q_run["raw"]["time_sec"]

    solved_both_deltas = []
    q_shorter_count = 0
    value_shorter_count = 0
    equal_length_count = 0
    q_only_solved_count = 0
    value_only_solved_count = 0

    for value_length, q_length in zip(value_lengths, q_lengths):
        if value_length is None and q_length is None:
            continue
        if value_length is None:
            q_only_solved_count += 1
            continue
        if q_length is None:
            value_only_solved_count += 1
            continue

        delta = q_length - value_length
        solved_both_deltas.append(delta)
        if delta < 0:
            q_shorter_count += 1
        elif delta > 0:
            value_shorter_count += 1
        else:
            equal_length_count += 1

    time_deltas = [q_time - value_time for value_time, q_time in zip(value_times, q_times)]
    q_faster_count = sum(1 for delta in time_deltas if delta < 0)
    value_faster_count = sum(1 for delta in time_deltas if delta > 0)
    equal_time_count = sum(1 for delta in time_deltas if delta == 0)

    return {
        "num_tests": len(value_lengths),
        "value_solved_count": value_run["summary"]["solved_count"],
        "q_solved_count": q_run["summary"]["solved_count"],
        "common_solved_count": len(solved_both_deltas),
        "q_only_solved_count": q_only_solved_count,
        "value_only_solved_count": value_only_solved_count,
        "mean_solution_length_delta_q_minus_value": safe_mean(solved_both_deltas),
        "median_solution_length_delta_q_minus_value": safe_median(solved_both_deltas),
        "q_shorter_count": q_shorter_count,
        "value_shorter_count": value_shorter_count,
        "equal_length_count": equal_length_count,
        "mean_time_delta_sec_q_minus_value": safe_mean(time_deltas),
        "median_time_delta_sec_q_minus_value": safe_median(time_deltas),
        "q_faster_count": q_faster_count,
        "value_faster_count": value_faster_count,
        "equal_time_count": equal_time_count,
    }


def default_output_path(args):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    return LOG_DIR / (
        f"compare_p_{time_str}_{int(args.group_id):03d}-t{int(args.target_id):03d}_"
        f"{dataset_label(args.rnd_depth, args.rnd_seed)}_"
        f"n{int(args.tests_num)}_Eb{int(args.eval_batch_size)}_"
        f"{int(args.value_model_id)}_vs_{int(args.q_model_id)}.json"
    )


def collect_model_runs(args, label, model_id, beam_exponents):
    info, info_path = load_model_info(LOG_DIR, args.group_id, args.target_id, model_id)
    model_name = info.get("model_name", f"p{int(args.group_id):03d}-t{int(args.target_id):03d}")
    runs = []

    for beam_size in powers_of_two(beam_exponents):
        log_path, command = ensure_test_log(
            args=args,
            model_name=model_name,
            model_id=model_id,
            beam_size=beam_size,
        )
        run_result = load_run_result(log_path=log_path, tests_num=args.tests_num)
        runs.append(
                {
                    "B": beam_size,
                    "source_log": str(log_path.relative_to(PACKAGE_ROOT)),
                    "command": sanitize_command(command),
                    "summary": run_result["summary"],
                    "raw": run_result["raw"],
                }
        )

    return {
        "label": label,
        "model_id": model_id,
        "model_name": model_name,
        "metadata_path": str(info_path.relative_to(PACKAGE_ROOT)),
        "runs": runs,
    }


def build_pairwise_section(value_model, q_model):
    value_by_beam = {run["B"]: run for run in value_model["runs"]}
    q_by_beam = {run["B"]: run for run in q_model["runs"]}
    common_beam_sizes = sorted(set(value_by_beam) & set(q_by_beam))

    return [
        {
            "B": beam_size,
            "comparison": pairwise_comparison(
                value_run=value_by_beam[beam_size],
                q_run=q_by_beam[beam_size],
            ),
        }
        for beam_size in common_beam_sizes
    ]


def main():
    args = parse_args()
    print_args(args)
    output_json = args.output_json.expanduser().resolve() if args.output_json else default_output_path(args)

    started_at = time.time()
    value_model = collect_model_runs(args=args, label="value", model_id=args.value_model_id, beam_exponents=args.value_beam_exponents)
    q_model = collect_model_runs(args=args, label="q", model_id=args.q_model_id, beam_exponents=args.q_beam_exponents)

    report = {
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "duration_sec": round(time.time() - started_at, 2),
        "config": {
            "group_id": args.group_id,
            "target_id": args.target_id,
            "tests_num": args.tests_num,
            "rnd_depth": args.rnd_depth,
            "rnd_seed": args.rnd_seed,
            "search_seed": args.search_seed,
            "eval_batch_size": args.eval_batch_size,
            "num_attempts": args.num_attempts,
            "num_steps": args.num_steps,
            "gpu_ids": args.gpu_ids,
            "value_beam_exponents": args.value_beam_exponents,
            "q_beam_exponents": args.q_beam_exponents,
        },
        "models": [value_model, q_model],
        "pairwise_comparison": build_pairwise_section(value_model, q_model),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)

    print(f"[compare_models] saved aggregated report to {output_json}")


if __name__ == "__main__":
    main()
