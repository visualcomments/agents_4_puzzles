from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load_local_module(name: str, filename: str):
    module_path = _HERE / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {filename}')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


sm = _load_local_module(f'{__name__}_solve_module', 'solve_module.py')
search_v3 = _load_local_module(f'{__name__}_search_v3', 'search_improver_v3.py')
from row_scoreboard import build_row_scoreboard, summarize_scoreboard


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Megaminx hard-row routed search wrapper around search_improver_v3')
    parser.add_argument('--submission', default=str(_HERE / 'submissions' / 'optimized_submission.csv'))
    parser.add_argument('--test-csv', default=str(_HERE / 'data' / 'test.csv'))
    parser.add_argument('--out', default=str(_HERE / 'submissions' / 'submission_hard_row_routed.csv'))
    parser.add_argument('--stats-out', default=str(_HERE / 'submissions' / 'submission_hard_row_routed.stats.json'))
    parser.add_argument('--profiles-out', default=str(_HERE / 'submissions' / 'submission_hard_row_routed.profiles.json'))
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--min-improvement', type=int, default=2)
    parser.add_argument('--light-min-path-len', type=int, default=520)
    parser.add_argument('--aggressive-min-path-len', type=int, default=640)
    parser.add_argument('--force-aggressive-top-n', type=int, default=64)
    parser.add_argument('--light-time-budget-per-row', type=float, default=0.05)
    parser.add_argument('--aggressive-time-budget-per-row', type=float, default=0.15)
    parser.add_argument('--light-beam-width', type=int, default=64)
    parser.add_argument('--aggressive-beam-width', type=int, default=128)
    parser.add_argument('--light-max-steps', type=int, default=6)
    parser.add_argument('--aggressive-max-steps', type=int, default=10)
    parser.add_argument('--light-history-depth', type=int, default=1)
    parser.add_argument('--aggressive-history-depth', type=int, default=3)
    parser.add_argument('--light-mitm-depth', type=int, default=2)
    parser.add_argument('--aggressive-mitm-depth', type=int, default=3)
    parser.add_argument('--light-window-lengths', type=str, default='14,18,22')
    parser.add_argument('--aggressive-window-lengths', type=str, default='18,24,30,36')
    parser.add_argument('--light-window-samples', type=int, default=8)
    parser.add_argument('--aggressive-window-samples', type=int, default=16)
    parser.add_argument('--light-beam-mode', choices=['simple', 'advanced'], default='advanced')
    parser.add_argument('--aggressive-beam-mode', choices=['simple', 'advanced'], default='advanced')
    parser.add_argument('--disable-cayleypy', action='store_true')
    parser.add_argument('--profile-mode', choices=['none', 'compact', 'full'], default='compact')
    parser.add_argument('--gc-every', type=int, default=10)
    parser.add_argument('--trim-adapter-cache-every', type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    submission_rows = search_v3._load_rows(Path(args.submission))
    test_rows = search_v3._load_rows(Path(args.test_csv))
    central, generators = sm.load_puzzle_bundle()
    started = time.perf_counter()
    improved_rows, stats, profiles = search_v3.improve_submission_rows(
        submission_rows=submission_rows,
        test_rows=test_rows,
        central=central,
        generators=generators,
        args=args,
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    search_v3._write_rows(Path(args.out), improved_rows)
    scoreboard = build_row_scoreboard(submission_rows=improved_rows, test_rows=test_rows)
    summary = summarize_scoreboard(scoreboard)
    stats['hard_row_wrapper'] = {
        'top_k': int(args.top_k),
        'saved_moves_per_cpu_hour': float((stats['baseline_score'] - stats['final_score']) / (elapsed / 3600.0)),
        'elapsed_seconds': elapsed,
        'summary_score': summary['score'],
    }
    Path(args.stats_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(args.profiles_out).write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'baseline_score': stats['baseline_score'], 'final_score': stats['final_score'], 'saved_moves': stats['baseline_score'] - stats['final_score']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
