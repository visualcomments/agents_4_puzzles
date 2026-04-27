from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

FIELDNAMES = ['initial_state_id', 'path']


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def score_rows(rows: Sequence[Dict[str, str]]) -> int:
    total = 0
    for row in rows:
        path = str(row.get('path') or '').strip()
        total += 0 if not path else len([tok for tok in path.split('.') if tok])
    return total


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def choose_baseline(repo_root: Path, explicit: str | None = None) -> Path:
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(candidate)
    candidates = [
        repo_root / 'competitions' / 'cayley-py-megaminx' / 'submissions' / 'optimized_submission.csv',
        repo_root / 'submissions' / 'optimized_submission.csv',
        repo_root / 'competition_files' / 'sample_submission.csv',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('Could not find a baseline submission CSV')


def chunk_ranges(n: int, chunk_size: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        out.append((start, end))
        start = end
    return out


def run_cmd(cmd: Sequence[str], cwd: Path) -> None:
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    subprocess.run(list(cmd), cwd=str(cwd), env=env, check=True)


def kaggle_submit(csv_path: Path, message: str, cwd: Path) -> None:
    run_cmd(['kaggle', 'competitions', 'submit', 'cayley-py-megaminx', '-f', str(csv_path), '-m', message], cwd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Low-RAM Colab runner for Megaminx full refine')
    p.add_argument('--repo-root', type=Path, default=Path('.'))
    p.add_argument('--run-name', type=str, default='megaminx_full_colab_lowmem')
    p.add_argument('--baseline', type=str, default=None)
    p.add_argument('--test-csv', type=str, default=None)
    p.add_argument('--chunk-size', type=int, default=24)
    p.add_argument('--max-passes', type=int, default=2)
    p.add_argument('--profile-mode', choices=['none', 'lite', 'full'], default='none')
    p.add_argument('--submit', action='store_true')
    p.add_argument('--submit-message', type=str, default='colab low-mem refine validated')
    p.add_argument('--disable-cayleypy', action='store_true')
    p.add_argument('--min-improvement', type=int, default=2)
    p.add_argument('--light-min-path-len', type=int, default=560)
    p.add_argument('--aggressive-min-path-len', type=int, default=700)
    p.add_argument('--force-aggressive-top-n', type=int, default=24)
    p.add_argument('--light-time-budget-per-row', type=float, default=0.20)
    p.add_argument('--aggressive-time-budget-per-row', type=float, default=0.60)
    p.add_argument('--light-beam-width', type=int, default=72)
    p.add_argument('--aggressive-beam-width', type=int, default=128)
    p.add_argument('--light-max-steps', type=int, default=8)
    p.add_argument('--aggressive-max-steps', type=int, default=10)
    p.add_argument('--light-history-depth', type=int, default=0)
    p.add_argument('--aggressive-history-depth', type=int, default=1)
    p.add_argument('--light-mitm-depth', type=int, default=0)
    p.add_argument('--aggressive-mitm-depth', type=int, default=0)
    p.add_argument('--light-window-lengths', type=str, default='14,18,22')
    p.add_argument('--aggressive-window-lengths', type=str, default='18,24,30')
    p.add_argument('--light-window-samples', type=int, default=6)
    p.add_argument('--aggressive-window-samples', type=int, default=10)
    p.add_argument('--light-beam-mode', choices=['simple', 'advanced'], default='advanced')
    p.add_argument('--aggressive-beam-mode', choices=['simple', 'advanced'], default='advanced')
    p.add_argument('--gc-every', type=int, default=1)
    p.add_argument('--trim-adapter-cache-every', type=int, default=1)
    p.add_argument('--search-v3-top-k', type=int, default=150)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    comp_dir = repo_root / 'competitions' / 'cayley-py-megaminx'
    search_script = comp_dir / 'search_improver_v3.py'
    if not search_script.exists():
        raise FileNotFoundError(search_script)
    if args.test_csv:
        test_csv = Path(args.test_csv)
        if not test_csv.is_absolute():
            test_csv = repo_root / test_csv
    else:
        test_csv = comp_dir / 'data' / 'test.csv'
        if not test_csv.exists():
            fallback_test = repo_root / 'competition_files' / 'test.csv'
            if fallback_test.exists():
                test_csv = fallback_test
    source_submission = choose_baseline(repo_root, args.baseline)

    run_dir = repo_root / 'colab_runs' / args.run_name
    temp_dir = run_dir / 'temp'
    run_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    current_rows = load_rows(source_submission)
    test_rows = load_rows(test_csv)
    if len(current_rows) != len(test_rows):
        raise ValueError(f'submission/test length mismatch: {len(current_rows)} vs {len(test_rows)}')

    summary: dict[str, Any] = {
        'run_name': args.run_name,
        'source_submission': str(source_submission),
        'passes': [],
        'profile_mode': args.profile_mode,
        'chunk_size': args.chunk_size,
        'max_passes': args.max_passes,
    }
    start_run = time.perf_counter()

    current_path = run_dir / 'submission_initial.csv'
    write_rows(current_path, current_rows)
    initial_score = score_rows(current_rows)
    summary['initial_score'] = initial_score

    for pass_idx in range(1, args.max_passes + 1):
        pass_start = time.perf_counter()
        before_score = score_rows(current_rows)
        chunks_meta: list[dict[str, Any]] = []

        for chunk_idx, (start, end) in enumerate(chunk_ranges(len(current_rows), args.chunk_size), start=1):
            chunk_rows = current_rows[start:end]
            chunk_test_rows = test_rows[start:end]
            chunk_prefix = temp_dir / f'pass_{pass_idx:02d}_chunk_{chunk_idx:03d}'
            chunk_submission = chunk_prefix.with_suffix('.submission.csv')
            chunk_test = chunk_prefix.with_suffix('.test.csv')
            chunk_out = chunk_prefix.with_suffix('.out.csv')
            chunk_stats = chunk_prefix.with_suffix('.stats.json')
            chunk_profiles = chunk_prefix.with_suffix('.profiles.json')
            write_rows(chunk_submission, chunk_rows)
            with chunk_test.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(chunk_test_rows[0].keys()))
                writer.writeheader()
                writer.writerows(chunk_test_rows)
            cmd = [
                sys.executable,
                str(search_script),
                '--submission', str(chunk_submission),
                '--test-csv', str(chunk_test),
                '--out', str(chunk_out),
                '--stats-out', str(chunk_stats),
                '--profile-out', str(chunk_profiles),
                '--top-k', str(args.search_v3_top_k),
                '--min-improvement', str(args.min_improvement),
                '--profile-mode', args.profile_mode,
                '--gc-every', str(args.gc_every),
                '--trim-adapter-cache-every', str(args.trim_adapter_cache_every),
                '--light-min-path-len', str(args.light_min_path_len),
                '--aggressive-min-path-len', str(args.aggressive_min_path_len),
                '--force-aggressive-top-n', str(args.force_aggressive_top_n),
                '--light-time-budget-per-row', str(args.light_time_budget_per_row),
                '--aggressive-time-budget-per-row', str(args.aggressive_time_budget_per_row),
                '--light-beam-width', str(args.light_beam_width),
                '--aggressive-beam-width', str(args.aggressive_beam_width),
                '--light-max-steps', str(args.light_max_steps),
                '--aggressive-max-steps', str(args.aggressive_max_steps),
                '--light-history-depth', str(args.light_history_depth),
                '--aggressive-history-depth', str(args.aggressive_history_depth),
                '--light-mitm-depth', str(args.light_mitm_depth),
                '--aggressive-mitm-depth', str(args.aggressive_mitm_depth),
                '--light-window-lengths', args.light_window_lengths,
                '--aggressive-window-lengths', args.aggressive_window_lengths,
                '--light-window-samples', str(args.light_window_samples),
                '--aggressive-window-samples', str(args.aggressive_window_samples),
                '--light-beam-mode', args.light_beam_mode,
                '--aggressive-beam-mode', args.aggressive_beam_mode,
            ]
            if args.disable_cayleypy:
                cmd.append('--disable-cayleypy')
            chunk_started = time.perf_counter()
            run_cmd(cmd, repo_root)
            improved_chunk = load_rows(chunk_out)
            current_rows[start:end] = improved_chunk
            chunk_meta = read_json(chunk_stats, {})
            chunk_meta.update({
                'chunk_idx': chunk_idx,
                'range': [start, end],
                'elapsed_seconds': round(time.perf_counter() - chunk_started, 3),
            })
            chunks_meta.append(chunk_meta)
            for path in [chunk_submission, chunk_test, chunk_out, chunk_stats, chunk_profiles]:
                if path.exists():
                    path.unlink()
            del chunk_rows, chunk_test_rows, improved_chunk, chunk_meta
            gc.collect()

        after_score = score_rows(current_rows)
        pass_path = run_dir / f'submission_pass_{pass_idx:02d}.csv'
        write_rows(pass_path, current_rows)
        pass_summary = {
            'pass_idx': pass_idx,
            'score_before': before_score,
            'score_after': after_score,
            'saved_moves': before_score - after_score,
            'elapsed_seconds': round(time.perf_counter() - pass_start, 3),
            'chunks': chunks_meta,
        }
        summary['passes'].append(pass_summary)
        current_path = pass_path
        summary['latest_submission'] = str(current_path)
        (run_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        if after_score >= before_score:
            break
        gc.collect()

    final_submission = run_dir / 'submission_final.csv'
    write_rows(final_submission, current_rows)
    final_score = score_rows(current_rows)
    summary['final_submission'] = str(final_submission)
    summary['final_validation'] = {
        'rows': len(current_rows),
        'score': final_score,
    }
    summary['elapsed_seconds'] = round(time.perf_counter() - start_run, 3)
    (run_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    if args.submit:
        kaggle_submit(final_submission, args.submit_message, repo_root)

    print(json.dumps({
        'run_dir': str(run_dir),
        'final_submission': str(final_submission),
        'final_score': final_score,
        'passes': len(summary['passes']),
        'elapsed_seconds': summary['elapsed_seconds'],
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
