from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import solve_module as sm
from cayley_adapter import MegaminxSearchAdapter
from search_policy_v3 import PolicyConfig, RowFeatures, classify_row, tier_params


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        writer.writerows(rows)


def _score_rows(rows: Sequence[Dict[str, str]]) -> int:
    total = 0
    for row in rows:
        path = (row.get('path') or '').strip()
        total += 0 if not path else len(path.split('.'))
    return total


def _row_len(row: Dict[str, str]) -> int:
    path = (row.get('path') or '').strip()
    return 0 if not path else len(path.split('.'))


def _sample_starts(path_len: int, window_len: int, sample_count: int) -> list[int]:
    max_start = path_len - window_len
    if max_start < 0:
        return []
    if max_start == 0:
        return [0]
    sample_count = max(1, sample_count)
    if max_start + 1 <= sample_count:
        return list(range(max_start + 1))
    out = {0, max_start}
    for idx in range(sample_count):
        frac = idx / max(1, sample_count - 1)
        out.add(int(round(frac * max_start)))
    return sorted(out)


def _candidate_segments(path_len: int, window_lengths: Sequence[int], samples_per_window: int) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for window_len in window_lengths:
        if window_len <= 1 or window_len > path_len:
            continue
        for start in _sample_starts(path_len, window_len, samples_per_window):
            seg = (start, start + window_len)
            if seg in seen:
                continue
            seen.add(seg)
            out.append(seg)
    out.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    return out


def _choose_better_candidate(
    *,
    start_state: Sequence[int],
    baseline_moves: Sequence[str],
    candidate_moves: Sequence[str],
    central: Sequence[int],
    generators: Dict[str, List[int]],
) -> list[str] | None:
    optimized = sm.optimize_moves(candidate_moves, generators)
    if len(optimized) >= len(baseline_moves):
        return None
    if not sm.validate_solution(start_state, optimized, central, generators):
        return None
    return optimized


def _attempt_segment_rewrites(
    *,
    start_state: Sequence[int],
    current_best: list[str],
    states: list[list[int]],
    central: Sequence[int],
    generators: Dict[str, List[int]],
    adapter: MegaminxSearchAdapter,
    cfg: Dict[str, Any],
    min_improvement: int,
    row_profile: dict[str, Any],
) -> tuple[list[str], list[list[int]], dict[str, Any]]:
    profile = {
        'segments_considered': 0,
        'segments_attempted': 0,
        'segments_improved': 0,
        'search_hits': 0,
        'backend_counts': {},
    }
    time_budget_s = float(cfg['time_budget_s'])
    started = time.perf_counter()
    segments = _candidate_segments(len(current_best), cfg['window_lengths'], cfg['window_samples'])
    profile['segments_considered'] = len(segments)
    detailed_hits: list[dict[str, Any]] = []

    for seg_idx, (start, end) in enumerate(segments):
        elapsed = time.perf_counter() - started
        remaining = time_budget_s - elapsed
        if remaining <= 0:
            profile['timed_out_segments'] = True
            break
        baseline_segment_len = end - start
        if baseline_segment_len <= min_improvement:
            continue
        source_state = states[start]
        target_state = states[end]
        max_total = baseline_segment_len - min_improvement
        outcome = adapter.search(
            start_state=source_state,
            target_state=target_state,
            beam_mode=str(cfg['beam_mode']),
            beam_width=int(cfg['beam_width']),
            max_steps=min(int(cfg['max_steps']), max_total),
            history_depth=int(cfg['history_depth']),
            mitm_depth=int(cfg['mitm_depth']),
            time_budget_s=min(remaining, time_budget_s / max(1, len(segments))),
            verbose=0,
            max_total_path_len=max_total,
        )
        profile['segments_attempted'] += 1
        profile['backend_counts'][outcome.backend] = profile['backend_counts'].get(outcome.backend, 0) + 1
        if outcome.path is None or len(outcome.path) >= baseline_segment_len:
            continue
        profile['search_hits'] += 1
        candidate = list(current_best[:start]) + list(outcome.path) + list(current_best[end:])
        better = _choose_better_candidate(
            start_state=start_state,
            baseline_moves=current_best,
            candidate_moves=candidate,
            central=central,
            generators=generators,
        )
        if better is None:
            continue
        profile['segments_improved'] += 1
        detailed_hits.append({
            'start': start,
            'end': end,
            'baseline_segment_len': baseline_segment_len,
            'replacement_len': len(outcome.path),
            'saved_moves': len(current_best) - len(better),
            'backend': outcome.backend,
            'search_profile': outcome.profile,
        })
        current_best = better
        states = sm.trajectory_states(start_state, current_best, generators)
        if len(current_best) <= min_improvement:
            break

    profile['hits'] = detailed_hits[:25]
    row_profile['search_profile'] = profile
    return current_best, states, row_profile


def improve_submission_rows(
    *,
    submission_rows: Sequence[Dict[str, str]],
    test_rows: Sequence[Dict[str, str]],
    central: Sequence[int],
    generators: Dict[str, List[int]],
    args: argparse.Namespace,
) -> tuple[List[Dict[str, str]], dict[str, Any], list[dict[str, Any]]]:
    rows = [dict(row) for row in submission_rows]
    indexed_lengths = sorted(((idx, _row_len(row)) for idx, row in enumerate(rows)), key=lambda item: item[1], reverse=True)
    top_k = min(int(args.top_k), len(indexed_lengths)) if int(args.top_k) > 0 else len(indexed_lengths)
    target_indices = {idx for idx, _length in indexed_lengths[:top_k]}
    adapter = MegaminxSearchAdapter(central, generators, prefer_cayleypy=not bool(args.disable_cayleypy))
    profiles: list[dict[str, Any]] = []
    improved_rows = [dict(row) for row in rows]
    total_saved = 0
    row_improved_count = 0
    start_run = time.perf_counter()
    policy_cfg = PolicyConfig(
        light_min_path_len=int(args.light_min_path_len),
        aggressive_min_path_len=int(args.aggressive_min_path_len),
        force_aggressive_top_n=int(args.force_aggressive_top_n),
        min_improvement_to_skip=int(args.min_improvement),
    )

    for rank, (row_idx, baseline_len) in enumerate(indexed_lengths[:top_k]):
        row = improved_rows[row_idx]
        test_row = test_rows[row_idx]
        start_state = [int(x) for x in str(test_row.get('initial_state') or '').split(',') if x != '']
        baseline_moves = sm.path_to_moves(row.get('path'))
        row_profile: dict[str, Any] = {
            'initial_state_id': str(row.get('initial_state_id') or ''),
            'row_index': row_idx,
            'row_rank': rank,
            'baseline_len': len(baseline_moves),
            'backend_preference': adapter.backend_name(),
        }
        t0 = time.perf_counter()

        preopt_moves = sm.optimize_moves(baseline_moves, generators)
        if sm.validate_solution(start_state, preopt_moves, central, generators) and len(preopt_moves) <= len(baseline_moves):
            current_best = preopt_moves
        else:
            current_best = baseline_moves
        row_profile['preopt_len'] = len(preopt_moves)
        row_profile['preopt_saved'] = len(baseline_moves) - len(current_best)
        row_profile['stage'] = 'none'

        features = RowFeatures(
            baseline_len=len(baseline_moves),
            current_best_len=len(current_best),
            preopt_saved=len(baseline_moves) - len(current_best),
            row_rank=rank,
        )
        tier = classify_row(features, policy_cfg)
        row_profile['tier'] = tier

        states = sm.trajectory_states(start_state, current_best, generators)
        if tier != 'skip':
            cfg = tier_params(tier, args)
            current_best, states, row_profile = _attempt_segment_rewrites(
                start_state=start_state,
                current_best=list(current_best),
                states=states,
                central=central,
                generators=generators,
                adapter=adapter,
                cfg=cfg,
                min_improvement=int(args.min_improvement),
                row_profile=row_profile,
            )

        if len(current_best) < len(baseline_moves):
            row_profile['stage'] = 'search' if row_profile.get('search_profile', {}).get('segments_improved', 0) > 0 else 'pre-opt'
            total_saved += len(baseline_moves) - len(current_best)
            row_improved_count += 1
        row_profile['final_len'] = len(current_best)
        row_profile['saved_moves'] = len(baseline_moves) - len(current_best)
        row_profile['elapsed_ms'] = round((time.perf_counter() - t0) * 1000.0, 3)
        improved_rows[row_idx]['path'] = sm.moves_to_path(current_best)
        profiles.append(row_profile)

    stats = {
        'top_k': top_k,
        'baseline_score': _score_rows(rows),
        'final_score': _score_rows(improved_rows),
        'score_delta': _score_rows(rows) - _score_rows(improved_rows),
        'rows_improved': row_improved_count,
        'total_saved_moves': total_saved,
        'elapsed_ms_total': round((time.perf_counter() - start_run) * 1000.0, 3),
        'cayleypy_enabled': adapter.prefer_cayleypy,
        'backend_preference': adapter.backend_name(),
    }
    return improved_rows, stats, profiles


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Megaminx offline search improver v3')
    parser.add_argument('--submission', type=Path, default=_HERE / 'submissions' / 'optimized_submission.csv')
    parser.add_argument('--out', type=Path, default=_HERE / 'submissions' / 'submission_search_improved_v3.csv')
    parser.add_argument('--stats-out', type=Path, default=_HERE / 'submissions' / 'submission_search_improved_v3.stats.json')
    parser.add_argument('--profile-out', type=Path, default=_HERE / 'submissions' / 'submission_search_improved_v3.profiles.json')
    parser.add_argument('--top-k', type=int, default=150)
    parser.add_argument('--min-improvement', type=int, default=2)
    parser.add_argument('--disable-cayleypy', action='store_true')

    parser.add_argument('--light-min-path-len', type=int, default=560)
    parser.add_argument('--aggressive-min-path-len', type=int, default=700)
    parser.add_argument('--force-aggressive-top-n', type=int, default=24)

    parser.add_argument('--light-time-budget-per-row', type=float, default=0.25)
    parser.add_argument('--aggressive-time-budget-per-row', type=float, default=0.75)

    parser.add_argument('--light-beam-width', type=int, default=96)
    parser.add_argument('--aggressive-beam-width', type=int, default=192)

    parser.add_argument('--light-max-steps', type=int, default=8)
    parser.add_argument('--aggressive-max-steps', type=int, default=12)

    parser.add_argument('--light-history-depth', type=int, default=0)
    parser.add_argument('--aggressive-history-depth', type=int, default=2)

    parser.add_argument('--light-mitm-depth', type=int, default=2)
    parser.add_argument('--aggressive-mitm-depth', type=int, default=3)

    parser.add_argument('--light-window-lengths', type=str, default='14,18,22')
    parser.add_argument('--aggressive-window-lengths', type=str, default='18,24,30,36')
    parser.add_argument('--light-window-samples', type=int, default=8)
    parser.add_argument('--aggressive-window-samples', type=int, default=14)

    parser.add_argument('--light-beam-mode', type=str, default='simple', choices=['simple', 'advanced'])
    parser.add_argument('--aggressive-beam-mode', type=str, default='advanced', choices=['simple', 'advanced'])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    central, generators = sm.load_puzzle_bundle()
    data_dir = sm._find_data_dir()
    test_rows = _load_rows(data_dir / 'test.csv')
    submission_rows = _load_rows(args.submission)

    improved_rows, stats, profiles = improve_submission_rows(
        submission_rows=submission_rows,
        test_rows=test_rows,
        central=central,
        generators=generators,
        args=args,
    )

    _write_rows(args.out, improved_rows)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    args.profile_out.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'stats_out': str(args.stats_out), 'profile_out': str(args.profile_out), **stats}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
