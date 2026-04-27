from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys
import importlib.util

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load_local_module(name: str, filename: str):
    module_path = _HERE / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {filename} from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


sm = _load_local_module(f'{__name__}_solve_module', 'solve_module.py')
from search_improver_v3 import build_arg_parser as build_v3_arg_parser
from search_improver_v3 import improve_submission_rows

SEED = 12345
RANDOM_ORDERS_PER_ROW = 50
BASE_ORDERS: Tuple[Tuple[str, ...], ...] = (
    ('FL', 'R', 'FR', 'U', 'L', 'F', 'DR', 'D', 'BL', 'DL', 'B', 'BR'),
    ('DR', 'FR', 'D', 'BL', 'B', 'U', 'R', 'FL', 'BR', 'L', 'F', 'DL'),
    ('BL', 'DL', 'DR', 'D', 'FR', 'B', 'BR', 'R', 'FL', 'F', 'L', 'U'),
    ('D', 'BR', 'B', 'FR', 'DL', 'BL', 'FL', 'DR', 'L', 'R', 'F', 'U'),
    ('U', 'L', 'DL', 'R', 'FL', 'F', 'BL', 'BR', 'D', 'DR', 'B', 'FR'),
    ('BL', 'R', 'L', 'BR', 'U', 'F', 'DL', 'B', 'DR', 'FL', 'D', 'FR'),
)


def _load_submission_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _score_rows(rows: Sequence[Dict[str, str]]) -> int:
    total = 0
    for row in rows:
        path = (row.get('path') or '').strip()
        total += 0 if not path else len(path.split('.'))
    return total


def _commuting_pairs(generators: Dict[str, List[int]]) -> tuple[list[str], set[tuple[str, str]]]:
    faces = [name for name in generators if not name.startswith('-')]
    commute: set[tuple[str, str]] = set()
    for a in faces:
        for b in faces:
            if sm._compose_perm(generators[a], generators[b]) == sm._compose_perm(generators[b], generators[a]):
                commute.add((a, b))
    return faces, commute


def _parse_path(path: str) -> list[tuple[str, int]]:
    if not path.strip():
        return []
    out: list[tuple[str, int]] = []
    for token in path.split('.'):
        token = token.strip()
        if not token:
            continue
        if token.startswith('-'):
            out.append((token[1:], -1))
        else:
            out.append((token, 1))
    return out


def _reduce_word(parsed_word: Sequence[tuple[str, int]], order: Dict[str, int], commute: set[tuple[str, str]]) -> list[str]:
    seq: list[list[object]] = []
    for face, exp in parsed_word:
        seq.append([face, exp])
        i = len(seq) - 1
        while i > 0:
            f1, e1 = seq[i - 1]
            f2, e2 = seq[i]
            if f1 == f2:
                seq[i - 1][1] = (int(e1) + int(e2)) % 5
                seq.pop(i)
                if int(seq[i - 1][1]) == 0:
                    seq.pop(i - 1)
                    i -= 1
                i = max(i - 1, 0)
                continue
            if (str(f1), str(f2)) in commute and order[str(f1)] > order[str(f2)]:
                seq[i - 1], seq[i] = seq[i], seq[i - 1]
                i -= 1
                continue
            break

    out: list[str] = []
    for face, exp in seq:
        value = int(exp) % 5
        if value == 0:
            continue
        if value <= 2:
            out.extend([str(face)] * value)
        else:
            out.extend(['-' + str(face)] * (5 - value))
    return out


def _optimize_submission_rows(rows: Sequence[Dict[str, str]], generators: Dict[str, List[int]]) -> tuple[List[Dict[str, str]], int]:
    faces, commute = _commuting_pairs(generators)
    fixed_orders = [tuple(order) for order in BASE_ORDERS] + [tuple(faces)]
    fixed_order_dicts = [{face: idx for idx, face in enumerate(order)} for order in fixed_orders]
    rng = random.Random(SEED)
    optimized_rows: list[Dict[str, str]] = []

    for row in rows:
        parsed = _parse_path((row.get('path') or '').strip())
        best = sm.path_to_moves(row.get('path'))
        best_len = len(best)

        for order_dict in fixed_order_dicts:
            candidate = _reduce_word(parsed, order_dict, commute)
            if len(candidate) < best_len:
                best_len = len(candidate)
                best = candidate

        for _ in range(RANDOM_ORDERS_PER_ROW):
            order = faces[:]
            rng.shuffle(order)
            candidate = _reduce_word(parsed, {face: idx for idx, face in enumerate(order)}, commute)
            if len(candidate) < best_len:
                best_len = len(candidate)
                best = candidate

        optimized_rows.append({'initial_state_id': str(row.get('initial_state_id') or ''), 'path': sm.moves_to_path(best)})
    return optimized_rows, len(fixed_orders)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Rebuild Megaminx optimized lookup/submission assets')
    parser.add_argument('--search-version', choices=['none', 'v3'], default='none')
    parser.add_argument('--search-top-k', type=int, default=0)
    parser.add_argument('--search-disable-cayleypy', action='store_true')
    parser.add_argument('--search-light-min-path-len', type=int, default=560)
    parser.add_argument('--search-aggressive-min-path-len', type=int, default=700)
    parser.add_argument('--search-force-aggressive-top-n', type=int, default=24)
    parser.add_argument('--search-min-improvement', type=int, default=2)
    parser.add_argument('--search-light-time-budget-per-row', type=float, default=0.25)
    parser.add_argument('--search-aggressive-time-budget-per-row', type=float, default=0.75)
    parser.add_argument('--search-light-beam-width', type=int, default=96)
    parser.add_argument('--search-aggressive-beam-width', type=int, default=192)
    parser.add_argument('--search-light-max-steps', type=int, default=8)
    parser.add_argument('--search-aggressive-max-steps', type=int, default=12)
    parser.add_argument('--search-light-history-depth', type=int, default=0)
    parser.add_argument('--search-aggressive-history-depth', type=int, default=2)
    parser.add_argument('--search-light-mitm-depth', type=int, default=2)
    parser.add_argument('--search-aggressive-mitm-depth', type=int, default=3)
    parser.add_argument('--search-light-window-lengths', type=str, default='14,18,22')
    parser.add_argument('--search-aggressive-window-lengths', type=str, default='18,24,30,36')
    parser.add_argument('--search-light-window-samples', type=int, default=8)
    parser.add_argument('--search-aggressive-window-samples', type=int, default=14)
    parser.add_argument('--search-light-beam-mode', choices=['simple', 'advanced'], default='simple')
    parser.add_argument('--search-aggressive-beam-mode', choices=['simple', 'advanced'], default='advanced')
    return parser


def _build_v3_namespace(args: argparse.Namespace, comp_dir: Path) -> argparse.Namespace:
    # Reuse the v3 CLI schema so both entrypoints stay aligned.
    parser = build_v3_arg_parser()
    v3_args = parser.parse_args([])
    v3_args.submission = comp_dir / 'submissions' / 'optimized_submission.csv'
    v3_args.out = comp_dir / 'submissions' / 'optimized_submission.csv'
    v3_args.stats_out = comp_dir / 'submissions' / 'optimized_submission.v3.stats.json'
    v3_args.profile_out = comp_dir / 'submissions' / 'optimized_submission.v3.profiles.json'
    v3_args.top_k = int(args.search_top_k)
    v3_args.disable_cayleypy = bool(args.search_disable_cayleypy)
    v3_args.light_min_path_len = int(args.search_light_min_path_len)
    v3_args.aggressive_min_path_len = int(args.search_aggressive_min_path_len)
    v3_args.force_aggressive_top_n = int(args.search_force_aggressive_top_n)
    v3_args.min_improvement = int(args.search_min_improvement)
    v3_args.light_time_budget_per_row = float(args.search_light_time_budget_per_row)
    v3_args.aggressive_time_budget_per_row = float(args.search_aggressive_time_budget_per_row)
    v3_args.light_beam_width = int(args.search_light_beam_width)
    v3_args.aggressive_beam_width = int(args.search_aggressive_beam_width)
    v3_args.light_max_steps = int(args.search_light_max_steps)
    v3_args.aggressive_max_steps = int(args.search_aggressive_max_steps)
    v3_args.light_history_depth = int(args.search_light_history_depth)
    v3_args.aggressive_history_depth = int(args.search_aggressive_history_depth)
    v3_args.light_mitm_depth = int(args.search_light_mitm_depth)
    v3_args.aggressive_mitm_depth = int(args.search_aggressive_mitm_depth)
    v3_args.light_window_lengths = str(args.search_light_window_lengths)
    v3_args.aggressive_window_lengths = str(args.search_aggressive_window_lengths)
    v3_args.light_window_samples = int(args.search_light_window_samples)
    v3_args.aggressive_window_samples = int(args.search_aggressive_window_samples)
    v3_args.light_beam_mode = str(args.search_light_beam_mode)
    v3_args.aggressive_beam_mode = str(args.search_aggressive_beam_mode)
    return v3_args


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    comp_dir = sm._find_comp_dir()
    data_dir = sm._find_data_dir()
    central, generators = sm.load_puzzle_bundle()

    sample_csv = data_dir / 'sample_submission.csv'
    optimized_submission = comp_dir / 'submissions' / 'optimized_submission.csv'
    source_csv = optimized_submission if optimized_submission.exists() else sample_csv
    sample_rows = _load_submission_rows(sample_csv)
    sample_score = _score_rows(sample_rows)
    source_rows = _load_submission_rows(source_csv)
    source_score = _score_rows(source_rows)

    optimized_rows, num_fixed_orders = _optimize_submission_rows(source_rows, generators)
    optimized_score = _score_rows(optimized_rows)
    if optimized_score > source_score:
        optimized_rows = list(source_rows)
        optimized_score = source_score

    v3_stats = None
    if args.search_version == 'v3' and int(args.search_top_k) > 0:
        test_rows = _load_submission_rows(data_dir / 'test.csv')
        v3_args = _build_v3_namespace(args, comp_dir)
        improved_rows, v3_stats, v3_profiles = improve_submission_rows(
            submission_rows=optimized_rows,
            test_rows=test_rows,
            central=central,
            generators=generators,
            args=v3_args,
        )
        if _score_rows(improved_rows) <= optimized_score:
            optimized_rows = improved_rows
            optimized_score = _score_rows(optimized_rows)
            v3_args.stats_out.write_text(json.dumps(v3_stats, ensure_ascii=False, indent=2), encoding='utf-8')
            v3_args.profile_out.write_text(json.dumps(v3_profiles, ensure_ascii=False, indent=2), encoding='utf-8')

    optimized_submission.parent.mkdir(parents=True, exist_ok=True)
    with optimized_submission.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        writer.writerows(optimized_rows)

    test_csv = data_dir / 'test.csv'
    with test_csv.open(newline='', encoding='utf-8') as tf:
        test_rows = list(csv.DictReader(tf))
    lookup = {}
    for idx, row in enumerate(test_rows):
        if idx >= len(optimized_rows):
            break
        state_key = (row.get('initial_state') or '').strip()
        if state_key:
            lookup[state_key] = str(optimized_rows[idx].get('path') or '')

    payload = {
        'source_submission': str(source_csv.relative_to(comp_dir)),
        'sample_score': sample_score,
        'source_score': source_score,
        'score_optimized': optimized_score,
        'score_delta': source_score - optimized_score,
        'num_states': len(lookup),
        'optimizer': {
            'kind': 'commuting-order-sweep',
            'random_orders_per_row': RANDOM_ORDERS_PER_ROW,
            'seed': SEED,
            'num_fixed_orders': num_fixed_orders,
            'search_version': args.search_version,
            'search_top_k': int(args.search_top_k),
        },
        'lookup': lookup,
    }
    if v3_stats is not None:
        payload['v3_stats'] = v3_stats
    (data_dir / 'optimized_lookup.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    (data_dir / 'optimized_stats.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'source_submission': str(source_csv), 'source_score': source_score, 'score_optimized': optimized_score, 'optimized_submission': str(optimized_submission), 'search_version': args.search_version}, ensure_ascii=False))


if __name__ == '__main__':
    main()
