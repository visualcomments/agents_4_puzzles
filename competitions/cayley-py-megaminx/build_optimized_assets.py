from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from solve_module import (  # type: ignore
    _find_comp_dir,
    _find_data_dir,
)

SEED = 12345
RANDOM_ORDERS_PER_ROW = 50
BASE_ORDERS: Tuple[Tuple[str, ...], ...] = (
    ("FL", "R", "FR", "U", "L", "F", "DR", "D", "BL", "DL", "B", "BR"),
    ("DR", "FR", "D", "BL", "B", "U", "R", "FL", "BR", "L", "F", "DL"),
    ("BL", "DL", "DR", "D", "FR", "B", "BR", "R", "FL", "F", "L", "U"),
    ("D", "BR", "B", "FR", "DL", "BL", "FL", "DR", "L", "R", "F", "U"),
    ("U", "L", "DL", "R", "FL", "F", "BL", "BR", "D", "DR", "B", "FR"),
    ("BL", "R", "L", "BR", "U", "F", "DL", "B", "DR", "FL", "D", "FR"),
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
    perm = {m: tuple(p) for m, p in generators.items()}

    def compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(p[i] for i in q)

    commute: set[tuple[str, str]] = set()
    for a in faces:
        for b in faces:
            if a != b and compose(perm[a], perm[b]) == compose(perm[b], perm[a]):
                commute.add((a, b))
    return faces, commute


def _parse_path(path: str) -> list[tuple[str, int]]:
    if not path:
        return []
    out: list[tuple[str, int]] = []
    for move in path.split('.'):
        out.append((move[1:], 4) if move.startswith('-') else (move, 1))
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
        best = [] if not parsed else [m for m in (row.get('path') or '').split('.') if m]
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

        optimized_rows.append({
            'initial_state_id': str(row.get('initial_state_id') or ''),
            'path': '.'.join(best),
        })
    return optimized_rows, len(fixed_orders)


def main() -> None:
    comp_dir = _find_comp_dir()
    data_dir = _find_data_dir()
    puzzle = json.loads((data_dir / 'puzzle_info.json').read_text(encoding='utf-8'))
    generators = {str(k): list(v) for k, v in dict(puzzle['generators']).items()}

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
        optimized_rows = source_rows
        optimized_score = source_score

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
        },
        'lookup': lookup,
    }
    (data_dir / 'optimized_lookup.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    (data_dir / 'optimized_stats.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'source_submission': str(source_csv), 'source_score': source_score, 'score_optimized': optimized_score, 'optimized_submission': str(optimized_submission)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
