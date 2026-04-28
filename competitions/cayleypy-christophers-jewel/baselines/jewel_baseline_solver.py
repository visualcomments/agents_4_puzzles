#!/usr/bin/env python3
"""
Pipeline-compatible working baseline solver for Kaggle CayleyPy Christopher's Jewel.

Competition: https://www.kaggle.com/competitions/cayleypy-christophers-jewel

This baseline uses Kaggle's bundled sample_submission.csv as a known-valid
reference. The sample paths are intentionally non-optimal, but they replay from
all test.csv initial_state rows to puzzle_info.json central_state and therefore
are a safe validity floor for prompt-sweep improvement.

Supported modes:
1. pipeline solve(vec) -> (moves, sorted_array)
2. script smoke mode: python jewel_baseline_solver.py '[...]'
3. submission builder: python jewel_baseline_solver.py --output submission.csv

No third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

State = Tuple[int, ...]
MoveOut = Union[List[str], str]
_CACHE: Optional["BaselineData"] = None


def parse_state(value: str | Sequence[int]) -> State:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return tuple()
        return tuple(int(part.strip()) for part in text.split(',') if part.strip() != '')
    return tuple(int(x) for x in value)


def split_path(path: str | Sequence[str]) -> List[str]:
    if isinstance(path, (list, tuple)):
        return [str(x) for x in path if str(x)]
    text = str(path or '').strip()
    if not text:
        return []
    return [token for token in text.split('.') if token]


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


class BaselineData:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        puzzle_info = json.loads((data_dir / 'puzzle_info.json').read_text(encoding='utf-8'))
        self.central_state: State = parse_state(puzzle_info['central_state'])
        self.generators: Dict[str, Tuple[int, ...]] = {
            str(name): tuple(int(i) for i in perm)
            for name, perm in puzzle_info['generators'].items()
        }
        if len(self.central_state) != 48:
            raise ValueError(f"Expected Christopher's Jewel state length 48, got {len(self.central_state)}")
        if len(self.generators) != 12:
            raise ValueError(f"Expected 12 Christopher's Jewel generators, got {len(self.generators)}")
        for name, perm in self.generators.items():
            if len(perm) != len(self.central_state):
                raise ValueError(f'Generator {name!r} length {len(perm)} != state length {len(self.central_state)}')
            if sorted(perm) != list(range(len(self.central_state))):
                raise ValueError(f'Generator {name!r} is not a permutation')

        self.test_rows = read_csv_dicts(data_dir / 'test.csv')
        self.sample_rows = read_csv_dicts(data_dir / 'sample_submission.csv')
        self.sample_by_id = {str(row['initial_state_id']): str(row['path']) for row in self.sample_rows}
        self.path_by_state: Dict[State, str] = {}
        self.id_by_state: Dict[State, str] = {}
        for row in self.test_rows:
            sid = str(row['initial_state_id'])
            state = parse_state(row['initial_state'])
            if sid not in self.sample_by_id:
                raise KeyError(f'initial_state_id={sid} is absent from sample_submission.csv')
            self.path_by_state[state] = self.sample_by_id[sid]
            self.id_by_state[state] = sid

    def apply_move(self, state: Sequence[int], move: str) -> State:
        try:
            perm = self.generators[move]
        except KeyError as exc:
            raise ValueError(f'Unknown move {move!r}') from exc
        # Official CayleyPy convention for these arrays: new[i] = old[perm[i]].
        return tuple(int(state[j]) for j in perm)

    def apply_path(self, state: Sequence[int], path: str | Sequence[str]) -> State:
        current: State = tuple(int(x) for x in state)
        for move in split_path(path):
            current = self.apply_move(current, move)
        return current

    def solve_state(self, vec: Sequence[int]) -> Tuple[List[str], List[int]]:
        state = tuple(int(x) for x in vec)
        if state == self.central_state:
            return [], list(self.central_state)
        path = self.path_by_state.get(state)
        if path is None:
            raise KeyError('This baseline only knows official bundled test.csv states and central_state')
        final_state = self.apply_path(state, path)
        if final_state != self.central_state:
            sid = self.id_by_state.get(state, '?')
            raise ValueError(f'Baseline path for initial_state_id={sid} does not replay to central_state')
        return split_path(path), list(final_state)

    def build_submission_rows(self, validate: bool = True) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
        rows: List[Dict[str, str]] = []
        total_moves = 0
        max_moves = 0
        prev = -1
        for row in self.test_rows:
            sid_text = str(row['initial_state_id'])
            sid = int(sid_text)
            if sid != prev + 1:
                raise ValueError(f'test.csv ids must be ordered and contiguous; expected {prev + 1}, got {sid}')
            prev = sid
            path = self.sample_by_id[sid_text]
            if path.strip().upper() == 'UNSOLVED' or not path.strip():
                raise ValueError(f'Invalid baseline path for row {sid_text}')
            moves = split_path(path)
            total_moves += len(moves)
            max_moves = max(max_moves, len(moves))
            if validate:
                initial = parse_state(row['initial_state'])
                if self.apply_path(initial, path) != self.central_state:
                    raise ValueError(f'Row {sid_text}: baseline path does not solve to central_state')
            rows.append({'initial_state_id': sid_text, 'path': path})
        summary = {
            'row_count': len(rows),
            'total_moves_score_baseline': total_moves,
            'max_path_moves': max_moves,
            'mean_path_moves': (total_moves / len(rows)) if rows else 0.0,
            'state_length': len(self.central_state),
            'generator_count': len(self.generators),
            'validated_by_replay': validate,
        }
        return rows, summary


def find_data_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / 'competitions' / 'cayleypy-christophers-jewel' / 'data',
        here.parent / 'data',
        here / 'data',
        Path.cwd(),
        Path.cwd() / 'data',
        Path('/kaggle/input/cayleypy-christophers-jewel'),
    ]
    required = {'puzzle_info.json', 'test.csv', 'sample_submission.csv'}
    for candidate in candidates:
        try:
            if candidate.exists() and required.issubset({p.name for p in candidate.iterdir() if p.is_file()}):
                return candidate
        except Exception:
            continue
    raise FileNotFoundError(
        'Could not find puzzle_info.json, test.csv and sample_submission.csv. '
        'Run from repository root or Kaggle input directory.'
    )


def get_data() -> BaselineData:
    global _CACHE
    if _CACHE is None:
        _CACHE = BaselineData(find_data_dir())
    return _CACHE


def solve(vec: Sequence[int]) -> Tuple[MoveOut, List[int]]:
    """Return a known-valid baseline path and the replayed final state."""
    return get_data().solve_state(vec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a valid baseline submission for cayleypy-christophers-jewel.")
    parser.add_argument('vector', nargs='?', default=None, help='Optional JSON vector for solve() smoke mode.')
    parser.add_argument('--output', default='', help='Write baseline submission CSV to this path.')
    parser.add_argument('--no-validate', action='store_true', help='Skip full replay validation when building CSV.')
    parser.add_argument('--print-summary', action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data = get_data()
    if args.output:
        rows, summary = data.build_submission_rows(validate=not args.no_validate)
        out = Path(args.output).expanduser().resolve()
        write_csv_dicts(out, rows, ['initial_state_id', 'path'])
        summary['output'] = str(out)
        if args.print_summary:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.vector is None:
        raise SystemExit('Pass a JSON vector or --output submission.csv')
    vec = json.loads(args.vector)
    moves, final_state = solve(vec)
    print(json.dumps({'moves': moves, 'sorted_array': final_state}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
