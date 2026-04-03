from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


HERE = Path(__file__).resolve().parent


def _candidate_repo_roots() -> list[Path]:
    candidates = [
        HERE / 'work_v3' / 'repo' / 'agents_4_puzzles-main',
        HERE / 'repo_work' / 'fullrepo' / 'agents_4_puzzles-main',
        HERE / 'repo_work' / 'agents_4_puzzles-main',
        HERE / 'agents_4_puzzles-main',
        HERE,
        HERE.parent,
        HERE.parent.parent,
        HERE.parent.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _find_comp_dir() -> Path:
    for root in _candidate_repo_roots():
        comp = root / 'competitions' / 'cayley-py-megaminx'
        if (comp / 'solve_module.py').exists() and (comp / 'data' / 'test.csv').exists():
            return comp
    searched = '\n'.join(str(p / 'competitions' / 'cayley-py-megaminx') for p in _candidate_repo_roots())
    raise FileNotFoundError(f'Could not locate cayley-py-megaminx competition directory. Searched:\n{searched}')


def _find_repo_root(comp_dir: Path) -> Path:
    return comp_dir.parent.parent


def _load_solve_module(comp_dir: Path):
    module_path = comp_dir / 'solve_module.py'
    spec = importlib.util.spec_from_file_location('megaminx_comp_solve_module', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load solve_module from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('megaminx_comp_solve_module', module)
    spec.loader.exec_module(module)
    return module


COMP_DIR = _find_comp_dir()
REPO_ROOT = _find_repo_root(COMP_DIR)
SM = _load_solve_module(COMP_DIR)
DATA_DIR = COMP_DIR / 'data'
RUN_RESULTS_DIR = REPO_ROOT.parent.parent / 'run_results' if REPO_ROOT.name == 'agents_4_puzzles-main' else HERE / 'run_results'

_CENTRAL, _GENERATORS = SM.load_puzzle_bundle()
_TEST_ROWS: list[dict[str, str]] | None = None
_LOOKUP_CACHE: tuple[Dict[str, str], str] | None = None


def _load_test_rows() -> list[dict[str, str]]:
    global _TEST_ROWS
    if _TEST_ROWS is None:
        with (DATA_DIR / 'test.csv').open(newline='', encoding='utf-8') as f:
            _TEST_ROWS = list(csv.DictReader(f))
    return _TEST_ROWS


def _submission_candidates() -> list[Path]:
    return [
        RUN_RESULTS_DIR / 'submission_search_improved_v3_top300.csv',
        COMP_DIR / 'submissions' / 'submission_search_improved_v3_top300.csv',
        COMP_DIR / 'submissions' / 'submission_search_top300_t035.csv',
        COMP_DIR / 'submissions' / 'optimized_submission.csv',
        DATA_DIR / 'sample_submission.csv',
    ]


def _load_lookup_from_submission(submission_csv: Path) -> Dict[str, str]:
    test_rows = _load_test_rows()
    with submission_csv.open(newline='', encoding='utf-8') as f:
        sub_rows = list(csv.DictReader(f))
    limit = min(len(test_rows), len(sub_rows))
    lookup: Dict[str, str] = {}
    for idx in range(limit):
        state_key = (test_rows[idx].get('initial_state') or '').strip()
        path = (sub_rows[idx].get('path') or '').strip()
        if state_key:
            lookup[state_key] = path
    return lookup


def _best_lookup() -> tuple[Dict[str, str], str]:
    global _LOOKUP_CACHE
    if _LOOKUP_CACHE is not None:
        return _LOOKUP_CACHE
    for candidate in _submission_candidates():
        if candidate.exists():
            lookup = _load_lookup_from_submission(candidate)
            if lookup:
                _LOOKUP_CACHE = (lookup, str(candidate))
                return _LOOKUP_CACHE
    raise FileNotFoundError('No submission CSV found for best-tested lookup.')


def state_from_text(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    if text.startswith('['):
        obj = json.loads(text)
        if not isinstance(obj, list):
            raise ValueError('JSON input must be a list of integers')
        return [int(x) for x in obj]
    return [int(x) for x in text.split(',') if x != '']


def state_key(vec: Sequence[int]) -> str:
    return ','.join(str(int(x)) for x in vec)


def solve_state(vec: Sequence[int]) -> dict[str, Any]:
    state = [int(x) for x in vec]
    lookup, source = _best_lookup()
    key = state_key(state)
    path = lookup.get(key)

    if path is None:
        moves, final_state = SM.solve(state)
        if isinstance(moves, str):
            return {
                'moves': moves,
                'path': moves,
                'sorted_array': list(final_state),
                'solved': list(final_state) == list(_CENTRAL),
                'source': 'fallback:solve_module',
            }
        final_state = SM.apply_moves(state, moves, _GENERATORS)
        return {
            'moves': list(moves),
            'path': SM.moves_to_path(moves),
            'sorted_array': list(final_state),
            'solved': list(final_state) == list(_CENTRAL),
            'source': 'fallback:solve_module',
        }

    moves = SM.path_to_moves(path)
    final_state = SM.apply_moves(state, moves, _GENERATORS)
    return {
        'moves': moves,
        'path': path,
        'sorted_array': list(final_state),
        'solved': list(final_state) == list(_CENTRAL),
        'source': f'lookup:{source}',
    }



def solve(vec: Sequence[int]) -> tuple[list[str] | str, list[int]]:
    result = solve_state(vec)
    moves = result['moves']
    if isinstance(moves, list):
        return list(moves), list(result['sorted_array'])
    return str(moves), list(result['sorted_array'])

def build_submission(out_csv: Path) -> dict[str, Any]:
    test_rows = _load_test_rows()
    rows: list[dict[str, str]] = []
    total_len = 0
    solved_count = 0
    for row in test_rows:
        state = state_from_text(str(row.get('initial_state') or ''))
        result = solve_state(state)
        path = str(result['path'])
        rows.append({'initial_state_id': str(row.get('initial_state_id') or ''), 'path': path})
        total_len += 0 if not path else len(path.split('.'))
        solved_count += int(bool(result['solved']))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        writer.writerows(rows)

    _lookup, source = _best_lookup()
    return {
        'rows': len(rows),
        'solved_rows': solved_count,
        'score': total_len,
        'source': source,
        'out_csv': str(out_csv),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Best tested Megaminx solver: best submission lookup + deterministic pre-opt fallback.'
    )
    parser.add_argument(
        'vector',
        nargs='?',
        help='Compatibility mode for validators: puzzle state as JSON list or comma-separated integers.',
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--state-json', help='State as JSON list, e.g. "[0,1,2,...]"')
    group.add_argument('--state-csv', help='State as comma-separated integers')
    group.add_argument('--build-submission', help='Write a full submission CSV to this path')
    group.add_argument('--print-source', action='store_true', help='Print which best-tested submission source is used')
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.print_source:
        _lookup, source = _best_lookup()
        print(json.dumps({'source': source}, ensure_ascii=False))
        return

    if args.build_submission:
        payload = build_submission(Path(args.build_submission))
        print(json.dumps(payload, ensure_ascii=False))
        return

    vector_arg = args.vector
    if vector_arg is None:
        vector_arg = args.state_json if args.state_json is not None else args.state_csv

    if vector_arg is None:
        parser.error('provide either a positional VECTOR, --state-json, --state-csv, --build-submission, or --print-source')

    state = state_from_text(str(vector_arg))
    result = solve_state(state)
    payload = {
        'moves': result['moves'],
        'sorted_array': result['sorted_array'],
        'path': result['path'],
        'solved': result['solved'],
        'source': result['source'],
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()
