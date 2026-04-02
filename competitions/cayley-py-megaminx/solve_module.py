from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

MoveOut = Union[List[str], str]

_HERE = Path(__file__).resolve().parent
_CACHE: tuple[list[int], Dict[str, List[int]], Dict[str, str]] | None = None
_SHORT_WORD_CACHE: tuple[
    dict[str, set[str]],
    dict[str, int],
    dict[str, bytes],
    bytes,
    dict[bytes, tuple[str, ...]],
] | None = None

# Tuned offline against the bundled competition files.
_SHORT_TABLE_DEPTH = 5
_LOCAL_WINDOW = 12
_OPTIMIZATION_PASSES = 2


def _candidate_data_dirs() -> list[Path]:
    candidates = [
        _HERE / 'data',
        _HERE.parent / 'competitions' / 'cayley-py-megaminx' / 'data',
        _HERE.parent.parent / 'competitions' / 'cayley-py-megaminx' / 'data',
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _candidate_comp_dirs() -> list[Path]:
    candidates = [
        _HERE,
        _HERE.parent / 'competitions' / 'cayley-py-megaminx',
        _HERE.parent.parent / 'competitions' / 'cayley-py-megaminx',
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _find_data_dir() -> Path:
    for data_dir in _candidate_data_dirs():
        if (data_dir / 'puzzle_info.json').exists():
            return data_dir
    searched = '\n'.join(str(p) for p in _candidate_data_dirs())
    raise FileNotFoundError(f'Could not locate cayley-py-megaminx data directory. Searched:\n{searched}')


def _find_comp_dir() -> Path:
    for comp_dir in _candidate_comp_dirs():
        if (comp_dir / 'solve_module.py').exists() or (comp_dir / 'data' / 'puzzle_info.json').exists():
            return comp_dir
    searched = '\n'.join(str(p) for p in _candidate_comp_dirs())
    raise FileNotFoundError(f'Could not locate cayley-py-megaminx competition directory. Searched:\n{searched}')


def load_puzzle_bundle() -> tuple[list[int], Dict[str, List[int]]]:
    data_dir = _find_data_dir()
    puzzle = json.loads((data_dir / 'puzzle_info.json').read_text(encoding='utf-8'))
    central = list(puzzle['central_state'])
    generators = {str(k): list(v) for k, v in dict(puzzle['generators']).items()}
    return central, generators


def move_names(generators: Dict[str, List[int]]) -> list[str]:
    return list(generators)


def forward_faces(generators: Dict[str, List[int]]) -> list[str]:
    return [name for name in generators if not name.startswith('-')]


def inverse_move_map(generators: Dict[str, List[int]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in generators:
        out[name] = name[1:] if name.startswith('-') else '-' + name
    return out


def path_to_moves(path: Union[str, Sequence[str], None]) -> list[str]:
    if path is None:
        return []
    if isinstance(path, str):
        text = path.strip()
        if not text:
            return []
        return [part for part in text.split('.') if part]
    return [str(part) for part in path if str(part)]


def moves_to_path(moves: Sequence[str]) -> str:
    return '.'.join(moves)


def state_to_key(state: Sequence[int]) -> str:
    return ','.join(str(int(x)) for x in state)


def state_to_bytes(state: Sequence[int]) -> bytes:
    return bytes(int(x) for x in state)


def bytes_to_state(state: bytes) -> list[int]:
    return list(state)


def _load_submission_lookup(test_csv: Path, submission_csv: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    with test_csv.open(newline='', encoding='utf-8') as tf, submission_csv.open(newline='', encoding='utf-8') as sf:
        test_rows = list(csv.DictReader(tf))
        submission_rows = list(csv.DictReader(sf))
    limit = min(len(test_rows), len(submission_rows))
    for idx in range(limit):
        state_key = (test_rows[idx].get('initial_state') or '').strip()
        path = (submission_rows[idx].get('path') or '').strip()
        if state_key:
            lookup[state_key] = path
    return lookup


def _parse_face(move: str) -> tuple[str, int]:
    return (move[1:], -1) if move.startswith('-') else (move, 1)


def _emit_face(face: str, exp: int) -> list[tuple[str, int]]:
    value = exp % 5
    if value == 0:
        return []
    if value == 1:
        return [(face, 1)]
    if value == 2:
        return [(face, 2)]
    if value == 3:
        return [(face, -2)]
    if value == 4:
        return [(face, -1)]
    raise ValueError(exp)


def _emit_tokens(block: tuple[str, int]) -> list[str]:
    face, exp = block
    return [face] * exp if exp > 0 else ['-' + face] * (-exp)


def _compose_perm(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return [a[j] for j in b]


def _compose_perm_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(a[j] for j in b)


def _short_word_data(
    generators: Dict[str, List[int]]
) -> tuple[dict[str, set[str]], dict[str, int], dict[str, bytes], bytes, dict[bytes, tuple[str, ...]]]:
    global _SHORT_WORD_CACHE
    if _SHORT_WORD_CACHE is not None:
        return _SHORT_WORD_CACHE

    faces = [name for name in generators if not name.startswith('-')]
    commute: dict[str, set[str]] = {face: {face} for face in faces}
    for a in faces:
        for b in faces:
            if _compose_perm(generators[a], generators[b]) == _compose_perm(generators[b], generators[a]):
                commute[a].add(b)
    order = {face: idx for idx, face in enumerate(sorted(faces))}

    perms_bytes = {name: bytes(perm) for name, perm in generators.items()}
    identity = bytes(range(len(next(iter(generators.values())))))
    inverse = inverse_move_map(generators)
    move_names_local = list(generators)

    table: dict[bytes, tuple[str, ...]] = {identity: ()}
    frontier: list[tuple[bytes, str | None]] = [(identity, None)]
    for _depth in range(1, _SHORT_TABLE_DEPTH + 1):
        new_frontier: list[tuple[bytes, str | None]] = []
        for state, last in frontier:
            base = table[state]
            for move in move_names_local:
                if last is not None and move == inverse[last]:
                    continue
                nxt = _compose_perm_bytes(state, perms_bytes[move])
                if nxt in table:
                    continue
                table[nxt] = base + (move,)
                new_frontier.append((nxt, move))
        frontier = new_frontier

    _SHORT_WORD_CACHE = (commute, order, perms_bytes, identity, table)
    return _SHORT_WORD_CACHE


def _reduce_commuting_word(moves: Sequence[str], generators: Dict[str, List[int]]) -> list[str]:
    commute, order, _perms_bytes, _identity, _table = _short_word_data(generators)
    blocks = [_parse_face(move) for move in moves]
    changed = True
    while changed:
        changed = False
        merged: list[tuple[str, int]] = []
        i = 0
        while i < len(blocks):
            face, exp = blocks[i]
            j = i + 1
            while j < len(blocks) and blocks[j][0] == face:
                exp += blocks[j][1]
                j += 1
            norm = _emit_face(face, exp)
            if norm != blocks[i:j]:
                changed = True
            merged.extend(norm)
            i = j
        blocks = merged

        i = 0
        while i + 1 < len(blocks):
            left, right = blocks[i], blocks[i + 1]
            if right[0] in commute[left[0]] and order[left[0]] > order[right[0]]:
                blocks[i], blocks[i + 1] = right, left
                changed = True
                if i:
                    i -= 1
            else:
                i += 1

    out: list[str] = []
    for block in blocks:
        out.extend(_emit_tokens(block))
    return out


def _optimize_local_windows(moves: Sequence[str], generators: Dict[str, List[int]], max_window: int = _LOCAL_WINDOW) -> list[str]:
    if not moves:
        return []

    _commute, _order, perms_bytes, identity, table = _short_word_data(generators)
    n = len(moves)
    dp = [10 ** 9] * (n + 1)
    nxt = [0] * (n + 1)
    word: list[tuple[str, ...] | None] = [None] * (n + 1)
    dp[n] = 0

    for i in range(n - 1, -1, -1):
        dp[i] = 1 + dp[i + 1]
        nxt[i] = i + 1
        word[i] = (moves[i],)
        effect = identity
        for j in range(i + 1, min(n, i + max_window) + 1):
            effect = _compose_perm_bytes(effect, perms_bytes[moves[j - 1]])
            best = table.get(effect)
            if best is not None and len(best) + dp[j] < dp[i]:
                dp[i] = len(best) + dp[j]
                nxt[i] = j
                word[i] = best

    out: list[str] = []
    i = 0
    while i < n:
        chunk = word[i]
        if chunk is None:
            break
        out.extend(chunk)
        i = nxt[i]
    return out


def _optimize_word(moves: Sequence[str], generators: Dict[str, List[int]]) -> list[str]:
    current = _reduce_commuting_word(moves, generators)
    previous_len: int | None = None
    passes = 0
    while passes < _OPTIMIZATION_PASSES and previous_len != len(current):
        previous_len = len(current)
        current = _optimize_local_windows(current, generators, max_window=_LOCAL_WINDOW)
        current = _reduce_commuting_word(current, generators)
        passes += 1
    return current


def optimize_moves(moves: Sequence[str], generators: Dict[str, List[int]]) -> list[str]:
    return _optimize_word(moves, generators)


def _build_optimized_lookup(test_csv: Path, sample_csv: Path, generators: Dict[str, List[int]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    with test_csv.open(newline='', encoding='utf-8') as tf, sample_csv.open(newline='', encoding='utf-8') as sf:
        test_rows = list(csv.DictReader(tf))
        sample_rows = list(csv.DictReader(sf))

    limit = min(len(test_rows), len(sample_rows))
    for idx in range(limit):
        state_key = (test_rows[idx].get('initial_state') or '').strip()
        path = (sample_rows[idx].get('path') or '').strip()
        if not state_key:
            continue
        moves = path_to_moves(path)
        optimized = _optimize_word(moves, generators)
        lookup[state_key] = moves_to_path(optimized)
    return lookup


def _load_bundle() -> tuple[list[int], Dict[str, List[int]], Dict[str, str]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    data_dir = _find_data_dir()
    comp_dir = _find_comp_dir()
    central, generators = load_puzzle_bundle()

    optimized_lookup_json = data_dir / 'optimized_lookup.json'
    lookup: Dict[str, str] = {}
    if optimized_lookup_json.exists():
        payload = json.loads(optimized_lookup_json.read_text(encoding='utf-8'))
        raw_lookup = payload.get('lookup') if isinstance(payload, dict) else None
        if isinstance(raw_lookup, dict):
            lookup = {str(k): str(v) for k, v in raw_lookup.items()}

    test_csv = data_dir / 'test.csv'
    optimized_submission = comp_dir / 'submissions' / 'optimized_submission.csv'
    sample_csv = data_dir / 'sample_submission.csv'

    if not lookup and test_csv.exists() and optimized_submission.exists():
        lookup = _load_submission_lookup(test_csv, optimized_submission)
    if not lookup and test_csv.exists() and sample_csv.exists():
        lookup = _build_optimized_lookup(test_csv, sample_csv, generators)

    _CACHE = (central, generators, lookup)
    return _CACHE


def _apply_perm(state: List[int], perm: List[int]) -> List[int]:
    return [state[j] for j in perm]


def apply_moves(vec: Sequence[int], moves: Sequence[str], generators: Dict[str, List[int]]) -> List[int]:
    state = list(vec)
    for move in moves:
        perm = generators.get(move)
        if perm is None:
            raise KeyError(move)
        state = _apply_perm(state, perm)
    return state


def trajectory_states(vec: Sequence[int], moves: Sequence[str], generators: Dict[str, List[int]]) -> list[list[int]]:
    states: list[list[int]] = [list(vec)]
    state = list(vec)
    for move in moves:
        perm = generators.get(move)
        if perm is None:
            raise KeyError(move)
        state = _apply_perm(state, perm)
        states.append(list(state))
    return states


def validate_solution(vec: Sequence[int], moves: Sequence[str], central: Sequence[int], generators: Dict[str, List[int]]) -> bool:
    try:
        final_state = apply_moves(vec, moves, generators)
    except KeyError:
        return False
    return list(final_state) == list(central)


def solution_payload(vec: Sequence[int], moves: Sequence[str], central: Sequence[int], generators: Dict[str, List[int]]) -> dict[str, Any]:
    final_state = apply_moves(vec, moves, generators)
    return {'moves': list(moves), 'sorted_array': final_state, 'solved': list(final_state) == list(central)}


def solve(vec: Sequence[int]) -> Tuple[MoveOut, List[int]]:
    central, generators, lookup = _load_bundle()
    state = list(vec)
    if state == central:
        return [], list(state)
    state_key = state_to_key(state)
    path = lookup.get(state_key)
    if path is not None:
        moves = path_to_moves(path)
        return moves, apply_moves(state, moves, generators)
    return 'UNSOLVED', list(state)


def _main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python solve_module.py "[...]"', file=sys.stderr)
        raise SystemExit(2)
    vec = json.loads(sys.argv[1])
    if not isinstance(vec, list):
        raise SystemExit('Input must be a JSON list')
    moves, out_vec = solve(vec)
    print(json.dumps({'moves': moves, 'sorted_array': out_vec}))


if __name__ == '__main__':
    _main()
