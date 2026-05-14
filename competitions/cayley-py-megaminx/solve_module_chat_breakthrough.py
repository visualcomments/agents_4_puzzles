from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

MoveOut = Union[List[str], str]

_HERE = Path(__file__).resolve().parent
_CACHE: tuple[list[int], Dict[str, List[int]], Dict[str, str]] | None = None
_LOOKUP_SOURCE: Dict[str, str] = {}
_CHAT_BREAKTHROUGH_STATS: dict[str, Any] = {
    'enabled': False,
    'roots': [],
    'candidate_files': [],
    'rows_seen': 0,
    'rows_applicable': 0,
    'accepted_updates': 0,
    'moves_saved_vs_incumbent': 0,
    'rejected_invalid': 0,
    'rejected_illegal': 0,
    'rejected_not_shorter': 0,
    'errors': [],
}

CHAT_BREAKTHROUGH_MANIFEST: dict[str, Any] = {
    'scenario': 'chat_breakthrough_artifact_lane',
    'source': 'ChatExport_2026-05-14.zip Megaminx discussions',
    'ideas_distilled': [
        'TPU teacher/student Q-shortlist beam results are treated as external artifacts, not runtime dependencies',
        'NISS/rescue/top-row portfolios are accepted only through exact replay and row-wise shortest-path merge',
        'non-backtracking/history-beam outputs are useful as candidate CSV lanes',
        'superflip/coset/PDB notes are preserved as prompt guidance and validation targets, not unverified shortcuts',
    ],
    'runtime_contract': 'standard-library only; bundled lookup first; optional external CSVs via MEGAMINX_CHAT_ARTIFACTS or /kaggle/input; exact replay before acceptance',
}
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
BASELINE_VERSION = 'chat_breakthrough_artifact_lane_v1'


def _candidate_data_dirs() -> list[Path]:
    candidates = [
        _HERE / 'data',
        Path.cwd() / 'data',
        Path.cwd() / 'competitions' / 'cayley-py-megaminx' / 'data',
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
        Path.cwd(),
        Path.cwd() / 'competitions' / 'cayley-py-megaminx',
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


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, '')
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _chat_breakthrough_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get('MEGAMINX_CHAT_ARTIFACTS', '')
    for raw in env.split(os.pathsep):
        raw = raw.strip()
        if raw:
            roots.append(Path(raw))
    # Kaggle datasets attached to a notebook normally appear here.  This keeps
    # the module portable while allowing the breakthrough chat scenario of
    # min-merging TPU/NISS/rescue CSV artifacts when they are available.
    roots.append(Path('/kaggle/input'))
    roots.append(_HERE / 'data' / 'chat_breakthrough_artifacts')
    roots.append(_HERE / 'chat_breakthrough_artifacts')

    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _looks_like_chat_artifact_csv(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() != '.csv':
        return False
    if name in {'sample_submission.csv'} or 'sample_submission' in name:
        return False
    tokens = ('submission', 'qshort', 'megaminx', 'merged', 'shortest', 'rescue', 'niss', 'beam')
    return any(token in name for token in tokens)


def _iter_chat_artifact_csvs(roots: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            candidates = [root] if root.is_file() else list(root.rglob('*.csv'))
        except OSError:
            continue
        for cand in candidates:
            try:
                resolved = cand.resolve()
                size = resolved.stat().st_size
            except OSError:
                continue
            if resolved in seen or not _looks_like_chat_artifact_csv(resolved):
                continue
            # Large enough for a submission but small enough to validate quickly.
            if size <= 0 or size > 256 * 1024 * 1024:
                continue
            seen.add(resolved)
            files.append(resolved)
    return sorted(files, key=lambda p: str(p))


def _load_test_state_maps(test_csv: Path) -> tuple[dict[str, list[int]], dict[str, str]]:
    pid_to_state: dict[str, list[int]] = {}
    pid_to_key: dict[str, str] = {}
    with test_csv.open(newline='', encoding='utf-8') as f:
        for idx, row in enumerate(csv.DictReader(f)):
            pid = str(row.get('initial_state_id') or idx).strip()
            state = _parse_state_text(str(row.get('initial_state') or ''))
            if not pid or not state:
                continue
            pid_to_state[pid] = state
            pid_to_key[pid] = state_to_key(state)
    return pid_to_state, pid_to_key


def _reset_chat_breakthrough_stats(roots: Sequence[Path]) -> None:
    _CHAT_BREAKTHROUGH_STATS.clear()
    _CHAT_BREAKTHROUGH_STATS.update({
        'enabled': bool(roots),
        'roots': [str(root) for root in roots],
        'candidate_files': [],
        'rows_seen': 0,
        'rows_applicable': 0,
        'accepted_updates': 0,
        'moves_saved_vs_incumbent': 0,
        'rejected_invalid': 0,
        'rejected_illegal': 0,
        'rejected_not_shorter': 0,
        'errors': [],
    })


def _load_chat_breakthrough_lookup(
    test_csv: Path,
    generators: Dict[str, List[int]],
    central: Sequence[int],
    incumbent_lookup: Dict[str, str],
) -> Dict[str, str]:
    roots = _chat_breakthrough_roots()
    _reset_chat_breakthrough_stats(roots)
    if not roots or not test_csv.exists():
        return incumbent_lookup

    try:
        pid_to_state, pid_to_key = _load_test_state_maps(test_csv)
    except Exception as exc:  # pragma: no cover - defensive artifact mode
        _CHAT_BREAKTHROUGH_STATS['errors'].append(f'load_test_state_maps: {exc}')
        return incumbent_lookup

    lengths: dict[str, int] = {key: len(path_to_moves(path)) for key, path in incumbent_lookup.items()}
    for key in incumbent_lookup:
        _LOOKUP_SOURCE.setdefault(key, 'optimized_lookup_depth5_atlas')

    artifact_files = _iter_chat_artifact_csvs(roots)
    _CHAT_BREAKTHROUGH_STATS['candidate_files'] = [str(p) for p in artifact_files]
    if not artifact_files:
        return incumbent_lookup

    legal_moves = set(generators)
    lookup = dict(incumbent_lookup)
    for csv_path in artifact_files:
        try:
            with csv_path.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fields = set(reader.fieldnames or [])
                id_field = next((name for name in ('initial_state_id', 'id', 'puzzle_id', 'pid') if name in fields), None)
                path_field = next((name for name in ('path', 'moves', 'solution') if name in fields), None)
                if id_field is None or path_field is None:
                    _CHAT_BREAKTHROUGH_STATS['errors'].append(f'{csv_path}: missing initial_state_id/path columns')
                    continue
                for row in reader:
                    _CHAT_BREAKTHROUGH_STATS['rows_seen'] += 1
                    pid = str(row.get(id_field) or '').strip()
                    if pid not in pid_to_state:
                        continue
                    _CHAT_BREAKTHROUGH_STATS['rows_applicable'] += 1
                    candidate_path = str(row.get(path_field) or '').strip()
                    moves = path_to_moves(candidate_path)
                    illegal = [move for move in moves if move not in legal_moves]
                    if illegal:
                        _CHAT_BREAKTHROUGH_STATS['rejected_illegal'] += 1
                        continue
                    state_key = pid_to_key[pid]
                    incumbent_len = lengths.get(state_key, 10 ** 9)
                    if len(moves) >= incumbent_len:
                        _CHAT_BREAKTHROUGH_STATS['rejected_not_shorter'] += 1
                        continue
                    if not validate_solution(pid_to_state[pid], moves, central, generators):
                        _CHAT_BREAKTHROUGH_STATS['rejected_invalid'] += 1
                        continue
                    saved = 0 if incumbent_len >= 10 ** 9 else incumbent_len - len(moves)
                    lookup[state_key] = moves_to_path(moves)
                    lengths[state_key] = len(moves)
                    _LOOKUP_SOURCE[state_key] = 'chat_breakthrough_external_artifact:' + csv_path.name
                    _CHAT_BREAKTHROUGH_STATS['accepted_updates'] += 1
                    _CHAT_BREAKTHROUGH_STATS['moves_saved_vs_incumbent'] += saved
        except Exception as exc:  # pragma: no cover - defensive artifact mode
            _CHAT_BREAKTHROUGH_STATS['errors'].append(f'{csv_path}: {exc}')
    return lookup


def chat_breakthrough_stats() -> dict[str, Any]:
    if _CACHE is None:
        _load_bundle()
    return dict(_CHAT_BREAKTHROUGH_STATS)


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

    _LOOKUP_SOURCE.clear()
    for state_key in lookup:
        _LOOKUP_SOURCE[state_key] = 'optimized_lookup_depth5_atlas'
    if test_csv.exists():
        lookup = _load_chat_breakthrough_lookup(test_csv, generators, central, lookup)

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
    result = solve_with_trace(vec)
    return result['moves'], result['sorted_array']


def solve_with_trace(vec: Sequence[int]) -> dict[str, Any]:
    central, generators, lookup = _load_bundle()
    state = list(int(x) for x in vec)
    if state == central:
        return {
            'moves': [],
            'path': '',
            'sorted_array': list(state),
            'solved': True,
            'selected_lane': 'identity',
            'candidate_len': 0,
            'baseline_version': BASELINE_VERSION,
        }
    state_key = state_to_key(state)
    path = lookup.get(state_key)
    if path is None:
        return {
            'moves': 'UNSOLVED',
            'path': 'UNSOLVED',
            'sorted_array': list(state),
            'solved': False,
            'selected_lane': 'missing_lookup',
            'candidate_len': None,
            'baseline_version': BASELINE_VERSION,
        }
    moves = path_to_moves(path)
    final_state = apply_moves(state, moves, generators)
    return {
        'moves': moves,
        'path': moves_to_path(moves),
        'sorted_array': final_state,
        'solved': final_state == list(central),
        'selected_lane': _LOOKUP_SOURCE.get(state_key, 'optimized_lookup_depth5_atlas'),
        'candidate_len': len(moves),
        'baseline_version': BASELINE_VERSION,
    }


def _parse_state_text(text: str) -> list[int]:
    raw = str(text or '').strip()
    if not raw:
        return []
    if raw.startswith('['):
        obj = json.loads(raw)
        if not isinstance(obj, list):
            raise ValueError('JSON state must be a list')
        return [int(x) for x in obj]
    return [int(x) for x in raw.split(',') if x != '']


def build_submission(out_csv: Union[str, Path], test_csv: Union[str, Path, None] = None) -> dict[str, Any]:
    data_dir = _find_data_dir()
    test_path = Path(test_csv) if test_csv is not None else data_dir / 'test.csv'
    with test_path.open(newline='', encoding='utf-8') as f:
        test_rows = list(csv.DictReader(f))
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    total_len = 0
    solved_rows = 0
    for idx, row in enumerate(test_rows):
        state = _parse_state_text(str(row.get('initial_state') or ''))
        trace = solve_with_trace(state)
        moves = trace.get('moves')
        if isinstance(moves, list):
            path = moves_to_path(moves)
            total_len += len(moves)
        else:
            path = str(moves or '')
        solved_rows += int(bool(trace.get('solved')))
        rows.append({'initial_state_id': str(row.get('initial_state_id') or idx), 'path': path})
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        writer.writerows(rows)
    return {
        'rows': len(rows),
        'solved_rows': solved_rows,
        'score': total_len,
        'out_csv': str(out_path),
        'baseline_version': BASELINE_VERSION,
        'chat_breakthrough_artifacts': dict(_CHAT_BREAKTHROUGH_STATS),
    }


def _main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in {'--version', 'version'}:
        print(json.dumps({'baseline_version': BASELINE_VERSION, 'chat_breakthrough_manifest': CHAT_BREAKTHROUGH_MANIFEST}, ensure_ascii=False))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == '--chat-breakthrough-stats':
        _load_bundle()
        print(json.dumps({'baseline_version': BASELINE_VERSION, 'chat_breakthrough_artifacts': _CHAT_BREAKTHROUGH_STATS}, ensure_ascii=False, indent=2))
        return
    if len(sys.argv) >= 3 and sys.argv[1] == '--build-submission':
        payload = build_submission(sys.argv[2])
        print(json.dumps(payload, ensure_ascii=False))
        return
    if len(sys.argv) < 2:
        print('Usage: python solve_module.py "[...]" OR python solve_module.py --build-submission submission.csv', file=sys.stderr)
        raise SystemExit(2)
    vec = json.loads(sys.argv[1])
    if not isinstance(vec, list):
        raise SystemExit('Input must be a JSON list')
    moves, out_vec = solve(vec)
    print(json.dumps({'moves': moves, 'sorted_array': out_vec}))


if __name__ == '__main__':
    _main()
