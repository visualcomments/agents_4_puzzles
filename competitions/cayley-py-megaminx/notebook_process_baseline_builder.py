from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

_HERE = Path(__file__).resolve().parent


def _load_solve_module():
    module_path = _HERE / 'solve_module.py'
    spec = importlib.util.spec_from_file_location('megaminx_notebook_process_solve_module', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load solve_module.py from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('megaminx_notebook_process_solve_module', module)
    spec.loader.exec_module(module)
    return module


sm = _load_solve_module()


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_submission(path: Path, rows: Sequence[Dict[str, str]]) -> None:
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


def _compose_perm_bytes(left: bytes, right: bytes) -> bytes:
    return bytes(left[j] for j in right)


class ShortEffectAtlas:
    def __init__(self, generators: Dict[str, List[int]], *, depth: int) -> None:
        self.generators = generators
        self.depth = int(depth)
        self.move_names = list(generators)
        self.inverse = sm.inverse_move_map(generators)
        self.perms = {name: bytes(perm) for name, perm in generators.items()}
        size = len(next(iter(generators.values())))
        self.identity = bytes(range(size))
        self.table: Dict[bytes, Tuple[str, ...]] = {self.identity: ()}

    def build(self) -> Dict[str, Any]:
        started = time.perf_counter()
        frontier: List[Tuple[bytes, str | None]] = [(self.identity, None)]
        layers: List[Dict[str, Any]] = []
        for depth_now in range(1, self.depth + 1):
            new_frontier: List[Tuple[bytes, str | None]] = []
            for state, last in frontier:
                base = self.table[state]
                for move in self.move_names:
                    if last is not None and move == self.inverse[last]:
                        continue
                    nxt = _compose_perm_bytes(state, self.perms[move])
                    if nxt in self.table:
                        continue
                    self.table[nxt] = base + (move,)
                    new_frontier.append((nxt, move))
            frontier = new_frontier
            layers.append({'depth': depth_now, 'frontier': len(frontier), 'table_size': len(self.table)})
        return {'depth': self.depth, 'table_size': len(self.table), 'layers': layers, 'elapsed_s': round(time.perf_counter() - started, 3)}

    def optimize_moves(self, moves: Sequence[str], *, max_window: int, passes: int) -> List[str]:
        current = list(moves)
        for _pass_idx in range(max(1, int(passes))):
            n = len(current)
            dp = [10**9] * (n + 1)
            next_pos = [0] * (n + 1)
            words: List[Tuple[str, ...] | None] = [None] * (n + 1)
            dp[n] = 0
            for i in range(n - 1, -1, -1):
                dp[i] = 1 + dp[i + 1]
                next_pos[i] = i + 1
                words[i] = (current[i],)
                effect = self.identity
                limit = min(n, i + int(max_window))
                for j in range(i + 1, limit + 1):
                    effect = _compose_perm_bytes(effect, self.perms[current[j - 1]])
                    best = self.table.get(effect)
                    if best is not None and len(best) + dp[j] < dp[i]:
                        dp[i] = len(best) + dp[j]
                        next_pos[i] = j
                        words[i] = best
            out: List[str] = []
            i = 0
            while i < n:
                chunk = words[i]
                if chunk is None:
                    break
                out.extend(chunk)
                i = next_pos[i]
            if len(out) >= len(current):
                break
            current = out
        return current


def build_notebook_process_baseline(
    *,
    source_submission: Path,
    test_csv: Path,
    out_csv: Path,
    stats_out: Path,
    depth: int = 5,
    max_window: int = 12,
    passes: int = 3,
) -> Dict[str, Any]:
    central, generators = sm.load_puzzle_bundle()
    source_rows = _load_rows(source_submission)
    test_rows = _load_rows(test_csv)
    atlas = ShortEffectAtlas(generators, depth=depth)
    atlas_stats = atlas.build()
    out_rows: List[Dict[str, str]] = []
    rows_improved = 0
    rows_regressed = 0
    saved_moves = 0
    illegal_or_invalid: List[int] = []
    started = time.perf_counter()
    for idx, row in enumerate(source_rows):
        baseline_moves = sm.path_to_moves(row.get('path'))
        best_moves = atlas.optimize_moves(baseline_moves, max_window=max_window, passes=passes)
        if len(best_moves) < len(baseline_moves) and idx < len(test_rows):
            start_state = [int(x) for x in str(test_rows[idx].get('initial_state') or '').split(',') if x != '']
            if sm.validate_solution(start_state, best_moves, central, generators):
                rows_improved += 1
                saved_moves += len(baseline_moves) - len(best_moves)
            else:
                illegal_or_invalid.append(idx)
                best_moves = baseline_moves
        elif len(best_moves) > len(baseline_moves):
            rows_regressed += 1
            best_moves = baseline_moves
        out_rows.append({'initial_state_id': str(row.get('initial_state_id') or idx), 'path': sm.moves_to_path(best_moves)})
    _write_submission(out_csv, out_rows)
    stats = {
        'source_submission': str(source_submission),
        'out_csv': str(out_csv),
        'source_score': _score_rows(source_rows),
        'final_score': _score_rows(out_rows),
        'score_delta': _score_rows(source_rows) - _score_rows(out_rows),
        'rows': len(out_rows),
        'rows_improved': rows_improved,
        'rows_regressed': rows_regressed,
        'saved_moves': saved_moves,
        'invalid_rows': illegal_or_invalid,
        'optimizer': {
            'kind': 'notebook-process-short-effect-atlas-local-dp',
            'short_table_depth': int(depth),
            'local_window': int(max_window),
            'passes': int(passes),
            'atlas': atlas_stats,
        },
        'elapsed_s': round(time.perf_counter() - started + float(atlas_stats.get('elapsed_s', 0.0)), 3),
    }
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build the Megaminx notebook-process depth-5 local-DP baseline submission.')
    parser.add_argument('--source-submission', type=Path, default=_HERE / 'submissions' / 'optimized_submission.csv')
    parser.add_argument('--test-csv', type=Path, default=_HERE / 'data' / 'test.csv')
    parser.add_argument('--out', type=Path, default=_HERE / 'submissions' / 'notebook_process_depth5_submission.csv')
    parser.add_argument('--stats-out', type=Path, default=_HERE / 'submissions' / 'notebook_process_depth5_submission.stats.json')
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--max-window', type=int, default=12)
    parser.add_argument('--passes', type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stats = build_notebook_process_baseline(
        source_submission=args.source_submission,
        test_csv=args.test_csv,
        out_csv=args.out,
        stats_out=args.stats_out,
        depth=args.depth,
        max_window=args.max_window,
        passes=args.passes,
    )
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
