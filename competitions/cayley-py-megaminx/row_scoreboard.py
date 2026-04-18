from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

_HERE = Path(__file__).resolve().parent


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def score_path(path: str | None) -> int:
    text = (path or '').strip()
    return 0 if not text else len(text.split('.'))


def submission_score(rows: Sequence[Dict[str, str]]) -> int:
    return sum(score_path(row.get('path')) for row in rows)


def _length_buckets(lengths: Sequence[int], bucket_count: int = 5) -> List[int]:
    if not lengths:
        return []
    ordered = sorted(lengths)
    cut_points: List[int] = []
    for bucket_idx in range(1, bucket_count):
        pos = int(round((len(ordered) - 1) * bucket_idx / bucket_count))
        cut_points.append(ordered[pos])
    out: List[int] = []
    for value in lengths:
        bucket = 0
        while bucket < len(cut_points) and value > cut_points[bucket]:
            bucket += 1
        out.append(bucket)
    return out


def build_row_scoreboard(
    *,
    submission_rows: Sequence[Dict[str, str]],
    test_rows: Sequence[Dict[str, str]] | None = None,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in submission_rows]
    lengths = [score_path(row.get('path')) for row in rows]
    buckets = _length_buckets(lengths)
    indexed = sorted(range(len(rows)), key=lambda idx: lengths[idx], reverse=True)
    rank_by_index = {idx: rank for rank, idx in enumerate(indexed, start=1)}
    scoreboard: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        test_row = test_rows[idx] if test_rows is not None and idx < len(test_rows) else {}
        scoreboard.append({
            'row_index': idx,
            'row_rank': rank_by_index[idx],
            'initial_state_id': str(row.get('initial_state_id') or test_row.get('initial_state_id') or ''),
            'initial_state': str(test_row.get('initial_state') or ''),
            'path': str(row.get('path') or ''),
            'path_len': lengths[idx],
            'length_bucket': buckets[idx] if idx < len(buckets) else 0,
        })
    return scoreboard


def build_shadow_splits(
    scoreboard: Sequence[Dict[str, Any]],
    *,
    dev_ratio: float = 0.15,
    holdout_ratio: float = 0.15,
    seed: int = 13,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    by_bucket: Dict[int, List[int]] = {}
    for row in scoreboard:
        bucket = int(row.get('length_bucket') or 0)
        by_bucket.setdefault(bucket, []).append(int(row['row_index']))
    dev: List[int] = []
    holdout: List[int] = []
    train: List[int] = []
    for bucket, indices in sorted(by_bucket.items()):
        local = list(indices)
        rng.shuffle(local)
        n = len(local)
        n_dev = min(n, max(1, int(round(n * dev_ratio)))) if n else 0
        n_hold = min(max(0, n - n_dev), max(1, int(round(n * holdout_ratio)))) if n - n_dev > 1 else max(0, n - n_dev)
        dev.extend(local[:n_dev])
        holdout.extend(local[n_dev:n_dev + n_hold])
        train.extend(local[n_dev + n_hold:])
    dev.sort(); holdout.sort(); train.sort()
    return {
        'seed': seed,
        'dev_ratio': dev_ratio,
        'holdout_ratio': holdout_ratio,
        'train_indices': train,
        'dev_indices': dev,
        'holdout_indices': holdout,
        'counts': {
            'train': len(train),
            'dev': len(dev),
            'holdout': len(holdout),
            'total': len(scoreboard),
        },
    }


def summarize_scoreboard(scoreboard: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [int(row.get('path_len') or 0) for row in scoreboard]
    ordered = sorted(lengths)
    def q(p: float) -> int:
        if not ordered:
            return 0
        pos = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return int(ordered[pos])
    return {
        'rows': len(lengths),
        'score': int(sum(lengths)),
        'min_len': int(ordered[0]) if ordered else 0,
        'median_len': q(0.5),
        'p90_len': q(0.9),
        'p95_len': q(0.95),
        'max_len': int(ordered[-1]) if ordered else 0,
    }


def write_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Megaminx row-level scoreboard and shadow split builder')
    parser.add_argument('--submission', required=True, help='Submission CSV to profile')
    parser.add_argument('--test-csv', default=str(_HERE / 'data' / 'test.csv'), help='Optional test.csv for state ids')
    parser.add_argument('--out-json', default=str(_HERE / 'submissions' / 'row_scoreboard.json'))
    parser.add_argument('--summary-out', default=str(_HERE / 'submissions' / 'row_scoreboard.summary.json'))
    parser.add_argument('--splits-out', default=str(_HERE / 'shadow_splits.json'))
    parser.add_argument('--dev-ratio', type=float, default=0.15)
    parser.add_argument('--holdout-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=13)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    submission_rows = load_rows(Path(args.submission))
    test_rows = load_rows(Path(args.test_csv)) if str(args.test_csv).strip() else None
    scoreboard = build_row_scoreboard(submission_rows=submission_rows, test_rows=test_rows)
    summary = summarize_scoreboard(scoreboard)
    splits = build_shadow_splits(scoreboard, dev_ratio=float(args.dev_ratio), holdout_ratio=float(args.holdout_ratio), seed=int(args.seed))
    write_json(Path(args.out_json), scoreboard)
    write_json(Path(args.summary_out), summary)
    write_json(Path(args.splits_out), splits)
    print(json.dumps({'scoreboard': str(args.out_json), 'summary': summary, 'splits_out': str(args.splits_out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
