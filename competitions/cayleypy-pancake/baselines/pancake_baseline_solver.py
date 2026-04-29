#!/usr/bin/env python3
"""Working baseline solver for Kaggle CayleyPy-pancake.

Competition slug: CayleyPy-pancake / cayleypy-pancake.

Data format:
- test.csv columns: id,n,permutation
- submission.csv columns: id,permutation,solution
- permutation is a comma-separated integer permutation
- solution is a dot-separated list of moves Rk
- move Rk reverses the first k elements of the current permutation

The solver uses the classic constructive pancake-sort baseline:
for size n..2, bring the maximum remaining element to the front, then flip it
into its final position. It is not optimal, but it solves every row without
UNSOLVED and is a valid baseline for prompt-sweep improvement.

This file is intentionally usable in two modes:

1. Pipeline module mode:
   solve(vec) -> (moves, sorted_array)

2. Submission builder:
   python pancake_baseline_solver.py --test-csv data/test.csv --output submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Move = str
MOVE_RE = re.compile(r"^R(\d+)$")


def _flip_prefix_inplace(a: List[int], k: int) -> None:
    if k < 0 or k > len(a):
        raise ValueError(f"invalid prefix length {k} for n={len(a)}")
    i, j = 0, k - 1
    while i < j:
        a[i], a[j] = a[j], a[i]
        i += 1
        j -= 1


def solve(vec: Sequence[int]) -> Tuple[List[Move], List[int]]:
    """Return a valid sequence of Rk moves and the resulting sorted array."""
    a = [int(x) for x in vec]
    moves: List[Move] = []
    n = len(a)

    for size in range(n, 1, -1):
        max_idx = 0
        for i in range(1, size):
            if a[i] > a[max_idx]:
                max_idx = i

        if max_idx == size - 1:
            continue

        if max_idx != 0:
            k = max_idx + 1
            _flip_prefix_inplace(a, k)
            moves.append(f"R{k}")

        _flip_prefix_inplace(a, size)
        moves.append(f"R{size}")

    return moves, a


def apply_moves(vec: Sequence[int], moves: Iterable[str]) -> List[int]:
    a = [int(x) for x in vec]
    for move in moves:
        move = str(move).strip()
        if not move:
            continue
        m = MOVE_RE.fullmatch(move)
        if not m:
            raise ValueError(f"illegal move: {move!r}")
        k = int(m.group(1))
        if k < 2 or k > len(a):
            raise ValueError(f"illegal move {move!r}: expected 2 <= k <= {len(a)}")
        _flip_prefix_inplace(a, k)
    return a


def parse_permutation(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def load_test_rows(test_csv: Path | None = None, competition_zip: Path | None = None) -> list[dict[str, str]]:
    if competition_zip is not None:
        with zipfile.ZipFile(competition_zip) as zf:
            with zf.open("test.csv") as f:
                return list(csv.DictReader(line.decode("utf-8") for line in f))
    if test_csv is None:
        raise ValueError("Either --test-csv or --competition-zip is required")
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_submission(rows: list[dict[str, str]], output: Path, *, validate: bool = True) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    total_moves = 0
    max_moves = 0
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "permutation", "solution"])
        writer.writeheader()
        for row in rows:
            vec = parse_permutation(row["permutation"])
            moves, sorted_array = solve(vec)
            if validate:
                replay = apply_moves(vec, moves)
                target = sorted(vec)
                if sorted_array != target or replay != target:
                    raise AssertionError(f"baseline failed for id={row.get('id')}: replay={replay}, target={target}")
            total_moves += len(moves)
            max_moves = max(max_moves, len(moves))
            writer.writerow({"id": row["id"], "permutation": row["permutation"], "solution": ".".join(moves)})
    return {"rows": len(rows), "total_moves": total_moves, "max_moves": max_moves}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or smoke-test a CayleyPy-pancake baseline submission.")
    parser.add_argument("vector", nargs="?", help="Optional JSON vector for solve(vec) smoke mode.")
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--competition-zip", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)

    if args.vector and not args.output:
        vec = json.loads(args.vector)
        moves, sorted_array = solve(vec)
        print(json.dumps({"moves": moves, "sorted_array": sorted_array}))
        return 0

    if not args.output:
        parser.error("--output is required unless a JSON vector is provided")

    rows = load_test_rows(
        test_csv=Path(args.test_csv) if args.test_csv else None,
        competition_zip=Path(args.competition_zip) if args.competition_zip else None,
    )
    stats = build_submission(rows, Path(args.output), validate=not args.no_validate)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
