#!/usr/bin/env python3
"""Strict submission validator for Kaggle CayleyPy-pancake.

Validates:
- columns id,permutation,solution;
- every test id is present exactly once;
- no blank/UNSOLVED solutions;
- every move is Rk with 2 <= k <= n;
- replaying dot-separated moves sorts the permutation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List

MOVE_RE = re.compile(r"^R(\d+)$")


def parse_perm(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def flip(a: List[int], k: int) -> None:
    a[:k] = reversed(a[:k])


def split_solution(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    if text.upper() == "UNSOLVED":
        return ["UNSOLVED"]
    return [part.strip() for part in text.split(".") if part.strip()]


def apply_moves(vec: List[int], moves: Iterable[str]) -> List[int]:
    a = list(vec)
    for move in moves:
        if move == "UNSOLVED":
            raise ValueError("UNSOLVED is not accepted by this strict validator")
        m = MOVE_RE.fullmatch(move)
        if not m:
            raise ValueError(f"illegal move {move!r}")
        k = int(m.group(1))
        if k < 2 or k > len(a):
            raise ValueError(f"illegal move {move!r}: expected 2 <= k <= {len(a)}")
        flip(a, k)
    return a


def load_test_rows(test_csv: Path | None, competition_zip: Path | None) -> list[dict[str, str]]:
    if competition_zip is not None:
        with zipfile.ZipFile(competition_zip) as zf:
            with zf.open("test.csv") as f:
                return list(csv.DictReader(line.decode("utf-8") for line in f))
    if test_csv is None:
        raise ValueError("test_csv or competition_zip is required")
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_submission(submission_csv: Path, test_rows: list[dict[str, str]], *, max_errors: int = 20) -> dict:
    expected = {str(row["id"]): row for row in test_rows}
    seen: set[str] = set()
    errors: list[str] = []
    total_moves = 0
    max_moves = 0
    row_count = 0

    with submission_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "permutation", "solution"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"submission must contain columns {sorted(required)}, got {reader.fieldnames}")
        for row in reader:
            row_count += 1
            rid = str(row.get("id", "")).strip()
            if rid not in expected:
                errors.append(f"unexpected id {rid!r}")
                if len(errors) >= max_errors:
                    break
                continue
            if rid in seen:
                errors.append(f"duplicate id {rid!r}")
                if len(errors) >= max_errors:
                    break
                continue
            seen.add(rid)

            expected_perm = str(expected[rid]["permutation"])
            if str(row.get("permutation", "")) != expected_perm:
                errors.append(f"id={rid}: permutation mismatch")
                if len(errors) >= max_errors:
                    break
                continue

            moves = split_solution(str(row.get("solution", "")))
            if not moves:
                errors.append(f"id={rid}: blank solution")
                if len(errors) >= max_errors:
                    break
                continue
            if any(m.upper() == "UNSOLVED" for m in moves):
                errors.append(f"id={rid}: UNSOLVED solution")
                if len(errors) >= max_errors:
                    break
                continue

            vec = parse_perm(expected_perm)
            try:
                result = apply_moves(vec, moves)
            except Exception as exc:
                errors.append(f"id={rid}: {type(exc).__name__}: {exc}")
                if len(errors) >= max_errors:
                    break
                continue
            target = sorted(vec)
            if result != target:
                errors.append(f"id={rid}: replay did not sort permutation")
                if len(errors) >= max_errors:
                    break
                continue
            total_moves += len(moves)
            max_moves = max(max_moves, len(moves))

    missing = sorted(set(expected) - seen, key=lambda x: int(x) if x.isdigit() else x)
    if missing:
        errors.append(f"missing ids count={len(missing)}, first={missing[:10]}")

    report = {
        "ok": not errors,
        "row_count": row_count,
        "expected_rows": len(expected),
        "total_moves": total_moves,
        "max_moves": max_moves,
        "errors": errors[:max_errors],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_csv")
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--competition-zip", default=None)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args(argv)

    test_rows = load_test_rows(
        Path(args.test_csv) if args.test_csv else None,
        Path(args.competition_zip) if args.competition_zip else None,
    )
    report = validate_submission(Path(args.submission_csv), test_rows, max_errors=args.max_errors)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
