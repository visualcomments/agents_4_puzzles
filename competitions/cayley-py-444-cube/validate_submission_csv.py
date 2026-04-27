#!/usr/bin/env python3
"""Validate a cayley-py-444-cube Kaggle submission by replaying every path.

This checks the actual competition contract:
- `test.csv` columns: initial_state_id, initial_state, comment
- `submission.csv` columns: initial_state_id, path
- path is dot-separated legal generator names
- applying path to the matching initial_state must reach central_state
- permutation convention: new[i] = old[perm[i]]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List


def parse_state(text: str) -> List[int]:
    return [int(x) for x in str(text).split(",") if str(x).strip() != ""]


def apply_perm(state: List[int], perm: List[int]) -> List[int]:
    return [state[j] for j in perm]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--submission", required=True)
    ap.add_argument("--test-csv", default=str(here / "data" / "test.csv"))
    ap.add_argument("--puzzle-info", default=str(here / "data" / "puzzle_info.json"))
    ap.add_argument("--max-errors", type=int, default=10)
    args = ap.parse_args()

    submission_csv = Path(args.submission)
    test_csv = Path(args.test_csv)
    puzzle_info = Path(args.puzzle_info)

    puzzle = json.loads(puzzle_info.read_text(encoding="utf-8"))
    central_state: List[int] = puzzle["central_state"]
    generators: Dict[str, List[int]] = puzzle["generators"]

    test_rows = load_rows(test_csv)
    sub_rows = load_rows(submission_csv)

    if not sub_rows:
        raise SystemExit("[!] submission has no data rows")
    if len(test_rows) != len(sub_rows):
        raise SystemExit(f"[!] row count mismatch: test={len(test_rows)} submission={len(sub_rows)}")

    for required in ("initial_state_id", "path"):
        if required not in sub_rows[0]:
            raise SystemExit(f"[!] submission missing column {required!r}")

    test_by_id = {str(r["initial_state_id"]): r for r in test_rows}
    errors: list[str] = []
    total_tokens = 0
    max_tokens = 0

    for idx, row in enumerate(sub_rows, start=1):
        row_id = str(row.get("initial_state_id", "")).strip()
        if row_id not in test_by_id:
            errors.append(f"row {idx}: unknown initial_state_id={row_id!r}")
            if len(errors) >= args.max_errors:
                break
            continue

        raw_path = str(row.get("path", "") or "").strip()
        if not raw_path:
            errors.append(f"row {idx}: blank path")
            if len(errors) >= args.max_errors:
                break
            continue
        if raw_path.upper() == "UNSOLVED":
            errors.append(f"row {idx}: UNSOLVED path is not accepted for generated success")
            if len(errors) >= args.max_errors:
                break
            continue

        tokens = [tok.strip() for tok in raw_path.split(".")]
        if any(not tok for tok in tokens):
            errors.append(f"row {idx}: empty token in path={raw_path!r}")
            if len(errors) >= args.max_errors:
                break
            continue
        unknown = [tok for tok in tokens if tok not in generators]
        if unknown:
            errors.append(f"row {idx}: unknown moves={unknown[:5]}")
            if len(errors) >= args.max_errors:
                break
            continue

        state = parse_state(test_by_id[row_id]["initial_state"])
        if len(state) != len(central_state):
            errors.append(f"row {idx}: state length {len(state)} != {len(central_state)}")
            if len(errors) >= args.max_errors:
                break
            continue

        for tok in tokens:
            state = apply_perm(state, generators[tok])

        total_tokens += len(tokens)
        max_tokens = max(max_tokens, len(tokens))

        if state != central_state:
            errors.append(f"row {idx}: path does not reach central_state")
            if len(errors) >= args.max_errors:
                break

    if errors:
        print("[!] submission replay validation failed", file=sys.stderr)
        for err in errors:
            print(" - " + err, file=sys.stderr)
        raise SystemExit(1)

    print(json.dumps({
        "ok": True,
        "rows": len(sub_rows),
        "total_move_tokens": total_tokens,
        "max_move_tokens": max_tokens,
        "state_len": len(central_state),
        "generator_count": len(generators),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
