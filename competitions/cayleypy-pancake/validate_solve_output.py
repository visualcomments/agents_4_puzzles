#!/usr/bin/env python3
"""Strict solve(vec) validator for CayleyPy-pancake solver scripts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List

MOVE_RE = re.compile(r"^R(\d+)$")


def timeout_s() -> float:
    raw = os.getenv("AGENTLAB_VALIDATOR_TIMEOUT_S", os.getenv("PIPELINE_SOLVER_TIMEOUT_S", "20"))
    try:
        return max(1.0, float(raw))
    except Exception:
        return 20.0


def flip(a: List[int], k: int) -> None:
    a[:k] = reversed(a[:k])


def validate_moves(vec: List[int], moves: list[str]) -> List[int]:
    a = list(vec)
    for move in moves:
        if str(move).upper() == "UNSOLVED":
            raise ValueError("UNSOLVED is not accepted for a working Pancake solver")
        m = MOVE_RE.fullmatch(str(move))
        if not m:
            raise ValueError(f"illegal move {move!r}")
        k = int(m.group(1))
        if k < 2 or k > len(a):
            raise ValueError(f"illegal move {move!r}: expected 2 <= k <= {len(a)}")
        flip(a, k)
    return a


def run_solver(solver: Path, vec: List[int]) -> dict:
    proc = subprocess.run(
        [sys.executable, str(solver), json.dumps(vec)],
        capture_output=True,
        text=True,
        timeout=timeout_s(),
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"solver exited with returncode={proc.returncode}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(f"solver did not print JSON on the last stdout line: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("solver")
    parser.add_argument("--vector", default="[3,2,0,1,4]")
    args = parser.parse_args(argv)

    vec = json.loads(args.vector)
    if not isinstance(vec, list):
        raise SystemExit("--vector must be a JSON list")
    vec = [int(x) for x in vec]

    payload = run_solver(Path(args.solver), vec)
    moves = payload.get("moves")
    sorted_array = payload.get("sorted_array")

    if not isinstance(moves, list):
        raise TypeError("solver JSON must contain moves as list[str]")
    replay = validate_moves(vec, [str(m) for m in moves])
    target = sorted(vec)
    if replay != target:
        raise AssertionError(f"replay did not sort permutation: got {replay}, expected {target}")
    if sorted_array != target:
        raise AssertionError(f"sorted_array mismatch: got {sorted_array}, expected {target}")
    print(json.dumps({"ok": True, "move_count": len(moves), "target": target}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
