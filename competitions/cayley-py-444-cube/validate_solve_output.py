#!/usr/bin/env python3
"""Strict validator for CayleyPy 444 Cube solver output.

Solver contract:
    python solve_module.py "[...]" -> {"moves": ..., "sorted_array": ...}

For cayley-py-444-cube a successful generated solver must emit a real legal
path that transforms the provided initial_state to puzzle_info.json central_state.
UNSOLVED is rejected by default because it is not a valid generated solution for
prompt-sweep success. It can be allowed only for explicit baseline smoke tests
via --allow-unsolved or CUBE444_ALLOW_UNSOLVED=1.

Permutation convention verified against the official sample_submission.csv:
    new[i] = old[perm[i]]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _solver_timeout_s() -> float:
    raw = os.getenv("AGENTLAB_VALIDATOR_TIMEOUT_S", os.getenv("PIPELINE_SOLVER_TIMEOUT_S", "20"))
    try:
        return max(1.0, float(raw))
    except Exception:
        return 20.0


def _run_solver(solver: Path, vec: List[int]) -> str:
    timeout_s = _solver_timeout_s()
    cmd = [sys.executable, str(solver), json.dumps(vec)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        print(f"[!] solver timed out after {timeout_s:g}s", file=sys.stderr)
        raise SystemExit(1)
    if proc.returncode != 0:
        print("[!] solver crashed", file=sys.stderr)
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(1)
    return proc.stdout


def _load_puzzle_info() -> Dict[str, Any]:
    here = Path(__file__).resolve().parent
    p = here / "data" / "puzzle_info.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing puzzle_info.json at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _apply_perm(state: List[int], perm: List[int]) -> List[int]:
    return [state[j] for j in perm]


def _parse_moves(moves: Any, *, allow_unsolved: bool) -> List[str]:
    if isinstance(moves, str):
        s = moves.strip()
        if s.upper() == "UNSOLVED":
            if allow_unsolved:
                return []
            raise ValueError("UNSOLVED is rejected for generated 444-cube solvers")
        if not s:
            return []
        return s.split(".")

    if isinstance(moves, list) and all(isinstance(m, str) for m in moves):
        return moves

    raise TypeError("moves must be list[str] or a dot-separated string")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--allow-unsolved", action="store_true")
    args = ap.parse_args()

    allow_unsolved = bool(args.allow_unsolved or os.getenv("CUBE444_ALLOW_UNSOLVED", "").lower() in {"1", "true", "yes", "on"})

    solver = Path(args.solver)
    vec = json.loads(args.vector)
    if not isinstance(vec, list) or not all(isinstance(x, int) for x in vec):
        print("[!] --vector must be a JSON list[int]", file=sys.stderr)
        raise SystemExit(1)

    puzzle = _load_puzzle_info()
    central_state = puzzle["central_state"]
    generators: Dict[str, List[int]] = puzzle["generators"]

    if len(vec) != len(central_state):
        print(f"[!] vector length {len(vec)} != central_state length {len(central_state)}", file=sys.stderr)
        raise SystemExit(1)
    for name, perm in generators.items():
        if len(perm) != len(central_state):
            print(f"[!] generator {name!r} length {len(perm)} != state length {len(central_state)}", file=sys.stderr)
            raise SystemExit(1)

    out = _run_solver(solver, vec)
    try:
        data = json.loads(out)
    except Exception:
        print("[!] solver output is not valid JSON", file=sys.stderr)
        print(out, file=sys.stderr)
        raise SystemExit(1)

    if not isinstance(data, dict):
        print("[!] solver output must be a JSON object", file=sys.stderr)
        raise SystemExit(1)
    if "moves" not in data or "sorted_array" not in data:
        print("[!] JSON must contain keys: moves, sorted_array", file=sys.stderr)
        raise SystemExit(1)

    sorted_array = data["sorted_array"]
    if not isinstance(sorted_array, list) or not all(isinstance(x, int) for x in sorted_array):
        print("[!] sorted_array must be a list[int]", file=sys.stderr)
        raise SystemExit(1)
    if len(sorted_array) != len(central_state):
        print(f"[!] sorted_array length {len(sorted_array)} != central_state length {len(central_state)}", file=sys.stderr)
        raise SystemExit(1)

    try:
        moves_list = _parse_moves(data["moves"], allow_unsolved=allow_unsolved)
    except Exception as exc:
        print(f"[!] invalid moves: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    state = list(vec)
    for m in moves_list:
        if not m:
            print("[!] empty move token in path", file=sys.stderr)
            raise SystemExit(1)
        if m not in generators:
            print(f"[!] invalid move token: {m}", file=sys.stderr)
            raise SystemExit(1)
        state = _apply_perm(state, generators[m])

    if state != central_state:
        print("[!] applying moves does not reach central_state", file=sys.stderr)
        raise SystemExit(1)

    if sorted_array != state:
        print("[!] sorted_array must equal the state after applying moves", file=sys.stderr)
        raise SystemExit(1)

    print(f"[validate] OK moves={len(moves_list)} state_len={len(state)}")


if __name__ == "__main__":
    main()
