#!/usr/bin/env python3
"""Placeholder validator for competitions where the baseline solver returns UNSOLVED.

We only check that:
- the solver runs,
- outputs valid JSON with keys moves and sorted_array,
- and that moves is either a list[str] or the string 'UNSOLVED'.

This keeps the AgentLaboratory pipeline usable as a template.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _solver_timeout_s() -> float:
    raw = os.getenv("AGENTLAB_VALIDATOR_TIMEOUT_S", os.getenv("PIPELINE_SOLVER_TIMEOUT_S", "20"))
    try:
        return max(1.0, float(raw))
    except Exception:
        return 20.0


def _run_solver(solver: Path, vec) -> str:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--vector", required=True)
    args = ap.parse_args()

    solver = Path(args.solver)
    vec = json.loads(args.vector)
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

    moves = data["moves"]
    if isinstance(moves, str):
        if moves.strip().upper() == "UNSOLVED":
            print("[validate] moves = UNSOLVED (accepted).")
            raise SystemExit(0)
        print("[!] moves string must be 'UNSOLVED'", file=sys.stderr)
        raise SystemExit(1)

    if isinstance(moves, list) and all(isinstance(m, str) for m in moves):
        print(f"[validate] OK (list[str]) moves={len(moves)}")
        raise SystemExit(0)

    print("[!] moves must be list[str] or 'UNSOLVED'", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
