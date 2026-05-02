from __future__ import annotations

import ast
import hashlib
import re
from typing import Any, Dict, Optional, Sequence


def solver_fingerprint(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()[:16]


def parse_small_int(code: str, name: str) -> Optional[int]:
    pattern = rf"^\s*{re.escape(name)}\s*=\s*([0-9]+)\s*$"
    match = re.search(pattern, code or "", re.MULTILINE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def safe_parse_ast(code: str) -> Optional[ast.Module]:
    try:
        return ast.parse(code or "")
    except SyntaxError:
        return None


def collect_ast_symbols(tree: Optional[ast.Module]) -> Dict[str, Any]:
    if tree is None:
        return {
            "ast_parse_ok": False,
            "functions": [],
            "classes": [],
            "assigned_names": [],
            "string_literals": [],
            "call_names": [],
        }
    functions: list[str] = []
    classes: list[str] = []
    assigned_names: list[str] = []
    string_literals: list[str] = []
    call_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                assigned_names.append(node.target.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_literals.append(node.value.lower())
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                call_names.append(func.id)
            elif isinstance(func, ast.Attribute):
                call_names.append(func.attr)
    return {
        "ast_parse_ok": True,
        "functions": sorted(set(functions)),
        "classes": sorted(set(classes)),
        "assigned_names": sorted(set(assigned_names)),
        "string_literals": string_literals[:300],
        "call_names": sorted(set(call_names)),
    }


def has_any_symbol(symbols: Dict[str, Any], names: Sequence[str]) -> bool:
    haystack = (
        set(symbols.get("functions", []))
        | set(symbols.get("classes", []))
        | set(symbols.get("assigned_names", []))
        | set(symbols.get("call_names", []))
    )
    lowered = {str(item).lower() for item in haystack}
    return any(str(name).lower() in lowered for name in names)


def capability_probe_from_ast_and_text(code: str, symbols: Dict[str, Any]) -> Dict[str, Any]:
    low = (code or "").lower()
    literals = " ".join(symbols.get("string_literals", []))

    def has_token(*tokens: str) -> bool:
        return any(token in low or token in literals for token in tokens)

    return {
        "has_trace_entrypoint": has_any_symbol(symbols, ["solve_with_trace", "trace_solve", "candidate_trace"]) or has_token("solve_with_trace", "trace_solve", "candidate_trace"),
        "has_lane_selection_trace": has_token("selected_lane", "lane_id", "policy_lane", "winner_lane", "patch_lane", "fresh_lane"),
        "has_rollback_trace": has_token("rollback_reason", "rollback", "kept_baseline", "fallback_to_incumbent", "no_regression"),
        "has_policy_bank_trace": has_token("policy_bank", "policies", "policy_lanes", "multi_policy"),
        "has_row_candidate_trace": has_token("row_candidate", "best_candidate", "candidate_len", "baseline_len", "row_decision"),
        "has_shadow_split_metrics": has_token("train_score", "dev_score", "holdout_score", "shadow_split"),
        "has_exact_replay_gate": has_token("exact replay", "replay_to_final", "central_state", "final_state", "apply_moves"),
        "has_official_move_gate": has_token("allowed_moves", "legal_moves", "official move", "move-name", "generators"),
    }


def inspect_solver_code_ast_aware(code: str) -> Dict[str, Any]:
    tree = safe_parse_ast(code)
    symbols = collect_ast_symbols(tree)
    probe = capability_probe_from_ast_and_text(code, symbols)
    low = (code or "").lower()
    constants = {
        "short_table_depth": parse_small_int(code, "_SHORT_TABLE_DEPTH"),
        "local_window": parse_small_int(code, "_LOCAL_WINDOW"),
        "optimization_passes": parse_small_int(code, "_OPTIMIZATION_PASSES"),
    }
    uses_lookup_first = (
        "optimized_lookup" in low
        or "lookup.get(state_key)" in low
        or has_any_symbol(symbols, ["_best_lookup", "_load_lookup_from_submission", "_load_submission_lookup", "optimized_lookup", "load_lookup"])
        or any(name in low for name in ("_lookup_cache", "submission_candidates", "optimized_submission.csv"))
    )
    return {
        "inspection_mode": "ast_plus_capability_probe",
        "ast_parse_ok": symbols["ast_parse_ok"],
        "uses_lookup_first": uses_lookup_first,
        "has_exact_short_atlas": has_any_symbol(symbols, ["_short_word_data", "build_short_table", "short_word_atlas"]) or "short_table_depth" in low,
        "has_local_window_dp": has_any_symbol(symbols, ["_optimize_local_windows", "optimize_local_windows"]) or ("dp =" in low and "nxt =" in low),
        "has_multi_policy_sweep": probe["has_policy_bank_trace"] or has_any_symbol(symbols, ["policy_bank", "run_policy", "multi_policy_sweep"]),
        "has_bidirectional_rewrite": has_any_symbol(symbols, ["right_first", "bidirectional_rewrite", "reverse_pass"]) or any(token in low for token in ("right-to-left", "right_first", "reverse pass", "reversed(")),
        "has_macro_mining": has_any_symbol(symbols, ["mine_commutators", "mine_conjugates", "macro_atlas"]) or any(token in low for token in ("commutator", "conjugate", "macro atlas", "small_support", "small-support")),
        "has_candidate_bank_scoring": probe["has_row_candidate_trace"] or has_any_symbol(symbols, ["score_candidates", "candidate_bank", "best_candidate"]),
        "runtime_probe": probe,
        "ast_symbols": {
            "functions": symbols["functions"][:100],
            "classes": symbols["classes"][:40],
            "assigned_names": symbols["assigned_names"][:100],
            "call_names": symbols["call_names"][:100],
        },
        "fingerprint": solver_fingerprint(code),
        "constants": constants,
    }
