from __future__ import annotations

from typing import Any, Dict, List, Sequence


DIRECTIVE_EVIDENCE_RULES: Dict[str, Dict[str, Any]] = {
    "multi_policy_sweep": {"required_capabilities": ["has_multi_policy_sweep", "runtime_probe.has_policy_bank_trace"], "evidence": "trace/code must expose a bounded policy bank, 2+ deterministic policy lanes, and selected_lane"},
    "candidate_bank_scoring": {"required_capabilities": ["has_candidate_bank_scoring", "runtime_probe.has_row_candidate_trace"], "evidence": "row-level candidate winner must be visible with baseline_len/candidate_len"},
    "score_regression_guard": {"required_capabilities": ["runtime_probe.has_rollback_trace"], "evidence": "rollback_reason must be visible whenever a candidate is not promoted"},
    "shadow_split_benchmarking": {"required_capabilities": ["runtime_probe.has_shadow_split_metrics"], "evidence": "train/dev/holdout exact scores must be reported separately"},
    "exact_metric_acceptance": {"required_capabilities": ["runtime_probe.has_exact_replay_gate"], "evidence": "promotion must depend on exact replay and deterministic dev/full score"},
    "validator_triad_recheck": {"required_capabilities": ["runtime_probe.has_official_move_gate", "runtime_probe.has_exact_replay_gate"], "evidence": "compile/import, official move, and exact replay gates must be explicit"},
    "patch_fresh_lane_split": {"required_capabilities": ["runtime_probe.has_lane_selection_trace"], "evidence": "trace must distinguish patch lane, fresh lane, and winning lane"},
    "portfolio_orchestration": {"required_capabilities": ["runtime_probe.has_lane_selection_trace", "runtime_probe.has_row_candidate_trace"], "evidence": "row-wise best-of-lanes fusion and lane provenance must be visible"},
    "per_row_delta_acceptance": {"required_capabilities": ["runtime_probe.has_row_candidate_trace", "runtime_probe.has_rollback_trace"], "evidence": "row-level improved/regressed/unchanged decision must be auditable"},
    "hard_row_routing": {"required_capabilities": ["runtime_probe.has_row_candidate_trace"], "evidence": "code must target explicitly listed hard rows and report per-row decisions"},
}


def _get_nested(mapping: Dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def build_evidence_checks(selected_directives: Sequence[Any], feature_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for item in selected_directives or []:
        key = item.get("key") if isinstance(item, dict) else str(item)
        rule = DIRECTIVE_EVIDENCE_RULES.get(key)
        if not rule:
            continue
        required = list(rule.get("required_capabilities", []))
        satisfied = []
        missing = []
        for capability in required:
            if _get_nested(feature_snapshot, capability):
                satisfied.append(capability)
            else:
                missing.append(capability)
        checks.append({
            "directive": key,
            "required_capabilities": required,
            "satisfied": satisfied,
            "missing": missing,
            "evidence": rule.get("evidence"),
            "currently_satisfied": not missing,
        })
    return checks


def evidence_prompt_block(evidence_checks: Sequence[Dict[str, Any]]) -> str:
    if not evidence_checks:
        return "Executable evidence requirements:\n- no directive-specific evidence checks selected"
    lines = ["Executable evidence requirements:"]
    for check in evidence_checks:
        lines.append(f"- {check.get('directive')}: {check.get('evidence')} (missing_in_baseline={check.get('missing')})")
    lines.extend([
        "- A directive that only appears in comments/prose is not implemented.",
        "- The generated solver must expose enough code or trace metadata for the outer loop to verify each selected directive.",
        "- Prefer one exact-valid row-level improvement with visible evidence over a broad rewrite that leaves submission bytes unchanged.",
    ])
    return "\n".join(lines)
