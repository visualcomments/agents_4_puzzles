from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _safe_read_text(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _clip(text: str, limit: int = 1200) -> str:
    body = str(text or "")
    if len(body) <= limit:
        return body
    head = max(80, limit // 2)
    tail = max(80, limit - head)
    return body[:head] + "\n...\n" + body[-tail:]


def _solver_fingerprint(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()[:16]


def _parse_small_int(code: str, name: str) -> Optional[int]:
    pattern = rf"^\s*{re.escape(name)}\s*=\s*([0-9]+)\s*$"
    match = re.search(pattern, code or "", re.MULTILINE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


@dataclass(frozen=True)
class Directive:
    key: str
    title: str
    instruction: str
    rationale: str
    priority: int


DIRECTIVE_LIBRARY: Dict[str, Directive] = {
    "multi_policy_sweep": Directive(
        key="multi_policy_sweep",
        title="Multi-policy candidate sweep",
        instruction=(
            "Introduce a tiny fixed bank of deterministic commuting-order / rewrite policies and choose the shortest valid equivalent candidate per row, "
            "instead of relying on a single normalization order."
        ),
        rationale="Different Megaminx rows compress under different legal commuting orders, so one global order leaves score on the table.",
        priority=100,
    ),
    "bidirectional_window_rewrite": Directive(
        key="bidirectional_window_rewrite",
        title="Bidirectional bounded local rewrite",
        instruction=(
            "Add a stronger bounded local rewrite layer that evaluates both left-to-right and right-to-left exact window replacements with a tiny fixed pass budget."
        ),
        rationale="The current local DP often depends on scan direction; a deterministic bidirectional pass can improve local minima without changing asymptotic guarantees.",
        priority=90,
    ),
    "small_support_macro_mining": Directive(
        key="small_support_macro_mining",
        title="Small-support commutator/conjugate mining",
        instruction=(
            "Mine only bounded short commutators/conjugates with small official permutation support and use them strictly as exact local rewrite candidates, never as fake move names."
        ),
        rationale="Megaminx local word compression benefits from structured short effects that do not appear in a plain same-face / commuting rewrite atlas.",
        priority=80,
    ),
    "stronger_effect_atlas": Directive(
        key="stronger_effect_atlas",
        title="Stronger exact short-word atlas",
        instruction=(
            "Strengthen the fixed-radius exact effect atlas: keep the shortest representative per local effect, canonicalize equivalent short words before storage, and memoize packed effects."
        ),
        rationale="A weak exact atlas makes the downstream optimizer blind to many safe equivalences already available at small radius.",
        priority=70,
    ),
    "candidate_bank_scoring": Directive(
        key="candidate_bank_scoring",
        title="Deterministic candidate bank scoring",
        instruction=(
            "Construct a small deterministic bank of optimizer variants, score them with a local exact-validity-preserving proxy, and keep only the best valid result."
        ),
        rationale="The repository already supports keep-improving loops; the solver itself should also exploit a bounded candidate bank instead of committing to a single rewrite schedule.",
        priority=60,
    ),
}


def inspect_solver_code(code: str) -> Dict[str, Any]:
    low = (code or "").lower()
    constants = {
        "short_table_depth": _parse_small_int(code, "_SHORT_TABLE_DEPTH"),
        "local_window": _parse_small_int(code, "_LOCAL_WINDOW"),
        "optimization_passes": _parse_small_int(code, "_OPTIMIZATION_PASSES"),
    }
    features = {
        "uses_lookup_first": "optimized_lookup" in low or "lookup.get(state_key)" in low,
        "has_exact_short_atlas": "_short_word_data" in low or "short_table_depth" in low,
        "has_local_window_dp": "_optimize_local_windows" in low or ("dp =" in low and "nxt =" in low),
        "has_multi_policy_sweep": any(token in low for token in ("policy", "policies", "candidate optimizers", "candidate bank", "best-of-fixed-candidates", "first-occurrence", "last-occurrence")),
        "has_bidirectional_rewrite": any(token in low for token in ("right-to-left", "right_first", "reverse pass", "reversed(", "left-first", "right-first")),
        "has_macro_mining": any(token in low for token in ("commutator", "conjugate", "macro atlas", "small_support", "small-support")),
        "has_candidate_bank_scoring": any(token in low for token in ("candidate optimizers", "local score proxy", "best_candidate", "candidate_metric", "score proxy")),
        "fingerprint": _solver_fingerprint(code),
        "constants": constants,
    }
    return features


def _recent_directives(prompt_history: Sequence[Dict[str, Any]], *, limit: int = 2) -> set[str]:
    used: set[str] = set()
    for entry in list(prompt_history)[-max(0, limit):]:
        for item in entry.get("selected_directives", []) or []:
            key = str(item or "").strip()
            if key:
                used.add(key)
    return used


def select_directives(
    *,
    feature_snapshot: Dict[str, Any],
    round_idx: int,
    prompt_history: Sequence[Dict[str, Any]],
) -> List[Directive]:
    candidates: List[Directive] = []
    constants = feature_snapshot.get("constants") if isinstance(feature_snapshot.get("constants"), dict) else {}
    if not feature_snapshot.get("has_multi_policy_sweep"):
        candidates.append(DIRECTIVE_LIBRARY["multi_policy_sweep"])
    if not feature_snapshot.get("has_bidirectional_rewrite"):
        candidates.append(DIRECTIVE_LIBRARY["bidirectional_window_rewrite"])
    if not feature_snapshot.get("has_macro_mining"):
        candidates.append(DIRECTIVE_LIBRARY["small_support_macro_mining"])
    if not feature_snapshot.get("has_exact_short_atlas") or int(constants.get("short_table_depth") or 0) <= 5:
        candidates.append(DIRECTIVE_LIBRARY["stronger_effect_atlas"])
    if not feature_snapshot.get("has_candidate_bank_scoring"):
        candidates.append(DIRECTIVE_LIBRARY["candidate_bank_scoring"])

    if not candidates:
        candidates = [
            DIRECTIVE_LIBRARY["multi_policy_sweep"],
            DIRECTIVE_LIBRARY["bidirectional_window_rewrite"],
            DIRECTIVE_LIBRARY["stronger_effect_atlas"],
        ]

    used_recently = _recent_directives(prompt_history)
    fresh = [item for item in sorted(candidates, key=lambda item: (-item.priority, item.key)) if item.key not in used_recently]
    selected = fresh[:3]
    if len(selected) < 3:
        for item in sorted(candidates, key=lambda item: (-item.priority, item.key)):
            if item not in selected:
                selected.append(item)
            if len(selected) >= 3:
                break

    if round_idx >= 3 and DIRECTIVE_LIBRARY["candidate_bank_scoring"] not in selected:
        selected[-1] = DIRECTIVE_LIBRARY["candidate_bank_scoring"]
    return selected[:3]


def summarize_history(history: Sequence[Dict[str, Any]], *, limit: int = 3) -> List[str]:
    out: List[str] = []
    for entry in list(history)[-max(0, limit):]:
        round_idx = entry.get("round")
        if entry.get("accepted") is True:
            metric = entry.get("metric") if isinstance(entry.get("metric"), dict) else {}
            source = metric.get("source")
            value = metric.get("value")
            if source is not None and value is not None:
                out.append(f"round {round_idx}: accepted via {source}={value}")
            else:
                out.append(f"round {round_idx}: accepted validated solver")
            continue
        if entry.get("error"):
            out.append(f"round {round_idx}: rejected after failure {entry.get('error')}")
            continue
        metric = entry.get("metric") if isinstance(entry.get("metric"), dict) else {}
        metric_desc = f" {metric.get('source')}={metric.get('value')}" if metric else ""
        out.append(f"round {round_idx}: validated but not selected{metric_desc}")
    return out


def _insert_before_strict_contract(base_text: str, addition: str) -> str:
    marker = "STRICT OUTPUT REQUIREMENTS"
    idx = base_text.find(marker)
    if idx < 0:
        return (base_text.rstrip() + "\n\n" + addition.strip() + "\n").strip() + "\n"
    prefix = base_text[:idx].rstrip()
    suffix = base_text[idx:].lstrip()
    return prefix + "\n\n" + addition.strip() + "\n\n" + suffix


def _best_metric_text(best_metric: Optional[Dict[str, Any]]) -> str:
    if not isinstance(best_metric, dict) or not best_metric:
        return "unavailable"
    source = best_metric.get("source") or "metric"
    value = best_metric.get("value")
    if value is None:
        return str(source)
    return f"{source}={value}"


def _directive_block(directives: Sequence[Directive]) -> str:
    lines = ["Required architecture deltas for this round:"]
    for idx, directive in enumerate(directives, start=1):
        lines.append(f"{idx}. {directive.title}: {directive.instruction}")
        lines.append(f"   Why: {directive.rationale}")
    return "\n".join(lines)


def _feature_block(feature_snapshot: Dict[str, Any]) -> str:
    constants = feature_snapshot.get("constants") if isinstance(feature_snapshot.get("constants"), dict) else {}
    lines = [
        "Current baseline diagnosis:",
        f"- fingerprint: {feature_snapshot.get('fingerprint')}",
        f"- exact lookup first: {'yes' if feature_snapshot.get('uses_lookup_first') else 'no'}",
        f"- exact short-effect atlas: {'yes' if feature_snapshot.get('has_exact_short_atlas') else 'no'}",
        f"- bounded local DP: {'yes' if feature_snapshot.get('has_local_window_dp') else 'no'}",
        f"- multi-policy sweep: {'yes' if feature_snapshot.get('has_multi_policy_sweep') else 'no'}",
        f"- bidirectional rewrite: {'yes' if feature_snapshot.get('has_bidirectional_rewrite') else 'no'}",
        f"- macro mining: {'yes' if feature_snapshot.get('has_macro_mining') else 'no'}",
        f"- candidate-bank scoring: {'yes' if feature_snapshot.get('has_candidate_bank_scoring') else 'no'}",
    ]
    if constants:
        rendered = ", ".join(f"{k}={v}" for k, v in constants.items() if v is not None)
        if rendered:
            lines.append(f"- extracted constants: {rendered}")
    return "\n".join(lines)


def _history_block(history: Sequence[Dict[str, Any]]) -> str:
    items = summarize_history(history)
    if not items:
        return "Recent round memory:\n- no prior rounds yet"
    return "Recent round memory:\n- " + "\n- ".join(items)


def synthesize_round_prompt_text(
    *,
    base_prompt_text: str,
    baseline_code: str,
    round_idx: int,
    score_history: Sequence[Dict[str, Any]],
    best_metric: Optional[Dict[str, Any]],
    prompt_history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    feature_snapshot = inspect_solver_code(baseline_code)
    directives = select_directives(feature_snapshot=feature_snapshot, round_idx=round_idx, prompt_history=prompt_history)
    evolution_block = "\n\n".join(
        [
            f"SELF-IMPROVEMENT ROUND {round_idx}",
            "The injected baseline for this round is the previous best validated solver. Do not merely rename helpers or tweak constants; produce a materially stronger alternative while preserving the public Megaminx competition contract.",
            f"Current best selection metric: {_best_metric_text(best_metric)}",
            _feature_block(feature_snapshot),
            _history_block(score_history),
            _directive_block(directives),
            "Acceptance intent for this round: keep exact lookup first, preserve legal official move names only, preserve deterministic replay, and significantly revise the local optimization core so that the resulting solver has a real chance to beat the previous answer rather than paraphrase it.",
        ]
    )
    text = _insert_before_strict_contract(base_prompt_text, evolution_block)
    return {
        "prompt_text": text,
        "feature_snapshot": feature_snapshot,
        "selected_directives": [directive.key for directive in directives],
        "selected_titles": [directive.title for directive in directives],
    }


def synthesize_round_custom_prompts(
    *,
    base_custom_prompts_text: str,
    round_idx: int,
    feature_snapshot: Dict[str, Any],
    selected_directives: Sequence[str],
    best_metric: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not str(base_custom_prompts_text or "").strip():
        return None
    try:
        payload = json.loads(base_custom_prompts_text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    titles = [DIRECTIVE_LIBRARY[key].title for key in selected_directives if key in DIRECTIVE_LIBRARY]
    directive_summary = "; ".join(titles)
    best_metric_text = _best_metric_text(best_metric)
    per_role_additions = {
        "planner": (
            f"SELF-IMPROVEMENT ROUND {round_idx}: the injected baseline is the previous best validated solver with fingerprint {feature_snapshot.get('fingerprint')}. "
            f"Return a materially stronger plan rather than a paraphrase. Force the plan to rethink the optimization core around: {directive_summary}. "
            f"Current best metric: {best_metric_text}."
        ),
        "coder": (
            f"SELF-IMPROVEMENT ROUND {round_idx}: the injected baseline is the previous best validated solver. "
            f"Do not produce a superficial patch. Make the optimization core materially stronger around: {directive_summary}. "
            "Preserve lookup-first semantics, legal move names only, deterministic replay, and competition-safe bounded search."
        ),
        "fixer": (
            f"SELF-IMPROVEMENT ROUND {round_idx}: when repairing the candidate, preserve the newly requested architecture deltas ({directive_summary}) instead of collapsing back to the previous answer. "
            "Fix only what is necessary for correctness and validation."
        ),
    }
    for role, addition in per_role_additions.items():
        text = str(payload.get(role) or "").strip()
        if not text:
            continue
        payload[role] = _insert_before_strict_contract(text, addition)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_round_prompt_bundle(
    *,
    base_prompt_file: Path,
    base_custom_prompts: Optional[Path],
    baseline_solver: Path,
    round_idx: int,
    score_history: Sequence[Dict[str, Any]],
    best_metric: Optional[Dict[str, Any]],
    prompt_history: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    base_prompt_text = _safe_read_text(base_prompt_file)
    base_custom_prompts_text = _safe_read_text(base_custom_prompts)
    baseline_code = _safe_read_text(baseline_solver)

    prompt_result = synthesize_round_prompt_text(
        base_prompt_text=base_prompt_text,
        baseline_code=baseline_code,
        round_idx=round_idx,
        score_history=score_history,
        best_metric=best_metric,
        prompt_history=prompt_history,
    )
    custom_prompts_text = synthesize_round_custom_prompts(
        base_custom_prompts_text=base_custom_prompts_text,
        round_idx=round_idx,
        feature_snapshot=prompt_result["feature_snapshot"],
        selected_directives=prompt_result["selected_directives"],
        best_metric=best_metric,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / f"round_{round_idx:04d}_user_prompt.txt"
    prompt_file.write_text(prompt_result["prompt_text"], encoding="utf-8")
    custom_file: Optional[Path] = None
    if custom_prompts_text is not None:
        custom_file = output_dir / f"round_{round_idx:04d}_custom_prompts.json"
        custom_file.write_text(custom_prompts_text, encoding="utf-8")

    meta = {
        "round": int(round_idx),
        "baseline_solver": str(baseline_solver),
        "baseline_fingerprint": prompt_result["feature_snapshot"].get("fingerprint"),
        "best_metric": best_metric,
        "selected_directives": list(prompt_result["selected_directives"]),
        "selected_titles": list(prompt_result["selected_titles"]),
        "feature_snapshot": prompt_result["feature_snapshot"],
        "prompt_file": str(prompt_file),
        "custom_prompts_file": str(custom_file) if custom_file is not None else None,
        "history_summary": summarize_history(score_history),
    }
    meta_path = output_dir / f"round_{round_idx:04d}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "prompt_file": prompt_file,
        "custom_prompts_file": custom_file,
        "meta": meta,
        "meta_file": meta_path,
    }
