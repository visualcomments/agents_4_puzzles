from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _safe_read_text(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


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
    "score_regression_guard": Directive(
        key="score_regression_guard",
        title="Score-regression guard",
        instruction=(
            "Introduce an explicit rollback / acceptance guard: risky optimization stages must keep the previous exact-valid candidate when they fail to shorten a fixed deterministic evaluation shard or violate exact replay equivalence."
        ),
        rationale="Many prompt rounds are syntactically valid but score-regressive; the model should be forced to design safe fallback logic instead of assuming every new optimizer stage helps.",
        priority=98,
    ),
    "semantic_equivalence_replay": Directive(
        key="semantic_equivalence_replay",
        title="Semantic-equivalence replay",
        instruction=(
            "After each non-trivial rewrite family, replay the optimized local word under official permutations and keep the rewrite only when the exact local effect matches and the word is strictly shorter."
        ),
        rationale="Algorithmic prompting works better when the solver proves semantic preservation programmatically instead of trusting heuristic transformations.",
        priority=95,
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
    "validator_triad_recheck": Directive(
        key="validator_triad_recheck",
        title="Validator triad recheck",
        instruction=(
            "Harden the final contract with a deterministic three-part check: compile/import health, legal official move-name check, and exact replay-to-final-state verification before returning the candidate result."
        ),
        rationale="For this repository, the best process supervision signal is a strict validator-aware triad.",
        priority=88,
    ),
    "compile_first_then_optimize": Directive(
        key="compile_first_then_optimize",
        title="Compile first, then optimize",
        instruction=(
            "Stage the patch so that the lookup-first baseline path stays intact and import-safe before adding heavier optimizer deltas; when a later stage is uncertain, fail back to the exact baseline-compatible path rather than breaking the whole solver."
        ),
        rationale="Rounds that mix architectural rewrites and fragile plumbing often fail for avoidable reasons; forcing a compile-safe baseline-preserving layer improves outer-loop efficiency.",
        priority=86,
    ),
    "policy_ablation_search": Directive(
        key="policy_ablation_search",
        title="Policy ablation search",
        instruction=(
            "Use a tiny fixed ablation bank of optimizer policy packages and keep per-row or per-window winners under a deterministic local score proxy, rather than betting on a single composite heuristic."
        ),
        rationale="APE-style prompt search is a strong fit for this repository because different deterministic policy mixes win on different bundled rows.",
        priority=84,
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
        priority=72,
    ),
    "candidate_bank_scoring": Directive(
        key="candidate_bank_scoring",
        title="Deterministic candidate bank scoring",
        instruction=(
            "Construct a small deterministic bank of optimizer variants, score them with a local exact-validity-preserving proxy, and keep only the best valid result."
        ),
        rationale="The solver itself should exploit a bounded candidate bank instead of committing to a single rewrite schedule.",
        priority=65,
    ),
    "prompt_population_search": Directive(
        key="prompt_population_search",
        title="Prompt-population search",
        instruction=(
            "Treat this round as one member of a small prompt population instead of a single linear patch chain: produce a candidate that is deliberately differentiated from recent accepted rounds in search policy, acceptance guard design, and candidate-ranking logic."
        ),
        rationale="Promptbreeder-style diversity is a better fit for algorithmic plateau breaking than repeatedly paraphrasing the last accepted prompt.",
        priority=110,
    ),
    "exact_evaluator_shard": Directive(
        key="exact_evaluator_shard",
        title="Exact evaluator shard",
        instruction=(
            "Introduce or strengthen a tiny exact evaluation shard inside the generated solver workflow: every risky optimizer stage must justify itself against a fixed deterministic replay-based micro-benchmark before replacing the previous exact-valid candidate."
        ),
        rationale="Algorithmic improvement should be driven by executable validation and score deltas, not by prose confidence alone.",
        priority=108,
    ),
    "solver_archive_lineage": Directive(
        key="solver_archive_lineage",
        title="Solver archive and lineage",
        instruction=(
            "Preserve explicit lineage and candidate provenance in the design: keep track of baseline, patch lane, fresh lane, acceptance metric, and why the winning candidate survived."
        ),
        rationale="AlphaEvolve-style improvement works better when each accepted candidate is grounded in verifiable lineage rather than hidden prompt drift.",
        priority=104,
    ),
    "patch_fresh_lane_split": Directive(
        key="patch_fresh_lane_split",
        title="Patch/fresh lane split",
        instruction=(
            "Ask for two bounded candidate lanes: a minimal patch over the best validated solver and a deliberately fresh candidate that preserves the public contract but is not anchored to the same local rewrite ordering."
        ),
        rationale="A mixed patch-vs-fresh proposal bank reduces premature convergence while keeping the repository's contract-safe bias.",
        priority=102,
    ),
    "pareto_candidate_selection": Directive(
        key="pareto_candidate_selection",
        title="Pareto candidate selection",
        instruction=(
            "Select winners using a simple Pareto-style view over exact validity, score delta, runtime risk, and novelty instead of a one-dimensional preference for whichever patch sounds strongest."
        ),
        rationale="Megaminx prompt evolution should trade off score, safety, and diversity explicitly when search begins to plateau.",
        priority=101,
    ),
}


def inspect_solver_code(code: str) -> Dict[str, Any]:
    low = (code or "").lower()
    constants = {
        "short_table_depth": _parse_small_int(code, "_SHORT_TABLE_DEPTH"),
        "local_window": _parse_small_int(code, "_LOCAL_WINDOW"),
        "optimization_passes": _parse_small_int(code, "_OPTIMIZATION_PASSES"),
    }
    return {
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


def _recent_directives(prompt_history: Sequence[Dict[str, Any]], *, limit: int = 2) -> set[str]:
    used: set[str] = set()
    for entry in list(prompt_history)[-max(0, limit):]:
        for item in entry.get("selected_directives", []) or []:
            key = str(item or "").strip()
            if key:
                used.add(key)
    return used


def _error_bucket(entry: Dict[str, Any]) -> str | None:
    text = str(entry.get("error") or "").lower()
    if not text:
        return None
    if any(token in text for token in ("json", "decode", "strict output", "fence")):
        return "json_contract"
    if any(token in text for token in ("validator", "illegal move", "unsolved", "final state", "replay")):
        return "validation"
    if any(token in text for token in ("syntax", "import", "traceback", "exception", "compile", "typeerror", "nameerror")):
        return "runtime"
    return "other"


def analyze_history_signals(history: Sequence[Dict[str, Any]], *, limit: int = 6) -> Dict[str, Any]:
    window = list(history)[-max(0, limit):]
    signals: Dict[str, Any] = {
        "window": len(window),
        "recent_accepts": 0,
        "recent_failures": 0,
        "json_contract_failures": 0,
        "validation_failures": 0,
        "runtime_failures": 0,
        "other_failures": 0,
        "validated_not_selected": 0,
        "plateau": False,
        "score_regressions": 0,
        "recent_failure_labels": [],
        "accepted_rounds": 0,
        "accepted_metric_values": [],
        "accepted_metric_plateau": False,
    }
    if not window:
        return signals

    labels: List[str] = []
    for entry in window:
        if entry.get("accepted") is True:
            signals["recent_accepts"] += 1
            signals["accepted_rounds"] += 1
            metric = entry.get("metric") if isinstance(entry.get("metric"), dict) else {}
            if metric.get("source") == "local_score" and metric.get("value") is not None:
                signals["accepted_metric_values"].append(metric.get("value"))
            continue
        bucket = _error_bucket(entry)
        if bucket is not None:
            signals["recent_failures"] += 1
            labels.append(bucket)
            if bucket == "json_contract":
                signals["json_contract_failures"] += 1
            elif bucket == "validation":
                signals["validation_failures"] += 1
            elif bucket == "runtime":
                signals["runtime_failures"] += 1
            else:
                signals["other_failures"] += 1
            continue
        signals["validated_not_selected"] += 1
        metric = entry.get("metric") if isinstance(entry.get("metric"), dict) else {}
        if metric.get("source") == "local_score" and metric.get("value") is not None:
            signals["score_regressions"] += 1

    tail = window[-3:]
    if tail and all(entry.get("accepted") is False for entry in tail):
        signals["plateau"] = True
    accepted_metric_values = [float(v) for v in signals.get("accepted_metric_values") or [] if v is not None]
    if len(accepted_metric_values) >= 2 and max(accepted_metric_values) - min(accepted_metric_values) <= 1.0:
        signals["accepted_metric_plateau"] = True
    signals["recent_failure_labels"] = labels[-3:]
    return signals


def select_directives(
    *,
    feature_snapshot: Dict[str, Any],
    round_idx: int,
    prompt_history: Sequence[Dict[str, Any]],
    score_history: Sequence[Dict[str, Any]],
) -> List[Directive]:
    candidates: Dict[str, Directive] = {}
    constants = feature_snapshot.get("constants") if isinstance(feature_snapshot.get("constants"), dict) else {}
    history_signals = analyze_history_signals(score_history)

    def add(key: str) -> None:
        candidates[key] = DIRECTIVE_LIBRARY[key]

    if not feature_snapshot.get("has_multi_policy_sweep"):
        add("multi_policy_sweep")
    if not feature_snapshot.get("has_bidirectional_rewrite"):
        add("bidirectional_window_rewrite")
    if not feature_snapshot.get("has_macro_mining"):
        add("small_support_macro_mining")
    if not feature_snapshot.get("has_exact_short_atlas") or int(constants.get("short_table_depth") or 0) <= 5:
        add("stronger_effect_atlas")
    if not feature_snapshot.get("has_candidate_bank_scoring"):
        add("candidate_bank_scoring")

    plateau = bool(history_signals.get("plateau") or history_signals.get("accepted_metric_plateau") or history_signals.get("validated_not_selected"))
    if plateau or round_idx >= 2:
        add("prompt_population_search")
        add("exact_evaluator_shard")
        add("solver_archive_lineage")
        add("patch_fresh_lane_split")
        add("pareto_candidate_selection")

    if history_signals.get("plateau"):
        add("policy_ablation_search")
        add("score_regression_guard")
        add("semantic_equivalence_replay")
    if history_signals.get("validation_failures") or history_signals.get("runtime_failures"):
        add("compile_first_then_optimize")
        add("validator_triad_recheck")
        add("exact_evaluator_shard")
    if history_signals.get("json_contract_failures"):
        add("validator_triad_recheck")
        add("compile_first_then_optimize")
    if history_signals.get("score_regressions"):
        add("score_regression_guard")
        add("semantic_equivalence_replay")
        add("pareto_candidate_selection")

    if not candidates:
        for key in ("multi_policy_sweep", "bidirectional_window_rewrite", "stronger_effect_atlas"):
            add(key)

    ordered = sorted(candidates.values(), key=lambda item: (-item.priority, item.key))
    used_recently = _recent_directives(prompt_history)
    fresh = [item for item in ordered if item.key not in used_recently]
    core_required: list[str] = []
    for key in ("multi_policy_sweep", "bidirectional_window_rewrite", "small_support_macro_mining"):
        if key in candidates:
            core_required.append(key)
    if history_signals.get("score_regressions") or history_signals.get("plateau"):
        core_required.append("score_regression_guard")
        if "semantic_equivalence_replay" in candidates:
            core_required.append("semantic_equivalence_replay")
        elif "policy_ablation_search" in candidates:
            core_required.append("policy_ablation_search")
    target_size = 8 if (plateau or history_signals.get("recent_failures") or round_idx >= 2) else 4
    target_size = max(target_size, len(core_required))
    selected = fresh[:target_size]
    if len(selected) < target_size:
        for item in ordered:
            if item not in selected:
                selected.append(item)
            if len(selected) >= target_size:
                break
    for key in core_required:
        directive = DIRECTIVE_LIBRARY[key]
        if directive in selected:
            continue
        if len(selected) < target_size:
            selected.append(directive)
            continue
        replaced = False
        for idx in range(len(selected) - 1, -1, -1):
            if selected[idx].key not in core_required:
                selected[idx] = directive
                replaced = True
                break
        if not replaced:
            selected.append(directive)
    deduped: list[Directive] = []
    seen_keys: set[str] = set()
    for item in selected:
        if item.key in seen_keys:
            continue
        deduped.append(item)
        seen_keys.add(item.key)
    return deduped


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
    rendered = ", ".join(f"{k}={v}" for k, v in constants.items() if v is not None)
    if rendered:
        lines.append(f"- extracted constants: {rendered}")
    return "\n".join(lines)


def _history_block(history: Sequence[Dict[str, Any]]) -> str:
    items = summarize_history(history)
    if not items:
        return "Recent round memory:\n- no prior rounds yet"
    return "Recent round memory:\n- " + "\n- ".join(items)


def _history_signal_block(history: Sequence[Dict[str, Any]]) -> str:
    signals = analyze_history_signals(history)
    labels = ", ".join(signals.get("recent_failure_labels") or []) or "none"
    return "\n".join(
        [
            "Recent failure / plateau signals:",
            f"- recent_accepts: {signals.get('recent_accepts')}",
            f"- recent_failures: {signals.get('recent_failures')} (json={signals.get('json_contract_failures')}, validation={signals.get('validation_failures')}, runtime={signals.get('runtime_failures')})",
            f"- validated_not_selected: {signals.get('validated_not_selected')}",
            f"- plateau_detected: {'yes' if signals.get('plateau') else 'no'}",
            f"- accepted_metric_plateau: {'yes' if signals.get('accepted_metric_plateau') else 'no'}",
            f"- score_regressions: {signals.get('score_regressions')}",
            f"- recent_failure_labels: {labels}",
        ]
    )


def _algorithm_search_block(round_idx: int) -> str:
    return "\n".join(
        [
            "Algorithm-search operating mode:",
            f"- round: {round_idx}",
            "- think in terms of a tiny population of bounded candidates, not one fragile linear rewrite",
            "- require an exact evaluator shard or replay-based acceptance gate before risky optimizer stages survive",
            "- keep two lanes when useful: minimal patch lane and fresh-contract-safe lane",
            "- preserve lineage: baseline fingerprint, candidate type, acceptance metric, and rollback reason",
            "- optimize for a simple Pareto frontier over exact validity, score delta, runtime risk, and novelty",
            "- never use Kaggle leaderboard probing as an inner-loop reward; rely on bundled deterministic evaluation only",
        ]
    )


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
    directives = select_directives(
        feature_snapshot=feature_snapshot,
        round_idx=round_idx,
        prompt_history=prompt_history,
        score_history=score_history,
    )
    evolution_block = "\n\n".join(
        [
            f"SELF-IMPROVEMENT ROUND {round_idx}",
            "The injected baseline for this round is the previous best validated solver. Do not merely rename helpers or tweak constants; produce a materially stronger alternative while preserving the public Megaminx competition contract.",
            f"Current best selection metric: {_best_metric_text(best_metric)}",
            _feature_block(feature_snapshot),
            _history_block(score_history),
            _history_signal_block(score_history),
            _algorithm_search_block(round_idx),
            _directive_block(directives),
            "Acceptance intent for this round: keep exact lookup first, preserve legal official move names only, preserve deterministic replay, add an explicit anti-regression fallback, maintain candidate lineage, and materially revise the local optimization core so that the resulting solver has a real chance to beat the previous answer rather than paraphrase it.",
            "Planner quality bar for this round: prefer compile-safe staged patches, require exact semantic-equivalence replay for every new local rewrite family, design a tiny prompt/candidate population instead of a single chain, and build a deterministic policy bank instead of trusting one heuristic ordering.",
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
            f"Current best metric: {best_metric_text}. Include an anti-regression story, an exact evaluator-shard story, and a small prompt-population / candidate-lineage story."
        ),
        "coder": (
            f"SELF-IMPROVEMENT ROUND {round_idx}: the injected baseline is the previous best validated solver. "
            f"Do not produce a superficial patch. Make the optimization core materially stronger around: {directive_summary}. "
            "Preserve lookup-first semantics, legal move names only, deterministic replay, explicit rollback-safe score guarding, competition-safe bounded search, and a bounded patch-vs-fresh candidate split backed by exact replay-based acceptance."
        ),
        "fixer": (
            f"SELF-IMPROVEMENT ROUND {round_idx}: when repairing the candidate, preserve the newly requested architecture deltas ({directive_summary}) instead of collapsing back to the previous answer. "
            "Fix only what is necessary for correctness and validation, and preserve any replay-equivalence, evaluator-shard, lineage, or score-guard logic that was added on purpose."
        ),
    }
    for role, addition in per_role_additions.items():
        role_text = str(payload.get(role) or "").strip()
        if not role_text:
            continue
        payload[role] = _insert_before_strict_contract(role_text, addition)
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
        "history_signals": analyze_history_signals(score_history),
    }
    meta_path = output_dir / f"round_{round_idx:04d}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "prompt_file": prompt_file,
        "custom_prompts_file": custom_file,
        "meta": meta,
        "meta_file": meta_path,
    }
