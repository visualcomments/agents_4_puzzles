from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MOD = SourceFileLoader(
    "megaminx_prompt_self_improver_score_guarded",
    str(ROOT / "competitions" / "cayley-py-megaminx" / "prompt_self_improver.py"),
).load_module()


def test_analyze_history_signals_detects_plateau_and_failures():
    history = [
        {"round": 1, "accepted": False, "error": "json decode error"},
        {"round": 2, "accepted": False, "error": "validator illegal move"},
        {"round": 3, "accepted": False, "metric": {"source": "local_score", "value": 123}},
    ]
    signals = MOD.analyze_history_signals(history)
    assert signals["plateau"] is True
    assert signals["json_contract_failures"] == 1
    assert signals["validation_failures"] == 1
    assert signals["validated_not_selected"] == 1


def test_select_directives_uses_history_signals():
    feature_snapshot = {
        "has_multi_policy_sweep": False,
        "has_bidirectional_rewrite": False,
        "has_macro_mining": False,
        "has_exact_short_atlas": False,
        "has_candidate_bank_scoring": False,
        "constants": {"short_table_depth": 5},
    }
    history = [
        {"round": 1, "accepted": False, "error": "validator illegal move"},
        {"round": 2, "accepted": False, "metric": {"source": "local_score", "value": 123}},
        {"round": 3, "accepted": False, "metric": {"source": "local_score", "value": 124}},
    ]
    directives = MOD.select_directives(
        feature_snapshot=feature_snapshot,
        round_idx=3,
        prompt_history=[],
        score_history=history,
    )
    keys = {item.key for item in directives}
    assert "score_regression_guard" in keys
    assert "semantic_equivalence_replay" in keys or "policy_ablation_search" in keys
