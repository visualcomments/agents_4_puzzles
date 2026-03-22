import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
AGENTLAB_ROOT = ROOT / 'AgentLaboratory'
if str(AGENTLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTLAB_ROOT))

from perm_pipeline import run_perm_pipeline as rpp  # type: ignore


def _payload(edit_targets, proposed_changes, validation_plan, *, notes=''):
    return {
        'strategy_family': 'stronger_exact_table',
        'goal': 'Improve local exact replacements while preserving polynomial-time bounded behavior.',
        'patch_scope': 'Minimal patch over baseline solver.',
        'edit_targets': list(edit_targets),
        'must_preserve': list(rpp.COMMON_PLAN_MUST_PRESERVE),
        'complexity_claim': {
            'precompute': 'O(1) fixed-radius tables only.',
            'per_row': 'O(n) bounded local rewrites.',
            'why_polynomial': 'All radii, pass counts, and candidate counts are constants.',
        },
        'proposed_changes': list(proposed_changes),
        'validation_plan': list(validation_plan),
        'forbidden': list(rpp.COMMON_PLAN_FORBIDDEN),
        'notes': notes,
    }


def test_combined_plan_score_rewards_stricter_refinement():
    parent_payload = _payload(
        ['_short_word_data'],
        ['Strengthen fixed-depth table construction.'],
        ['Compile solver and validate bundled rows.'],
    )
    parent_text = rpp._render_structured_plan(parent_payload)
    parent_score = rpp._combined_plan_score(parent_text, parent_payload)
    parent = rpp.PlanCandidate(
        plan_text=parent_text,
        planner_model='g4f:test',
        score=parent_score,
        variant_index=1,
        planner_payload=parent_payload,
        strategy_package=rpp._strategy_package_for_variant(1),
        prompt_score=parent_score,
    )

    richer_payload = _payload(
        ['_short_word_data', '_optimize_word'],
        ['Strengthen fixed-depth table construction.', 'Canonicalize equivalent short effects before storing them.'],
        ['Compile solver and validate bundled rows.', 'Replay bundled rows and compare final_state against baseline semantics.'],
        notes='Avoid the previous regression where the validator passed but the plan lacked stronger validation coverage.',
    )
    richer_text = rpp._render_structured_plan(richer_payload)
    richer_score = rpp._combined_plan_score(richer_text, richer_payload, parent=parent, archive_summary='previous failure: weak validation plan')
    assert richer_score > parent.score + 4.0

    same_score = rpp._combined_plan_score(parent_text, parent_payload, parent=parent, archive_summary='same failure memory')
    assert same_score < parent.score + 4.0


def test_generate_refined_plan_candidates_rejects_non_improving_and_accepts_improving(monkeypatch):
    parent_payload = _payload(
        ['_short_word_data'],
        ['Strengthen fixed-depth table construction.'],
        ['Compile solver and validate bundled rows.'],
    )
    parent_text = rpp._render_structured_plan(parent_payload)
    parent_score = rpp._combined_plan_score(parent_text, parent_payload)
    parent = rpp.PlanCandidate(
        plan_text=parent_text,
        planner_model='g4f:test',
        score=parent_score,
        variant_index=1,
        planner_payload=parent_payload,
        strategy_package=rpp._strategy_package_for_variant(1),
        prompt_score=parent_score,
    )

    same_payload_json = json.dumps(parent_payload)
    monkeypatch.setattr(rpp, '_query_model_stable', lambda *args, **kwargs: same_payload_json)
    rejected = rpp.generate_refined_plan_candidates(
        ['g4f:test'],
        'user task',
        'planner system',
        parent_candidates=[parent],
        beam_width=1,
        archive_summary='failure: plan was not concrete enough',
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
    )
    assert rejected == []

    improved_payload = _payload(
        ['_short_word_data', '_optimize_word'],
        ['Strengthen fixed-depth table construction.', 'Canonicalize equivalent short effects before storing them.'],
        ['Compile solver and validate bundled rows.', 'Replay bundled rows and compare final_state against baseline semantics.'],
        notes='Avoid the prior weak-validation failure by explicitly replaying bundled rows.',
    )
    improved_json = json.dumps(improved_payload)
    monkeypatch.setattr(rpp, '_query_model_stable', lambda *args, **kwargs: improved_json)
    accepted = rpp.generate_refined_plan_candidates(
        ['g4f:test'],
        'user task',
        'planner system',
        parent_candidates=[parent],
        beam_width=1,
        archive_summary='failure: weak validation plan',
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
    )
    assert len(accepted) == 1
    assert accepted[0].score > parent.score + 4.0
    assert accepted[0].parent_signature
