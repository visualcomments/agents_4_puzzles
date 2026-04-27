from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "AgentLaboratory"))
sys.path.insert(0, str(ROOT / "AgentLaboratory" / "perm_pipeline"))

import run_perm_pipeline as rpp  # type: ignore


def test_hybrid_refinement_does_not_index_missing_round(monkeypatch, tmp_path):
    initial_plan = rpp.PlanCandidate(
        plan_text="initial plan",
        planner_model="g4f:r1-1776",
        score=1.0,
        variant_index=1,
    )
    responses = [[initial_plan], []]

    def fake_generate_plan_candidates(*args, **kwargs):
        return responses.pop(0) if responses else []

    monkeypatch.setattr(rpp, "generate_plan_candidates", fake_generate_plan_candidates)
    monkeypatch.setattr(rpp, "build_plan_model_frontier", lambda plans, coder_models, frontier_width: [(plans[0], coder_models[0])])
    monkeypatch.setattr(rpp, "try_generate_with_model", lambda **kwargs: (False, "g4f:r1-1776: fixer iteration 353 credentials required (Add a \"api_key\")"))

    ok, reports, archive, plan, planner_model, winner_model = rpp.run_hybrid_codegen_search(
        planner_models=["g4f:r1-1776"],
        coder_models=["g4f:r1-1776"],
        fixer_models=["g4f:r1-1776"],
        user_prompt="solve it",
        prompts={"planner": "planner", "coder": "coder", "fixer": "fixer"},
        out_path=tmp_path / "solve.py",
        validator_path=tmp_path / "validator.py",
        tests=[[1, 2, 3]],
        max_iters=2,
        baseline_code="def solve(vec):\n    return [], list(vec)\n",
        plan_beam_width=1,
        frontier_width=1,
        archive_size=2,
        refine_rounds=1,
    )

    assert ok is False
    assert winner_model is None
    assert plan == "initial plan"
    assert planner_model == "g4f:r1-1776"
    assert reports
    assert "credentials required" in reports[0].lower()
