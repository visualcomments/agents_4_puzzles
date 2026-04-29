from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(repo_root: Path):
    path = repo_root / "competitions" / "cayley-py-megaminx" / "prompt_self_improver.py"
    spec = importlib.util.spec_from_file_location("prompt_self_improver_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_aware_prompt_includes_failed_candidate_autopsy(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent
    module = _load_module(repo_root)

    base_prompt = tmp_path / "user_prompt.txt"
    base_prompt.write_text("Megaminx prompt\n\nSTRICT OUTPUT REQUIREMENTS\nreturn json\n", encoding="utf-8")
    custom = tmp_path / "custom_prompts.json"
    custom.write_text(json.dumps({"planner": "plan", "coder": "code", "fixer": "fix"}), encoding="utf-8")
    baseline = tmp_path / "baseline_solver.py"
    baseline.write_text("def solve(vec):\n    return [], list(vec)\n", encoding="utf-8")
    failed = tmp_path / "failed_solver.py"
    failed.write_text(
        "from __future__ import annotations\n"
        "def solve(vec):\n"
        "    # attempted exact window rewrite but broke JSON stdout and move legality\n"
        "    window_rewrite_policy = True\n"
        "    return ['FAKE_MOVE'], list(vec)\n",
        encoding="utf-8",
    )

    out = module.build_round_prompt_bundle(
        base_prompt_file=base_prompt,
        base_custom_prompts=custom,
        baseline_solver=baseline,
        round_idx=2,
        score_history=[{
            "round": 1,
            "accepted": False,
            "path": str(failed),
            "error": "validator failed: illegal move FAKE_MOVE",
        }],
        best_metric={"source": "local_score", "value": 100},
        prompt_history=[],
        output_dir=tmp_path / "prompt_rounds",
    )

    prompt_text = Path(out["prompt_file"]).read_text(encoding="utf-8")
    assert "Failure-aware repair memory" in prompt_text
    assert "failure_bucket: validation" in prompt_text
    assert "FAKE_MOVE" in prompt_text
    assert "Compile-validation ladder" in prompt_text
    assert "Failure-autopsy repair loop" in prompt_text

    meta = out["meta"]
    assert "failure_repair_report" in meta
    assert meta["failure_repair_report"]["candidate_code_available"] is True
    assert "failure_autopsy_repair_loop" in meta["selected_directives"]
    assert "compile_validation_ladder" in meta["selected_directives"]
