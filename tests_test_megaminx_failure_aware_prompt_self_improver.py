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


def _write_base_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    base_prompt = tmp_path / "user_prompt.txt"
    base_prompt.write_text("Megaminx prompt\n\nSTRICT OUTPUT REQUIREMENTS\nreturn json\n", encoding="utf-8")
    custom = tmp_path / "custom_prompts.json"
    custom.write_text(json.dumps({"planner": "plan", "coder": "code", "fixer": "fix"}), encoding="utf-8")
    baseline = tmp_path / "baseline_solver.py"
    baseline.write_text("def solve(vec):\n    return [], list(vec)\n", encoding="utf-8")
    return base_prompt, custom, baseline


def test_failure_aware_prompt_includes_failed_candidate_autopsy(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent
    module = _load_module(repo_root)
    base_prompt, custom, baseline = _write_base_inputs(tmp_path)
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
    assert "failure_bucket: illegal_move" in prompt_text
    assert "FAKE_MOVE" in prompt_text
    assert "Compile-validation ladder" in prompt_text
    assert "Failure-autopsy repair loop" in prompt_text

    custom_text = Path(out["custom_prompts_file"]).read_text(encoding="utf-8")
    assert "Failure context:" in custom_text
    assert "failure_bucket=illegal_move" in custom_text

    meta = out["meta"]
    assert "failure_repair_report" in meta
    assert meta["failure_repair_report"]["candidate_code_available"] is True
    assert meta["failure_repair_report"]["bucket"] == "illegal_move"
    assert "failure_autopsy_repair_loop" in meta["selected_directives"]
    assert "compile_validation_ladder" in meta["selected_directives"]


def test_no_per_row_improvement_is_failure_not_validated_not_selected(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent
    module = _load_module(repo_root)
    base_prompt, custom, baseline = _write_base_inputs(tmp_path)

    out = module.build_round_prompt_bundle(
        base_prompt_file=base_prompt,
        base_custom_prompts=custom,
        baseline_solver=baseline,
        round_idx=3,
        score_history=[{
            "round": 2,
            "accepted": False,
            "failure_kind": "no_per_row_improvement",
            "rejection_reasons": ["no_per_row_improvement"],
            "novelty_report": {
                "identical_solver": True,
                "identical_submission": True,
                "per_row_delta": {
                    "improved_rows": 0,
                    "regressed_rows": 0,
                    "net_delta_moves": 0,
                },
            },
        }],
        best_metric={"source": "local_score", "value": 414305},
        prompt_history=[],
        output_dir=tmp_path / "prompt_rounds",
    )

    meta = out["meta"]
    signals = meta["history_signals"]
    assert signals["recent_failures"] == 1
    assert signals["validated_not_selected"] == 0
    assert signals["no_novelty_failures"] == 1
    assert signals["no_per_row_improvement_failures"] == 1
    assert meta["failure_repair_report"]["bucket"] == "no_per_row_improvement"
    assert meta["failure_repair_report"]["suppress_code_excerpt"] is True

    selected = set(meta["selected_directives"])
    assert {"failure_autopsy_repair_loop", "no_novelty_rejection", "per_row_delta_acceptance", "patch_fresh_lane_split"} <= selected

    prompt_text = Path(out["prompt_file"]).read_text(encoding="utf-8")
    assert "failure_bucket: no_per_row_improvement" in prompt_text
    assert "Create exactly one fresh isolated lane" in prompt_text
    assert "suppress_code_excerpt: yes" in prompt_text


def test_solver_path_is_used_for_failed_candidate_code(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent
    module = _load_module(repo_root)
    base_prompt, custom, baseline = _write_base_inputs(tmp_path)
    failed = tmp_path / "failed_solver_alt_path.py"
    failed.write_text(
        "def solve(vec):\n"
        "    candidate_bank = True\n"
        "    return [], list(vec)\n",
        encoding="utf-8",
    )

    out = module.build_round_prompt_bundle(
        base_prompt_file=base_prompt,
        base_custom_prompts=custom,
        baseline_solver=baseline,
        round_idx=4,
        score_history=[{
            "round": 3,
            "accepted": False,
            "solver_path": str(failed),
            "failure_kind": "score_regression",
            "rejection_reasons": ["score regression"],
        }],
        best_metric={"source": "local_score", "value": 414305},
        prompt_history=[],
        output_dir=tmp_path / "prompt_rounds",
    )

    report = out["meta"]["failure_repair_report"]
    assert report["candidate_code_available"] is True
    assert report["path"] == str(failed)
    assert report["bucket"] == "score_regression"
    assert report["features"]["has_optimizer_delta"] is True
