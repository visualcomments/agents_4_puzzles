from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llm-puzzles"))
sys.path.insert(0, str(ROOT))

from src.comp_registry import get_config  # type: ignore
import pipeline_cli  # type: ignore


def test_current_cayleypy_sample_backed_schema_configs():
    for slug in ["cayleypy-pancake", "cayleypy-glushkov", "cayleypy-rapapport-m2", "CayleyPy-pancake"]:
        cfg = get_config(slug)
        assert cfg.submission_headers == ["id", "permutation", "solution"]
        assert cfg.header_keys == ["id", "permutation", "moves"]
        assert cfg.puzzles_id_field == "id"


def test_preferred_kaggle_cli_submit_cmd_uses_positional_competition(tmp_path):
    out = tmp_path / "submission.csv"
    cmd = pipeline_cli._preferred_kaggle_cli_submit_cmd("cayleypy-rapapport-m2", out, "msg")
    assert cmd[:4] == ["kaggle", "competitions", "submit", "cayleypy-rapapport-m2"]
    assert "-c" not in cmd


def test_bundled_sample_submission_matches_competition_zip_schema():
    for comp in ["cayleypy-pancake", "cayleypy-glushkov", "cayleypy-rapapport-m2"]:
        sp = ROOT / "competitions" / comp / "data" / "sample_submission.csv"
        with sp.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
        assert header == ["id", "permutation", "solution"]
        assert row[0].isdigit()
        assert "," in row[1]


def test_infer_format_slug_from_sample_header(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,permutation,solution\n0,3,UNSOLVED\n", encoding="utf-8")
    assert pipeline_cli._infer_format_slug_from_sample(sample) == "format/id+permutation+solution"



def test_megaminx_baseline_returns_valid_optimized_path_for_bundled_state():
    import importlib.util
    import json as _json

    solver_path = ROOT / "competitions" / "cayley-py-megaminx" / "solve_module.py"
    spec = importlib.util.spec_from_file_location("megaminx_solve_module", solver_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    test_csv = ROOT / "competitions" / "cayley-py-megaminx" / "data" / "test.csv"
    sample_csv = ROOT / "competitions" / "cayley-py-megaminx" / "data" / "sample_submission.csv"
    with test_csv.open(newline="", encoding="utf-8") as tf, sample_csv.open(newline="", encoding="utf-8") as sf:
        test_row = next(csv.DictReader(tf))
        sample_row = next(csv.DictReader(sf))

    vec = [int(x) for x in test_row["initial_state"].split(",")]
    moves, out_vec = module.solve(vec)
    assert isinstance(moves, list)
    assert len(moves) <= len(sample_row["path"].split("."))

    puzzle_info = _json.loads((ROOT / "competitions" / "cayley-py-megaminx" / "data" / "puzzle_info.json").read_text(encoding="utf-8"))
    assert out_vec == puzzle_info["central_state"]

def test_megaminx_copied_baseline_solver_can_still_find_data():
    import json as _json
    import shutil
    import subprocess
    import tempfile

    solver_src = ROOT / "competitions" / "cayley-py-megaminx" / "solve_module.py"
    with tempfile.TemporaryDirectory(dir=ROOT) as tmpdir:
        copied = Path(tmpdir) / "solve_megaminx_copy.py"
        shutil.copyfile(solver_src, copied)
        vec = list(range(120))
        out = subprocess.check_output([sys.executable, str(copied), _json.dumps(vec)], text=True)
        payload = _json.loads(out)
    assert payload["moves"] == []
    assert payload["sorted_array"] == list(range(120))

def test_cmd_run_forwards_print_generation_flags_to_agent_lab(monkeypatch, tmp_path):
    captured = {}

    def fake_run_agent_laboratory(**kwargs):
        captured.update(kwargs)
        Path(kwargs["out_path"]).write_text("def solve(vec):\n    return [], list(vec)\n", encoding="utf-8")

    def fake_validate_solver(*args, **kwargs):
        return None

    def fake_build_submission(**kwargs):
        sample = ROOT / "competitions" / "cayley-py-megaminx" / "data" / "sample_submission.csv"
        Path(kwargs["out_csv"]).write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(pipeline_cli, "_run_agent_laboratory", fake_run_agent_laboratory)
    monkeypatch.setattr(pipeline_cli, "_validate_solver", fake_validate_solver)
    monkeypatch.setattr(pipeline_cli, "_build_submission", fake_build_submission)
    monkeypatch.setattr(pipeline_cli, "_load_allowed_moves_from_validator", lambda _p: None)
    monkeypatch.setattr(pipeline_cli, "_gpu_diag_hint", lambda *_a, **_k: None)

    out_csv = tmp_path / "submission.csv"
    args = type("Args", (), {})()
    args.competition = "cayley-py-megaminx"
    args.puzzles = None
    args.output = str(out_csv)
    args.run_log = str(tmp_path / "run_log.json")
    args.no_llm = False
    args.prompt_file = None
    args.custom_prompts = None
    args.models = "gpt-4o-mini"
    args.agent_models = None
    args.planner_models = None
    args.coder_models = None
    args.fixer_models = None
    args.search_mode = None
    args.plan_beam_width = None
    args.frontier_width = None
    args.archive_size = None
    args.refine_rounds = None
    args.max_iters = 3
    args.allow_baseline = True
    args.g4f_recovery_rounds = None
    args.g4f_recovery_max_iters = None
    args.g4f_recovery_sleep = None
    args.worker_no_kill_process_group = False
    args.print_generation = True
    args.print_generation_max_chars = 777
    args.g4f_async = True
    args.max_response_chars = 12345
    args.g4f_request_timeout = 66.0
    args.g4f_stop_at_python_fence = True
    args.format = None
    args.vector_col = None
    args.max_rows = None
    args.no_progress = True
    args.submit = False
    args.message = None
    args.kaggle_json = None
    args.kaggle_config_dir = None
    args.submit_via = "auto"
    args.submit_competition = None
    args.schema_check = False
    args.no_schema_check = False
    args.no_schema_check_ids = False
    args.no_run_log = True

    pipeline_cli.cmd_run(args)

    assert captured["print_generation"] is True
    assert captured["print_generation_max_chars"] == 777
    assert captured["g4f_async"] is True
    assert captured["max_response_chars"] == 12345
    assert captured["g4f_request_timeout"] == 66.0
    assert captured["g4f_stop_at_python_fence"] is True
    assert out_csv.exists()



def test_validate_submission_move_tokens_rejects_unsolved_for_megaminx(tmp_path):
    sub = tmp_path / "submission.csv"
    sub.write_text("initial_state_id,path\n0,UNSOLVED\n", encoding="utf-8")
    allowed = {"U", "-U"}
    try:
        pipeline_cli._validate_submission_move_tokens(
            submission_csv=sub,
            move_column="path",
            allowed_moves=allowed,
            joiner=".",
        )
    except ValueError as exc:
        assert "UNSOLVED" in str(exc)
    else:
        raise AssertionError("expected UNSOLVED to be rejected")


def test_resolve_smoke_vectors_for_megaminx_includes_nontrivial_test_state():
    from pipeline_registry import get_pipeline

    spec = get_pipeline("cayley-py-megaminx")
    vectors = pipeline_cli._resolve_smoke_vectors(spec)
    assert len(vectors) >= 2
    assert vectors[0] == list(range(120))
    assert any(vec != list(range(120)) for vec in vectors[1:])

def test_megaminx_optimized_lookup_beats_sample_for_known_row():
    import importlib.util

    solver_path = ROOT / "competitions" / "cayley-py-megaminx" / "solve_module.py"
    spec = importlib.util.spec_from_file_location("megaminx_solve_module_opt", solver_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    test_csv = ROOT / "competitions" / "cayley-py-megaminx" / "data" / "test.csv"
    sample_csv = ROOT / "competitions" / "cayley-py-megaminx" / "data" / "sample_submission.csv"
    optimized_csv = ROOT / "competitions" / "cayley-py-megaminx" / "submissions" / "optimized_submission.csv"

    with test_csv.open(newline="", encoding="utf-8") as tf, sample_csv.open(newline="", encoding="utf-8") as sf, optimized_csv.open(newline="", encoding="utf-8") as of:
        test_rows = list(csv.DictReader(tf))
        sample_rows = list(csv.DictReader(sf))
        optimized_rows = list(csv.DictReader(of))

    idx = 100
    vec = [int(x) for x in test_rows[idx]["initial_state"].split(",")]
    sample_len = len(sample_rows[idx]["path"].split("."))
    optimized_len = len(optimized_rows[idx]["path"].split("."))

    moves, out_vec = module.solve(vec)
    assert isinstance(moves, list)
    assert len(moves) == optimized_len
    assert optimized_len < sample_len
    assert out_vec == list(range(120))


def _make_cmd_run_args(tmp_path):
    args = type('Args', (), {})()
    args.competition = 'cayley-py-megaminx'
    args.puzzles = None
    args.output = str(tmp_path / 'submission.csv')
    args.run_log = str(tmp_path / 'run_log.json')
    args.no_llm = False
    args.prompt_file = None
    args.custom_prompts = None
    args.models = 'gpt-4o-mini'
    args.agent_models = None
    args.planner_models = None
    args.coder_models = None
    args.fixer_models = None
    args.search_mode = 'hybrid'
    args.plan_beam_width = 3
    args.frontier_width = 6
    args.archive_size = 6
    args.refine_rounds = 1
    args.max_iters = 3
    args.allow_baseline = True
    args.g4f_recovery_rounds = None
    args.g4f_recovery_max_iters = None
    args.g4f_recovery_sleep = None
    args.worker_no_kill_process_group = False
    args.print_generation = False
    args.print_generation_max_chars = None
    args.g4f_async = None
    args.max_response_chars = None
    args.g4f_request_timeout = None
    args.g4f_stop_at_python_fence = False
    args.keep_improving = False
    args.improvement_rounds = 1
    args.format = None
    args.vector_col = None
    args.max_rows = None
    args.no_progress = True
    args.submit = True
    args.message = 'test submit'
    args.kaggle_json = None
    args.kaggle_config_dir = None
    args.submit_via = 'auto'
    args.submit_competition = None
    args.schema_check = False
    args.no_schema_check = False
    args.no_schema_check_ids = False
    args.no_run_log = True
    return args


def test_cmd_run_skips_submit_when_credentials_missing(monkeypatch, tmp_path):
    called = {'submit': 0}

    def fake_run_agent_laboratory(**kwargs):
        Path(kwargs['out_path']).write_text('ROUND0', encoding='utf-8')

    def fake_validate_solver(*args, **kwargs):
        return None

    def fake_build_submission(**kwargs):
        marker = Path(kwargs['solver_path']).read_text(encoding='utf-8').strip()
        Path(kwargs['out_csv']).write_text(f'initial_state_id,path\n0,{marker}\n', encoding='utf-8')

    def fake_kaggle_submit(**kwargs):
        called['submit'] += 1
        raise AssertionError('kaggle submit should not be called when credentials are unavailable')

    monkeypatch.setattr(pipeline_cli, '_run_agent_laboratory', fake_run_agent_laboratory)
    monkeypatch.setattr(pipeline_cli, '_validate_solver', fake_validate_solver)
    monkeypatch.setattr(pipeline_cli, '_build_submission', fake_build_submission)
    monkeypatch.setattr(pipeline_cli, '_load_allowed_moves_from_validator', lambda _p: None)
    monkeypatch.setattr(pipeline_cli, '_validate_submission_schema', lambda **_: {'header_ok': True, 'row_count_ok': True})
    monkeypatch.setattr(pipeline_cli, '_gpu_diag_hint', lambda *_a, **_k: None)
    monkeypatch.setattr(
        pipeline_cli,
        '_resolve_kaggle_submit_availability',
        lambda **_: {
            'enabled': False,
            'source': 'missing_credentials',
            'credentials_path': None,
            'reason': 'No Kaggle credentials were found. Live submission will be skipped.',
            'nonfatal': True,
        },
    )
    monkeypatch.setattr(pipeline_cli, '_kaggle_submit', fake_kaggle_submit)

    args = _make_cmd_run_args(tmp_path)
    pipeline_cli.cmd_run(args)

    assert called['submit'] == 0
    assert Path(args.output).read_text(encoding='utf-8') == 'initial_state_id,path\n0,ROUND0\n'


def test_cmd_run_keep_improving_preserves_validated_solver_on_round_submit_systemexit(monkeypatch, tmp_path):
    def fake_run_agent_laboratory(**kwargs):
        Path(kwargs['out_path']).write_text('ROUND1', encoding='utf-8')

    def fake_validate_solver(*args, **kwargs):
        return None

    def fake_build_submission(**kwargs):
        marker = Path(kwargs['solver_path']).read_text(encoding='utf-8').strip()
        Path(kwargs['out_csv']).write_text(f'initial_state_id,path\n0,{marker}\n', encoding='utf-8')

    def fake_score_solver_with_submission(**kwargs):
        text = Path(kwargs['solver_path']).read_text(encoding='utf-8')
        return 90 if text == 'ROUND1' else 100

    def fake_kaggle_submit(**kwargs):
        raise SystemExit('Kaggle submission failed: /root/.kaggle/kaggle.json')

    monkeypatch.setattr(pipeline_cli, '_run_agent_laboratory', fake_run_agent_laboratory)
    monkeypatch.setattr(pipeline_cli, '_validate_solver', fake_validate_solver)
    monkeypatch.setattr(pipeline_cli, '_build_submission', fake_build_submission)
    monkeypatch.setattr(pipeline_cli, '_score_solver_with_submission', fake_score_solver_with_submission)
    monkeypatch.setattr(pipeline_cli, '_load_allowed_moves_from_validator', lambda _p: None)
    monkeypatch.setattr(pipeline_cli, '_validate_submission_schema', lambda **_: {'header_ok': True, 'row_count_ok': True})
    monkeypatch.setattr(pipeline_cli, '_gpu_diag_hint', lambda *_a, **_k: None)
    monkeypatch.setattr(
        pipeline_cli,
        '_resolve_kaggle_submit_availability',
        lambda **_: {
            'enabled': True,
            'source': 'environment',
            'credentials_path': None,
        },
    )
    monkeypatch.setattr(pipeline_cli, '_kaggle_submit', fake_kaggle_submit)

    args = _make_cmd_run_args(tmp_path)
    args.keep_improving = True
    args.improvement_rounds = 1
    pipeline_cli.cmd_run(args)

    assert Path(args.output).read_text(encoding='utf-8') == 'initial_state_id,path\n0,ROUND1\n'
