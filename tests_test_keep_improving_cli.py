from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'AgentLaboratory'))

import pipeline_cli  # type: ignore
from pipeline_registry import PipelineSpec, get_pipeline  # type: ignore
from perm_pipeline import run_perm_pipeline as rpp  # type: ignore


def test_build_initial_codegen_prompt_uses_reference_baseline_for_from_scratch_prompt():
    prompt = (
        'Write the code FROM SCRATCH. '\
        'Do NOT rely on, patch, wrap, or extend any baseline implementation. '\
        'Enable CREATIVE_SCORE_SEARCH.'
    )
    out = rpp.build_initial_codegen_prompt(
        prompt,
        'planner summary',
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
    )
    assert '## REFERENCE BASELINE (compatibility and score target only)' in out
    assert 'Do NOT patch, wrap, subclass, or extend the baseline.' in out
    assert 'Modify the baseline minimally' not in out
    assert 'synthesize a small bank of deterministic candidate optimizers' in out


def test_build_parser_accepts_keep_improving_flags_for_generate_and_run():
    parser = pipeline_cli.build_parser()

    gen_args = parser.parse_args([
        'generate-solver',
        '--competition', 'cayley-py-megaminx',
        '--out', 'generated/solve.py',
        '--keep-improving',
        '--improvement-rounds', '4',
    ])
    assert gen_args.keep_improving is True
    assert gen_args.improvement_rounds == 4

    run_args = parser.parse_args([
        'run',
        '--competition', 'cayley-py-megaminx',
        '--output', 'submissions/submission.csv',
        '--keep-improving',
        '--improvement-rounds', '5',
    ])
    assert run_args.keep_improving is True
    assert run_args.improvement_rounds == 5


def test_generate_solver_with_optional_improvement_keeps_best_local_score(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline.py'
    baseline.write_text('baseline', encoding='utf-8')
    validator = tmp_path / 'validator.py'
    validator.write_text('# validator', encoding='utf-8')
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('FROM SCRATCH\nCREATIVE_SCORE_SEARCH', encoding='utf-8')
    out_path = tmp_path / 'solve_best.py'

    spec = PipelineSpec(
        key='demo',
        competition='demo',
        format_slug='format/moves-dot',
        baseline_solver=baseline,
        validator=validator,
        prompt_file=prompt_file,
        custom_prompts_file=None,
        state_columns=['vector'],
        smoke_vector=[1, 2, 3],
    )

    rounds = {'count': 0}

    def fake_run_agent_laboratory(**kwargs):
        rounds['count'] += 1
        Path(kwargs['out_path']).write_text(f'round-{rounds["count"]}', encoding='utf-8')

    def fake_validate_solver(*args, **kwargs):
        return None

    def fake_score_solver_with_submission(**kwargs):
        solver_path = Path(kwargs['solver_path'])
        text = solver_path.read_text(encoding='utf-8')
        if text == 'baseline':
            return 100
        if text == 'round-1':
            return 90
        if text == 'round-2':
            return 95
        raise AssertionError(text)

    monkeypatch.setattr(pipeline_cli, '_run_agent_laboratory', fake_run_agent_laboratory)
    monkeypatch.setattr(pipeline_cli, '_validate_solver', fake_validate_solver)
    monkeypatch.setattr(pipeline_cli, '_score_solver_with_submission', fake_score_solver_with_submission)

    result = pipeline_cli._generate_solver_with_optional_improvement(
        spec=spec,
        out_path=out_path,
        prompt_file=prompt_file,
        custom_prompts=None,
        llm='gpt-4o-mini',
        agent_models=None,
        planner_models=None,
        coder_models=None,
        fixer_models=None,
        search_mode='hybrid',
        plan_beam_width=3,
        frontier_width=6,
        archive_size=6,
        refine_rounds=1,
        max_iters=2,
        allow_baseline=False,
        g4f_recovery_rounds=None,
        g4f_recovery_max_iters=None,
        g4f_recovery_sleep=None,
        worker_no_kill_process_group=False,
        print_generation=False,
        print_generation_max_chars=None,
        g4f_async=None,
        max_response_chars=None,
        g4f_request_timeout=None,
        g4f_stop_at_python_fence=None,
        keep_improving=True,
        improvement_rounds=3,
        puzzles_csv_for_score=tmp_path / 'test.csv',
        competition_format_slug='format/moves-dot',
    )

    assert out_path.read_text(encoding='utf-8') == 'round-1'
    assert result['best_score'] == 90
    assert result['best_round'] == 1
    assert rounds['count'] == 3


def test_megaminx_regular_prompt_bundle_is_from_scratch_and_creative():
    spec = get_pipeline('cayley-py-megaminx')
    assert spec is not None
    args = argparse.Namespace(prompt_variant='regular', prompt_file=None, custom_prompts=None)
    prompt_file, custom_prompts = pipeline_cli._resolve_prompt_bundle(spec, args)
    prompt_text = prompt_file.read_text(encoding='utf-8')
    custom_text = custom_prompts.read_text(encoding='utf-8') if custom_prompts is not None else ''

    assert 'NO_BASELINE_PATCH_BIAS' in prompt_text
    assert 'CREATIVE_SCORE_SEARCH' in prompt_text
    assert 'Write the code from scratch' in prompt_text
    assert 'baseline, if shown, is only a compatibility and score reference' in custom_text



def test_generate_solver_with_optional_improvement_runs_submission_hook_before_next_round(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline.py'
    baseline.write_text('baseline', encoding='utf-8')
    validator = tmp_path / 'validator.py'
    validator.write_text('# validator', encoding='utf-8')
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('FROM SCRATCH\nCREATIVE_SCORE_SEARCH', encoding='utf-8')
    out_path = tmp_path / 'solve_best.py'

    spec = PipelineSpec(
        key='demo',
        competition='demo',
        format_slug='format/moves-dot',
        baseline_solver=baseline,
        validator=validator,
        prompt_file=prompt_file,
        custom_prompts_file=None,
        state_columns=['vector'],
        smoke_vector=[1, 2, 3],
    )

    events: list[str] = []
    rounds = {'count': 0}

    def fake_run_agent_laboratory(**kwargs):
        rounds['count'] += 1
        events.append(f"run{rounds['count']}")
        Path(kwargs['out_path']).write_text(f'round-{rounds["count"]}', encoding='utf-8')

    def fake_validate_solver(path, *_args, **_kwargs):
        text = Path(path).read_text(encoding='utf-8')
        events.append(f"validate{text.split('-')[-1]}")
        return None

    def round_hook(round_idx: int, _path: Path):
        events.append(f'hook{round_idx}')
        return {'submitted': True, 'round': round_idx}

    monkeypatch.setattr(pipeline_cli, '_run_agent_laboratory', fake_run_agent_laboratory)
    monkeypatch.setattr(pipeline_cli, '_validate_solver', fake_validate_solver)

    result = pipeline_cli._generate_solver_with_optional_improvement(
        spec=spec,
        out_path=out_path,
        prompt_file=prompt_file,
        custom_prompts=None,
        llm='gpt-4o-mini',
        agent_models=None,
        planner_models=None,
        coder_models=None,
        fixer_models=None,
        search_mode='hybrid',
        plan_beam_width=3,
        frontier_width=6,
        archive_size=6,
        refine_rounds=1,
        max_iters=2,
        allow_baseline=False,
        g4f_recovery_rounds=None,
        g4f_recovery_max_iters=None,
        g4f_recovery_sleep=None,
        worker_no_kill_process_group=False,
        print_generation=False,
        print_generation_max_chars=None,
        g4f_async=None,
        max_response_chars=None,
        g4f_request_timeout=None,
        g4f_stop_at_python_fence=None,
        keep_improving=True,
        improvement_rounds=2,
        puzzles_csv_for_score=None,
        competition_format_slug='format/moves-dot',
        validated_round_hook=round_hook,
    )

    assert events == ['run1', 'validate1', 'hook1', 'run2', 'validate2', 'hook2']
    assert result['submitted_rounds'] == [1, 2]
    assert result['selected_round_already_submitted'] is True



def test_cmd_run_skips_duplicate_final_submit_when_best_round_was_already_submitted(tmp_path, monkeypatch):
    parser = pipeline_cli.build_parser()
    args = parser.parse_args([
        'run',
        '--competition', 'demo-competition',
        '--output', str(tmp_path / 'submission.csv'),
        '--keep-improving',
        '--improvement-rounds', '2',
        '--submit',
        '--message', 'demo submission',
    ])

    baseline = tmp_path / 'baseline.py'
    baseline.write_text('def solve(vec):\n    return [], list(vec)\n', encoding='utf-8')
    validator = tmp_path / 'validator.py'
    validator.write_text('# validator', encoding='utf-8')
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('FROM SCRATCH\nCREATIVE_SCORE_SEARCH', encoding='utf-8')
    puzzles_csv = tmp_path / 'test.csv'
    puzzles_csv.write_text('id,vector\n1,"[1,2,3]"\n', encoding='utf-8')
    sample_csv = tmp_path / 'sample_submission.csv'
    sample_csv.write_text('id,moves\n1,\n', encoding='utf-8')

    spec = PipelineSpec(
        key='demo',
        competition='demo-kaggle',
        format_slug='format/moves-dot',
        baseline_solver=baseline,
        validator=validator,
        prompt_file=prompt_file,
        custom_prompts_file=None,
        state_columns=['vector'],
        smoke_vector=[1, 2, 3],
    )

    kaggle_calls: list[dict[str, str]] = []

    def fake_generate_solver(**kwargs):
        out_path = Path(kwargs['out_path'])
        out_path.write_text('def solve(vec):\n    return [], list(vec)\n', encoding='utf-8')
        hook = kwargs.get('validated_round_hook')
        if hook is not None:
            hook(1, out_path)
        return {
            'best_round': 1,
            'best_score': 10,
            'history': [],
            'submitted_rounds': [1],
            'selected_round_already_submitted': True,
        }

    def fake_build_submission(**kwargs):
        Path(kwargs['out_csv']).write_text('id,moves\n1,\n', encoding='utf-8')

    def fake_kaggle_submit(**kwargs):
        kaggle_calls.append({
            'competition': kwargs['competition'],
            'submission_csv': str(kwargs['submission_csv']),
            'message': kwargs['message'],
        })
        return {'mode': 'fake'}

    monkeypatch.setattr(pipeline_cli, 'get_pipeline', lambda _key: spec)
    monkeypatch.setattr(pipeline_cli, '_gpu_diag_hint', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_cli, '_resolve_default_puzzles', lambda _spec: puzzles_csv)
    monkeypatch.setattr(pipeline_cli, '_resolve_sample_submission', lambda _spec: sample_csv)
    monkeypatch.setattr(pipeline_cli, '_resolve_prompt_bundle', lambda _spec, _args: (prompt_file, None))
    monkeypatch.setattr(pipeline_cli, '_generate_solver_with_optional_improvement', fake_generate_solver)
    monkeypatch.setattr(pipeline_cli, '_validate_solver', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_cli, '_resolve_smoke_vectors', lambda _spec: [[1, 2, 3]])
    monkeypatch.setattr(pipeline_cli, '_load_allowed_moves_from_validator', lambda _validator: None)
    monkeypatch.setattr(pipeline_cli, '_resolve_submission_move_column', lambda _fmt: ('moves', '.'))
    monkeypatch.setattr(pipeline_cli, '_build_submission', fake_build_submission)
    monkeypatch.setattr(pipeline_cli, '_finalize_submission_output', lambda src, dst: Path(dst).write_text(Path(src).read_text(encoding='utf-8'), encoding='utf-8'))
    monkeypatch.setattr(pipeline_cli, '_kaggle_submit', fake_kaggle_submit)
    monkeypatch.setattr(pipeline_cli, '_attach_io_stats', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_cli, '_append_run_log', lambda *_args, **_kwargs: None)

    pipeline_cli.cmd_run(args)

    assert len(kaggle_calls) == 1
    assert kaggle_calls[0]['competition'] == 'demo-kaggle'
    assert '[round 1/2]' in kaggle_calls[0]['message']
    assert Path(args.output).exists()


def test_generate_solver_with_optional_improvement_does_not_stop_on_first_non_improving_round(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline.py'
    baseline.write_text('baseline', encoding='utf-8')
    validator = tmp_path / 'validator.py'
    validator.write_text('# validator', encoding='utf-8')
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('FROM SCRATCH\nCREATIVE_SCORE_SEARCH', encoding='utf-8')
    out_path = tmp_path / 'solve_best.py'

    spec = PipelineSpec(
        key='demo',
        competition='demo',
        format_slug='format/moves-dot',
        baseline_solver=baseline,
        validator=validator,
        prompt_file=prompt_file,
        custom_prompts_file=None,
        state_columns=['vector'],
        smoke_vector=[1, 2, 3],
    )

    rounds = {'count': 0}

    def fake_run_agent_laboratory(**kwargs):
        rounds['count'] += 1
        Path(kwargs['out_path']).write_text(f'round-{rounds["count"]}', encoding='utf-8')

    def fake_validate_solver(*args, **kwargs):
        return None

    def fake_score_solver_with_submission(**kwargs):
        solver_path = Path(kwargs['solver_path'])
        text = solver_path.read_text(encoding='utf-8')
        if text == 'baseline':
            return 100
        if text == 'round-1':
            return 90
        if text == 'round-2':
            return 95
        if text == 'round-3':
            return 85
        raise AssertionError(text)

    monkeypatch.setattr(pipeline_cli, '_run_agent_laboratory', fake_run_agent_laboratory)
    monkeypatch.setattr(pipeline_cli, '_validate_solver', fake_validate_solver)
    monkeypatch.setattr(pipeline_cli, '_score_solver_with_submission', fake_score_solver_with_submission)

    result = pipeline_cli._generate_solver_with_optional_improvement(
        spec=spec,
        out_path=out_path,
        prompt_file=prompt_file,
        custom_prompts=None,
        llm='gpt-4o-mini',
        agent_models=None,
        planner_models=None,
        coder_models=None,
        fixer_models=None,
        search_mode='hybrid',
        plan_beam_width=3,
        frontier_width=6,
        archive_size=6,
        refine_rounds=1,
        max_iters=2,
        allow_baseline=False,
        g4f_recovery_rounds=None,
        g4f_recovery_max_iters=None,
        g4f_recovery_sleep=None,
        worker_no_kill_process_group=False,
        print_generation=False,
        print_generation_max_chars=None,
        g4f_async=None,
        max_response_chars=None,
        g4f_request_timeout=None,
        g4f_stop_at_python_fence=None,
        keep_improving=True,
        improvement_rounds=3,
        puzzles_csv_for_score=tmp_path / 'test.csv',
        competition_format_slug='format/moves-dot',
    )

    assert rounds['count'] == 3
    assert out_path.read_text(encoding='utf-8') == 'round-3'
    assert result['best_score'] == 85
    assert result['best_round'] == 3



def test_generate_solver_with_optional_improvement_continues_after_round_hook_system_exit(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline.py'
    baseline.write_text('baseline', encoding='utf-8')
    validator = tmp_path / 'validator.py'
    validator.write_text('# validator', encoding='utf-8')
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('FROM SCRATCH\nCREATIVE_SCORE_SEARCH', encoding='utf-8')
    out_path = tmp_path / 'solve_best.py'

    spec = PipelineSpec(
        key='demo',
        competition='demo',
        format_slug='format/moves-dot',
        baseline_solver=baseline,
        validator=validator,
        prompt_file=prompt_file,
        custom_prompts_file=None,
        state_columns=['vector'],
        smoke_vector=[1, 2, 3],
    )

    rounds = {'count': 0}

    def fake_run_agent_laboratory(**kwargs):
        rounds['count'] += 1
        Path(kwargs['out_path']).write_text(f'round-{rounds["count"]}', encoding='utf-8')

    def fake_validate_solver(*args, **kwargs):
        return None

    def round_hook(round_idx: int, _path: Path):
        if round_idx == 1:
            raise SystemExit('simulated submit failure')
        return {'submitted': True, 'round': round_idx}

    monkeypatch.setattr(pipeline_cli, '_run_agent_laboratory', fake_run_agent_laboratory)
    monkeypatch.setattr(pipeline_cli, '_validate_solver', fake_validate_solver)

    result = pipeline_cli._generate_solver_with_optional_improvement(
        spec=spec,
        out_path=out_path,
        prompt_file=prompt_file,
        custom_prompts=None,
        llm='gpt-4o-mini',
        agent_models=None,
        planner_models=None,
        coder_models=None,
        fixer_models=None,
        search_mode='hybrid',
        plan_beam_width=3,
        frontier_width=6,
        archive_size=6,
        refine_rounds=1,
        max_iters=2,
        allow_baseline=False,
        g4f_recovery_rounds=None,
        g4f_recovery_max_iters=None,
        g4f_recovery_sleep=None,
        worker_no_kill_process_group=False,
        print_generation=False,
        print_generation_max_chars=None,
        g4f_async=None,
        max_response_chars=None,
        g4f_request_timeout=None,
        g4f_stop_at_python_fence=None,
        keep_improving=True,
        improvement_rounds=2,
        puzzles_csv_for_score=None,
        competition_format_slug='format/moves-dot',
        validated_round_hook=round_hook,
    )

    assert rounds['count'] == 2
    assert out_path.read_text(encoding='utf-8') == 'round-2'
    assert result['best_round'] == 2
    assert result['submitted_rounds'] == [2]
    assert result['selected_round_already_submitted'] is True
    assert result['history'][0]['error'] == 'simulated submit failure'
