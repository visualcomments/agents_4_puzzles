from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pipeline_cli  # type: ignore
from pipeline_registry import PipelineSpec  # type: ignore

COMP = ROOT / 'competitions' / 'cayley-py-megaminx'
MODULE_PATH = COMP / 'prompt_self_improver.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('megaminx_prompt_self_improver_test', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules.setdefault('megaminx_prompt_self_improver_test', module)
    spec.loader.exec_module(module)
    return module


def test_prompt_self_improver_targets_missing_megaminx_families(tmp_path):
    mod = _load_module()
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text(
        'Megaminx bounded optimizer.\n\nSTRICT OUTPUT REQUIREMENTS:\n- Return exactly one JSON object and no prose outside JSON.\n',
        encoding='utf-8',
    )
    custom = tmp_path / 'custom.json'
    custom.write_text(
        '{"planner": "planner\\n\\nSTRICT OUTPUT REQUIREMENTS:\\n- Return exactly one JSON object and no prose outside JSON.", '
        '"coder": "coder\\n\\nSTRICT OUTPUT REQUIREMENTS:\\n- Return exactly one JSON object and no prose outside JSON.", '
        '"fixer": "fixer\\n\\nSTRICT OUTPUT REQUIREMENTS:\\n- Return exactly one JSON object and no prose outside JSON."}',
        encoding='utf-8',
    )
    baseline = tmp_path / 'baseline.py'
    baseline.write_text(
        'from __future__ import annotations\n\n_SHORT_TABLE_DEPTH = 5\n_LOCAL_WINDOW = 12\n_OPTIMIZATION_PASSES = 2\n\n'
        'def _short_word_data():\n    return {}\n\n'
        'def _optimize_local_windows():\n    return []\n',
        encoding='utf-8',
    )

    result = mod.build_round_prompt_bundle(
        base_prompt_file=prompt,
        base_custom_prompts=custom,
        baseline_solver=baseline,
        round_idx=2,
        score_history=[{'round': 1, 'accepted': True, 'metric': {'source': 'local_score', 'value': 100}}],
        best_metric={'source': 'local_score', 'value': 100},
        prompt_history=[],
        output_dir=tmp_path / 'rounds',
    )

    prompt_text = Path(result['prompt_file']).read_text(encoding='utf-8')
    custom_text = Path(result['custom_prompts_file']).read_text(encoding='utf-8')
    meta = result['meta']

    assert 'SELF-IMPROVEMENT ROUND 2' in prompt_text
    assert 'materially stronger alternative' in prompt_text
    assert 'Current baseline diagnosis:' in prompt_text
    assert 'Required architecture deltas for this round:' in prompt_text
    assert 'multi_policy_sweep' in meta['selected_directives']
    assert 'bidirectional_window_rewrite' in meta['selected_directives']
    assert 'small_support_macro_mining' in meta['selected_directives']
    assert 'STRICT OUTPUT REQUIREMENTS' in custom_text
    assert 'the injected baseline is the previous best validated solver' in custom_text


def test_generate_solver_with_self_improving_prompts_uses_round_specific_bundle(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline.py'
    baseline.write_text('baseline solver body', encoding='utf-8')
    validator = tmp_path / 'validator.py'
    validator.write_text('# validator', encoding='utf-8')
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text(
        'Base prompt text.\n\nSTRICT OUTPUT REQUIREMENTS:\n- Return exactly one JSON object and no prose outside JSON.\n',
        encoding='utf-8',
    )
    custom_prompts = tmp_path / 'custom_prompts.json'
    custom_prompts.write_text(
        '{"planner": "planner\\n\\nSTRICT OUTPUT REQUIREMENTS:\\n- Return exactly one JSON object and no prose outside JSON.", '
        '"coder": "coder\\n\\nSTRICT OUTPUT REQUIREMENTS:\\n- Return exactly one JSON object and no prose outside JSON.", '
        '"fixer": "fixer\\n\\nSTRICT OUTPUT REQUIREMENTS:\\n- Return exactly one JSON object and no prose outside JSON."}',
        encoding='utf-8',
    )
    out_path = tmp_path / 'solve_best.py'

    spec = PipelineSpec(
        key='cayley-py-megaminx',
        competition='cayley-py-megaminx',
        format_slug='cayley-py-megaminx',
        baseline_solver=baseline,
        validator=validator,
        prompt_file=prompt_file,
        custom_prompts_file=custom_prompts,
        state_columns=['vector'],
        smoke_vector=[1, 2, 3],
    )

    seen_prompts: list[str] = []
    rounds = {'count': 0}

    def fake_run_agent_laboratory(**kwargs):
        rounds['count'] += 1
        seen_prompts.append(Path(kwargs['prompt_file']).read_text(encoding='utf-8'))
        Path(kwargs['out_path']).write_text(f'round-{rounds["count"]}', encoding='utf-8')

    monkeypatch.setattr(pipeline_cli, '_run_agent_laboratory', fake_run_agent_laboratory)
    monkeypatch.setattr(pipeline_cli, '_validate_solver', lambda *args, **kwargs: None)

    result = pipeline_cli._generate_solver_with_optional_improvement(
        spec=spec,
        out_path=out_path,
        prompt_file=prompt_file,
        custom_prompts=custom_prompts,
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
        competition_format_slug='cayley-py-megaminx',
        self_improve_prompts=True,
    )

    assert result['rounds_requested'] == 2
    assert len(seen_prompts) == 2
    assert 'SELF-IMPROVEMENT ROUND 1' in seen_prompts[0]
    assert 'SELF-IMPROVEMENT ROUND 2' in seen_prompts[1]
    history_path = out_path.with_name(out_path.stem + '_prompt_evolution.json')
    assert history_path.exists()
