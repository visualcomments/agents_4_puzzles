from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / 'competitions' / 'cayley-py-megaminx' / 'prompt_self_improver.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('megaminx_prompt_self_improver_algorithmic_population', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules.setdefault('megaminx_prompt_self_improver_algorithmic_population', module)
    spec.loader.exec_module(module)
    return module


def test_select_directives_adds_population_and_evaluator_bias_on_plateau():
    mod = _load_module()
    feature_snapshot = {
        'has_multi_policy_sweep': True,
        'has_bidirectional_rewrite': True,
        'has_macro_mining': True,
        'has_exact_short_atlas': True,
        'has_candidate_bank_scoring': True,
        'constants': {'short_table_depth': 6},
    }
    history = [
        {'round': 1, 'accepted': False, 'metric': {'source': 'local_score', 'value': 120}},
        {'round': 2, 'accepted': False, 'metric': {'source': 'local_score', 'value': 120}},
        {'round': 3, 'accepted': False, 'metric': {'source': 'local_score', 'value': 121}},
    ]
    directives = mod.select_directives(
        feature_snapshot=feature_snapshot,
        round_idx=3,
        prompt_history=[],
        score_history=history,
    )
    keys = {item.key for item in directives}
    assert 'prompt_population_search' in keys
    assert 'exact_evaluator_shard' in keys
    assert 'solver_archive_lineage' in keys
    assert 'patch_fresh_lane_split' in keys
    assert 'pareto_candidate_selection' in keys


def test_round_prompt_mentions_algorithm_search_operating_mode(tmp_path):
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
        round_idx=3,
        score_history=[
            {'round': 1, 'accepted': False, 'metric': {'source': 'local_score', 'value': 100}},
            {'round': 2, 'accepted': False, 'metric': {'source': 'local_score', 'value': 100}},
            {'round': 3, 'accepted': False, 'metric': {'source': 'local_score', 'value': 101}},
        ],
        best_metric={'source': 'local_score', 'value': 100},
        prompt_history=[],
        output_dir=tmp_path / 'rounds',
    )

    prompt_text = Path(result['prompt_file']).read_text(encoding='utf-8')
    custom_text = Path(result['custom_prompts_file']).read_text(encoding='utf-8')

    assert 'Algorithm-search operating mode:' in prompt_text
    assert 'tiny population of bounded candidates' in prompt_text
    assert 'Pareto frontier' in prompt_text
    assert 'prompt-population / candidate-lineage story' in custom_text
