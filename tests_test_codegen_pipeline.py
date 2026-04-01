from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'AgentLaboratory'))
sys.path.insert(0, str(ROOT / 'AgentLaboratory' / 'perm_pipeline'))

import run_perm_pipeline as rpp  # type: ignore


def test_extract_python_from_fenced_block():
    text = 'before\n```python\ndef solve(vec):\n    return [], list(vec)\n```\nafter'
    code = rpp.extract_python(text)
    assert code is not None
    assert 'def solve' in code


def test_compile_python_reports_syntax_error():
    ok, reason = rpp.compile_python('def solve(:\n    pass')
    assert ok is False
    assert 'SyntaxError' in reason


def test_rank_models_prefers_stronger_code_models():
    ranked = rpp.rank_models_for_codegen([
        'g4f:aria',
        'g4f:gpt-4o-mini',
        'g4f:command-r',
    ])
    assert ranked[0] == 'g4f:gpt-4o-mini'


def test_default_max_rss_mb_is_disabled_outside_explicit_or_colab(monkeypatch):
    monkeypatch.delenv('AGENTLAB_MAX_RSS_MB', raising=False)
    monkeypatch.delenv('COLAB_GPU', raising=False)
    monkeypatch.delenv('COLAB_RELEASE_TAG', raising=False)
    monkeypatch.setattr(rpp, '_system_total_mb', lambda: 10000.0)
    assert rpp._default_max_rss_mb() == 0


def test_default_max_rss_mb_uses_explicit_env_or_colab_default(monkeypatch):
    monkeypatch.setenv('AGENTLAB_MAX_RSS_MB', '4096')
    assert rpp._default_max_rss_mb() == 4096

    monkeypatch.delenv('AGENTLAB_MAX_RSS_MB', raising=False)
    monkeypatch.setenv('COLAB_GPU', '1')
    monkeypatch.setattr(rpp, '_system_total_mb', lambda: 10000.0)
    assert rpp._default_max_rss_mb() == 7200


def test_remote_worker_timeout_scales_with_stream_budget(monkeypatch):
    monkeypatch.setenv('AGENTLAB_G4F_STREAM_TIMEOUT_S', '250')
    monkeypatch.setenv('AGENTLAB_G4F_STREAM_IDLE_TIMEOUT_S', '80')
    timeout_s = rpp._remote_worker_timeout_s(tries=5, timeout=20.0, model='g4f:gpt-4')
    assert timeout_s >= 1285


def test_query_model_stable_uses_worker_result(monkeypatch, tmp_path):
    monkeypatch.setenv('AGENTLAB_REMOTE_SUBPROCESS', '1')
    captured = {}

    def fake_worker(**kwargs):
        captured['proc_timeout'] = kwargs['proc_timeout']
        kwargs['out_json'].write_text(json.dumps({'ok': True, 'answer': '```python\ndef solve(vec):\n    return [], list(vec)\n```'}), encoding='utf-8')
        return {'ok': True, 'answer': '```python\ndef solve(vec):\n    return [], list(vec)\n```'}

    monkeypatch.setattr(rpp, '_run_json_worker_subprocess', fake_worker)
    answer = rpp._query_model_stable('g4f:gpt-4', 'prompt', 'system', tries=1, timeout=5.0)
    assert 'def solve' in answer
    assert captured['proc_timeout'] >= 30




def test_query_model_stable_bypasses_worker_for_local(monkeypatch):
    monkeypatch.setenv('AGENTLAB_REMOTE_SUBPROCESS', '1')

    def fake_query_model(model, prompt, system_prompt, **kwargs):
        return 'LOCAL_OK'

    monkeypatch.setattr(rpp, 'query_model', fake_query_model)
    answer = rpp._query_model_stable('local:demo-model', 'prompt', 'system')
    assert answer == 'LOCAL_OK'


def test_attempt_recovery_rounds_retries_models(monkeypatch, tmp_path):
    reports = [
        'g4f:gpt-4: fixer iteration 44 did not return a python file',
        'g4f:gpt-4: baseline-patcher did not return a python file',
    ]
    monkeypatch.setenv('AGENTLAB_G4F_RECOVERY_ROUNDS', '1')
    monkeypatch.setenv('AGENTLAB_G4F_RECOVERY_MAX_ITERS', '1')
    monkeypatch.setenv('AGENTLAB_G4F_RECOVERY_SLEEP_S', '0')

    calls = []

    def fake_try_generate_with_model(**kwargs):
        calls.append((kwargs['model'], kwargs['stage_label'], kwargs['plan'], kwargs['baseline_code']))
        return True, 'g4f:gpt-4: recovery round 1 output validated immediately'

    monkeypatch.setattr(rpp, 'try_generate_with_model', fake_try_generate_with_model)
    monkeypatch.setattr(rpp, '_best_effort_release_memory', lambda clear_local_cache=True: None)

    ok, report = rpp.attempt_recovery_rounds(
        recovery_models=['g4f:gpt-4'],
        fixer_models=['g4f:gpt-4'],
        user_prompt='solve it',
        plan='initial plan',
        prompts={'coder': 'coder', 'fixer': 'fixer'},
        out_path=tmp_path / 'solve.py',
        validator_path=tmp_path / 'validator.py',
        tests=[[1, 2, 3]],
        max_iters=4,
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
        generation_reports=reports,
    )

    assert ok is True
    assert 'recovery round 1' in report
    assert calls
    assert calls[0][1] == 'recovery round 1'
    assert 'RECOVERY MODE' in calls[0][2]
    assert 'did not return a python file' in calls[0][2]


def test_attempt_recovery_rounds_skips_nonrecoverable_reports(monkeypatch, tmp_path):
    monkeypatch.setenv('AGENTLAB_G4F_RECOVERY_ROUNDS', '1')
    called = []

    def fake_try_generate_with_model(**kwargs):
        called.append(kwargs)
        return False, 'should not run'

    monkeypatch.setattr(rpp, 'try_generate_with_model', fake_try_generate_with_model)
    ok, report = rpp.attempt_recovery_rounds(
        recovery_models=['g4f:gpt-4'],
        fixer_models=['g4f:gpt-4'],
        user_prompt='solve it',
        plan='plan',
        prompts={'coder': 'coder', 'fixer': 'fixer'},
        out_path=tmp_path / 'solve.py',
        validator_path=tmp_path / 'validator.py',
        tests=[[1, 2, 3]],
        max_iters=4,
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
        generation_reports=['compile check failed'],
    )
    assert ok is False
    assert report is None
    assert called == []


def test_print_generation_preview_respects_env(monkeypatch, capsys):
    monkeypatch.setenv('AGENTLAB_PRINT_GENERATION', '1')
    monkeypatch.setenv('AGENTLAB_PRINT_GENERATION_MAX_CHARS', '12')
    rpp._print_generation_preview('coder', 'g4f:gpt-4', '1234567890abcdef')
    out = capsys.readouterr().out
    assert '[generation:coder]' in out
    assert '1234567890ab' in out





def test_extract_python_from_json_code_envelope():
    text = json.dumps({
        'version': 'code_response.v2',
        'artifact_type': 'python_module',
        'language': 'python',
        'filename': 'solve_module.py',
        'code': 'import json\n\n\ndef solve(vec):\n    return [], list(vec)\n',
    })
    code = rpp.extract_python(text)
    assert code is not None
    assert 'def solve' in code
    ok, reason = rpp.compile_python(code)
    assert ok is True, reason


def test_extract_python_from_fenced_json_code_envelope_with_prose():
    text = """before
```json
{"version":"code_response.v2","artifact_type":"python_module","language":"python","filename":"solve_module.py","code":"#!/usr/bin/env python3\\nfrom __future__ import annotations\\n\\nimport json\\n\\n\\ndef solve(vec):\\n    return [], list(vec)\\n"}
```
after"""
    code = rpp.extract_python(text)
    assert code is not None
    assert code.startswith('#!/usr/bin/env python3')
    assert 'def solve' in code

def test_extract_python_strips_comments_docstrings_and_explanations():
    text = '''[fixer] iteration 4 trying model: g4f:gpt-4
[generation:fixer iteration 4] model=g4f:gpt-4
```python
import json

def solve(vec):
    """temporary docstring that should be removed"""
    # inline comment that should be removed
    return [], list(vec)
```

### Explanation of Fix:
- this trailing explanation must not leak into the extracted file
'''
    code = rpp.extract_python(text)
    assert code is not None
    assert 'def solve' in code
    assert 'Explanation of Fix' not in code
    assert '# inline comment' not in code




def test_extract_python_preserves_valid_docstring_only_class_body_when_stripping():
    text = '''```python
class SolverError(Exception):
    """Raised when the solver contract is violated."""


def solve(vec):
    return [], list(vec)
```'''
    code = rpp.extract_python(text)
    assert code is not None
    assert 'class SolverError' in code
    assert '"""Raised when the solver contract is violated."""' in code
    ok, reason = rpp.compile_python(code)
    assert ok is True, reason
def test_extract_python_handles_prose_wrapped_raw_python_module():
    text = '''Content of solve_module.py
- This is a complete, self-contained module ready to drop into your repository.
- The script prints JSON when run directly.

Note: The following code is designed to be drop-in.

Code starts here (save as solve_module.py):

#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def solve(vec):
    return [], list(vec)


if __name__ == "__main__":
    print(json.dumps({"moves": [], "sorted_array": []}))

Additional explanation after code that must be ignored.
'''
    code = rpp.extract_python(text)
    assert code is not None
    assert code.startswith('#!/usr/bin/env python3')
    assert 'def solve' in code
    assert 'Additional explanation after code' not in code
    ok, reason = rpp.compile_python(code)
    assert ok is True, reason


def test_extract_python_prefers_real_solver_over_intro_text_block():
    text = '''```text
Content of solve_module.py
- explanatory text
```

Content of solve_module.py
- explanatory text

#!/usr/bin/env python3
import json


def solve(vec):
    return [], list(vec)
'''
    code = rpp.extract_python(text)
    assert code is not None
    assert code.startswith('#!/usr/bin/env python3')
    assert 'def solve' in code
    ok, reason = rpp.compile_python(code)
    assert ok is True, reason


def test_generate_plan_candidates_deduplicates_and_ranks(monkeypatch):
    outputs = iter([
        'Algorithm family: constructive\nInvariants: maintain parity\nComplexity: O(n)',
        'Algorithm family: constructive\nInvariants: maintain parity\nComplexity: O(n)',
        'Algorithm family: BFS\nComplexity: exponential brute force',
    ])

    monkeypatch.setattr(rpp, '_query_model_stable', lambda *args, **kwargs: next(outputs))

    plans = rpp.generate_plan_candidates(
        ['local:planner-a', 'g4f:planner-b'],
        'solve it',
        'planner system',
        beam_width=2,
    )

    assert len(plans) == 2
    assert 'constructive' in plans[0].plan_text.lower()
    assert plans[0].score > plans[1].score
    assert plans[1].score < 0



def test_run_hybrid_codegen_search_uses_archive_for_refinement(monkeypatch, tmp_path):
    calls = []

    def fake_generate(planner_models, user_prompt, planner_system_prompt, *, beam_width, archive_summary='', baseline_code=None):
        calls.append(archive_summary)
        if not archive_summary:
            return [rpp.PlanCandidate(plan_text='bad plan', planner_model='g4f:planner', score=1.0, variant_index=1)]
        assert 'compile check failed' in archive_summary.lower()
        return [rpp.PlanCandidate(plan_text='good plan with invariants', planner_model='g4f:planner', score=2.0, variant_index=2)]

    def fake_try_generate_with_model(**kwargs):
        if 'good plan' in kwargs['plan']:
            return True, 'g4f:coder: coder refine 1 variant 2 output validated immediately'
        return False, 'Initial compile check failed. missing colon'

    monkeypatch.setattr(rpp, 'generate_plan_candidates', fake_generate)
    monkeypatch.setattr(rpp, 'try_generate_with_model', fake_try_generate_with_model)

    ok, reports, archive, plan, planner_model, winner_model = rpp.run_hybrid_codegen_search(
        planner_models=['g4f:planner'],
        coder_models=['local:coder'],
        fixer_models=['local:fixer'],
        user_prompt='solve it',
        prompts={'planner': 'planner', 'coder': 'coder', 'fixer': 'fixer'},
        out_path=tmp_path / 'solve.py',
        validator_path=tmp_path / 'validator.py',
        tests=[[1, 2, 3]],
        max_iters=2,
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
        plan_beam_width=1,
        frontier_width=1,
        archive_size=4,
        refine_rounds=1,
    )

    assert ok is True
    assert winner_model == 'local:coder'
    assert planner_model == 'g4f:planner'
    assert 'good plan' in plan
    assert len(reports) == 2
    assert calls[0] == ''
    assert 'compile check failed' in calls[1].lower()
    assert archive.best_failures(limit=1)


def test_run_validator_times_out_on_hanging_validator(monkeypatch, tmp_path):
    validator = tmp_path / 'validator.py'
    solver = tmp_path / 'solver.py'
    validator.write_text('import time\ntime.sleep(5)\n', encoding='utf-8')
    solver.write_text('print("{}")\n', encoding='utf-8')

    monkeypatch.setenv('AGENTLAB_VALIDATOR_TIMEOUT_S', '0.1')
    rc, out, err = rpp.run_validator(validator, solver, [1, 2, 3])

    assert rc == 124
    assert out == ''
    assert 'validator timed out' in err


def test_validate_solver_suite_reports_timeout(monkeypatch, tmp_path):
    validator = tmp_path / 'validator.py'
    solver = tmp_path / 'solver.py'
    validator.write_text('import time\ntime.sleep(5)\n', encoding='utf-8')
    solver.write_text('print("{}")\n', encoding='utf-8')

    monkeypatch.setenv('AGENTLAB_VALIDATOR_TIMEOUT_S', '0.1')
    ok, report = rpp.validate_solver_suite(validator, solver, [[1, 2, 3]])

    assert ok is False
    assert 'TEST 0 FAILED' in report
    assert 'validator timed out' in report

def test_lenient_load_json_object_extracts_fenced_json():
    text = 'before\n```json\n{"strategy_family":"bounded_window_dp","goal":"x"}\n```\nafter'
    payload = rpp._lenient_load_json_object(text)
    assert isinstance(payload, dict)
    assert payload['strategy_family'] == 'bounded_window_dp'


def test_ask_first_structured_plan_returns_payload_and_package(monkeypatch):
    monkeypatch.setattr(
        rpp,
        '_query_model_stable',
        lambda *args, **kwargs: '{"strategy_family":"stronger_exact_table","goal":"keep exact lookup first","edit_targets":["_short_word_data"],"must_preserve":["exact_lookup_first","solve_signature","script_json_output"],"complexity_claim":{"precompute":"constant","per_row":"O(L)","why_polynomial":"fixed constants only"},"proposed_changes":["tighten table","reuse cached effects"],"validation_plan":["compile","validator"],"forbidden":["BFS","DFS","beam search"]}'
    )

    plan_text, planner_model, plan_payload, strategy_package = rpp.ask_first_structured_plan(
        ['g4f:planner-a'],
        'solve it',
        'planner system',
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
    )

    assert planner_model == 'g4f:planner-a'
    assert plan_payload is not None
    assert strategy_package is not None
    assert plan_payload['strategy_family'] == 'stronger_exact_table'
    assert 'Algorithm family' in plan_text


def test_build_initial_codegen_prompt_embeds_structured_plan_and_baseline():
    prompt = rpp.build_initial_codegen_prompt(
        'task',
        'Algorithm family: stronger_exact_table',
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
        plan_payload={'strategy_family': 'stronger_exact_table', 'goal': 'goal', 'edit_targets': ['_short_word_data'], 'must_preserve': ['exact_lookup_first'], 'complexity_claim': {'precompute': 'constant', 'per_row': 'O(L)', 'why_polynomial': 'fixed constants'}, 'proposed_changes': ['a', 'b'], 'validation_plan': ['compile', 'validator'], 'forbidden': ['BFS']},
        strategy_package={'strategy_family': 'stronger_exact_table', 'label': 'Variant A', 'goal': 'goal'},
    )
    assert '## STRATEGY PACKAGE' in prompt
    assert '## PLANNER JSON' in prompt
    assert 'KNOWN-GOOD BASELINE SOLVER' in prompt


def test_try_generate_with_model_forwards_plan_payload_to_fixer(monkeypatch, tmp_path):
    bad_code = 'def solve(:\n    pass\n'
    monkeypatch.setattr(rpp, '_query_code_block_with_rescue', lambda **kwargs: (bad_code, None))
    captured = {}

    def fake_fixer(**kwargs):
        captured.update(kwargs)
        return False, kwargs['last_report']

    monkeypatch.setattr(rpp, '_run_fixer_loop', fake_fixer)

    ok, report = rpp.try_generate_with_model(
        model='g4f:gpt-4',
        fixer_models=['g4f:gpt-4'],
        user_prompt='solve it',
        plan='Algorithm family: stronger_exact_table',
        prompts={'coder': 'coder', 'fixer': 'fixer'},
        out_path=tmp_path / 'solve.py',
        validator_path=tmp_path / 'validator.py',
        tests=[[1, 2, 3]],
        max_iters=2,
        baseline_code='def solve(vec):\n    return [], list(vec)\n',
        plan_payload={'strategy_family': 'stronger_exact_table', 'goal': 'goal', 'edit_targets': ['_short_word_data'], 'must_preserve': ['exact_lookup_first'], 'complexity_claim': {'precompute': 'constant', 'per_row': 'O(L)', 'why_polynomial': 'fixed constants'}, 'proposed_changes': ['a', 'b'], 'validation_plan': ['compile', 'validator'], 'forbidden': ['BFS']},
        strategy_package={'strategy_family': 'stronger_exact_table', 'label': 'Variant A', 'goal': 'goal'},
    )

    assert ok is False
    assert 'Initial compile check failed' in report
    assert captured['plan_payload']['strategy_family'] == 'stronger_exact_table'
    assert captured['strategy_package']['strategy_family'] == 'stronger_exact_table'
    assert 'Algorithm family' in captured['plan']


def test_resolve_validator_smoke_vectors_prefers_competition_dataset():
    validator = (ROOT / 'competitions' / 'cayley-py-megaminx' / 'validate_solve_output.py').resolve()
    vectors = rpp.resolve_validator_smoke_vectors(validator)
    assert len(vectors) >= 2
    assert all(len(vec) == 120 for vec in vectors[:2])


def test_megaminx_baseline_fails_generic_tests_but_passes_resolved_smoke_vectors():
    validator = (ROOT / 'competitions' / 'cayley-py-megaminx' / 'validate_solve_output.py').resolve()
    solver = (ROOT / 'competitions' / 'cayley-py-megaminx' / 'solve_module.py').resolve()

    generic_tests = [
        [3, 1, 2, 5, 4],
        [1, 2, 3, 4],
    ]
    generic_ok, _ = rpp.validate_solver_suite(validator, solver, generic_tests)
    assert generic_ok is False

    resolved_ok, report = rpp.validate_solver_suite(validator, solver, rpp.resolve_validator_smoke_vectors(validator))
    assert resolved_ok is True, report
