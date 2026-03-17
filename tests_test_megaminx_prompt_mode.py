from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'AgentLaboratory' / 'perm_pipeline'))

import run_perm_pipeline  # type: ignore


def test_megaminx_regular_prompt_is_from_scratch() -> None:
    prompt_path = ROOT / 'competitions' / 'cayley-py-megaminx' / 'prompts' / 'user_prompt.txt'
    text = prompt_path.read_text(encoding='utf-8')
    assert 'Write the solver fully from scratch for this prompt.' in text
    assert 'Paste the current baseline solve_module.py here.' not in text
    assert 'Start from the existing baseline code below' not in text


def test_build_initial_codegen_prompt_from_scratch_ignores_baseline_code() -> None:
    prompt = run_perm_pipeline.build_initial_codegen_prompt(
        'task',
        'plan',
        baseline_code='def solve(vec):\n    return []\n',
        from_scratch=True,
    )
    assert 'KNOWN-GOOD BASELINE SOLVER:' not in prompt
    assert 'def solve(vec):' not in prompt
    assert 'from scratch' in prompt.lower()
