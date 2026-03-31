from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / 'competitions' / 'cayley-py-megaminx' / 'prompts'

CUSTOM_PROMPT_FILES = sorted(PROMPTS.glob('custom_prompts*.json'))
USER_PROMPT_FILES = sorted(PROMPTS.glob('user_prompt*.txt'))

REQUIRED_SNIPPETS = [
    'STRICT OUTPUT REQUIREMENTS',
    'Return exactly one JSON object and no prose outside JSON.',
    'The JSON object must contain exactly these keys: version, artifact_type, language, filename, code.',
    "Set version='code_response.v2', artifact_type='python_module', language='python', filename='solve_module.py'.",
    'Put the entire Python file only inside the code string.',
    'Do not wrap the code in markdown fences inside the JSON string.',
]

FORBIDDEN_OUTPUT_SNIPPETS = [
    'Return exactly one complete Python file inside one ```python``` block',
    'Return the full `solve_module.py` in exactly one ```python ...``` block.',
    'Return the full improved `solve_module.py` in exactly one ```python ...``` block.',
    'Do not include any text outside that code block.',
]


def test_megaminx_custom_prompt_variants_use_json_output_contract():
    assert CUSTOM_PROMPT_FILES
    for path in CUSTOM_PROMPT_FILES:
        data = json.loads(path.read_text(encoding='utf-8'))
        for role in ('coder', 'fixer'):
            text = data[role]
            for snippet in REQUIRED_SNIPPETS:
                assert snippet in text, f'{path.name}:{role} missing {snippet!r}'
            for snippet in FORBIDDEN_OUTPUT_SNIPPETS:
                assert snippet not in text, f'{path.name}:{role} still contains legacy output instruction {snippet!r}'


def test_megaminx_user_prompt_variants_require_json_output_contract():
    assert USER_PROMPT_FILES
    for path in USER_PROMPT_FILES:
        text = path.read_text(encoding='utf-8')
        for snippet in REQUIRED_SNIPPETS:
            assert snippet in text, f'{path.name} missing {snippet!r}'
        for snippet in FORBIDDEN_OUTPUT_SNIPPETS:
            assert snippet not in text, f'{path.name} still contains legacy output instruction {snippet!r}'
