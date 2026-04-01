from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pipeline_cli  # type: ignore
from pipeline_registry import get_pipeline  # type: ignore
from AgentLaboratory.perm_pipeline import run_perm_pipeline as rpp  # type: ignore
from AgentLaboratory import inference  # type: ignore


def test_megaminx_variants_request_json_code_envelope():
    spec = get_pipeline('cayley-py-megaminx')
    assert spec is not None
    for variant in ('structured', 'heuristic_boosted', 'improved'):
        args = argparse.Namespace(prompt_variant=variant, prompt_file=None, custom_prompts=None)
        prompt_file, custom_prompts = pipeline_cli._resolve_prompt_bundle(spec, args)
        assert pipeline_cli._prompt_bundle_requires_json_code_envelope(prompt_file, custom_prompts)


def test_build_initial_codegen_prompt_frontloads_output_contract():
    prompt = rpp.build_initial_codegen_prompt('task', 'plan', baseline_code='def solve(vec):\n    return [], list(vec)\n')
    first_chunk = prompt[:300]
    assert '## OUTPUT CONTRACT (HIGHEST PRIORITY)' in first_chunk
    assert 'Return JSON only.' in first_chunk


def test_response_format_for_code_envelope_prefers_json_schema_for_4o_models():
    prompt = 'Return JSON only. code_response.v2 solve_module.py code'
    system = 'Return exactly one JSON object and no prose outside JSON.'
    fmt = inference._response_format_for_code_envelope('gpt-4o-2024-08-06', prompt, system)
    assert isinstance(fmt, dict)
    assert fmt.get('type') == 'json_schema'
    json_schema = fmt.get('json_schema') or {}
    assert json_schema.get('strict') is True


def test_response_format_for_code_envelope_falls_back_to_json_object_for_other_models():
    prompt = 'Return JSON only. code_response.v2 solve_module.py code'
    system = 'Return exactly one JSON object and no prose outside JSON.'
    fmt = inference._response_format_for_code_envelope('deepseek-chat', prompt, system)
    assert fmt == {'type': 'json_object'}
