from __future__ import annotations

import argparse

import pipeline_cli  # type: ignore
from pipeline_registry import get_pipeline  # type: ignore


def test_neighbour_model_prompt_variant_resolves():
    spec = get_pipeline('cayley-py-megaminx')
    assert spec is not None
    args = argparse.Namespace(prompt_variant='neighbour_model_hybrid', prompt_file=None, custom_prompts=None)
    prompt_file, custom_prompts = pipeline_cli._resolve_prompt_bundle(spec, args)
    assert prompt_file.name == 'user_prompt_neighbour_model_hybrid.txt'
    assert custom_prompts is not None
    assert custom_prompts.name == 'custom_prompts_neighbour_model_hybrid.json'
