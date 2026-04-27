from __future__ import annotations

import argparse
from pathlib import Path

import pipeline_cli  # type: ignore
from pipeline_registry import get_pipeline  # type: ignore


ROOT = Path(__file__).resolve().parent


def _resolve(variant: str):
    spec = get_pipeline('cayley-py-megaminx')
    assert spec is not None
    args = argparse.Namespace(prompt_variant=variant, prompt_file=None, custom_prompts=None)
    return pipeline_cli._resolve_prompt_bundle(spec, args)


def test_breakthrough_prompt_variants_resolve():
    for variant in ('portfolio_orchestrated', 'hard_row_routed', 'exact_score_population'):
        prompt_file, custom_prompts = _resolve(variant)
        assert prompt_file.name == f'user_prompt_{variant}.txt'
        assert custom_prompts is not None
        assert custom_prompts.name == f'custom_prompts_{variant}.json'
