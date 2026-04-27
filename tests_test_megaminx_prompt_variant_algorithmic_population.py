from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pipeline_cli  # type: ignore
from pipeline_registry import get_pipeline  # type: ignore


def test_megaminx_algorithmic_population_prompt_bundle_resolves():
    spec = get_pipeline('cayley-py-megaminx')
    assert spec is not None
    args = argparse.Namespace(prompt_variant='algorithmic_population', prompt_file=None, custom_prompts=None)
    prompt_file, custom_prompts = pipeline_cli._resolve_prompt_bundle(spec, args)
    assert prompt_file.name == 'user_prompt_algorithmic_population.txt'
    assert prompt_file.exists()
    assert custom_prompts is not None
    assert custom_prompts.name == 'custom_prompts_algorithmic_population.json'
    assert custom_prompts.exists()
