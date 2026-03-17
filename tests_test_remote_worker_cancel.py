from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'AgentLaboratory'))

import inference  # type: ignore


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g4f_async_cancelled_error_is_wrapped_as_runtime_error(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            raise asyncio.CancelledError()

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(inference, '_load_g4f_async_client_class', lambda: FakeAsyncClient)
    monkeypatch.setattr(inference, '_g4f_async_stream_enabled', lambda: True)

    with pytest.raises(RuntimeError, match='cancelled'):
        asyncio.run(
            inference._g4f_async_create_text(
                model_str='r1-1776',
                messages=[{'role': 'user', 'content': 'ping'}],
                provider_name='Blackbox',
                timeout_s=1.0,
                stream_timeout_s=1.0,
                stream_idle_timeout_s=1.0,
                max_resp_chars=100,
                stop_at_python_fence=False,
            )
        )


@pytest.mark.parametrize(
    'module_path',
    [
        ROOT / 'AgentLaboratory' / 'query_model_worker.py',
        ROOT / 'AgentLaboratory' / 'perm_pipeline' / 'query_model_worker.py',
    ],
)
def test_worker_serializes_cancelled_error_payload(monkeypatch, module_path, tmp_path):
    mod = _load_module(f'test_worker_{module_path.stem}_{module_path.parent.name}', module_path)

    def fake_query_model(*args, **kwargs):
        raise asyncio.CancelledError('provider cancelled request')

    monkeypatch.setattr(mod, 'query_model', fake_query_model)

    prompt_file = tmp_path / 'prompt.txt'
    system_file = tmp_path / 'system.txt'
    out_json = tmp_path / 'result.json'
    prompt_file.write_text('hello', encoding='utf-8')
    system_file.write_text('system', encoding='utf-8')

    monkeypatch.setattr(
        sys,
        'argv',
        [
            str(module_path),
            '--model',
            'g4f:r1-1776',
            '--prompt-file',
            str(prompt_file),
            '--system-file',
            str(system_file),
            '--out-json',
            str(out_json),
        ],
    )

    rc = mod.main()
    payload = json.loads(out_json.read_text(encoding='utf-8'))
    assert rc == 1
    assert payload['ok'] is False
    assert payload['error_type'] == 'CancelledError'
    assert 'cancelled' in payload['error'].lower()
