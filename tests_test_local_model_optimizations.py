from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'AgentLaboratory'))

import inference  # type: ignore


def test_local_model_runtime_config_reflects_env(monkeypatch):
    monkeypatch.setenv('AGENTLAB_LOCAL_QUANT', '4bit')
    monkeypatch.setenv('AGENTLAB_ATTN_IMPL', 'flash_attention_2')
    monkeypatch.setenv('AGENTLAB_LOCAL_DTYPE', 'bf16')
    monkeypatch.setenv('AGENTLAB_LOCAL_MAX_MEMORY', 'cuda:6GiB,cpu:24GiB')
    monkeypatch.setenv('AGENTLAB_ENABLE_OFFLOAD', '1')
    cfg = inference.local_model_runtime_config()
    assert cfg['quant'] == '4bit'
    assert cfg['attn_impl'] == 'flash_attention_2'
    assert cfg['dtype'] == 'bf16'
    assert cfg['offload_enabled'] is True
    assert cfg['max_memory'] == {'cuda': '6GiB', 'cpu': '24GiB'}


def test_local_transformers_load_applies_quant_attn_and_offload(monkeypatch):
    inference._LOCAL_LM_CACHE.clear()

    captured = {}

    fake_torch = types.ModuleType('torch')
    fake_torch.float16 = 'float16'
    fake_torch.bfloat16 = 'bfloat16'
    class FakeCuda:
        @staticmethod
        def is_available():
            return True
    fake_torch.cuda = FakeCuda()
    def fake_compile(model, mode=None):
        captured['compile_mode'] = mode
        return model
    fake_torch.compile = fake_compile

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 42
        @classmethod
        def from_pretrained(cls, model_id, use_fast=True):
            captured['tok_model_id'] = model_id
            return cls()

    class FakeModel:
        def eval(self):
            captured['eval_called'] = True
            return self

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured['model_id'] = model_id
            captured['kwargs'] = kwargs
            return FakeModel()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_transformers = types.ModuleType('transformers')
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig

    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setenv('AGENTLAB_DEVICE', 'cuda')
    monkeypatch.setenv('AGENTLAB_LOCAL_QUANT', '4bit')
    monkeypatch.setenv('AGENTLAB_ATTN_IMPL', 'flash_attention_2')
    monkeypatch.setenv('AGENTLAB_LOCAL_DTYPE', 'bf16')
    monkeypatch.setenv('AGENTLAB_ENABLE_OFFLOAD', '1')
    monkeypatch.setenv('AGENTLAB_OFFLOAD_DIR', '/tmp/offload-test')
    monkeypatch.setenv('AGENTLAB_LOCAL_MAX_MEMORY', 'cuda:6GiB,cpu:24GiB')
    monkeypatch.setenv('AGENTLAB_TORCH_COMPILE', '1')

    tok, model = inference._local_transformers_load('local-demo-model')
    assert tok is not None and model is not None
    assert captured['kwargs']['attn_implementation'] == 'flash_attention_2'
    assert captured['kwargs']['torch_dtype'] == 'bfloat16'
    assert captured['kwargs']['device_map'] == 'auto'
    assert captured['kwargs']['offload_folder'] == '/tmp/offload-test'
    assert captured['kwargs']['max_memory'] == {'cuda': '6GiB', 'cpu': '24GiB'}
    assert captured['compile_mode'] == 'reduce-overhead'
    assert captured['kwargs']['quantization_config'].kwargs['load_in_4bit'] is True


class _FakeTensor:
    def __init__(self, values=None, device='cpu'):
        self.values = values or [1, 2, 3]
        self.device = device

    def to(self, device):
        return _FakeTensor(self.values, device=device)

    @property
    def shape(self):
        return (1, len(self.values))


class _FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeModelForChat:
    def parameters(self):
        yield _FakeTensor(device='cpu')

    def generate(self, **kwargs):
        return [[1, 2, 3, 4, 5]]


def test_local_transformers_chat_uses_input_truncation(monkeypatch):
    fake_torch = types.ModuleType('torch')
    fake_torch.inference_mode = lambda: _FakeInferenceMode()
    class FakeCuda:
        @staticmethod
        def is_available():
            return False
    fake_torch.cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    calls = {}

    class FakeTokenizer:
        eos_token_id = 99
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return 'PROMPT'
        def __call__(self, text, **kwargs):
            calls['tok_kwargs'] = kwargs
            return {'input_ids': _FakeTensor([1, 2, 3])}
        def decode(self, tokens, skip_special_tokens=True):
            return 'decoded'

    monkeypatch.setattr(inference, '_local_transformers_load', lambda model_id: (FakeTokenizer(), _FakeModelForChat()))
    monkeypatch.setenv('AGENTLAB_LOCAL_MAX_INPUT_TOKENS', '128')

    out = inference._local_transformers_chat('local-demo', 'user prompt', 'system prompt')
    assert out == 'decoded'
    assert calls['tok_kwargs']['truncation'] is True
    assert calls['tok_kwargs']['max_length'] == 128


def test_openai_compatible_ollama_backend_uses_expected_defaults(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key):
            captured['base_url'] = base_url
            captured['api_key'] = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            captured['kwargs'] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='pong'))])

    monkeypatch.setattr(inference, 'OpenAI', FakeOpenAI)

    out = inference.query_model('ollama:qwen2.5-coder:7b', 'user prompt', 'system prompt', tries=1, timeout=9.0)
    assert out == 'pong'
    assert captured['base_url'] == 'http://localhost:11434/v1/'
    assert captured['api_key'] == 'ollama'
    assert captured['kwargs']['model'] == 'qwen2.5-coder:7b'
    assert captured['kwargs']['messages'][0]['role'] == 'system'
    assert captured['kwargs']['timeout'] == 9.0


def test_openai_compatible_generic_backend_requires_base_url(monkeypatch):
    monkeypatch.delenv('AGENTLAB_OPENAI_COMPAT_BASE_URL', raising=False)
    monkeypatch.delenv('OPENAI_COMPAT_BASE_URL', raising=False)

    try:
        inference.query_model('openai-compatible:demo-model', 'user prompt', 'system prompt', tries=1)
        assert False, 'expected RuntimeError'
    except RuntimeError as exc:
        assert 'AGENTLAB_OPENAI_COMPAT_BASE_URL' in str(exc)


def test_g4f_provider_failover_caches_last_good_provider(monkeypatch):
    calls = []
    inference._G4F_PROVIDER_SUCCESS_CACHE.clear()

    class FakeChatCompletion:
        @staticmethod
        def create(*, model, messages, timeout, provider=None, stream=False, **kwargs):
            calls.append(getattr(provider, 'name', provider))
            provider_name = getattr(provider, 'name', provider)
            if provider_name == 'Alpha':
                raise RuntimeError('alpha failed')
            return 'provider-ok'

    fake_g4f = SimpleNamespace(
        ChatCompletion=FakeChatCompletion,
        Provider=SimpleNamespace(Alpha=SimpleNamespace(name='Alpha'), Beta=SimpleNamespace(name='Beta')),
    )

    monkeypatch.setattr(inference, '_load_g4f_module', lambda: fake_g4f)
    monkeypatch.setattr(inference, '_g4f_async_enabled', lambda: False)
    monkeypatch.setattr(inference, '_best_effort_release_memory', lambda clear_local_cache=False: None)
    monkeypatch.setenv('G4F_PROVIDER_LIST', 'Alpha,Beta')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)

    out = inference.query_model('g4f:demo-model', 'user prompt', 'system prompt', tries=1, timeout=5.0)
    assert out == 'provider-ok'
    assert calls == ['Alpha', 'Beta']
    assert inference._G4F_PROVIDER_SUCCESS_CACHE.get('demo-model') == 'Beta'
