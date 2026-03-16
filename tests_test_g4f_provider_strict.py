from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "AgentLaboratory"))

import inference  # type: ignore


def test_g4f_provider_candidates_respect_explicit_list_without_auto(monkeypatch):
    monkeypatch.delenv("G4F_PROVIDER", raising=False)
    monkeypatch.setenv("AGENTLAB_G4F_PROVIDER_LIST", "Blackbox,PollinationsAI,Glider")
    monkeypatch.delenv("AGENTLAB_G4F_PROVIDER_ALLOW_AUTO_FALLBACK", raising=False)
    inference._G4F_PROVIDER_SUCCESS_CACHE.clear()
    inference._G4F_PROVIDER_SUCCESS_CACHE["g4f:r1-1776"] = "Perplexity"

    providers = inference._g4f_provider_candidates("g4f:r1-1776")

    assert providers == ["Blackbox", "PollinationsAI", "Glider"]


def test_g4f_provider_candidates_can_opt_into_auto_fallback(monkeypatch):
    monkeypatch.delenv("G4F_PROVIDER", raising=False)
    monkeypatch.setenv("AGENTLAB_G4F_PROVIDER_LIST", "Blackbox,PollinationsAI")
    monkeypatch.setenv("AGENTLAB_G4F_PROVIDER_ALLOW_AUTO_FALLBACK", "1")
    inference._G4F_PROVIDER_SUCCESS_CACHE.clear()

    providers = inference._g4f_provider_candidates("g4f:r1-1776")

    assert providers[:2] == ["Blackbox", "PollinationsAI"]
    assert providers[-1] is None
