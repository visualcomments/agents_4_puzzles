from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / 'competitions' / 'cayley-py-megaminx' / 'megaminx_neighbour_model_lane.py'


spec = importlib.util.spec_from_file_location('test_megaminx_neighbour_model_lane', MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules.setdefault('test_megaminx_neighbour_model_lane', module)
spec.loader.exec_module(module)



def test_move_name_roundtrip():
    assert module.NeighbourModelRuntime.training_to_official("U'") == '-U'
    assert module.NeighbourModelRuntime.training_to_official('BR') == 'BR'
    assert module.NeighbourModelRuntime.official_to_training('-DL') == "DL'"
    assert module.NeighbourModelRuntime.official_to_training('F') == 'F'



def test_vendored_repo_candidate_present():
    cfg = module.NeighbourModelConfig()
    runtime = module.NeighbourModelRuntime(cfg)
    candidates = runtime._candidate_repo_dirs()
    expected = (ROOT / 'tp' / 'cayleypy-neighbour-model-training-main').resolve()
    assert expected in candidates
