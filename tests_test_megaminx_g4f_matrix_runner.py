from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / 'scripts' / 'run_megaminx_g4f_matrix.py'


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_infer_repo_root_from_cwd_when_script_is_outside_repo(tmp_path, monkeypatch):
    external_script = tmp_path / 'run_megaminx_g4f_matrix.py'
    shutil.copy2(SCRIPT_PATH, external_script)
    module = _load_module(external_script, 'megaminx_matrix_runner_external')

    monkeypatch.chdir(REPO_ROOT)

    resolved = module.infer_repo_root(None)
    assert resolved == REPO_ROOT


def test_build_detector_command_uses_absolute_pipeline_cli_path():
    module = _load_module(SCRIPT_PATH, 'megaminx_matrix_runner_repo')

    parser = module.build_parser()
    args = parser.parse_args(['--models', 'gpt-4o-mini'])
    command = module.build_detector_command(args.python_exe, REPO_ROOT, args)

    assert command[1] == str(REPO_ROOT / 'pipeline_cli.py')
