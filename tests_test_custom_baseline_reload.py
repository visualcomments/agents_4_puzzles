from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pipeline_cli

ROOT = Path(__file__).resolve().parent
RUNNER_PATH = ROOT / 'colab' / 'megaminx_full_colab_runner_lowmem.py'


def _load_runner_module():
    spec = importlib.util.spec_from_file_location('megaminx_colab_runner_test', RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_solve_fn_reloads_overwritten_same_path(tmp_path):
    solver = tmp_path / 'baseline_override.py'
    solver.write_text(
        'def solve(vec):\n'
        '    return ["A"], list(vec)\n',
        encoding='utf-8',
    )

    solve_a = pipeline_cli._load_solve_fn(solver)
    assert solve_a([1, 2, 3])[0] == ['A']

    solver.write_text(
        'def solve(vec):\n'
        '    return ["B"], list(vec)\n',
        encoding='utf-8',
    )

    solve_b = pipeline_cli._load_solve_fn(solver)
    assert solve_b([1, 2, 3])[0] == ['B']



def test_colab_runner_accepts_explicit_solver_py_baseline(tmp_path, monkeypatch):
    runner = _load_runner_module()

    pipeline_cli_stub = tmp_path / 'pipeline_cli.py'
    pipeline_cli_stub.write_text('# stub pipeline_cli\n', encoding='utf-8')

    solver = tmp_path / 'custom_baseline.py'
    solver.write_text(
        'def solve(vec):\n'
        '    return [], list(vec)\n',
        encoding='utf-8',
    )

    test_csv = tmp_path / 'test.csv'
    test_csv.write_text('initial_state_id,state\n0,[]\n', encoding='utf-8')

    calls: list[list[str]] = []

    def fake_run_cmd(cmd, cwd):
        calls.append(list(cmd))
        output_path = Path(cmd[cmd.index('--output') + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
            writer.writeheader()
            writer.writerow({'initial_state_id': '0', 'path': 'A.B'})

    monkeypatch.setattr(runner, 'run_cmd', fake_run_cmd)

    run_dir = tmp_path / 'colab_runs' / 'case'
    path, meta = runner.resolve_source_submission(tmp_path, test_csv, run_dir, explicit=str(solver))

    assert path == run_dir / 'baseline_from_solver.csv'
    assert path.exists()
    assert meta['kind'] == 'solver_py'
    assert meta['explicit_baseline'] == str(solver)
    assert meta['generated_submission'] == str(path)
    assert calls, 'expected pipeline_cli build-submission to be invoked'
    assert 'build-submission' in calls[0]
    assert '--competition' in calls[0]
    assert calls[0][calls[0].index('--competition') + 1] == 'cayley-py-megaminx'
    assert calls[0][calls[0].index('--solver') + 1] == str(solver)
    assert calls[0][calls[0].index('--puzzles') + 1] == str(test_csv)
    assert calls[0][calls[0].index('--output') + 1] == str(path)
