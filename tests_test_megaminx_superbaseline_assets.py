import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def test_megaminx_superbaseline_score_and_solver_lookup():
    comp = ROOT / 'competitions' / 'cayley-py-megaminx'
    optimized_csv = comp / 'submissions' / 'optimized_submission.csv'
    with optimized_csv.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    score = sum(0 if not (row.get('path') or '').strip() else len(row['path'].split('.')) for row in rows)
    assert score == 414305

    solver_path = comp / 'solve_module.py'
    spec = importlib.util.spec_from_file_location('megaminx_superbaseline', solver_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    test_csv = comp / 'data' / 'test.csv'
    with test_csv.open(newline='', encoding='utf-8') as f:
        first = next(csv.DictReader(f))
    vec = [int(x) for x in first['initial_state'].split(',')]
    moves, out_vec = module.solve(vec)
    assert isinstance(moves, list)
    assert out_vec == list(range(120))
    assert len(moves) > 0
