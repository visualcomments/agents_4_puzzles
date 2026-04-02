import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMP = ROOT / 'competitions' / 'cayley-py-megaminx'


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_megaminx_v3_smoke_no_degradation_on_small_slice():
    sm = _load_module('megaminx_sm_v3', COMP / 'solve_module.py')
    search_v3 = _load_module('megaminx_search_v3', COMP / 'search_improver_v3.py')

    with (COMP / 'submissions' / 'optimized_submission.csv').open(newline='', encoding='utf-8') as f:
        submission_rows = list(csv.DictReader(f))[:5]
    with (COMP / 'data' / 'test.csv').open(newline='', encoding='utf-8') as f:
        test_rows = list(csv.DictReader(f))[:5]

    parser = search_v3.build_arg_parser()
    args = parser.parse_args([
        '--top-k', '5',
        '--disable-cayleypy',
        '--light-time-budget-per-row', '0.02',
        '--aggressive-time-budget-per-row', '0.02',
        '--light-window-samples', '2',
        '--aggressive-window-samples', '2',
        '--light-window-lengths', '14',
        '--aggressive-window-lengths', '18',
        '--light-max-steps', '4',
        '--aggressive-max-steps', '5',
        '--light-beam-width', '16',
        '--aggressive-beam-width', '24',
    ])

    central, generators = sm.load_puzzle_bundle()
    improved_rows, stats, profiles = search_v3.improve_submission_rows(
        submission_rows=submission_rows,
        test_rows=test_rows,
        central=central,
        generators=generators,
        args=args,
    )
    baseline = sum(0 if not (r.get('path') or '').strip() else len(r['path'].split('.')) for r in submission_rows)
    improved = sum(0 if not (r.get('path') or '').strip() else len(r['path'].split('.')) for r in improved_rows)
    assert improved <= baseline
    assert stats['final_score'] <= stats['baseline_score']
    assert len(profiles) == 5
    assert all('tier' in profile for profile in profiles)
