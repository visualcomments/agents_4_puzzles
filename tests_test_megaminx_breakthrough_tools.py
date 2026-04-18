from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMP = ROOT / 'competitions' / 'cayley-py-megaminx'


def _write_submission(path: Path, lengths: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        for idx, length in enumerate(lengths):
            writer.writerow({'initial_state_id': str(idx), 'path': '.'.join(['U'] * length)})


def test_row_scoreboard_and_shadow_splits(tmp_path):
    submission = tmp_path / 'submission.csv'
    _write_submission(submission, [10, 20, 30, 40, 50])
    out_json = tmp_path / 'scoreboard.json'
    splits_json = tmp_path / 'splits.json'
    subprocess.run([
        sys.executable, str(COMP / 'row_scoreboard.py'),
        '--submission', str(submission),
        '--test-csv', str(COMP / 'data' / 'test.csv'),
        '--out-json', str(out_json),
        '--splits-out', str(splits_json),
    ], check=True)
    scoreboard = json.loads(out_json.read_text(encoding='utf-8'))
    splits = json.loads(splits_json.read_text(encoding='utf-8'))
    assert len(scoreboard) == 5
    assert splits['counts']['total'] == 5
    assert sorted(splits['train_indices'] + splits['dev_indices'] + splits['holdout_indices']) == [0, 1, 2, 3, 4]


def test_portfolio_orchestrator_picks_shorter_rows(tmp_path):
    a = tmp_path / 'a.csv'
    b = tmp_path / 'b.csv'
    _write_submission(a, [10, 40, 30])
    _write_submission(b, [20, 20, 50])
    out = tmp_path / 'merged.csv'
    subprocess.run([
        sys.executable, str(COMP / 'portfolio_orchestrator.py'),
        '--candidate', f'a={a}',
        '--candidate', f'b={b}',
        '--test-csv', str(COMP / 'data' / 'test.csv'),
        '--out', str(out),
        '--lineage-out', str(tmp_path / 'lineage.json'),
        '--summary-out', str(tmp_path / 'summary.json'),
        '--scoreboard-out', str(tmp_path / 'scoreboard.json'),
    ], check=True)
    with out.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    lengths = [0 if not r['path'] else len(r['path'].split('.')) for r in rows]
    assert lengths == [10, 20, 30]


def test_prompt_population_runner_selects_lowest_dev_score(tmp_path):
    a = tmp_path / 'a.csv'
    b = tmp_path / 'b.csv'
    _write_submission(a, [10, 10, 10, 10])
    _write_submission(b, [9, 9, 20, 20])
    splits = {
        'train_indices': [0],
        'dev_indices': [1, 2],
        'holdout_indices': [3],
        'counts': {'train': 1, 'dev': 2, 'holdout': 1, 'total': 4},
    }
    splits_path = tmp_path / 'splits.json'
    splits_path.write_text(json.dumps(splits), encoding='utf-8')
    out = tmp_path / 'results.json'
    subprocess.run([
        sys.executable, str(COMP / 'prompt_population_runner.py'),
        '--candidate', f'a={a}',
        '--candidate', f'b={b}',
        '--splits', str(splits_path),
        '--out', str(out),
    ], check=True)
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['champion']['label'] == 'a'



def test_external_adapter_lane_normalizes_and_validates(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location('meg_solve_module', COMP / 'solve_module.py')
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    central, generators = module.load_puzzle_bundle()
    moved = module.apply_moves(central, ['U'], generators)

    test_csv = tmp_path / 'test.csv'
    with test_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'initial_state'])
        writer.writeheader()
        writer.writerow({'initial_state_id': '0', 'initial_state': ','.join(map(str, central))})
        writer.writerow({'initial_state_id': '1', 'initial_state': ','.join(map(str, moved))})

    fallback = tmp_path / 'fallback.csv'
    with fallback.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        writer.writerow({'initial_state_id': '0', 'path': ''})
        writer.writerow({'initial_state_id': '1', 'path': '-U'})

    external = tmp_path / 'external.jsonl'
    external.write_text('\n'.join([
        json.dumps({'initial_state_id': '1', 'path': 'U4'}),
    ]), encoding='utf-8')

    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'label': 'extlane',
        'generator': {'type': 'jsonl', 'path': str(external)},
        'row_key': 'initial_state_id',
        'path_key': 'path',
        'order': 5,
    }), encoding='utf-8')

    out_dir = tmp_path / 'external_out'
    subprocess.run([
        sys.executable, str(COMP / 'external_adapter_lane.py'),
        '--manifest', str(manifest),
        '--test-csv', str(test_csv),
        '--fallback-submission', str(fallback),
        '--out-dir', str(out_dir),
        '--summary-out', str(tmp_path / 'summary.json'),
    ], check=True)

    out_csv = out_dir / 'extlane.adapter.csv'
    with out_csv.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['path'] == ''
    assert rows[1]['path'] == '-U'


def test_portfolio_orchestrator_accepts_external_manifest(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location('meg_solve_module', COMP / 'solve_module.py')
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    central, generators = module.load_puzzle_bundle()
    moved = module.apply_moves(central, ['U'], generators)

    test_csv = tmp_path / 'test.csv'
    with test_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'initial_state'])
        writer.writeheader()
        writer.writerow({'initial_state_id': '0', 'initial_state': ','.join(map(str, central))})
        writer.writerow({'initial_state_id': '1', 'initial_state': ','.join(map(str, moved))})

    bundled = tmp_path / 'bundled.csv'
    with bundled.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        writer.writerow({'initial_state_id': '0', 'path': ''})
        writer.writerow({'initial_state_id': '1', 'path': 'U.U.U.U'})

    external = tmp_path / 'external.jsonl'
    external.write_text(json.dumps({'initial_state_id': '1', 'path': 'U4'}) + '\n', encoding='utf-8')

    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'label': 'extlane',
        'generator': {'type': 'jsonl', 'path': str(external)},
        'row_key': 'initial_state_id',
        'path_key': 'path',
        'order': 5,
    }), encoding='utf-8')

    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({
        'train_indices': [],
        'dev_indices': [0, 1],
        'holdout_indices': [],
        'counts': {'train': 0, 'dev': 2, 'holdout': 0, 'total': 2},
    }), encoding='utf-8')

    out = tmp_path / 'merged.csv'
    subprocess.run([
        sys.executable, str(COMP / 'portfolio_orchestrator.py'),
        '--candidate', f'bundled={bundled}',
        '--external-manifest', str(manifest),
        '--fallback-submission', str(bundled),
        '--external-out-dir', str(tmp_path / 'ext_out'),
        '--test-csv', str(test_csv),
        '--splits', str(splits),
        '--candidate-eval-out', str(tmp_path / 'candidate_eval.json'),
        '--out', str(out),
        '--lineage-out', str(tmp_path / 'lineage.json'),
        '--summary-out', str(tmp_path / 'summary.json'),
        '--scoreboard-out', str(tmp_path / 'scoreboard.json'),
    ], check=True)

    with out.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert rows[1]['path'] == '-U'
    payload = json.loads((tmp_path / 'candidate_eval.json').read_text(encoding='utf-8'))
    assert payload['champion']['label'] == 'extlane'



def test_megaminxolver_wrapper_runs_repo_specific_export_script(tmp_path):
    repo = tmp_path / 'MegaminXolver-master'
    repo.mkdir()
    script = repo / 'export_bundle_candidates.py'
    script.write_text(
        """
import argparse, csv
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--test-csv', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
rows = list(csv.DictReader(Path(args.test_csv).open(newline='', encoding='utf-8')))
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
    writer.writeheader()
    for row in rows[:3]:
        writer.writerow({'initial_state_id': row['initial_state_id'], 'path': ''})
""",
        encoding='utf-8',
    )
    out_csv = tmp_path / 'megaminxolver.csv'
    subprocess.run([
        sys.executable,
        str(COMP / 'external_solver_adapters' / 'wrappers' / 'run_megaminxolver_wrapper.py'),
        '--repo', str(repo),
        '--test-csv', str(COMP / 'data' / 'test.csv'),
        '--out', str(out_csv),
        '--summary-out', str(tmp_path / 'summary.json'),
    ], check=True)
    with out_csv.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]['initial_state_id'] == '0'


def test_llminxsolver_wrapper_accepts_jsonl_export(tmp_path):
    repo = tmp_path / 'llminxsolver-master'
    repo.mkdir()
    script = repo / 'export_candidates.py'
    script.write_text(
        "\n".join([
            "import argparse, csv, json",
            "from pathlib import Path",
            "parser = argparse.ArgumentParser()",
            "parser.add_argument('--test-csv', required=True)",
            "parser.add_argument('--jsonl-out', required=True)",
            "args = parser.parse_args()",
            "rows = list(csv.DictReader(Path(args.test_csv).open(newline='', encoding='utf-8')))",
            "out = Path(args.jsonl_out)",
            "out.parent.mkdir(parents=True, exist_ok=True)",
            "with out.open('w', encoding='utf-8') as f:",
            "    for row in rows[:2]:",
            "        f.write(json.dumps({'initial_state_id': row['initial_state_id'], 'path': ''}) + '\\n')",
            "",
        ]),
        encoding='utf-8',
    )
    out_csv = tmp_path / 'llminxsolver.csv'
    subprocess.run([
        sys.executable,
        str(COMP / 'external_solver_adapters' / 'wrappers' / 'run_llminxsolver_wrapper.py'),
        '--repo', str(repo),
        '--test-csv', str(COMP / 'data' / 'test.csv'),
        '--out', str(out_csv),
        '--summary-out', str(tmp_path / 'summary.json'),
    ], check=True)
    with out_csv.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[1]['initial_state_id'] == '1'
