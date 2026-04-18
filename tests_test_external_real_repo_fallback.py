from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / 'competitions' / 'cayley-py-megaminx' / 'external_adapter_lane.py'
spec = importlib.util.spec_from_file_location('external_adapter_lane_testmod', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_external_manifest_clone_and_fallback(tmp_path: Path) -> None:
    src_repo = tmp_path / 'src_repo'
    src_repo.mkdir()
    (src_repo / 'README.md').write_text('hello external repo\n', encoding='utf-8')
    subprocess.run(['git', 'init'], cwd=src_repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=src_repo, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=src_repo, check=True)
    subprocess.run(['git', 'add', 'README.md'], cwd=src_repo, check=True)
    subprocess.run(['git', 'commit', '-m', 'init'], cwd=src_repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    test_csv = tmp_path / 'test.csv'
    fallback_csv = tmp_path / 'fallback.csv'
    _write_csv(
        test_csv,
        ['initial_state_id', 'initial_state'],
        [
            {'initial_state_id': 'row-1', 'initial_state': '1,2,3'},
            {'initial_state_id': 'row-2', 'initial_state': '4,5,6'},
        ],
    )
    _write_csv(
        fallback_csv,
        ['initial_state_id', 'path'],
        [
            {'initial_state_id': 'row-1', 'path': 'A.B'},
            {'initial_state_id': 'row-2', 'path': 'C'},
        ],
    )

    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'label': 'local_clone_fail_safe',
        'allow_failure': True,
        'repo': {
            'url': str(src_repo),
            'ref': 'HEAD',
            'checkout_dir': str(tmp_path / 'cloned_repo'),
            'allow_failure': True,
        },
        'generator': {
            'type': 'command',
            'command': [sys.executable, '-c', 'import sys; sys.exit(2)'],
            'cwd': '{repo_root}',
            'timeout_seconds': 30,
            'output_mode': 'stdout',
            'output_format': 'submission_csv',
            'allow_failure': True,
        },
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    label, out_csv, summary = mod.adapt_external_manifest(
        manifest,
        test_csv=test_csv,
        fallback_submission=fallback_csv,
        out_dir=tmp_path / 'out',
    )

    assert label == 'local_clone_fail_safe'
    assert Path(summary['repo']['repo_dir']).exists()
    assert summary['fallback_rows_used'] == 2
    assert summary['valid_rows'] == 0
    assert 'External generator failed' in summary['generator_error'] or summary['generator_error']

    with out_csv.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {'initial_state_id': 'row-1', 'path': 'A.B'},
        {'initial_state_id': 'row-2', 'path': 'C'},
    ]
