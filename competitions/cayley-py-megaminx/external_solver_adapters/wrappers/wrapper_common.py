from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_submission(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'initial_state_id': str(row.get('initial_state_id') or row.get('row_index') or ''),
                'path': str(row.get('path') or ''),
            })


def _render(parts: Sequence[str], context: Mapping[str, str]) -> list[str]:
    return [str(part).format(**context) for part in parts]


def _load_rows_from_text(text: str, fmt: str, *, row_key: str = 'initial_state_id', path_key: str = 'path') -> list[dict[str, str]]:
    if fmt == 'submission_csv':
        rows = []
        for row in csv.DictReader(text.splitlines()):
            rows.append({
                'initial_state_id': str(row.get(row_key) or row.get('initial_state_id') or row.get('row_index') or ''),
                'path': str(row.get(path_key) or row.get('path') or ''),
            })
        return rows
    if fmt == 'jsonl':
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            rows.append({
                'initial_state_id': str(row.get(row_key) or row.get('initial_state_id') or row.get('row_index') or ''),
                'path': str(row.get(path_key) or row.get('path') or ''),
            })
        return rows
    if fmt == 'pathlist':
        return [
            {'initial_state_id': str(idx), 'path': line.strip()}
            for idx, line in enumerate(text.splitlines()) if line.strip()
        ]
    raise ValueError(f'Unsupported text format: {fmt}')


def _load_rows_from_file(path: Path, fmt: str) -> list[dict[str, str]]:
    return _load_rows_from_text(path.read_text(encoding='utf-8'), fmt)


def _discover_existing_output(repo: Path, candidates: Sequence[tuple[str, str]]) -> tuple[Path, str] | None:
    for rel, fmt in candidates:
        p = repo / rel
        if p.exists() and p.is_file():
            return p, fmt
    for pattern, fmt in [
        ('**/*candidate*.csv', 'submission_csv'),
        ('**/*submission*.csv', 'submission_csv'),
        ('**/*candidate*.jsonl', 'jsonl'),
        ('**/*solution*.csv', 'submission_csv'),
    ]:
        matches = sorted(repo.glob(pattern))
        for p in matches:
            if p.is_file():
                return p, fmt
    return None


def run_with_autodiscovery(
    *,
    repo: Path,
    output_csv: Path,
    test_csv: Path,
    label: str,
    command_specs: Sequence[dict],
    existing_outputs: Sequence[tuple[str, str]],
    env_overrides: Mapping[str, str] | None = None,
) -> dict:
    repo = repo.resolve()
    output_csv = output_csv.resolve()
    test_csv = test_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    context = {
        'repo': str(repo),
        'repo_root': str(repo),
        'test_csv': str(test_csv),
        'out': str(output_csv),
        'output_csv': str(output_csv),
        'label': label,
        'work_dir': str(output_csv.parent),
    }

    preexisting = _discover_existing_output(repo, existing_outputs)
    if preexisting is not None:
        source_path, fmt = preexisting
        rows = _load_rows_from_file(source_path, fmt)
        _write_submission(output_csv, rows)
        return {
            'label': label,
            'mode': 'existing_output',
            'repo': str(repo),
            'source': str(source_path),
            'rows': len(rows),
            'format': fmt,
        }

    errors: list[dict[str, str]] = []
    for spec in command_specs:
        cmd = _render(spec['cmd'], context)
        cwd = Path(str(spec.get('cwd') or '{repo}').format(**context)).resolve()
        env = os.environ.copy()
        env['PYTHONPATH'] = str(repo) + os.pathsep + env.get('PYTHONPATH', '')
        for key, value in dict(env_overrides or {}).items():
            env[str(key)] = str(value).format(**context)
        for key, value in dict(spec.get('env') or {}).items():
            env[str(key)] = str(value).format(**context)
        timeout = float(spec.get('timeout_seconds', 180))
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except Exception as exc:
            errors.append({'cmd': ' '.join(cmd), 'error': repr(exc)})
            continue
        if proc.returncode != 0:
            errors.append({'cmd': ' '.join(cmd), 'error': f'returncode={proc.returncode}', 'stderr': proc.stderr[-500:]})
            continue

        output_mode = str(spec.get('output_mode', 'file'))
        fmt = str(spec.get('format', 'submission_csv'))
        try:
            if output_mode == 'stdout':
                rows = _load_rows_from_text(proc.stdout, fmt)
            else:
                produced = Path(str(spec.get('output_path', output_csv)).format(**context))
                if not produced.is_absolute():
                    produced = cwd / produced
                rows = _load_rows_from_file(produced, fmt)
            _write_submission(output_csv, rows)
            return {
                'label': label,
                'mode': 'command',
                'repo': str(repo),
                'cmd': cmd,
                'rows': len(rows),
                'format': fmt,
            }
        except Exception as exc:
            errors.append({'cmd': ' '.join(cmd), 'error': repr(exc), 'stderr': proc.stderr[-500:]})
            continue

    discovered = _discover_existing_output(repo, existing_outputs)
    if discovered is not None:
        source_path, fmt = discovered
        rows = _load_rows_from_file(source_path, fmt)
        _write_submission(output_csv, rows)
        return {
            'label': label,
            'mode': 'post_run_existing_output',
            'repo': str(repo),
            'source': str(source_path),
            'rows': len(rows),
            'format': fmt,
            'errors': errors,
        }

    raise RuntimeError(
        f'Could not materialize candidates for {label} from {repo}. ' \
        f'Tried {len(command_specs)} commands and {len(existing_outputs)} output patterns. Errors: {errors[:4]}'
    )


def build_basic_parser(description: str, default_repo: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--repo', default=default_repo, help='Path to the external repository checkout')
    parser.add_argument('--test-csv', required=True, help='Megaminx test.csv')
    parser.add_argument('--out', required=True, help='Path to output Kaggle-style submission CSV')
    parser.add_argument('--summary-out', default='', help='Optional JSON summary output path')
    return parser


def finish(summary: Mapping[str, object], summary_out: str | None) -> None:
    payload = json.dumps(summary, ensure_ascii=False)
    if summary_out and str(summary_out).strip():
        Path(summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(payload)
