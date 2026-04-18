from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load_local_module(name: str, filename: str):
    module_path = _HERE / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {filename}')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


sm = _load_local_module(f'{__name__}_solve_module', 'solve_module.py')


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_submission(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        writer.writerows(rows)


def _parse_state(text: str | None) -> List[int]:
    raw = (text or '').strip()
    if not raw:
        return []
    return [int(part) for part in raw.split(',') if part != '']


def _tokenize_path(text: str | None) -> List[str]:
    raw = (text or '').strip()
    if not raw:
        return []
    raw = raw.replace('\n', ' ').replace('\t', ' ')
    raw = raw.replace('.', ' ')
    raw = raw.replace(',', ' ')
    raw = raw.replace(';', ' ')
    tokens = [tok.strip() for tok in raw.split(' ') if tok.strip()]
    return tokens


def _expand_face_power(face: str, exp: int, order: int, forward_faces: set[str], official_moves: set[str]) -> List[str]:
    if face not in forward_faces:
        if face in official_moves and exp == 1:
            return [face]
        raise KeyError(face)
    value = exp % order
    if value == 0:
        return []
    inverse = '-' + face
    if value <= order // 2:
        return [face] * value
    return [inverse] * (order - value)


def _normalize_token(
    token: str,
    *,
    official_moves: set[str],
    forward_faces: set[str],
    move_aliases: Mapping[str, str],
    face_aliases: Mapping[str, str],
    order: int,
) -> List[str]:
    if not token:
        return []
    raw = token.strip()
    raw = raw.strip('()[]{}')
    raw = raw.replace('’', "'").replace('`', "'")
    if raw in move_aliases:
        return sm.path_to_moves(move_aliases[raw])
    if raw in official_moves:
        return [raw]
    if raw.startswith('.'):
        raw = raw[1:]
    if raw.endswith('.'):
        raw = raw[:-1]
    if raw in official_moves:
        return [raw]

    plus_minus = re.fullmatch(r'([A-Za-z][A-Za-z0-9_]*)([+-]{2,})', raw)
    if plus_minus:
        base = plus_minus.group(1)
        signs = plus_minus.group(2)
        base = face_aliases.get(base, base)
        if len(set(signs)) == 1:
            step = 1 if signs[0] == '+' else -1
            return _expand_face_power(base, step * len(signs), order, forward_faces, official_moves)

    match = re.fullmatch(r'(-)?([A-Za-z]+?)(?:([\'i-])?(\d+)?)?$', raw)
    if not match:
        raise ValueError(f'Unsupported token: {token!r}')
    lead_inv, face, suffix_inv, digits = match.groups()
    face = face_aliases.get(face, face)
    exp = int(digits) if digits else 1
    sign = -1 if (lead_inv or suffix_inv in {"'", 'i', '-'}) else 1
    return _expand_face_power(face, sign * exp, order, forward_faces, official_moves)


def normalize_external_path(
    path_text: str | None,
    *,
    official_moves: set[str],
    forward_faces: set[str],
    move_aliases: Mapping[str, str] | None = None,
    face_aliases: Mapping[str, str] | None = None,
    order: int = 5,
) -> Tuple[str, List[str]]:
    aliases = dict(move_aliases or {})
    face_map = dict(face_aliases or {})
    normalized: List[str] = []
    failures: List[str] = []
    for token in _tokenize_path(path_text):
        try:
            normalized.extend(
                _normalize_token(
                    token,
                    official_moves=official_moves,
                    forward_faces=forward_faces,
                    move_aliases=aliases,
                    face_aliases=face_map,
                    order=order,
                )
            )
        except Exception:
            failures.append(token)
    if failures:
        raise ValueError(f'Could not normalize tokens: {failures}')
    return sm.moves_to_path(normalized), normalized


def _load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise TypeError(f'Manifest must be a JSON object: {path}')
    payload.setdefault('label', path.stem)
    payload.setdefault('generator', {})
    return payload


def _render_template(value: str, context: Mapping[str, str]) -> str:
    try:
        return value.format(**context)
    except KeyError:
        return value


def _materialize_command_output(generator: Mapping[str, Any], context: Mapping[str, str], work_dir: Path) -> Tuple[str, str]:
    command = generator.get('command')
    if not command:
        raise ValueError('command generator requires a command')
    if isinstance(command, str):
        cmd = shlex.split(_render_template(command, context))
    elif isinstance(command, list):
        cmd = [_render_template(str(part), context) for part in command]
    else:
        raise TypeError('command must be string or list')
    cwd_raw = str(generator.get('cwd') or context['repo_root'])
    cwd = Path(_render_template(cwd_raw, context))
    env = os.environ.copy()
    for key, value in dict(generator.get('env') or {}).items():
        env[str(key)] = _render_template(str(value), context)
    timeout = float(generator.get('timeout_seconds') or 60.0)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'External generator failed for {context.get("label")}: '
            f'code={result.returncode} stderr={result.stderr[-400:]}'
        )
    output_mode = str(generator.get('output_mode') or 'stdout')
    output_format = str(generator.get('output_format') or 'submission_csv')
    if output_mode == 'file':
        output_path = Path(_render_template(str(generator.get('output_path') or ''), context))
        if not output_path.is_absolute():
            output_path = cwd / output_path
        return output_path.read_text(encoding='utf-8'), output_format
    return result.stdout, output_format


def _rows_from_text(text: str, fmt: str, row_key: str, path_key: str) -> Dict[str, str]:
    if fmt == 'submission_csv':
        reader = csv.DictReader(text.splitlines())
        return {str(row.get(row_key) or row.get('initial_state_id') or row.get('row_index') or ''): str(row.get(path_key) or row.get('path') or '') for row in reader}
    if fmt == 'jsonl':
        out: Dict[str, str] = {}
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                continue
            key = str(row.get(row_key) or row.get('initial_state_id') or row.get('row_index') or '')
            out[key] = str(row.get(path_key) or row.get('path') or '')
        return out
    if fmt == 'pathlist':
        rows: Dict[str, str] = {}
        for idx, raw_line in enumerate(text.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            rows[str(idx)] = line
        return rows
    raise ValueError(f'Unsupported output format: {fmt}')


def _load_generator_rows(
    generator: Mapping[str, Any],
    *,
    context: Mapping[str, str],
    row_key: str,
    path_key: str,
    work_dir: Path,
) -> Dict[str, str]:
    gtype = str(generator.get('type') or 'submission_csv')
    if gtype == 'submission_csv':
        source = Path(_render_template(str(generator.get('path') or ''), context))
        rows = _read_csv_rows(source)
        return {
            str(row.get(row_key) or row.get('initial_state_id') or row.get('row_index') or ''): str(row.get(path_key) or row.get('path') or '')
            for row in rows
        }
    if gtype == 'jsonl':
        source = Path(_render_template(str(generator.get('path') or ''), context))
        return _rows_from_text(source.read_text(encoding='utf-8'), 'jsonl', row_key, path_key)
    if gtype == 'command':
        text, fmt = _materialize_command_output(generator, context, work_dir)
        return _rows_from_text(text, fmt, row_key, path_key)
    raise ValueError(f'Unsupported generator type: {gtype}')


def adapt_external_manifest(
    manifest_path: Path,
    *,
    test_csv: Path,
    fallback_submission: Path | None = None,
    out_dir: Path | None = None,
) -> Tuple[str, Path, Dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    label = str(manifest.get('label') or manifest_path.stem)
    generator = dict(manifest.get('generator') or {})
    row_key = str(manifest.get('row_key') or 'initial_state_id')
    path_key = str(manifest.get('path_key') or 'path')
    order = int(manifest.get('order') or 5)
    move_aliases = {str(k): str(v) for k, v in dict(manifest.get('move_aliases') or {}).items()}
    face_aliases = {str(k): str(v) for k, v in dict(manifest.get('face_aliases') or {}).items()}

    out_base = out_dir or (_HERE / 'submissions' / 'external_adapter_lane')
    out_base.mkdir(parents=True, exist_ok=True)
    out_csv = out_base / f'{label}.adapter.csv'
    out_summary = out_base / f'{label}.adapter.summary.json'

    test_rows = _read_csv_rows(test_csv)
    fallback_rows = _read_csv_rows(fallback_submission) if fallback_submission and fallback_submission.exists() else []
    fallback_by_id = {
        str(row.get('initial_state_id') or ''): str(row.get('path') or '')
        for row in fallback_rows
    }

    central, generators = sm.load_puzzle_bundle()
    official_moves = set(sm.move_names(generators))
    forward_faces = set(sm.forward_faces(generators))

    with tempfile.TemporaryDirectory(prefix=f'{label}_adapter_') as tmpdir:
        work_dir = Path(tmpdir)
        context = {
            'label': label,
            'repo_root': str(_HERE.parent.parent.resolve()),
            'comp_dir': str(_HERE.resolve()),
            'test_csv': str(test_csv.resolve()),
            'output_csv': str((work_dir / f'{label}.generated.csv').resolve()),
            'work_dir': str(work_dir.resolve()),
        }
        source_rows = _load_generator_rows(generator, context=context, row_key=row_key, path_key=path_key, work_dir=work_dir)

    adapted_rows: List[Dict[str, str]] = []
    conversions = 0
    valid_rows = 0
    fallback_rows_used = 0
    invalid_rows = 0
    missing_rows = 0
    failures: List[Dict[str, Any]] = []

    for idx, test_row in enumerate(test_rows):
        row_id = str(test_row.get('initial_state_id') or '')
        raw_path = source_rows.get(row_id)
        initial_state = _parse_state(test_row.get('initial_state'))
        chosen_path = ''
        source = 'external'
        try:
            if raw_path is None:
                raise KeyError('missing')
            normalized_path, moves = normalize_external_path(
                raw_path,
                official_moves=official_moves,
                forward_faces=forward_faces,
                move_aliases=move_aliases,
                face_aliases=face_aliases,
                order=order,
            )
            if normalized_path != str(raw_path).strip():
                conversions += 1
            if not sm.validate_solution(initial_state, moves, central, generators):
                raise ValueError('normalized path does not solve official state')
            chosen_path = normalized_path
            valid_rows += 1
        except Exception as exc:
            if raw_path is None:
                missing_rows += 1
            else:
                invalid_rows += 1
            fallback = fallback_by_id.get(row_id, '')
            chosen_path = fallback
            source = 'fallback' if fallback else 'empty'
            if fallback:
                fallback_rows_used += 1
            failures.append({
                'row_index': idx,
                'initial_state_id': row_id,
                'raw_path': raw_path,
                'error': str(exc),
                'fallback_used': bool(fallback),
            })
        adapted_rows.append({'initial_state_id': row_id, 'path': chosen_path})

    summary = {
        'label': label,
        'manifest': str(manifest_path),
        'test_csv': str(test_csv),
        'fallback_submission': str(fallback_submission) if fallback_submission else '',
        'rows': len(adapted_rows),
        'valid_rows': valid_rows,
        'invalid_rows': invalid_rows,
        'missing_rows': missing_rows,
        'fallback_rows_used': fallback_rows_used,
        'converted_rows': conversions,
        'failures': failures[:200],
    }
    _write_submission(out_csv, adapted_rows)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return label, out_csv, summary


def materialize_external_candidates(
    manifest_paths: Sequence[str | Path],
    *,
    test_csv: Path,
    fallback_submission: Path | None = None,
    out_dir: Path | None = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    candidate_specs: List[str] = []
    summaries: List[Dict[str, Any]] = []
    for manifest in manifest_paths:
        label, csv_path, summary = adapt_external_manifest(
            Path(manifest),
            test_csv=test_csv,
            fallback_submission=fallback_submission,
            out_dir=out_dir,
        )
        candidate_specs.append(f'{label}={csv_path}')
        summaries.append(summary)
    return candidate_specs, summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Normalize and validate external Megaminx candidate generators against the official bundle')
    parser.add_argument('--manifest', action='append', required=True, help='JSON manifest describing one external generator')
    parser.add_argument('--test-csv', default=str(_HERE / 'data' / 'test.csv'))
    parser.add_argument('--fallback-submission', default=str(_HERE / 'submissions' / 'optimized_submission.csv'))
    parser.add_argument('--out-dir', default=str(_HERE / 'submissions' / 'external_adapter_lane'))
    parser.add_argument('--summary-out', default='')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    candidate_specs, summaries = materialize_external_candidates(
        args.manifest,
        test_csv=Path(args.test_csv),
        fallback_submission=Path(args.fallback_submission) if str(args.fallback_submission).strip() else None,
        out_dir=Path(args.out_dir),
    )
    payload = {'candidate_specs': candidate_specs, 'summaries': summaries}
    if str(args.summary_out).strip():
        Path(args.summary_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()
