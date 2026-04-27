from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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
    return [tok.strip() for tok in raw.split(' ') if tok.strip()]


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

    match = re.fullmatch(r"(-)?([A-Za-z]+?)(?:([\'i-])?(\d+)?)?$", raw)
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


def _slugify(text: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text).strip())
    return slug.strip('._-') or 'external_repo'


def _run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _clone_or_refresh_repo(repo_cfg: Mapping[str, Any], *, repo_root: Path) -> Dict[str, Any]:
    url = str(repo_cfg.get('url') or repo_cfg.get('repo_url') or '').strip()
    local_path_raw = str(repo_cfg.get('path') or repo_cfg.get('checkout_dir') or '').strip()
    ref = str(repo_cfg.get('ref') or '').strip()
    refresh = bool(repo_cfg.get('refresh'))
    timeout = float(repo_cfg.get('clone_timeout_seconds') or 600.0)

    if local_path_raw:
        repo_dir = Path(_render_template(local_path_raw, {'repo_root': str(repo_root)}))
        if not repo_dir.is_absolute():
            repo_dir = repo_root / repo_dir
    elif url:
        repo_dir = repo_root / 'external_real' / _slugify(Path(url).stem or Path(url).name)
    else:
        return {
            'repo_dir': '',
            'repo_bootstrapped': False,
            'repo_errors': ['no repo url or path configured'],
            'repo_sha': '',
        }

    repo_dir = repo_dir.resolve()
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    actions: List[str] = []

    if url:
        if refresh and repo_dir.exists():
            shutil.rmtree(repo_dir)
            actions.append('removed_existing_checkout')
        if not repo_dir.exists():
            proc = _run_subprocess(['git', 'clone', url, str(repo_dir)], timeout=timeout)
            if proc.returncode != 0:
                errors.append(f'git clone failed: {proc.stderr[-400:]}')
                return {
                    'repo_dir': str(repo_dir),
                    'repo_bootstrapped': False,
                    'repo_errors': errors,
                    'repo_sha': '',
                    'repo_actions': actions,
                }
            actions.append('cloned')
        else:
            actions.append('reused_existing_checkout')
        if ref:
            proc = _run_subprocess(['git', '-C', str(repo_dir), 'checkout', ref], timeout=timeout)
            if proc.returncode != 0:
                fetch = _run_subprocess(['git', '-C', str(repo_dir), 'fetch', '--all', '--tags'], timeout=timeout)
                if fetch.returncode == 0:
                    proc = _run_subprocess(['git', '-C', str(repo_dir), 'checkout', ref], timeout=timeout)
            if proc.returncode != 0:
                errors.append(f'git checkout {ref} failed: {proc.stderr[-400:]}')
            else:
                actions.append(f'checkout:{ref}')
    elif not repo_dir.exists():
        errors.append(f'Configured local repo path does not exist: {repo_dir}')

    sha = ''
    if repo_dir.exists() and (repo_dir / '.git').exists():
        proc = _run_subprocess(['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'], timeout=60)
        if proc.returncode == 0:
            sha = proc.stdout.strip()
        else:
            errors.append(f'git rev-parse failed: {proc.stderr[-200:]}')

    return {
        'repo_dir': str(repo_dir),
        'repo_bootstrapped': repo_dir.exists() and not errors,
        'repo_errors': errors,
        'repo_sha': sha,
        'repo_actions': actions,
    }


def _run_repo_build_steps(repo_cfg: Mapping[str, Any], *, repo_dir: Path, context: Mapping[str, str]) -> Dict[str, Any]:
    build_steps = repo_cfg.get('build_steps') or []
    if not isinstance(build_steps, list):
        build_steps = [build_steps]
    if not build_steps:
        return {'build_steps': [], 'build_errors': []}
    step_logs: List[Dict[str, Any]] = []
    build_errors: List[str] = []
    for idx, step in enumerate(build_steps):
        if isinstance(step, str):
            cmd = shlex.split(_render_template(step, context))
            cwd = repo_dir
            env = os.environ.copy()
            timeout = 900.0
            optional = False
        elif isinstance(step, Mapping):
            raw_cmd = step.get('cmd') or step.get('command') or []
            if isinstance(raw_cmd, str):
                cmd = shlex.split(_render_template(raw_cmd, context))
            else:
                cmd = [_render_template(str(part), context) for part in raw_cmd]
            cwd_raw = str(step.get('cwd') or '{external_repo}')
            cwd = Path(_render_template(cwd_raw, context))
            env = os.environ.copy()
            for key, value in dict(step.get('env') or {}).items():
                env[str(key)] = _render_template(str(value), context)
            timeout = float(step.get('timeout_seconds') or 900.0)
            optional = bool(step.get('optional'))
        else:
            build_errors.append(f'Unsupported build step at index {idx}: {step!r}')
            continue
        proc = _run_subprocess(cmd, cwd=cwd, env=env, timeout=timeout)
        step_logs.append({
            'cmd': list(cmd),
            'cwd': str(cwd),
            'returncode': proc.returncode,
            'stdout_tail': proc.stdout[-400:],
            'stderr_tail': proc.stderr[-400:],
            'optional': optional,
        })
        if proc.returncode != 0 and not optional:
            build_errors.append(f'Build step failed: {cmd!r}')
            break
    return {'build_steps': step_logs, 'build_errors': build_errors}


def _prepare_cases_jsonl(test_rows: Sequence[Mapping[str, str]], *, work_dir: Path, adapter_name: str) -> Path:
    cases_path = work_dir / f'cases_{_slugify(adapter_name)}.jsonl'
    with cases_path.open('w', encoding='utf-8') as f:
        for idx, row in enumerate(test_rows):
            state_csv = str(row.get('initial_state') or '')
            state = _parse_state(state_csv)
            payload: Dict[str, Any] = {
                'row_index': idx,
                'initial_state_id': str(row.get('initial_state_id') or idx),
                'initial_state_csv': state_csv,
                'initial_state': state,
                'repo_input': state_csv,
                'repo_state': state,
                'adapter': adapter_name,
            }
            if adapter_name == 'odder_last2faces':
                payload['repo_state'] = {'official_state': state, 'adapter': adapter_name, 'supported': False}
            elif adapter_name == 'llminx_last_layer':
                payload['repo_state'] = {'official_state': state, 'adapter': adapter_name, 'supported': False}
            elif adapter_name == 'abgolev_stickers_12x10':
                payload['repo_input'] = state_csv
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
    return cases_path


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
        return {
            str(row.get(row_key) or row.get('initial_state_id') or row.get('row_index') or ''): str(row.get(path_key) or row.get('path') or '')
            for row in reader
        }
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
    repo_cfg = dict(manifest.get('repo') or {})
    row_key = str(manifest.get('row_key') or 'initial_state_id')
    path_key = str(manifest.get('path_key') or 'path')
    order = int(manifest.get('order') or 5)
    move_aliases = {str(k): str(v) for k, v in dict(manifest.get('move_aliases') or {}).items()}
    face_aliases = {str(k): str(v) for k, v in dict(manifest.get('face_aliases') or {}).items()}
    allow_failure = bool(manifest.get('allow_failure') or repo_cfg.get('allow_failure') or generator.get('allow_failure'))
    state_adapter = str(manifest.get('state_adapter') or 'official_state_vector')

    out_base = out_dir or (_HERE / 'submissions' / 'external_adapter_lane')
    out_base.mkdir(parents=True, exist_ok=True)
    out_csv = out_base / f'{label}.adapter.csv'
    out_summary = out_base / f'{label}.adapter.summary.json'

    test_rows = _read_csv_rows(test_csv)
    fallback_rows = _read_csv_rows(fallback_submission) if fallback_submission and fallback_submission.exists() else []
    fallback_by_id = {str(row.get('initial_state_id') or ''): str(row.get('path') or '') for row in fallback_rows}

    central, generators = sm.load_puzzle_bundle()
    official_moves = set(sm.move_names(generators))
    forward_faces = set(sm.forward_faces(generators))

    repo_root = _HERE.parent.parent.resolve()
    repo_summary = _clone_or_refresh_repo(repo_cfg, repo_root=repo_root)
    repo_dir = Path(repo_summary['repo_dir']) if repo_summary.get('repo_dir') else None
    build_summary: Dict[str, Any] = {'build_steps': [], 'build_errors': []}
    source_rows: Dict[str, str] = {}
    generator_error = ''

    with tempfile.TemporaryDirectory(prefix=f'{label}_adapter_') as tmpdir:
        work_dir = Path(tmpdir)
        cases_jsonl = _prepare_cases_jsonl(test_rows, work_dir=work_dir, adapter_name=state_adapter)
        context = {
            'label': label,
            'repo_root': str(repo_root),
            'comp_dir': str(_HERE.resolve()),
            'test_csv': str(test_csv.resolve()),
            'output_csv': str((work_dir / f'{label}.generated.csv').resolve()),
            'output_jsonl': str((work_dir / f'{label}.generated.jsonl').resolve()),
            'work_dir': str(work_dir.resolve()),
            'cases_jsonl': str(cases_jsonl.resolve()),
            'external_repo': str(repo_dir.resolve()) if repo_dir else '',
            'repo_checkout': str(repo_dir.resolve()) if repo_dir else '',
            'repo_sha': str(repo_summary.get('repo_sha') or ''),
        }
        if repo_dir and repo_dir.exists():
            build_summary = _run_repo_build_steps(repo_cfg, repo_dir=repo_dir, context=context)
        if repo_summary.get('repo_errors') or build_summary.get('build_errors'):
            generator_error = '; '.join(list(repo_summary.get('repo_errors') or []) + list(build_summary.get('build_errors') or []))
        if not generator_error:
            try:
                source_rows = _load_generator_rows(generator, context=context, row_key=row_key, path_key=path_key, work_dir=work_dir)
            except Exception as exc:
                generator_error = str(exc)
                if not allow_failure:
                    raise

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
        'state_adapter': state_adapter,
        'allow_failure': allow_failure,
        'generator_error': generator_error,
        'repo': repo_summary,
        'build': build_summary,
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
