from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import gradio as gr

PROMPT_VARIANTS = [
    'regular',
    'improved',
    'dataset_adapted',
    'structured',
    'heuristic_boosted',
    'master_hybrid',
]

PIPELINES = [
    'baseline_no_llm',
    'best_tested_solver',
    'optimized_assets_only',
    'optimized_assets_v3_top150',
    'optimized_assets_v3_top300',
    *[f'prompt_run:{variant}' for variant in PROMPT_VARIANTS],
]

COMMON_G4F_MODELS = [
    'gpt-4o-mini',
    'deepseek-chat',
    'claude-3.5-sonnet',
    'grok-2',
    'gemini-1.5-flash',
]


@dataclass
class PromptModelSpec:
    tag: str
    label: str
    models: str | None = None
    planner_models: str | None = None
    coder_models: str | None = None
    fixer_models: str | None = None


@dataclass
class PipelineResult:
    name: str
    submission_path: str
    score: int
    rows: int
    note: str = ''
    run_log: str | None = None


def _repo_paths(repo_dir: Path) -> dict[str, Path]:
    comp_dir = repo_dir / 'competitions' / 'cayley-py-megaminx'
    submissions_dir = comp_dir / 'submissions'
    generated_dir = repo_dir / 'generated' / 'megaminx_gui_runs'
    generated_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    return {
        'repo_dir': repo_dir,
        'comp_dir': comp_dir,
        'submissions_dir': submissions_dir,
        'generated_dir': generated_dir,
        'pipeline_cli': repo_dir / 'pipeline_cli.py',
        'best_solver': comp_dir / 'megaminx_best_tested_solver.py',
        'build_opt': comp_dir / 'build_optimized_assets.py',
    }


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
    merged = os.environ.copy()
    if env:
        merged.update({k: str(v) for k, v in env.items() if v is not None})
    started = time.time()
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=merged,
        text=True,
        capture_output=True,
    )
    elapsed = time.time() - started
    shell_line = ' '.join(shlex.quote(str(x)) for x in cmd)
    out = f'$ {shell_line}\n[exit={proc.returncode} elapsed={elapsed:.2f}s]\n'
    if proc.stdout:
        out += '\n[stdout]\n' + proc.stdout
    if proc.stderr:
        out += '\n[stderr]\n' + proc.stderr
    return proc.returncode, out


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        item = (raw or '').strip()
        if not item:
            continue
        if item.startswith('g4f:'):
            item = item.split(':', 1)[1].strip()
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _parse_csv_like(text: str | None) -> list[str]:
    if not text:
        return []
    parts: list[str] = []
    for chunk in text.replace('\n', ',').split(','):
        item = chunk.strip()
        if item:
            parts.append(item)
    return _dedupe_keep_order(parts)


def _slugify(text: str) -> str:
    import re
    text = (text or '').strip().lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = re.sub(r'-+', '-', text).strip('-')
    return text or 'item'


def _score_submission(csv_path: Path) -> tuple[int, int]:
    total = 0
    rows = 0
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            path = (row.get('path') or '').strip()
            total += len([tok for tok in path.split('.') if tok])
    return total, rows


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def _load_json_maybe(text: str) -> Any:
    text = (text or '').strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    for idx in range(len(text)):
        if text[idx] in '{[':
            try:
                return json.loads(text[idx:])
            except Exception:
                continue
    return None


def _results_to_rows(results: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for item in results:
        rows.append([
            item.get('name'),
            item.get('score'),
            item.get('rows'),
            item.get('note', ''),
            item.get('submission_path', ''),
            item.get('run_log', '') or '',
        ])
    return rows


def _best_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return sorted(results, key=lambda x: (int(x.get('score', 10**18)), x.get('name', '')))[0]


def _install_kaggle_json(upload_path: str | None) -> str:
    if not upload_path:
        return 'No kaggle.json uploaded.'
    source = Path(upload_path)
    if not source.exists():
        return f'Uploaded kaggle.json not found: {source}'
    target_dir = Path.home() / '.kaggle'
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / 'kaggle.json'
    shutil.copy2(source, target)
    target.chmod(0o600)
    return f'Installed Kaggle credentials to {target}'


def _install_dependencies(repo_dir_text: str, install_full_requirements: bool, install_g4f: str, install_cayleypy: str, extra_pip_packages: str) -> str:
    repo_dir = Path(repo_dir_text).expanduser().resolve()
    logs: list[str] = []
    commands: list[list[str]] = [
        [sys.executable, '-m', 'pip', 'install', '-U', 'pip'],
        [sys.executable, '-m', 'pip', 'install', 'gradio', 'kaggle'],
    ]
    if install_full_requirements:
        req = repo_dir / 'requirements-full.txt'
        if req.exists():
            commands.append([sys.executable, '-m', 'pip', 'install', '-r', str(req)])
    if install_g4f == 'github':
        commands.append([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/xtekky/gpt4free'])
    elif install_g4f == 'pypi':
        commands.append([sys.executable, '-m', 'pip', 'install', 'g4f'])
    if install_cayleypy == 'github':
        commands.append([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/cayleypy/cayleypy'])
    elif install_cayleypy == 'pypi':
        commands.append([sys.executable, '-m', 'pip', 'install', 'cayleypy'])
    extra_pkgs = _parse_csv_like(extra_pip_packages)
    if extra_pkgs:
        commands.append([sys.executable, '-m', 'pip', 'install', *extra_pkgs])
    for cmd in commands:
        code, log = _run(cmd, cwd=repo_dir)
        logs.append(log)
        if code != 0:
            break
    return '\n\n'.join(logs)


def _probe_models(repo_dir_text: str, probe_scope: str, models_text: str, provider: str, backend_api_url: str, timeout_s: float, max_models: int, probe_mode: str, concurrency: int):
    repo_dir = Path(repo_dir_text).expanduser().resolve()
    cmd = [
        sys.executable, 'pipeline_cli.py', 'check-g4f-models',
        '--json',
        '--timeout', str(timeout_s),
        '--probe-mode', probe_mode,
        '--concurrency', str(concurrency),
    ]
    if provider.strip():
        cmd.extend(['--provider', provider.strip()])
    if backend_api_url.strip():
        cmd.extend(['--backend-api-url', backend_api_url.strip()])
    if max_models and max_models > 0:
        cmd.extend(['--max-models', str(max_models)])
    scope = (probe_scope or 'selected').strip().lower()
    selected = _parse_csv_like(models_text)
    if scope == 'selected' and selected:
        cmd.extend(['--models', ','.join(selected)])
    elif scope == 'discover_only':
        cmd.append('--discover-only')
    elif scope == 'all_discovered':
        pass
    elif scope != 'selected':
        raise ValueError(f'Unsupported probe scope: {probe_scope}')
    code, log = _run(cmd, cwd=repo_dir)
    payload = _load_json_maybe(log.split('[stdout]\n', 1)[-1] if '[stdout]\n' in log else log)
    if not isinstance(payload, dict):
        payload = {'raw_output': log, 'return_code': code}
    working = _dedupe_keep_order(payload.get('working_models', []) if isinstance(payload, dict) else [])
    discovered = _dedupe_keep_order(payload.get('candidates', []) if isinstance(payload, dict) else [])
    choices = working or discovered or selected or COMMON_G4F_MODELS
    update_single = gr.update(choices=choices, value=(working[0] if working else None))
    update_multi = gr.update(choices=choices, value=(working[: min(3, len(working))] if working else None))
    return payload, '\n'.join(working), log, working, update_single, update_multi


def _build_prompt_model_specs(prompt_model_run_mode: str, single_model: str, selected_models: list[str] | None, working_models_state: list[str] | None, planner_model: str, coder_model: str, fixer_model: str) -> list[PromptModelSpec]:
    mode = (prompt_model_run_mode or 'single').strip().lower()
    planner = (planner_model or '').strip() or None
    coder = (coder_model or '').strip() or None
    fixer = (fixer_model or '').strip() or None
    if mode == 'single':
        if planner or coder or fixer:
            return [PromptModelSpec(tag='per-agent', label='per-agent overrides', planner_models=planner, coder_models=coder, fixer_models=fixer)]
        model = _dedupe_keep_order([single_model])
        if not model:
            return [PromptModelSpec(tag='default', label='default')]
        return [PromptModelSpec(tag=_slugify(model[0]), label=f'g4f:{model[0]}', models=model[0])]
    if mode == 'selected':
        models = _dedupe_keep_order(selected_models or [])
    elif mode == 'working':
        models = _dedupe_keep_order(working_models_state or [])
        if not models:
            raise RuntimeError('Model mode=working requires at least one working model from probe.')
    else:
        raise ValueError(f'Unsupported prompt_model_run_mode: {prompt_model_run_mode}')
    if not models:
        raise RuntimeError('No models selected for prompt pipelines.')
    return [PromptModelSpec(tag=_slugify(m), label=f'g4f:{m}', models=m) for m in models]


def _maybe_submit_args(cmd: list[str], submit: bool, submit_message: str, submit_via: str, kaggle_json_path: str | None) -> list[str]:
    if not submit:
        return cmd
    cmd = list(cmd)
    cmd.append('--submit')
    if submit_message.strip():
        cmd.extend(['--message', submit_message.strip()])
    if submit_via.strip():
        cmd.extend(['--submit-via', submit_via.strip()])
    if kaggle_json_path:
        cmd.extend(['--kaggle-json', kaggle_json_path])
    return cmd


def _run_baseline(paths: dict[str, Path], output_path: Path, run_log_path: Path, submit: bool, submit_message: str, submit_via: str, kaggle_json_path: str | None, max_rows: int) -> tuple[PipelineResult, str]:
    cmd = [sys.executable, 'pipeline_cli.py', 'run', '--competition', 'cayley-py-megaminx', '--output', str(output_path), '--run-log', str(run_log_path), '--no-llm']
    if max_rows > 0:
        cmd.extend(['--max-rows', str(max_rows)])
    cmd = _maybe_submit_args(cmd, submit, submit_message, submit_via, kaggle_json_path)
    code, log = _run(cmd, cwd=paths['repo_dir'])
    if code != 0:
        raise RuntimeError(log)
    score, rows = _score_submission(output_path)
    return PipelineResult('baseline_no_llm', str(output_path), score, rows, note='standard pipeline_cli --no-llm', run_log=str(run_log_path)), log


def _run_best_solver(paths: dict[str, Path], output_path: Path) -> tuple[PipelineResult, str]:
    cmd = [sys.executable, str(paths['best_solver']), '--build-submission', str(output_path)]
    code, log = _run(cmd, cwd=paths['repo_dir'])
    if code != 0:
        raise RuntimeError(log)
    score, rows = _score_submission(output_path)
    return PipelineResult('best_tested_solver', str(output_path), score, rows, note='megaminx_best_tested_solver.py'), log


def _run_build_assets(paths: dict[str, Path], name: str, search_version: str, search_top_k: int | None, output_path: Path, run_log_path: Path, disable_cayleypy: bool, light_budget: float, aggressive_budget: float, aggressive_min_len: int) -> tuple[PipelineResult, str]:
    cmd = [sys.executable, str(paths['build_opt']), '--search-version', search_version]
    if search_top_k is not None:
        cmd.extend(['--search-top-k', str(search_top_k)])
    if disable_cayleypy:
        cmd.append('--search-disable-cayleypy')
    if search_version == 'v3':
        cmd.extend([
            '--search-light-time-budget-per-row', str(light_budget),
            '--search-aggressive-time-budget-per-row', str(aggressive_budget),
            '--search-aggressive-min-path-len', str(aggressive_min_len),
        ])
    code, log = _run(cmd, cwd=paths['comp_dir'])
    if code != 0:
        raise RuntimeError(log)
    optimized = paths['submissions_dir'] / 'optimized_submission.csv'
    shutil.copy2(optimized, output_path)
    default_run_log = paths['submissions_dir'] / 'run_log.json'
    if default_run_log.exists():
        shutil.copy2(default_run_log, run_log_path)
    score, rows = _score_submission(output_path)
    return PipelineResult(name, str(output_path), score, rows, note=f'build_optimized_assets.py search_version={search_version}', run_log=str(run_log_path)), log


def _run_prompt_variant(paths: dict[str, Path], variant: str, spec: PromptModelSpec, output_path: Path, run_log_path: Path, submit: bool, submit_message: str, submit_via: str, kaggle_json_path: str | None, search_mode: str, keep_improving: bool, improvement_rounds: int, max_rows: int, g4f_provider: str) -> tuple[PipelineResult, str]:
    cmd = [sys.executable, 'pipeline_cli.py', 'run', '--competition', 'cayley-py-megaminx', '--output', str(output_path), '--run-log', str(run_log_path), '--prompt-variant', variant, '--search-mode', search_mode]
    if max_rows > 0:
        cmd.extend(['--max-rows', str(max_rows)])
    if spec.models:
        cmd.extend(['--models', spec.models])
    if spec.planner_models:
        cmd.extend(['--planner-models', spec.planner_models])
    if spec.coder_models:
        cmd.extend(['--coder-models', spec.coder_models])
    if spec.fixer_models:
        cmd.extend(['--fixer-models', spec.fixer_models])
    if keep_improving:
        cmd.append('--keep-improving')
        cmd.extend(['--improvement-rounds', str(improvement_rounds)])
    if variant != 'regular':
        cmd.append('--allow-baseline')
    cmd = _maybe_submit_args(cmd, submit, submit_message, submit_via, kaggle_json_path)
    env = {'G4F_PROVIDER': g4f_provider.strip()} if g4f_provider.strip() else None
    code, log = _run(cmd, cwd=paths['repo_dir'], env=env)
    if code != 0:
        raise RuntimeError(log)
    score, rows = _score_submission(output_path)
    note = 'regular is from-scratch' if variant == 'regular' else 'baseline-backed prompt variant'
    if spec.models:
        note += f' | model={spec.models}'
    elif spec.planner_models or spec.coder_models or spec.fixer_models:
        note += ' | per-agent models'
    return PipelineResult(f'prompt_run:{variant}@{spec.tag}', str(output_path), score, rows, note=note, run_log=str(run_log_path)), log


def _run_pipelines(repo_dir_text: str, selected_pipelines: list[str], prompt_model_run_mode: str, single_model: str, selected_models: list[str] | None, working_models_state: list[str] | None, planner_model: str, coder_model: str, fixer_model: str, search_mode: str, keep_improving: bool, improvement_rounds: int, submit: bool, submit_message: str, submit_via: str, kaggle_file_upload: str | None, kaggle_json_path_text: str, g4f_provider: str, disable_cayleypy: bool, light_budget: float, aggressive_budget: float, aggressive_min_len: int, max_rows: int, results_state: list[dict[str, Any]] | None):
    repo_dir = Path(repo_dir_text).expanduser().resolve()
    paths = _repo_paths(repo_dir)
    results = list(results_state or [])
    logs: list[str] = []
    if not selected_pipelines:
        raise RuntimeError('No pipelines selected.')
    kaggle_json_path = None
    if kaggle_file_upload:
        _install_kaggle_json(kaggle_file_upload)
        kaggle_json_path = str(Path.home() / '.kaggle' / 'kaggle.json')
    elif kaggle_json_path_text.strip():
        kaggle_json_path = str(Path(kaggle_json_path_text.strip()).expanduser().resolve())
    prompt_specs = _build_prompt_model_specs(prompt_model_run_mode, single_model, selected_models, working_models_state, planner_model, coder_model, fixer_model)
    stamp = _timestamp()
    for pipeline_name in selected_pipelines:
        if pipeline_name.startswith('prompt_run:'):
            variant = pipeline_name.split(':', 1)[1]
            for spec in prompt_specs:
                out_name = f'{pipeline_name.replace(":", "_")}_{spec.tag}_{stamp}.csv'
                output_path = paths['generated_dir'] / out_name
                run_log_path = paths['generated_dir'] / (out_name + '.run_log.json')
                result, log = _run_prompt_variant(paths, variant, spec, output_path, run_log_path, submit, submit_message, submit_via, kaggle_json_path, search_mode, keep_improving, improvement_rounds, max_rows, g4f_provider)
                results.append(asdict(result))
                logs.append(log)
        elif pipeline_name == 'baseline_no_llm':
            out_name = f'baseline_no_llm_{stamp}.csv'
            result, log = _run_baseline(paths, paths['generated_dir'] / out_name, paths['generated_dir'] / (out_name + '.run_log.json'), submit, submit_message, submit_via, kaggle_json_path, max_rows)
            results.append(asdict(result))
            logs.append(log)
        elif pipeline_name == 'best_tested_solver':
            out_name = f'best_tested_solver_{stamp}.csv'
            result, log = _run_best_solver(paths, paths['generated_dir'] / out_name)
            results.append(asdict(result))
            logs.append(log)
        elif pipeline_name == 'optimized_assets_only':
            out_name = f'optimized_assets_only_{stamp}.csv'
            result, log = _run_build_assets(paths, 'optimized_assets_only', 'none', None, paths['generated_dir'] / out_name, paths['generated_dir'] / (out_name + '.run_log.json'), disable_cayleypy, light_budget, aggressive_budget, aggressive_min_len)
            results.append(asdict(result))
            logs.append(log)
        elif pipeline_name == 'optimized_assets_v3_top150':
            out_name = f'optimized_assets_v3_top150_{stamp}.csv'
            result, log = _run_build_assets(paths, 'optimized_assets_v3_top150', 'v3', 150, paths['generated_dir'] / out_name, paths['generated_dir'] / (out_name + '.run_log.json'), disable_cayleypy, light_budget, aggressive_budget, aggressive_min_len)
            results.append(asdict(result))
            logs.append(log)
        elif pipeline_name == 'optimized_assets_v3_top300':
            out_name = f'optimized_assets_v3_top300_{stamp}.csv'
            result, log = _run_build_assets(paths, 'optimized_assets_v3_top300', 'v3', 300, paths['generated_dir'] / out_name, paths['generated_dir'] / (out_name + '.run_log.json'), disable_cayleypy, light_budget, aggressive_budget, aggressive_min_len)
            results.append(asdict(result))
            logs.append(log)
        else:
            raise RuntimeError(f'Unknown pipeline: {pipeline_name}')
    best = _best_result(results)
    best_md = 'Нет результатов.'
    best_file = None
    if best:
        best_md = (
            '### Лучший результат\n'
            f'- **name:** `{best.get("name")}`\n'
            f'- **score:** `{best.get("score")}`\n'
            f'- **rows:** `{best.get("rows")}`\n'
            f'- **note:** {best.get("note") or "-"}\n'
            f'- **submission:** `{best.get("submission_path")}`\n'
        )
        best_file = best.get('submission_path')
    return results, _results_to_rows(results), best_md, best_file, '\n\n'.join(logs)


def _refresh_results(repo_dir_text: str, results_state: list[dict[str, Any]] | None):
    results = list(results_state or [])
    best = _best_result(results)
    best_md = 'Нет результатов.'
    best_file = None
    if best:
        best_md = (
            '### Лучший результат\n'
            f'- **name:** `{best.get("name")}`\n'
            f'- **score:** `{best.get("score")}`\n'
            f'- **rows:** `{best.get("rows")}`\n'
            f'- **note:** {best.get("note") or "-"}\n'
            f'- **submission:** `{best.get("submission_path")}`\n'
        )
        best_file = best.get('submission_path')
    return _results_to_rows(results), best_md, best_file, json.dumps(results, ensure_ascii=False, indent=2)


def _submit_best(repo_dir_text: str, results_state: list[dict[str, Any]] | None, kaggle_file_upload: str | None, kaggle_json_path_text: str, message: str, submit_via: str):
    repo_dir = Path(repo_dir_text).expanduser().resolve()
    best = _best_result(list(results_state or []))
    if not best:
        raise RuntimeError('Нет результатов для отправки.')
    if kaggle_file_upload:
        _install_kaggle_json(kaggle_file_upload)
        kaggle_json_path = str(Path.home() / '.kaggle' / 'kaggle.json')
    elif kaggle_json_path_text.strip():
        kaggle_json_path = str(Path(kaggle_json_path_text.strip()).expanduser().resolve())
    else:
        kaggle_json_path = None
    preflight_cmd = [sys.executable, 'pipeline_cli.py', 'kaggle-preflight', '--competition', 'cayley-py-megaminx']
    if kaggle_json_path:
        preflight_cmd.extend(['--kaggle-json', kaggle_json_path])
    _, preflight_log = _run(preflight_cmd, cwd=repo_dir)
    submit_cmd = [sys.executable, 'pipeline_cli.py', 'run', '--competition', 'cayley-py-megaminx', '--output', str(best['submission_path']), '--no-llm', '--submit', '--submit-via', submit_via, '--message', message or f'GUI submit {datetime.now(timezone.utc).isoformat()}']
    if kaggle_json_path:
        submit_cmd.extend(['--kaggle-json', kaggle_json_path])
    code, submit_log = _run(submit_cmd, cwd=repo_dir)
    return preflight_log + '\n\n' + submit_log, (best['submission_path'] if code == 0 else None)


def create_demo(repo_dir: str | Path | None = None) -> gr.Blocks:
    resolved_repo = Path(repo_dir or Path(__file__).resolve().parent).expanduser().resolve()
    repo_text_default = str(resolved_repo)
    with gr.Blocks(title='Megaminx pipelines GUI') as demo:
        gr.Markdown(
            '# Megaminx pipelines GUI\n'
            'Интерфейс для запуска megaminx-пайплайнов, проверки **g4f** моделей, '
            'выбора одной или нескольких моделей и опциональной Kaggle-отправки.'
        )
        results_state = gr.State([])
        working_models_state = gr.State([])
        with gr.Tab('1) Setup'):
            with gr.Row():
                repo_dir_text = gr.Textbox(label='Путь к локальному репозиторию', value=repo_text_default, scale=3)
                kaggle_json_path_text = gr.Textbox(label='Путь к kaggle.json (опционально)', value='')
            with gr.Row():
                kaggle_json_upload = gr.File(label='Или загрузите kaggle.json', type='filepath')
                install_btn = gr.Button('Установить зависимости')
            with gr.Row():
                install_full_requirements = gr.Checkbox(label='Установить requirements-full.txt', value=True)
                install_g4f = gr.Dropdown(label='Установка g4f', choices=['none', 'github', 'pypi'], value='none')
                install_cayleypy = gr.Dropdown(label='Установка cayleypy', choices=['none', 'github', 'pypi'], value='none')
            extra_pip_packages = gr.Textbox(label='Доп. pip-пакеты через запятую', value='')
            setup_log = gr.Textbox(label='Лог setup/install', lines=18)
            kaggle_install_btn = gr.Button('Сохранить Kaggle credentials')
            kaggle_status = gr.Textbox(label='Статус Kaggle credentials', lines=3)
            install_btn.click(fn=_install_dependencies, inputs=[repo_dir_text, install_full_requirements, install_g4f, install_cayleypy, extra_pip_packages], outputs=[setup_log])
            kaggle_install_btn.click(fn=_install_kaggle_json, inputs=[kaggle_json_upload], outputs=[kaggle_status])
        with gr.Tab('2) g4f models'):
            gr.Markdown('Проверьте **все** или **только выбранные** g4f-модели. После probe рабочие модели автоматически появятся в селекторах ниже.')
            with gr.Row():
                probe_scope = gr.Dropdown(label='Режим проверки', choices=['selected', 'all_discovered', 'discover_only'], value='selected')
                probe_mode = gr.Dropdown(label='Как проверять', choices=['pipeline', 'async'], value='pipeline')
                probe_timeout = gr.Slider(label='Timeout на модель (сек)', minimum=2, maximum=60, value=12, step=1)
                probe_concurrency = gr.Slider(label='Async concurrency', minimum=1, maximum=32, value=5, step=1)
            with gr.Row():
                probe_max_models = gr.Slider(label='Макс. число моделей (0 = без лимита)', minimum=0, maximum=100, value=0, step=1)
                g4f_provider = gr.Textbox(label='G4F provider (опционально)', value='')
                backend_api_url = gr.Textbox(label='G4F backend API URL (опционально)', value='')
            models_text = gr.Textbox(label='Выбранные модели (через запятую или с новой строки)', value='gpt-4o-mini, deepseek-chat', lines=3)
            probe_btn = gr.Button('Проверить g4f-модели')
            probe_payload = gr.JSON(label='Результат probe')
            working_models_text = gr.Textbox(label='Рабочие модели', lines=4)
            probe_log = gr.Textbox(label='Лог probe', lines=16)
        with gr.Tab('3) Pipelines'):
            gr.Markdown('Выберите пайплайны, режим моделей и расширенные параметры. `regular` остаётся вариантом **from scratch**.')
            pipeline_selection = gr.CheckboxGroup(label='Какие пайплайны запускать', choices=PIPELINES, value=['best_tested_solver', 'optimized_assets_v3_top150'], show_select_all=True)
            with gr.Row():
                prompt_model_run_mode = gr.Dropdown(label='Режим выбора моделей для prompt-пайплайнов', choices=['single', 'selected', 'working'], value='single')
                single_model = gr.Dropdown(label='Одна g4f-модель', choices=COMMON_G4F_MODELS, value='gpt-4o-mini', allow_custom_value=True)
                selected_models = gr.Dropdown(label='Набор g4f-моделей', choices=COMMON_G4F_MODELS, value=['gpt-4o-mini'], multiselect=True, allow_custom_value=True)
            with gr.Row():
                planner_model = gr.Dropdown(label='Planner model override', choices=COMMON_G4F_MODELS, value=None, allow_custom_value=True)
                coder_model = gr.Dropdown(label='Coder model override', choices=COMMON_G4F_MODELS, value=None, allow_custom_value=True)
                fixer_model = gr.Dropdown(label='Fixer model override', choices=COMMON_G4F_MODELS, value=None, allow_custom_value=True)
            with gr.Row():
                search_mode = gr.Dropdown(label='search-mode для prompt-run', choices=['classic', 'hybrid'], value='hybrid')
                keep_improving = gr.Checkbox(label='keep-improving', value=False)
                improvement_rounds = gr.Slider(label='improvement-rounds', minimum=1, maximum=10, value=3, step=1)
                max_rows = gr.Slider(label='max-rows (0 = весь test.csv)', minimum=0, maximum=1001, value=0, step=1)
            with gr.Accordion('v3 / advanced options', open=False):
                with gr.Row():
                    disable_cayleypy = gr.Checkbox(label='Отключить cayleypy backend', value=False)
                    light_budget = gr.Slider(label='v3 light budget / row', minimum=0.05, maximum=2.0, value=0.20, step=0.05)
                    aggressive_budget = gr.Slider(label='v3 aggressive budget / row', minimum=0.10, maximum=3.0, value=0.75, step=0.05)
                    aggressive_min_len = gr.Slider(label='v3 aggressive min path len', minimum=100, maximum=1000, value=700, step=10)
                with gr.Row():
                    submit = gr.Checkbox(label='Live submit на Kaggle во время run', value=False)
                    submit_via = gr.Dropdown(label='submit-via', choices=['auto', 'api', 'cli'], value='auto')
                submit_message = gr.Textbox(label='Kaggle submit message', value=f'megaminx gui run {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")}')
            run_btn = gr.Button('Запустить выбранные пайплайны', variant='primary')
            run_log = gr.Textbox(label='Лог запуска', lines=20)
        with gr.Tab('4) Results'):
            results_table = gr.Dataframe(headers=['name', 'score', 'rows', 'note', 'submission_path', 'run_log'], datatype=['str', 'number', 'number', 'str', 'str', 'str'], label='Результаты')
            best_markdown = gr.Markdown('Нет результатов.')
            best_file = gr.File(label='Лучший submission.csv')
            raw_results_json = gr.Code(label='Raw results JSON', language='json', lines=16)
            refresh_btn = gr.Button('Обновить результаты')
            submit_best_btn = gr.Button('Отправить лучший результат на Kaggle')
            submit_best_log = gr.Textbox(label='Лог preflight / submit', lines=18)
        probe_btn.click(fn=_probe_models, inputs=[repo_dir_text, probe_scope, models_text, g4f_provider, backend_api_url, probe_timeout, probe_max_models, probe_mode, probe_concurrency], outputs=[probe_payload, working_models_text, probe_log, working_models_state, single_model, selected_models])
        run_btn.click(fn=_run_pipelines, inputs=[repo_dir_text, pipeline_selection, prompt_model_run_mode, single_model, selected_models, working_models_state, planner_model, coder_model, fixer_model, search_mode, keep_improving, improvement_rounds, submit, submit_message, submit_via, kaggle_json_upload, kaggle_json_path_text, g4f_provider, disable_cayleypy, light_budget, aggressive_budget, aggressive_min_len, max_rows, results_state], outputs=[results_state, results_table, best_markdown, best_file, run_log])
        refresh_btn.click(fn=_refresh_results, inputs=[repo_dir_text, results_state], outputs=[results_table, best_markdown, best_file, raw_results_json])
        submit_best_btn.click(fn=_submit_best, inputs=[repo_dir_text, results_state, kaggle_json_upload, kaggle_json_path_text, submit_message, submit_via], outputs=[submit_best_log, best_file])
    return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.queue(default_concurrency_limit=1).launch(server_name='0.0.0.0', server_port=7860, share=False)
