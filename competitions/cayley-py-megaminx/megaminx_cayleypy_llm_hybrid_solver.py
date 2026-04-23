from __future__ import annotations

"""Comprehensive Megaminx strategy combining vendored CayleyPy + repository LLM flows.

Architecture
------------
1. Deterministic candidate layer:
   - best-tested lookup / fallback solver
   - optimized bundled submission
   - optional search_v3 refinement
2. LLM candidate layer:
   - reuse already-generated submissions when present
   - optionally generate new candidates through `pipeline_cli.py run`
3. Ensemble layer:
   - choose the shortest locally validated path per row across all candidates
4. CayleyPy-backed post-optimization:
   - run `search_improver_v3.improve_submission_rows` on the fused submission
     (which internally uses `MegaminxSearchAdapter` with exact BFS + beam + fallback)
5. Final validation and optional Kaggle submit.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for p in [REPO_ROOT / 'third_party' / 'cayleypy-main', REPO_ROOT, HERE]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import importlib.util


def _load_local_module(name: str, filename: str):
    module_path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {filename} from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


sm = _load_local_module('megaminx_hybrid_solve_module', 'solve_module.py')
v3 = _load_local_module('megaminx_hybrid_search_improver_v3', 'search_improver_v3.py')
best_tested = _load_local_module('megaminx_hybrid_best_tested_solver', 'megaminx_best_tested_solver.py')
neighbour_lane = _load_local_module('megaminx_hybrid_neighbour_model_lane', 'megaminx_neighbour_model_lane.py')

COMPETITION = 'cayley-py-megaminx'
SUBMISSIONS_DIR = HERE / 'submissions'
DATA_DIR = HERE / 'data'
DEFAULT_OUT = SUBMISSIONS_DIR / 'submission_cayleypy_llm_hybrid.csv'
DEFAULT_STATS = SUBMISSIONS_DIR / 'submission_cayleypy_llm_hybrid.stats.json'
DEFAULT_PROFILES = SUBMISSIONS_DIR / 'submission_cayleypy_llm_hybrid.profiles.json'


@dataclass
class CandidateSubmission:
    name: str
    path: Path
    rows: list[dict[str, str]]
    score: int
    generated: bool = False
    validated_rows: int = 0
    invalid_rows: int = 0


@dataclass
class ValidationResult:
    valid: bool
    final_state: list[int] | None
    error: str | None = None


CENTRAL, GENERATORS = sm.load_puzzle_bundle()
TEST_ROWS: list[dict[str, str]] | None = None
_VALIDATION_CACHE: dict[tuple[int, str], ValidationResult] = {}


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        writer.writerows(rows)


def _score_rows(rows: Sequence[dict[str, str]]) -> int:
    total = 0
    for row in rows:
        path = str(row.get('path') or '').strip()
        total += 0 if not path else len([tok for tok in path.split('.') if tok])
    return total


def _get_test_rows() -> list[dict[str, str]]:
    global TEST_ROWS
    if TEST_ROWS is None:
        TEST_ROWS = _load_csv_rows(DATA_DIR / 'test.csv')
    return TEST_ROWS


def _row_state(row_idx: int) -> list[int]:
    row = _get_test_rows()[row_idx]
    return [int(x) for x in str(row.get('initial_state') or '').split(',') if x != '']


def _validate_path(row_idx: int, path: str) -> ValidationResult:
    cache_key = (int(row_idx), str(path))
    cached = _VALIDATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    moves = sm.path_to_moves(path)
    state = _row_state(row_idx)
    try:
        final_state = sm.apply_moves(state, moves, GENERATORS)
    except KeyError as exc:
        result = ValidationResult(valid=False, final_state=None, error=f'unknown_move:{exc}')
        _VALIDATION_CACHE[cache_key] = result
        return result
    valid = list(final_state) == list(CENTRAL)
    result = ValidationResult(valid=valid, final_state=list(final_state), error=None if valid else 'not_solved')
    _VALIDATION_CACHE[cache_key] = result
    return result


def _load_candidate(path: Path, *, name: str, generated: bool = False, strict_validate: bool = False) -> CandidateSubmission | None:
    if not path.exists():
        return None
    rows = _load_csv_rows(path)
    invalid_rows = 0
    validated_rows = 0
    if strict_validate:
        limit = min(len(rows), len(_get_test_rows()))
        for idx in range(limit):
            vr = _validate_path(idx, str(rows[idx].get('path') or '').strip())
            validated_rows += 1
            if not vr.valid:
                invalid_rows += 1
    return CandidateSubmission(
        name=name,
        path=path,
        rows=rows,
        score=_score_rows(rows),
        generated=generated,
        validated_rows=validated_rows,
        invalid_rows=invalid_rows,
    )


def _build_best_tested_submission(out_path: Path) -> CandidateSubmission:
    payload = best_tested.build_submission(out_path)
    rows = _load_csv_rows(out_path)
    return CandidateSubmission(
        name='best_tested_lookup',
        path=out_path,
        rows=rows,
        score=int(payload.get('score') or _score_rows(rows)),
        generated=True,
    )


def _maybe_copy(src: Path, dst: Path) -> Path:
    if src.resolve() == dst.resolve():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def _run_command(cmd: Sequence[str], *, cwd: Path) -> None:
    printable = ' '.join(cmd)
    print(f'[cmd] {printable}', flush=True)
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def _variant_to_output_name(variant: str) -> str:
    safe = variant.replace(',', '_').replace(' ', '_')
    return f'submission_llm_{safe}.csv'


def _generate_llm_candidate(
    *,
    variant: str,
    output_path: Path,
    models: str | None,
    agent_models: str | None,
    planner_models: str | None,
    coder_models: str | None,
    fixer_models: str | None,
    keep_improving: bool,
    improvement_rounds: int,
    baseline: str | None,
    max_iters: int | None = None,
    baseline_patch_max_iters: int | None = None,
    g4f_recovery_max_iters: int | None = None,
) -> Path:
    cmd = [
        sys.executable,
        'pipeline_cli.py',
        'run',
        '--competition',
        COMPETITION,
        '--output',
        str(output_path),
        '--prompt-variant',
        variant,
    ]
    if models:
        cmd.extend(['--models', models])
    if agent_models:
        cmd.extend(['--agent-models', agent_models])
    if planner_models:
        cmd.extend(['--planner-models', planner_models])
    if coder_models:
        cmd.extend(['--coder-models', coder_models])
    if fixer_models:
        cmd.extend(['--fixer-models', fixer_models])
    if keep_improving:
        cmd.extend(['--keep-improving', '--improvement-rounds', str(improvement_rounds)])
    if max_iters is not None:
        cmd.extend(['--max-iters', str(max(1, int(max_iters)))])
    if baseline_patch_max_iters is not None:
        cmd.extend(['--baseline-patch-max-iters', str(max(1, int(baseline_patch_max_iters)))])
    if g4f_recovery_max_iters is not None:
        cmd.extend(['--g4f-recovery-max-iters', str(max(1, int(g4f_recovery_max_iters)))])
    if baseline:
        baseline_path = Path(baseline)
        if baseline_path.suffix.lower() == '.py':
            cmd.extend(['--baseline', baseline])
        else:
            print(
                f"[baseline] WARNING: skipping non-Python --baseline for solver generation: {baseline_path}",
                flush=True,
            )
    _run_command(cmd, cwd=REPO_ROOT)
    return output_path


def _run_search_v3(source_submission: Path, output_path: Path, args: argparse.Namespace) -> Path:
    cmd = [
        sys.executable,
        str(HERE / 'search_improver_v3.py'),
        '--submission',
        str(source_submission),
        '--out',
        str(output_path),
        '--stats-out',
        str(output_path.with_suffix('.stats.json')),
        '--profile-out',
        str(output_path.with_suffix('.profiles.json')),
        '--top-k',
        str(args.search_v3_top_k),
        '--min-improvement',
        str(args.search_v3_min_improvement),
        '--light-min-path-len',
        str(args.light_min_path_len),
        '--aggressive-min-path-len',
        str(args.aggressive_min_path_len),
        '--force-aggressive-top-n',
        str(args.force_aggressive_top_n),
        '--light-time-budget-per-row',
        str(args.light_time_budget_per_row),
        '--aggressive-time-budget-per-row',
        str(args.aggressive_time_budget_per_row),
        '--light-beam-width',
        str(args.light_beam_width),
        '--aggressive-beam-width',
        str(args.aggressive_beam_width),
        '--light-max-steps',
        str(args.light_max_steps),
        '--aggressive-max-steps',
        str(args.aggressive_max_steps),
        '--light-history-depth',
        str(args.light_history_depth),
        '--aggressive-history-depth',
        str(args.aggressive_history_depth),
        '--light-mitm-depth',
        str(args.light_mitm_depth),
        '--aggressive-mitm-depth',
        str(args.aggressive_mitm_depth),
        '--light-window-lengths',
        args.light_window_lengths,
        '--aggressive-window-lengths',
        args.aggressive_window_lengths,
        '--light-window-samples',
        str(args.light_window_samples),
        '--aggressive-window-samples',
        str(args.aggressive_window_samples),
        '--light-beam-mode',
        args.light_beam_mode,
        '--aggressive-beam-mode',
        args.aggressive_beam_mode,
    ]
    if args.disable_cayleypy:
        cmd.append('--disable-cayleypy')
    _run_command(cmd, cwd=REPO_ROOT)
    return output_path


def _build_namespace_for_v3(args: argparse.Namespace) -> argparse.Namespace:
    ns = v3.build_arg_parser().parse_args([])
    ns.top_k = args.search_v3_top_k
    ns.min_improvement = args.search_v3_min_improvement
    ns.disable_cayleypy = args.disable_cayleypy
    ns.light_min_path_len = args.light_min_path_len
    ns.aggressive_min_path_len = args.aggressive_min_path_len
    ns.force_aggressive_top_n = args.force_aggressive_top_n
    ns.light_time_budget_per_row = args.light_time_budget_per_row
    ns.aggressive_time_budget_per_row = args.aggressive_time_budget_per_row
    ns.light_beam_width = args.light_beam_width
    ns.aggressive_beam_width = args.aggressive_beam_width
    ns.light_max_steps = args.light_max_steps
    ns.aggressive_max_steps = args.aggressive_max_steps
    ns.light_history_depth = args.light_history_depth
    ns.aggressive_history_depth = args.aggressive_history_depth
    ns.light_mitm_depth = args.light_mitm_depth
    ns.aggressive_mitm_depth = args.aggressive_mitm_depth
    ns.light_window_lengths = args.light_window_lengths
    ns.aggressive_window_lengths = args.aggressive_window_lengths
    ns.light_window_samples = args.light_window_samples
    ns.aggressive_window_samples = args.aggressive_window_samples
    ns.light_beam_mode = args.light_beam_mode
    ns.aggressive_beam_mode = args.aggressive_beam_mode
    return ns


def _default_candidate_paths() -> list[tuple[str, Path]]:
    return [
        ('optimized_submission', SUBMISSIONS_DIR / 'optimized_submission.csv'),
        ('search_v3_top300', SUBMISSIONS_DIR / 'submission_search_improved_v3_top300.csv'),
        ('search_v3_default', SUBMISSIONS_DIR / 'submission_search_improved_v3.csv'),
        ('llm_structured', SUBMISSIONS_DIR / 'submission_structured.csv'),
        ('llm_heuristic_boosted', SUBMISSIONS_DIR / 'submission_heuristic_boosted.csv'),
        ('llm_master_hybrid', SUBMISSIONS_DIR / 'submission_master_hybrid.csv'),
        ('llm_neighbour_model_hybrid', SUBMISSIONS_DIR / 'submission_neighbour_model_hybrid.csv'),
        ('llm_notebook_structured_live', SUBMISSIONS_DIR / 'submission_notebook_structured_live.csv'),
    ]


def _fuse_candidates(candidates: Sequence[CandidateSubmission]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    test_rows = _get_test_rows()
    fused: list[dict[str, str]] = []
    source_counter: Counter[str] = Counter()
    invalid_counter: Counter[str] = Counter()
    per_row_stats: list[dict[str, Any]] = []

    for idx, test_row in enumerate(test_rows):
        chosen_path = None
        chosen_source = None
        chosen_len = None
        contenders: list[dict[str, Any]] = []
        for cand in candidates:
            if idx >= len(cand.rows):
                continue
            path = str(cand.rows[idx].get('path') or '').strip()
            vr = _validate_path(idx, path)
            clen = 0 if not path else len(sm.path_to_moves(path))
            contenders.append({
                'candidate': cand.name,
                'length': clen,
                'valid': vr.valid,
                'error': vr.error,
            })
            if not vr.valid:
                invalid_counter[cand.name] += 1
                continue
            if chosen_path is None or clen < int(chosen_len):
                chosen_path = path
                chosen_source = cand.name
                chosen_len = clen
        if chosen_path is None:
            # Robust fallback to best-tested direct solver for this row.
            state = _row_state(idx)
            result = best_tested.solve_state(state)
            chosen_path = str(result.get('path') or '')
            chosen_source = 'fallback_best_tested_runtime'
            chosen_len = 0 if not chosen_path else len(sm.path_to_moves(chosen_path))
        source_counter[str(chosen_source)] += 1
        fused.append({
            'initial_state_id': str(test_row.get('initial_state_id') or ''),
            'path': str(chosen_path),
        })
        per_row_stats.append({
            'row_index': idx,
            'winner': chosen_source,
            'winner_len': chosen_len,
            'candidates': contenders,
        })

    stats = {
        'rows': len(fused),
        'score': _score_rows(fused),
        'winner_counts': dict(source_counter),
        'invalid_counts': dict(invalid_counter),
        'per_row': per_row_stats,
    }
    return fused, stats


def _validate_submission_rows(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    valid = 0
    invalid_examples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        path = str(row.get('path') or '').strip()
        vr = _validate_path(idx, path)
        if vr.valid:
            valid += 1
        elif len(invalid_examples) < 20:
            invalid_examples.append({'row_index': idx, 'path': path[:200], 'error': vr.error})
    return {
        'rows': len(rows),
        'valid_rows': valid,
        'invalid_rows': len(rows) - valid,
        'invalid_examples': invalid_examples,
        'score': _score_rows(rows),
    }


def _submit_to_kaggle(submission_path: Path, message: str, *, kaggle_json: str | None = None, kaggle_config_dir: str | None = None) -> None:
    env = os.environ.copy()
    if kaggle_config_dir:
        env['KAGGLE_CONFIG_DIR'] = kaggle_config_dir
    if kaggle_json:
        kaggle_json_path = Path(kaggle_json)
        target_dir = Path(kaggle_config_dir) if kaggle_config_dir else (Path.home() / '.kaggle')
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / 'kaggle.json'
        if kaggle_json_path.resolve() != target_path.resolve():
            shutil.copyfile(kaggle_json_path, target_path)
            os.chmod(target_path, 0o600)
    cmd = [
        'kaggle',
        'competitions',
        'submit',
        COMPETITION,
        '-f',
        str(submission_path),
        '-m',
        message,
    ]
    print(f'[cmd] {' '.join(cmd)}', flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Advanced CayleyPy + LLM Megaminx hybrid orchestrator')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--stats-out', type=Path, default=DEFAULT_STATS)
    p.add_argument('--profiles-out', type=Path, default=DEFAULT_PROFILES)
    p.add_argument('--candidate', action='append', default=[], help='Extra candidate submission CSVs to include')
    p.add_argument('--skip-default-candidates', action='store_true')
    p.add_argument('--strict-validate-candidates', action='store_true')
    p.add_argument('--generate-llm', action='store_true', help='Generate fresh LLM candidate submissions before fusion')
    p.add_argument('--llm-variants', default='structured,heuristic_boosted,master_hybrid,neighbour_model_hybrid')
    p.add_argument('--models', default=None)
    p.add_argument('--agent-models', default=None)
    p.add_argument('--planner-models', default=None)
    p.add_argument('--coder-models', default=None)
    p.add_argument('--fixer-models', default=None)
    p.add_argument('--keep-improving-llm', action='store_true')
    p.add_argument('--llm-improvement-rounds', type=int, default=8)
    p.add_argument('--max-iters', type=int, default=None, help='Forwarded to pipeline_cli.py run / AgentLaboratory fixer iterations.')
    p.add_argument('--baseline-patch-max-iters', type=int, default=None, help='Forwarded to pipeline_cli.py run for baseline patch fixer iterations.')
    p.add_argument('--g4f-recovery-max-iters', type=int, default=None, help='Forwarded to pipeline_cli.py run for recovery fixer iterations.')
    p.add_argument('--baseline', default=None)
    p.add_argument('--generate-neighbour-model', action='store_true', help='Run the vendored CayleyPy neighbour-model lane against the fused submission before post-refinement')
    p.add_argument('--neighbour-model-repo', type=Path, default=None, help='Optional override for the vendored cayleypy-neighbour-model-training repo path')
    p.add_argument('--neighbour-model-model-id', type=int, default=1776581286)
    p.add_argument('--neighbour-model-device', default='cpu')
    p.add_argument('--neighbour-model-eval-batch-size', type=int, default=2048)
    p.add_argument('--neighbour-model-beam-width', type=int, default=128)
    p.add_argument('--neighbour-model-num-steps', type=int, default=18)
    p.add_argument('--neighbour-model-num-attempts', type=int, default=4)
    p.add_argument('--neighbour-model-max-rows', type=int, default=32, help='How many longest fused rows to try solving from scratch with the neighbour-model lane')
    p.add_argument('--run-search-v3', action='store_true', help='Generate a fresh search_v3 candidate from the current best deterministic source before fusion')
    p.add_argument('--search-v3-source', type=Path, default=None, help='If set, use this submission as input to search_v3')
    p.add_argument('--skip-post-refine', action='store_true', help='Skip the final v3 post-refinement on the fused submission')
    p.add_argument('--disable-cayleypy', action='store_true')
    p.add_argument('--search-v3-top-k', type=int, default=220)
    p.add_argument('--search-v3-min-improvement', type=int, default=2)
    p.add_argument('--light-min-path-len', type=int, default=560)
    p.add_argument('--aggressive-min-path-len', type=int, default=700)
    p.add_argument('--force-aggressive-top-n', type=int, default=24)
    p.add_argument('--light-time-budget-per-row', type=float, default=0.25)
    p.add_argument('--aggressive-time-budget-per-row', type=float, default=0.9)
    p.add_argument('--light-beam-width', type=int, default=96)
    p.add_argument('--aggressive-beam-width', type=int, default=224)
    p.add_argument('--light-max-steps', type=int, default=8)
    p.add_argument('--aggressive-max-steps', type=int, default=12)
    p.add_argument('--light-history-depth', type=int, default=0)
    p.add_argument('--aggressive-history-depth', type=int, default=2)
    p.add_argument('--light-mitm-depth', type=int, default=2)
    p.add_argument('--aggressive-mitm-depth', type=int, default=4)
    p.add_argument('--light-window-lengths', default='14,18,22')
    p.add_argument('--aggressive-window-lengths', default='18,24,30,36')
    p.add_argument('--light-window-samples', type=int, default=8)
    p.add_argument('--aggressive-window-samples', type=int, default=16)
    p.add_argument('--light-beam-mode', default='simple', choices=['simple', 'advanced'])
    p.add_argument('--aggressive-beam-mode', default='advanced', choices=['simple', 'advanced'])
    p.add_argument('--submit', action='store_true')
    p.add_argument('--message', default='cayleypy+llm hybrid validated')
    p.add_argument('--kaggle-json', default=None)
    p.add_argument('--kaggle-config-dir', default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = build_arg_parser().parse_known_args(argv)
    if extra:
        print(f"[compat] ignoring extra args: {' '.join(extra)}", flush=True)
    t0 = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.profiles_out.parent.mkdir(parents=True, exist_ok=True)

    candidate_records: list[CandidateSubmission] = []
    with tempfile.TemporaryDirectory(prefix='megaminx_hybrid_') as tmpdir_raw:
        tmpdir = Path(tmpdir_raw)

        # Deterministic anchor candidate: always build fresh from the best tested solver.
        best_path = tmpdir / 'submission_best_tested_lookup.csv'
        best_cand = _build_best_tested_submission(best_path)
        candidate_records.append(best_cand)

        if not args.skip_default_candidates:
            for name, path in _default_candidate_paths():
                cand = _load_candidate(path, name=name, strict_validate=args.strict_validate_candidates)
                if cand is not None:
                    candidate_records.append(cand)

        for raw_path in args.candidate:
            path = Path(raw_path)
            cand = _load_candidate(path, name=f'user:{path.name}', strict_validate=args.strict_validate_candidates)
            if cand is not None:
                candidate_records.append(cand)

        # Optional fresh LLM generations via repository pipeline.
        if args.generate_llm:
            variants = [v.strip() for v in str(args.llm_variants).split(',') if v.strip()]
            for variant in variants:
                out_path = tmpdir / _variant_to_output_name(variant)
                try:
                    _generate_llm_candidate(
                        variant=variant,
                        output_path=out_path,
                        models=args.models,
                        agent_models=args.agent_models,
                        planner_models=args.planner_models,
                        coder_models=args.coder_models,
                        fixer_models=args.fixer_models,
                        keep_improving=args.keep_improving_llm,
                        improvement_rounds=args.llm_improvement_rounds,
                        baseline=args.baseline,
                        max_iters=args.max_iters,
                        baseline_patch_max_iters=args.baseline_patch_max_iters,
                        g4f_recovery_max_iters=args.g4f_recovery_max_iters,
                    )
                    cand = _load_candidate(out_path, name=f'generated:{variant}', generated=True, strict_validate=True)
                    if cand is not None:
                        candidate_records.append(cand)
                except Exception as exc:
                    print(f'[warn] LLM candidate {variant} failed: {type(exc).__name__}: {exc}', flush=True)

        # Optional fresh search_v3 deterministic improver before fusion.
        if args.run_search_v3:
            if args.search_v3_source is not None:
                search_source = Path(args.search_v3_source)
            else:
                by_score = sorted(candidate_records, key=lambda c: c.score)
                search_source = by_score[0].path
            search_out = tmpdir / 'submission_search_v3_fresh.csv'
            try:
                _run_search_v3(search_source, search_out, args)
                cand = _load_candidate(search_out, name='generated:search_v3', generated=True, strict_validate=False)
                if cand is not None:
                    candidate_records.append(cand)
            except Exception as exc:
                print(f'[warn] search_v3 generation failed: {type(exc).__name__}: {exc}', flush=True)

        # Deduplicate by canonical path+score+name best-effort.
        dedup: dict[tuple[str, int], CandidateSubmission] = {}
        for cand in candidate_records:
            key = (cand.name, cand.score)
            dedup[key] = cand
        candidate_records = list(dedup.values())

        fused_rows, fusion_stats = _fuse_candidates(candidate_records)
        fused_path = tmpdir / 'submission_fused.csv'
        _write_rows(fused_path, fused_rows)

        final_rows = fused_rows
        neighbour_model_stats: dict[str, Any] = {}
        neighbour_model_profiles: list[dict[str, Any]] = []
        if args.generate_neighbour_model:
            neighbour_cfg = neighbour_lane.NeighbourModelConfig(
                repo_dir=args.neighbour_model_repo,
                model_id=int(args.neighbour_model_model_id),
                device=str(args.neighbour_model_device),
                eval_batch_size=int(args.neighbour_model_eval_batch_size),
                beam_width=int(args.neighbour_model_beam_width),
                num_steps=int(args.neighbour_model_num_steps),
                num_attempts=int(args.neighbour_model_num_attempts),
                max_rows=int(args.neighbour_model_max_rows),
                only_if_shorter=True,
            )
            try:
                nm_rows, nm_stats, nm_profiles = neighbour_lane.improve_submission_rows(
                    submission_rows=final_rows,
                    test_rows=_get_test_rows(),
                    central=CENTRAL,
                    generators=GENERATORS,
                    config=neighbour_cfg,
                )
                if _score_rows(nm_rows) <= _score_rows(final_rows):
                    final_rows = nm_rows
                neighbour_model_stats = nm_stats
                neighbour_model_profiles = nm_profiles
            except Exception as exc:
                neighbour_model_stats = {'error': f'{type(exc).__name__}: {exc}'}

        post_refine_stats: dict[str, Any] = {}
        post_refine_profiles: list[dict[str, Any]] = []
        if not args.skip_post_refine:
            ns = _build_namespace_for_v3(args)
            improved_rows, stats, profiles = v3.improve_submission_rows(
                submission_rows=final_rows,
                test_rows=_get_test_rows(),
                central=CENTRAL,
                generators=GENERATORS,
                args=ns,
            )
            if _score_rows(improved_rows) <= _score_rows(final_rows):
                final_rows = improved_rows
                post_refine_stats = stats
                post_refine_profiles = profiles

        _write_rows(args.out, final_rows)
        final_validation = _validate_submission_rows(final_rows)

        payload = {
            'competition': COMPETITION,
            'elapsed_seconds': round(time.perf_counter() - t0, 3),
            'candidates': [
                {
                    'name': c.name,
                    'path': str(c.path),
                    'score': c.score,
                    'generated': c.generated,
                    'validated_rows': c.validated_rows,
                    'invalid_rows': c.invalid_rows,
                }
                for c in sorted(candidate_records, key=lambda c: (c.score, c.name))
            ],
            'fusion': {
                'score': _score_rows(fused_rows),
                'winner_counts': fusion_stats.get('winner_counts', {}),
                'invalid_counts': fusion_stats.get('invalid_counts', {}),
            },
            'neighbour_model': neighbour_model_stats,
            'post_refine': post_refine_stats,
            'final_validation': final_validation,
            'output': str(args.out),
            'uses_cayleypy': not args.disable_cayleypy,
            'llm_enabled': bool(args.generate_llm),
            'neighbour_model_enabled': bool(args.generate_neighbour_model),
        }
        args.stats_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        args.profiles_out.write_text(
            json.dumps(
                {
                    'fusion_per_row': fusion_stats.get('per_row', []),
                    'neighbour_model_profiles': neighbour_model_profiles,
                    'post_refine_profiles': post_refine_profiles,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        print(json.dumps(payload, ensure_ascii=False))

        if final_validation['invalid_rows'] > 0:
            raise SystemExit('Final submission did not pass local validation.')
        if args.submit:
            _submit_to_kaggle(args.out, args.message, kaggle_json=args.kaggle_json, kaggle_config_dir=args.kaggle_config_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
