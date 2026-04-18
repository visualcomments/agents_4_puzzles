from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from row_scoreboard import build_row_scoreboard, build_shadow_splits, load_rows, summarize_scoreboard, write_json
from external_adapter_lane import materialize_external_candidates
from prompt_population_runner import evaluate_candidates
from portfolio_orchestrator import merge_candidates, write_rows


DEFAULT_MANIFESTS = [
    str(_HERE / 'external_solver_adapters' / 'manifest_real_odder_megaminxolver.json'),
    str(_HERE / 'external_solver_adapters' / 'manifest_real_sevilze_llminxsolver_cmp.json'),
    str(_HERE / 'external_solver_adapters' / 'manifest_real_abgolev_astar.json'),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Turnkey Megaminx breakthrough runner with real external repo lanes')
    parser.add_argument('--baseline-submission', default=str(_HERE / 'submissions' / 'optimized_submission.csv'))
    parser.add_argument('--test-csv', default=str(_HERE / 'data' / 'test.csv'))
    parser.add_argument('--work-dir', default=str(_HERE / 'runs' / 'turnkey_real_external'))
    parser.add_argument('--external-manifest', action='append', default=[])
    parser.add_argument('--skip-external', action='store_true')
    parser.add_argument('--skip-hard-row', action='store_true')
    parser.add_argument('--top-k', type=int, default=64)
    parser.add_argument('--light-time-budget-per-row', type=float, default=0.02)
    parser.add_argument('--aggressive-time-budget-per-row', type=float, default=0.05)
    parser.add_argument('--zip-out', default='')
    return parser


def _run(cmd: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    baseline_submission = Path(args.baseline_submission).resolve()
    test_csv = Path(args.test_csv).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    external_out_dir = work_dir / 'external_adapter'
    external_out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_rows(baseline_submission)
    test_rows = load_rows(test_csv)
    scoreboard = build_row_scoreboard(submission_rows=baseline_rows, test_rows=test_rows)
    summary = summarize_scoreboard(scoreboard)
    splits = build_shadow_splits(scoreboard)
    scoreboard_out = work_dir / 'row_scoreboard.json'
    scoreboard_summary_out = work_dir / 'row_scoreboard.summary.json'
    splits_out = work_dir / 'shadow_splits.json'
    write_json(scoreboard_out, scoreboard)
    write_json(scoreboard_summary_out, summary)
    write_json(splits_out, splits)

    hard_row_out = work_dir / 'submission_hard_row_routed.csv'
    hard_row_stats = work_dir / 'submission_hard_row_routed.stats.json'
    hard_row_profiles = work_dir / 'submission_hard_row_routed.profiles.json'
    if args.skip_hard_row:
        if not hard_row_out.exists():
            write_rows(hard_row_out, baseline_rows)
            hard_row_stats.write_text(json.dumps({'skipped': True, 'reason': 'skip-hard-row'}, ensure_ascii=False, indent=2), encoding='utf-8')
            hard_row_profiles.write_text(json.dumps({'skipped': True}, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        _run([
            sys.executable,
            str(_HERE / 'hard_row_routed_search.py'),
            '--submission', str(baseline_submission),
            '--test-csv', str(test_csv),
            '--out', str(hard_row_out),
            '--stats-out', str(hard_row_stats),
            '--profiles-out', str(hard_row_profiles),
            '--top-k', str(int(args.top_k)),
            '--light-time-budget-per-row', str(float(args.light_time_budget_per_row)),
            '--aggressive-time-budget-per-row', str(float(args.aggressive_time_budget_per_row)),
        ], cwd=_HERE.parent.parent)

    candidate_specs = [
        f'bundled={baseline_submission}',
        f'routed={hard_row_out}',
    ]
    external_summaries = []
    if not args.skip_external:
        manifests = list(args.external_manifest or DEFAULT_MANIFESTS)
        ext_specs, external_summaries = materialize_external_candidates(
            manifests,
            test_csv=test_csv,
            fallback_submission=baseline_submission,
            out_dir=external_out_dir,
        )
        candidate_specs.extend(ext_specs)

    candidate_eval = evaluate_candidates(candidate_specs, splits)
    candidate_eval['candidate_specs'] = candidate_specs
    candidate_eval['external_adapter_summaries'] = external_summaries
    candidate_eval_out = work_dir / 'prompt_population.results.json'
    candidate_eval_out.write_text(json.dumps(candidate_eval, ensure_ascii=False, indent=2), encoding='utf-8')

    merged, lineage = merge_candidates(candidate_specs)
    final_submission = work_dir / 'submission_portfolio_external_real.csv'
    write_rows(final_submission, merged)
    final_scoreboard = build_row_scoreboard(submission_rows=merged, test_rows=test_rows)
    final_summary = summarize_scoreboard(final_scoreboard)
    lineage['portfolio_summary'] = final_summary
    lineage['candidate_specs'] = candidate_specs
    lineage['external_adapter_summaries'] = external_summaries
    lineage['candidate_eval'] = candidate_eval
    final_lineage_out = work_dir / 'submission_portfolio_external_real.lineage.json'
    final_summary_out = work_dir / 'submission_portfolio_external_real.summary.json'
    final_scoreboard_out = work_dir / 'submission_portfolio_external_real.scoreboard.json'
    final_lineage_out.write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding='utf-8')
    final_summary_out.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    final_scoreboard_out.write_text(json.dumps(final_scoreboard, ensure_ascii=False, indent=2), encoding='utf-8')

    payload = {
        'baseline_submission': str(baseline_submission),
        'test_csv': str(test_csv),
        'work_dir': str(work_dir),
        'candidate_specs': candidate_specs,
        'candidate_eval_out': str(candidate_eval_out),
        'final_submission': str(final_submission),
        'final_summary': final_summary,
        'external_adapter_summaries': external_summaries,
    }
    (work_dir / 'turnkey_run_summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    zip_target = Path(args.zip_out).resolve() if str(args.zip_out).strip() else (work_dir.with_suffix('.zip'))
    with zipfile.ZipFile(zip_target, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(work_dir.rglob('*')):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(work_dir.parent))

    print(json.dumps({'work_dir': str(work_dir), 'zip_out': str(zip_target), 'final_submission': str(final_submission), 'score': final_summary['score']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
