from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from row_scoreboard import build_row_scoreboard, summarize_scoreboard
from external_adapter_lane import materialize_external_candidates
from prompt_population_runner import evaluate_candidates, load_splits

_HERE = Path(__file__).resolve().parent


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        writer.writerows(rows)


def score_path(path: str | None) -> int:
    text = (path or '').strip()
    return 0 if not text else len(text.split('.'))


def parse_candidate(spec: str) -> Tuple[str, Path]:
    if '=' not in spec:
        path = Path(spec)
        return path.stem, path
    label, raw_path = spec.split('=', 1)
    return label.strip(), Path(raw_path.strip())


def merge_candidates(candidate_specs: Sequence[str]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    parsed = [parse_candidate(spec) for spec in candidate_specs]
    candidate_rows = [(label, path, load_rows(path)) for label, path in parsed]
    if not candidate_rows:
        raise ValueError('No candidates provided')
    base_ids = [str(row.get('initial_state_id') or '') for row in candidate_rows[0][2]]
    merged: List[Dict[str, str]] = []
    lineage: List[Dict[str, Any]] = []
    wins = {label: 0 for label, _, _ in candidate_rows}
    for row_idx in range(len(base_ids)):
        winning_label = None
        winning_path = None
        winning_len = None
        winning_lane_idx = None
        for lane_idx, (label, path, rows) in enumerate(candidate_rows):
            if row_idx >= len(rows):
                raise IndexError(f'Candidate {path} is shorter than baseline rows')
            row_id = str(rows[row_idx].get('initial_state_id') or '')
            if row_id != base_ids[row_idx]:
                raise ValueError(f'Row mismatch at index {row_idx}: expected {base_ids[row_idx]!r}, got {row_id!r} from {path}')
            candidate_path = str(rows[row_idx].get('path') or '')
            candidate_len = score_path(candidate_path)
            if (
                winning_len is None
                or candidate_len < winning_len
                or (candidate_len == winning_len and (winning_lane_idx is None or lane_idx < winning_lane_idx))
            ):
                winning_label = label
                winning_path = candidate_path
                winning_len = candidate_len
                winning_lane_idx = lane_idx
        merged.append({'initial_state_id': base_ids[row_idx], 'path': str(winning_path or '')})
        wins[str(winning_label)] += 1
        lineage.append({
            'row_index': row_idx,
            'initial_state_id': base_ids[row_idx],
            'winner': winning_label,
            'path_len': int(winning_len or 0),
        })
    summary = {
        'wins': wins,
        'num_candidates': len(candidate_rows),
        'rows': len(merged),
    }
    return merged, {'lineage': lineage, 'summary': summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Megaminx best-of-lanes portfolio orchestrator')
    parser.add_argument('--candidate', action='append', default=[], help='label=/path/to/submission.csv or just /path/to/submission.csv')
    parser.add_argument('--external-manifest', action='append', default=[], help='JSON manifest for an external candidate generator')
    parser.add_argument('--fallback-submission', default=str(_HERE / 'submissions' / 'optimized_submission.csv'))
    parser.add_argument('--external-out-dir', default=str(_HERE / 'submissions' / 'external_adapter_lane'))
    parser.add_argument('--splits', default='', help='Optional shadow_splits.json for candidate-side exact score evaluation')
    parser.add_argument('--candidate-eval-out', default=str(_HERE / 'submissions' / 'submission_portfolio_orchestrated.candidate_eval.json'))
    parser.add_argument('--test-csv', default=str(_HERE / 'data' / 'test.csv'))
    parser.add_argument('--out', default=str(_HERE / 'submissions' / 'submission_portfolio_orchestrated.csv'))
    parser.add_argument('--lineage-out', default=str(_HERE / 'submissions' / 'submission_portfolio_orchestrated.lineage.json'))
    parser.add_argument('--summary-out', default=str(_HERE / 'submissions' / 'submission_portfolio_orchestrated.summary.json'))
    parser.add_argument('--scoreboard-out', default=str(_HERE / 'submissions' / 'submission_portfolio_orchestrated.scoreboard.json'))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    candidate_specs = list(args.candidate or [])
    external_summaries = []
    if args.external_manifest:
        generated_specs, external_summaries = materialize_external_candidates(
            args.external_manifest,
            test_csv=Path(args.test_csv),
            fallback_submission=Path(args.fallback_submission) if str(args.fallback_submission).strip() else None,
            out_dir=Path(args.external_out_dir),
        )
        candidate_specs.extend(generated_specs)
    if not candidate_specs:
        raise ValueError('No candidate lanes were provided')
    candidate_eval = None
    if str(args.splits).strip():
        candidate_eval = evaluate_candidates(candidate_specs, load_splits(Path(args.splits)))
        Path(args.candidate_eval_out).write_text(json.dumps(candidate_eval, ensure_ascii=False, indent=2), encoding='utf-8')
    merged, lineage = merge_candidates(candidate_specs)
    write_rows(Path(args.out), merged)
    test_rows = load_rows(Path(args.test_csv)) if str(args.test_csv).strip() else None
    scoreboard = build_row_scoreboard(submission_rows=merged, test_rows=test_rows)
    summary = summarize_scoreboard(scoreboard)
    lineage['portfolio_summary'] = summary
    lineage['candidate_specs'] = candidate_specs
    lineage['external_adapter_summaries'] = external_summaries
    if candidate_eval is not None:
        lineage['candidate_eval'] = candidate_eval
    Path(args.lineage_out).write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(args.scoreboard_out).write_text(json.dumps(scoreboard, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'score': summary['score'], 'rows': summary['rows'], 'candidates': len(candidate_specs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
