from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from external_adapter_lane import materialize_external_candidates

_HERE = Path(__file__).resolve().parent


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def score_path(path: str | None) -> int:
    text = (path or '').strip()
    return 0 if not text else len(text.split('.'))


def parse_candidate(spec: str) -> Tuple[str, Path]:
    if '=' not in spec:
        path = Path(spec)
        return path.stem, path
    label, raw_path = spec.split('=', 1)
    return label.strip(), Path(raw_path.strip())


def load_splits(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def score_subset(rows: Sequence[Dict[str, str]], indices: Sequence[int]) -> int:
    total = 0
    for idx in indices:
        if 0 <= int(idx) < len(rows):
            total += score_path(rows[int(idx)].get('path'))
    return total


def evaluate_candidates(candidate_specs: Sequence[str], splits: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    dev_indices = [int(x) for x in splits.get('dev_indices') or []]
    holdout_indices = [int(x) for x in splits.get('holdout_indices') or []]
    train_indices = [int(x) for x in splits.get('train_indices') or []]
    for spec in candidate_specs:
        label, path = parse_candidate(spec)
        rows = load_rows(path)
        result = {
            'label': label,
            'path': str(path),
            'full_score': score_subset(rows, range(len(rows))),
            'train_score': score_subset(rows, train_indices),
            'dev_score': score_subset(rows, dev_indices),
            'holdout_score': score_subset(rows, holdout_indices),
            'rows': len(rows),
        }
        results.append(result)
    ranked = sorted(results, key=lambda item: (item['dev_score'], item['holdout_score'], item['full_score'], item['label']))
    champion = ranked[0] if ranked else None
    return {'results': ranked, 'champion': champion, 'splits': splits}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Exact-score prompt population selector for Megaminx candidate submissions')
    parser.add_argument('--candidate', action='append', default=[], help='label=/path/to/submission.csv or /path/to/submission.csv')
    parser.add_argument('--external-manifest', action='append', default=[], help='JSON manifest for an external candidate generator')
    parser.add_argument('--test-csv', default=str(_HERE / 'data' / 'test.csv'))
    parser.add_argument('--fallback-submission', default=str(_HERE / 'submissions' / 'optimized_submission.csv'))
    parser.add_argument('--external-out-dir', default=str(_HERE / 'submissions' / 'external_adapter_lane'))
    parser.add_argument('--splits', default=str(_HERE / 'shadow_splits.json'))
    parser.add_argument('--out', default=str(_HERE / 'submissions' / 'prompt_population_runner.results.json'))
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
    payload = evaluate_candidates(candidate_specs, load_splits(Path(args.splits)))
    payload['candidate_specs'] = candidate_specs
    payload['external_adapter_summaries'] = external_summaries
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload.get('champion') or {}, ensure_ascii=False))


if __name__ == '__main__':
    main()
