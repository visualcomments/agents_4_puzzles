from __future__ import annotations

import csv
import json
from pathlib import Path

from solve_module import (  # type: ignore
    _LOCAL_WINDOW,
    _OPTIMIZATION_PASSES,
    _SHORT_TABLE_DEPTH,
    _build_optimized_lookup,
    _find_comp_dir,
    _find_data_dir,
)


def main() -> None:
    comp_dir = _find_comp_dir()
    data_dir = _find_data_dir()
    puzzle = json.loads((data_dir / 'puzzle_info.json').read_text(encoding='utf-8'))
    generators = {str(k): list(v) for k, v in dict(puzzle['generators']).items()}
    test_csv = data_dir / 'test.csv'
    sample_csv = data_dir / 'sample_submission.csv'
    lookup = _build_optimized_lookup(test_csv, sample_csv, generators)

    with test_csv.open(newline='', encoding='utf-8') as tf:
        test_rows = list(csv.DictReader(tf))
    optimized_rows = []
    for row in test_rows:
        state_key = (row.get('initial_state') or '').strip()
        optimized_rows.append((row['initial_state_id'], lookup.get(state_key, '')))

    score_optimized = sum(0 if not path else len(path.split('.')) for _id, path in optimized_rows)
    with sample_csv.open(newline='', encoding='utf-8') as sf:
        sample_rows = list(csv.DictReader(sf))
    score_original = sum(0 if not row.get('path') else len(str(row['path']).split('.')) for row in sample_rows)

    optimized_submission = comp_dir / 'submissions' / 'optimized_submission.csv'
    optimized_submission.parent.mkdir(parents=True, exist_ok=True)
    with optimized_submission.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['initial_state_id', 'path'])
        writer.writerows(optimized_rows)

    (data_dir / 'optimized_lookup.json').write_text(
        json.dumps(
            {
                'score_original': score_original,
                'score_optimized': score_optimized,
                'score_delta': score_original - score_optimized,
                'num_states': len(lookup),
                'optimizer': {
                    'kind': 'fixed-depth-word-dp',
                    'short_table_depth': _SHORT_TABLE_DEPTH,
                    'local_window': _LOCAL_WINDOW,
                    'passes': _OPTIMIZATION_PASSES,
                },
                'lookup': lookup,
            },
            ensure_ascii=False,
            separators=(',', ':'),
        ),
        encoding='utf-8',
    )

    (data_dir / 'optimized_stats.json').write_text(
        json.dumps(
            {
                'score_original': score_original,
                'score_optimized': score_optimized,
                'score_delta': score_original - score_optimized,
                'num_states': len(lookup),
                'short_table_depth': _SHORT_TABLE_DEPTH,
                'local_window': _LOCAL_WINDOW,
                'passes': _OPTIMIZATION_PASSES,
                'generated_submission': str(optimized_submission.relative_to(comp_dir)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    print(
        json.dumps(
            {
                'score_original': score_original,
                'score_optimized': score_optimized,
                'score_delta': score_original - score_optimized,
                'optimized_submission': str(optimized_submission),
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
