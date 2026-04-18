from __future__ import annotations

from pathlib import Path

from wrapper_common import build_basic_parser, finish, run_with_autodiscovery


DEFAULT_REPO = 'external/llminxsolver-master'


def main() -> None:
    parser = build_basic_parser('Run llminxsolver checkout and emit Kaggle-style submission CSV', DEFAULT_REPO)
    args = parser.parse_args()
    repo = Path(args.repo)
    jsonl_out = str(Path(args.out).with_suffix('.jsonl'))
    summary = run_with_autodiscovery(
        repo=repo,
        output_csv=Path(args.out),
        test_csv=Path(args.test_csv),
        label='llminxsolver_wrapper',
        existing_outputs=[
            ('out/llminxsolver.jsonl', 'jsonl'),
            ('llminxsolver.jsonl', 'jsonl'),
            ('out/candidates.jsonl', 'jsonl'),
            ('out/megaminx_candidates.csv', 'submission_csv'),
            ('submission.csv', 'submission_csv'),
        ],
        command_specs=[
            {
                'cmd': ['python3', 'export_candidates.py', '--test-csv', '{test_csv}', '--jsonl-out', jsonl_out],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': jsonl_out,
                'format': 'jsonl',
            },
            {
                'cmd': ['python3', 'main.py', '--test-csv', '{test_csv}', '--jsonl-out', jsonl_out],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': jsonl_out,
                'format': 'jsonl',
            },
            {
                'cmd': ['python3', 'main.py', '--test-csv', '{test_csv}', '--out', '{output_csv}'],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': '{output_csv}',
                'format': 'submission_csv',
            },
            {
                'cmd': ['python3', 'solve.py', '--test-csv', '{test_csv}', '--jsonl-out', jsonl_out],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': jsonl_out,
                'format': 'jsonl',
            },
        ],
    )
    finish(summary, args.summary_out)


if __name__ == '__main__':
    main()
