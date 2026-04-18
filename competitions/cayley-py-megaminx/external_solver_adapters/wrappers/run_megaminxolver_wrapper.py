from __future__ import annotations

from pathlib import Path

from wrapper_common import build_basic_parser, finish, run_with_autodiscovery


DEFAULT_REPO = 'external_real/Odder_MegaminXolver'


def main() -> None:
    parser = build_basic_parser('Run Odder/MegaminXolver checkout and emit Kaggle-style submission CSV', DEFAULT_REPO)
    args = parser.parse_args()
    repo = Path(args.repo)
    summary = run_with_autodiscovery(
        repo=repo,
        output_csv=Path(args.out),
        test_csv=Path(args.test_csv),
        label='megaminxolver_wrapper',
        existing_outputs=[
            ('out/megaminx_candidates.csv', 'submission_csv'),
            ('out/submission.csv', 'submission_csv'),
            ('megaminx_candidates.csv', 'submission_csv'),
            ('submission.csv', 'submission_csv'),
            ('out/candidates.jsonl', 'jsonl'),
        ],
        command_specs=[
            {
                'cmd': ['python3', 'export_bundle_candidates.py', '--test-csv', '{test_csv}', '--out', '{output_csv}'],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': '{output_csv}',
                'format': 'submission_csv',
            },
            {
                'cmd': ['python3', 'export_candidates.py', '--test-csv', '{test_csv}', '--out', '{output_csv}'],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': '{output_csv}',
                'format': 'submission_csv',
            },
            {
                'cmd': ['python3', 'main.py', '--test-csv', '{test_csv}', '--out', '{output_csv}'],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': '{output_csv}',
                'format': 'submission_csv',
            },
            {
                'cmd': ['python3', 'solve.py', '--test-csv', '{test_csv}', '--out', '{output_csv}'],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': '{output_csv}',
                'format': 'submission_csv',
            },
        ],
        fallback_empty_ok=True,
        smoke_files=['MegaminXolver.py'],
        smoke_cmd=['python3', '-c', 'import sys; sys.path.insert(0, r"{repo}"); import MegaminXolver; print("ok")'],
    )
    finish(summary, args.summary_out)


if __name__ == '__main__':
    main()
