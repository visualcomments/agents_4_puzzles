from __future__ import annotations

from pathlib import Path

from wrapper_common import build_basic_parser, finish, run_with_autodiscovery


DEFAULT_REPO = 'external/Megaminx-Simulator-AStar-main'


def main() -> None:
    parser = build_basic_parser('Run Megaminx-Simulator-AStar checkout and emit Kaggle-style submission CSV', DEFAULT_REPO)
    args = parser.parse_args()
    repo = Path(args.repo)
    summary = run_with_autodiscovery(
        repo=repo,
        output_csv=Path(args.out),
        test_csv=Path(args.test_csv),
        label='megaminx_simulator_astar_wrapper',
        existing_outputs=[
            ('out/megaminx_candidates.csv', 'submission_csv'),
            ('astar_out/megaminx_candidates.csv', 'submission_csv'),
            ('submission.csv', 'submission_csv'),
            ('solutions.jsonl', 'jsonl'),
        ],
        command_specs=[
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
                'cmd': ['python3', 'astar.py', '--test-csv', '{test_csv}', '--out', '{output_csv}'],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': '{output_csv}',
                'format': 'submission_csv',
            },
        ],
    )
    finish(summary, args.summary_out)


if __name__ == '__main__':
    main()
