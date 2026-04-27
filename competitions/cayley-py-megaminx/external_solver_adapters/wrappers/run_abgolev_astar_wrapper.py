from __future__ import annotations

from pathlib import Path

from wrapper_common import build_basic_parser, finish, run_with_autodiscovery


DEFAULT_REPO = 'external_real/abgolev_A-star-Megaminx-solver'


def main() -> None:
    parser = build_basic_parser('Run abgolev/A-star-Megaminx-solver checkout and emit Kaggle-style submission CSV', DEFAULT_REPO)
    args = parser.parse_args()
    repo = Path(args.repo)
    summary = run_with_autodiscovery(
        repo=repo,
        output_csv=Path(args.out),
        test_csv=Path(args.test_csv),
        label='abgolev_astar_wrapper',
        existing_outputs=[
            ('out/megaminx_candidates.csv', 'submission_csv'),
            ('submission.csv', 'submission_csv'),
            ('solutions.jsonl', 'jsonl'),
        ],
        command_specs=[
            {
                'cmd': ['bash', '-lc', 'g++ -O3 megaminx3.cpp -o megaminx3 && ./megaminx3 > /dev/null 2>&1 || true'],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': '{output_csv}',
                'format': 'submission_csv',
                'timeout_seconds': 300,
            },
        ],
        fallback_empty_ok=True,
        smoke_files=['megaminx3.cpp'],
        smoke_cmd=['bash', '-lc', 'g++ -O3 megaminx3.cpp -o megaminx3.smoke >/dev/null 2>&1'],
    )
    finish(summary, args.summary_out)


if __name__ == '__main__':
    main()
