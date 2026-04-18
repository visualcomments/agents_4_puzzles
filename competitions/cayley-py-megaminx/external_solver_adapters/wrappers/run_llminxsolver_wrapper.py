from __future__ import annotations

from pathlib import Path

from wrapper_common import build_basic_parser, finish, run_with_autodiscovery


DEFAULT_REPO = 'external_real/Sevilze_llminxsolver-cmp'


def main() -> None:
    parser = build_basic_parser('Run Sevilze/llminxsolver-cmp checkout and emit Kaggle-style submission CSV', DEFAULT_REPO)
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
                'cmd': ['cargo', 'run', '--release', '-p', 'llminxsolver-rs', '--example', 'colab_batch', '--', '{cases_jsonl}', jsonl_out],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': jsonl_out,
                'format': 'jsonl',
                'timeout_seconds': 1200,
            },
            {
                'cmd': ['python3', 'export_candidates.py', '--test-csv', '{test_csv}', '--jsonl-out', jsonl_out],
                'cwd': '{repo}',
                'output_mode': 'file',
                'output_path': jsonl_out,
                'format': 'jsonl',
            },
        ],
        fallback_empty_ok=True,
        smoke_files=['llminxsolver-rs/Cargo.toml'],
        smoke_cmd=['cargo', 'metadata', '--format-version', '1', '--no-deps'],
    )
    finish(summary, args.summary_out)


if __name__ == '__main__':
    main()
