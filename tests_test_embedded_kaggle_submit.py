import pipeline_cli


def test_rewrite_embedded_kaggle_submit_tail():
    argv = [
        'run',
        '--competition', 'cayley-py-megaminx',
        '--output', 'submissions/submission.csv',
        '--print-generation-max-chars', '4000',
        'kaggle', 'competitions', 'submit', 'cayley-py-megaminx',
        '-f', 'submissions/submission.csv',
        '-m', 'cayley-py-megaminx submission from pipeline_cli',
    ]
    rewritten, note = pipeline_cli._rewrite_embedded_kaggle_submit(argv)
    assert note is not None
    assert '--submit' in rewritten
    assert '--submit-via' in rewritten
    assert rewritten[rewritten.index('--submit-via') + 1] == 'cli'
    assert rewritten[rewritten.index('--submit-competition') + 1] == 'cayley-py-megaminx'
    assert rewritten[rewritten.index('--message') + 1] == 'cayley-py-megaminx submission from pipeline_cli'
    assert 'kaggle' not in rewritten


def test_split_joined_kaggle_token_after_line_continuation():
    argv = [
        'run',
        '--competition', 'cayley-py-megaminx',
        '--output', 'submissions/submission.csv',
        '--print-generation-max-chars', '4000kaggle',
        'competitions', 'submit', 'cayley-py-megaminx',
        '-f', 'submissions/submission.csv',
        '-m', 'hello',
    ]
    normalized = pipeline_cli._split_accidental_joined_kaggle_token(argv)
    assert normalized[normalized.index('--print-generation-max-chars') + 1] == '4000'
    assert 'kaggle' in normalized
    rewritten, note = pipeline_cli._rewrite_embedded_kaggle_submit(argv)
    assert note is not None
    assert rewritten[rewritten.index('--message') + 1] == 'hello'
