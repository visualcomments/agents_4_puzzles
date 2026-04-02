# Submissions

Recommended location for generated submissions.

Bundled best-known Megaminx submission in this repo:

    competitions/cayley-py-megaminx/submissions/optimized_submission.csv

Current bundled score in this repo:

    414305

Rebuild deterministic assets offline:

    cd competitions/cayley-py-megaminx
    python build_optimized_assets.py

Run the v3 offline post-optimizer on the longest rows:

    cd competitions/cayley-py-megaminx
    python search_improver_v3.py --submission submissions/optimized_submission.csv --out submissions/submission_search_improved_v3.csv --top-k 150

Rebuild deterministic assets and then chain v3 in one command:

    cd competitions/cayley-py-megaminx
    python build_optimized_assets.py --search-version v3 --search-top-k 150

Build via the repo pipeline (offline):

    python ../../pipeline_cli.py run --competition cayley-py-megaminx --output submissions/submission.csv --no-llm
