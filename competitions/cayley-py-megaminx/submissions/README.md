# Submissions

Recommended location for generated submissions.

Prebuilt optimized Megaminx submission:

    competitions/cayley-py-megaminx/submissions/optimized_submission.csv

Rebuild optimized assets offline:

    cd competitions/cayley-py-megaminx
    python build_optimized_assets.py

Build via the repo pipeline (offline):

    python ../../pipeline_cli.py run --competition cayley-py-megaminx --output submissions/submission.csv --no-llm

Submit to Kaggle (optional):

    python ../../pipeline_cli.py run --competition cayley-py-megaminx --output submissions/submission.csv --no-llm --submit --message optimized --kaggle-json /path/to/kaggle.json --submit-via api
