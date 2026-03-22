# Submissions

Recommended location for generated submissions.

Prebuilt optimized Megaminx submission:

    competitions/cayley-py-megaminx/submissions/optimized_submission.csv

Current bundled optimized score:

    415075

Rebuild optimized assets offline:

    cd competitions/cayley-py-megaminx
    python build_optimized_assets.py

Build via the repo pipeline (offline):

    python ../../pipeline_cli.py run --competition cayley-py-megaminx --output submissions/submission.csv --no-llm

Submit to Kaggle (optional):

    python ../../pipeline_cli.py run --competition cayley-py-megaminx --output submissions/submission.csv --no-llm --submit --message optimized --kaggle-json /path/to/kaggle.json --submit-via api

Generate solver with prompt variants:

    python ../../pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx_regular.py --prompt-variant regular
    python ../../pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx_improved.py --prompt-variant improved
    python ../../pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx_dataset_adapted.py --prompt-variant dataset_adapted
    python ../../pipeline_cli.py generate-solver --competition cayley-py-megaminx --out generated/solve_megaminx_heuristic_boosted.py --prompt-variant heuristic_boosted
