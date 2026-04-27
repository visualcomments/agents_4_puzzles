# Archive fix summary

This corrected archive includes:
- Megaminx neighbour-model + LLM integration
- CLI flag support already added in `pipeline_cli.py`
- no vendored `.pth` checkpoint files in `tp/cayleypy-neighbour-model-training-main/weights/`
- no Git LFS rule for those checkpoint paths
- runtime fetch instructions via `scripts/fetch_megaminx_neighbour_weights.py`

Use this archive as the basis for a fresh clone/repository if your previous local Git history was poisoned by missing Git LFS pointers.
