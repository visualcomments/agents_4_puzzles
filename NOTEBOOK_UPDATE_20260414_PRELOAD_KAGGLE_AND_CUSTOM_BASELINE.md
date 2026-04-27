# Notebook update: preload Kaggle credentials and custom baseline

This archive updates the Megaminx Colab runners so that:
- `kaggle.json` can be uploaded before the main run
- a custom baseline `solve.py` can be uploaded and passed via `--baseline`
- live logging to file with tail refresh remains enabled
