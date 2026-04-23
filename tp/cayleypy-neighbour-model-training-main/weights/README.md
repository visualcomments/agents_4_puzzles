# Megaminx neighbour-model weights

This vendored snapshot does **not** store `.pth` checkpoints in Git, to avoid Git LFS integrity failures when pushing.

Fetch the real checkpoints on demand:

```bash
python scripts/fetch_megaminx_neighbour_weights.py
```

Expected files after download:
- `weights/p900-t000_1776548012_best.pth`
- `weights/p900-t000-q_1776581286_best.pth`

If you already have a local clone that previously contained Git LFS pointers for these files, clean those pointers from Git history before pushing.
