# Git push fix for missing vendored neighbour-model weights

This repository does not commit the Megaminx checkpoint `.pth` files from the vendored
`tp/cayleypy-neighbour-model-training-main` snapshot.

The upstream training repository stores those files in Git LFS. If only the tiny LFS pointer
files are copied into this repository, GitHub rejects `git push` with a missing-object error.

Use this flow instead:

1. Keep the vendored `weights/` directory empty in Git.
2. After clone, fetch the real checkpoints only in the local runtime:

```bash
python scripts/fetch_megaminx_neighbour_weights.py
```

If you already committed the pointer files by mistake, remove them from the index and amend:

```bash
git rm --cached tp/cayleypy-neighbour-model-training-main/weights/*.pth
git add tp/cayleypy-neighbour-model-training-main/.gitattributes \
        tp/cayleypy-neighbour-model-training-main/weights/README.md \
        GIT_PUSH_LFS_FIX.md
git commit --amend --no-edit
```

If the bad commit is not the latest one, rewrite the unpushed history before pushing.
