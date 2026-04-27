# Changes 2026-04-19 — direct git-clone Colab notebook

Added a repository-integrated Google Colab notebook:

- `agents_4_puzzles_megaminx_real_external_gitclone_runall.ipynb`

The notebook is designed for `Run all` and intentionally clones the repo directly via:

```bash
 git clone https://github.com/visualcomments/agents_4_puzzles
```

It does not depend on an embedded patch bundle. Instead, it validates that all required
Megaminx breakthrough files already exist inside the cloned repository and then runs:

```bash
python competitions/cayley-py-megaminx/turnkey_real_external_run.py
```

This makes the notebook compatible with a repo-first workflow where all Colab-required
patches live inside the repository itself.
