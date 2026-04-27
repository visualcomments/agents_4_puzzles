# Git push / Git LFS fix

This archive removes vendored `.pth` checkpoint pointer files from the repository payload.

## Fresh start

If you initialize a **new** Git repository from this archive, pushes should not hit the previous Git LFS integrity error for the Megaminx neighbour-model checkpoints.

## If your existing local clone still fails to push

Your local Git history may still contain old Git LFS pointers. Clean them from history and remove the LFS tracking rule.

Recommended outline:

1. Remove the LFS tracking rule for `tp/cayleypy-neighbour-model-training-main/weights/*.pth`.
2. Remove the old checkpoint paths from Git history using `git filter-repo`.
3. Force-push the rewritten branch.

Reference documentation:
- GitHub Docs: Resolving Git Large File Storage upload failures
- GitHub Docs: Removing files from Git Large File Storage
