from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = ROOT / 'tp' / 'cayleypy-neighbour-model-training-main'

FILES = [
    (
        'weights/p900-t000_1776548012_best.pth',
        'https://github.com/AnanasClassic/cayleypy-neighbour-model-training/raw/refs/heads/main/weights/p900-t000_1776548012_best.pth',
    ),
    (
        'weights/p900-t000-q_1776581286_best.pth',
        'https://github.com/AnanasClassic/cayleypy-neighbour-model-training/raw/refs/heads/main/weights/p900-t000-q_1776581286_best.pth',
    ),
]


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        head = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return False
    return head.startswith('version https://git-lfs.github.com/spec/v1')


def _try_git_lfs_pull(repo_dir: Path) -> bool:
    git = shutil.which('git')
    if git is None:
        return False
    try:
        subprocess.run([git, 'lfs', 'version'], cwd=repo_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return False
    try:
        subprocess.run([git, 'lfs', 'pull'], cwd=repo_dir, check=True)
        return True
    except Exception:
        return False


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dst.open('wb') as handle:
        shutil.copyfileobj(response, handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Fetch real Megaminx neighbour-model weights into the vendored training snapshot')
    parser.add_argument('--repo-dir', type=Path, default=DEFAULT_REPO)
    args = parser.parse_args(argv)
    repo_dir = args.repo_dir.resolve()
    if not repo_dir.exists():
        raise SystemExit(f'Repo dir not found: {repo_dir}')

    if _try_git_lfs_pull(repo_dir):
        print('git lfs pull completed')

    missing = []
    for rel, url in FILES:
        path = repo_dir / rel
        if not path.exists() or _is_lfs_pointer(path):
            print(f'fetching {rel} from GitHub raw...')
            _download(url, path)
        if not path.exists() or _is_lfs_pointer(path):
            missing.append(rel)
        else:
            print(f'ok: {rel} ({path.stat().st_size} bytes)')

    if missing:
        print('still unresolved:', ', '.join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
