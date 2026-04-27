from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

for _p in (_HERE, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_local_module(name: str, filename: str):
    module_path = _HERE / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {filename} from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


sm = _load_local_module('megaminx_neighbour_model_lane_solve_module', 'solve_module.py')


DEFAULT_MODEL_ID = 1776581286
DEFAULT_REPO_REL = Path('tp') / 'cayleypy-neighbour-model-training-main'


@dataclass
class NeighbourModelConfig:
    repo_dir: Path | None = None
    model_id: int = DEFAULT_MODEL_ID
    use_best: bool = True
    device: str = 'cpu'
    eval_batch_size: int = 2048
    beam_width: int = 128
    num_steps: int = 18
    num_attempts: int = 4
    max_rows: int = 32
    only_if_shorter: bool = True
    sort_rows_by_length: bool = True


class NeighbourModelRuntime:
    def __init__(
        self,
        config: NeighbourModelConfig,
        *,
        official_generators: Mapping[str, Sequence[int]] | None = None,
        official_central: Sequence[int] | None = None,
    ) -> None:
        self.config = config
        self.official_generators = {str(k): list(int(x) for x in v) for k, v in dict(official_generators or {}).items()} or None
        self.official_central = list(int(x) for x in official_central) if official_central is not None else None
        self.repo_dir: Path | None = None
        self.available = False
        self.error: str | None = None
        self._loaded = False
        self._torch = None
        self.device = None
        self.info: dict[str, Any] = {}
        self.model = None
        self.searcher = None
        self.training_move_names: list[str] = []
        self.official_move_names: list[str] = []
        self._training_to_official: dict[str, str] = {}
        self._official_to_training: dict[str, str] = {}

    @staticmethod
    def official_to_training(move: str) -> str:
        move = str(move)
        return f"{move[1:]}'" if move.startswith('-') else move

    @staticmethod
    def training_to_official(move: str) -> str:
        move = str(move)
        return '-' + move[:-1] if move.endswith("'") else move

    def _candidate_repo_dirs(self) -> list[Path]:
        env_dir = os.environ.get('MEGAMINX_NEIGHBOUR_MODEL_REPO')
        raw_candidates = [
            Path(env_dir).expanduser() if env_dir else None,
            self.config.repo_dir,
            _REPO_ROOT / DEFAULT_REPO_REL,
            _HERE / DEFAULT_REPO_REL,
            _HERE / 'external_real' / 'cayleypy-neighbour-model-training-main',
        ]
        out: list[Path] = []
        seen: set[Path] = set()
        for item in raw_candidates:
            if item is None:
                continue
            resolved = Path(item).expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)
        return out

    def _find_repo_dir(self) -> Path:
        for path in self._candidate_repo_dirs():
            if (path / 'pilgrim' / '__init__.py').exists() and (path / 'weights').exists() and (path / 'logs').exists():
                return path
        searched = '\n'.join(str(p) for p in self._candidate_repo_dirs())
        raise FileNotFoundError(f'Could not locate cayleypy-neighbour-model-training repo. Searched:\n{searched}')

    def _import_training_modules(self):
        assert self.repo_dir is not None
        repo_text = str(self.repo_dir)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        pilgrim = importlib.import_module('pilgrim')
        return {
            'pilgrim': pilgrim,
            'build_model_from_info': getattr(pilgrim, 'build_model_from_info'),
            'generate_inverse_moves': getattr(pilgrim, 'generate_inverse_moves'),
            'parse_generator_spec': getattr(pilgrim, 'parse_generator_spec'),
            'QSearcher': getattr(pilgrim, 'QSearcher'),
            'Searcher': getattr(pilgrim, 'Searcher'),
        }

    @staticmethod
    def _looks_like_git_lfs_pointer(path: Path) -> bool:
        try:
            if not path.exists() or path.stat().st_size > 1024:
                return False
            head = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return False
        return head.startswith('version https://git-lfs.github.com/spec/v1')

    def _verify_assets(self, *, move_names: Sequence[str], moves: Sequence[Sequence[int]], target_state: Sequence[int]) -> None:
        training_official = [self.training_to_official(name) for name in move_names]
        if self.official_generators is not None:
            missing = [name for name in training_official if name not in self.official_generators]
            if missing:
                raise KeyError(f'Official generator set is missing converted training moves: {missing}')
            for train_name, official_name, perm in zip(move_names, training_official, moves):
                if list(int(x) for x in perm) != list(int(x) for x in self.official_generators[official_name]):
                    raise ValueError(f'Generator mismatch between neighbour-model repo and competition bundle for move {train_name}->{official_name}')
        if self.official_central is not None and list(int(x) for x in target_state) != self.official_central:
            raise ValueError('Neighbour-model target state does not match the Megaminx competition central_state')

    def load(self) -> bool:
        if self._loaded:
            return self.available
        self._loaded = True
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            self.error = f'torch_import_failed:{type(exc).__name__}:{exc}'
            self.available = False
            return False

        try:
            self.repo_dir = self._find_repo_dir()
            modules = self._import_training_modules()
            self._torch = torch
            self.device = torch.device(self.config.device)

            info_path = self.repo_dir / 'logs' / f'model_p900-t000-q_{int(self.config.model_id)}.json'
            if not info_path.exists():
                info_path = self.repo_dir / 'logs' / f'model_p900-t000_{int(self.config.model_id)}.json'
            if not info_path.exists():
                raise FileNotFoundError(f'Model metadata was not found for model_id={self.config.model_id}: {info_path}')
            self.info = json.loads(info_path.read_text(encoding='utf-8'))

            generator_spec = json.loads((self.repo_dir / 'generators' / 'p900.json').read_text(encoding='utf-8'))
            moves, move_names = modules['parse_generator_spec'](generator_spec)
            all_moves = torch.tensor(moves, dtype=torch.int64, device=self.device)
            target_state = torch.load(self.repo_dir / 'targets' / 'p900-t000.pt', weights_only=False, map_location=self.device)
            target_state = target_state.to(device=self.device, dtype=torch.int64).view(-1)
            state_size = int(all_moves.size(1))
            num_classes = int(torch.unique(target_state).numel())
            q_model = str(self.info.get('model_name') or '').endswith('-q') or str(self.info.get('training_mode') or '').startswith('q_')
            output_dim = int(len(move_names) if q_model else 1)
            model = modules['build_model_from_info'](
                self.info,
                num_classes=num_classes,
                state_size=state_size,
                output_dim=output_dim,
            )

            weights_rel = self.info.get('best_weights_file') if self.config.use_best else self.info.get('final_weights_file')
            if not weights_rel:
                model_name = str(self.info.get('model_name') or f'p900-t000-q')
                epoch_label = 'best' if self.config.use_best else 'final'
                weights_rel = f'weights/{model_name}_{int(self.config.model_id)}_{epoch_label}.pth'
            weights_path = self.repo_dir / str(weights_rel)
            if not weights_path.exists():
                raise FileNotFoundError(f'Weights file not found: {weights_path}')
            if self._looks_like_git_lfs_pointer(weights_path):
                helper = _REPO_ROOT / 'scripts' / 'fetch_megaminx_neighbour_weights.py'
                raise RuntimeError(
                    'Weights file is still a Git LFS pointer, not the real checkpoint binary: '
                    f'{weights_path}. Run `python {helper}` or `git lfs pull` inside the vendored repo.'
                )
            state_dict = torch.load(weights_path, weights_only=False, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            if self.device.type == 'cuda':
                model.half()
                model.dtype = torch.float16
            else:
                model.dtype = torch.float32
            if int(target_state.min().item()) < 0:
                model.z_add = -int(target_state.min().item())
            model.to(self.device)

            inverse_moves = torch.tensor(modules['generate_inverse_moves'](move_names), dtype=torch.int64, device=self.device)
            searcher_cls = modules['QSearcher'] if q_model else modules['Searcher']
            self.searcher = searcher_cls(
                model=model,
                all_moves=all_moves,
                V0=target_state,
                device=self.device,
                verbose=0,
                move_names=move_names,
                inverse_moves=inverse_moves,
                normalize_path=True,
                batch_size=max(1, int(self.config.eval_batch_size)),
                hash_seed=42,
            )
            self.model = model
            self.training_move_names = [str(x) for x in move_names]
            self.official_move_names = [self.training_to_official(name) for name in self.training_move_names]
            self._training_to_official = {src: self.training_to_official(src) for src in self.training_move_names}
            self._official_to_training = {dst: src for src, dst in self._training_to_official.items()}
            self._verify_assets(move_names=self.training_move_names, moves=moves, target_state=target_state.tolist())
            self.available = True
            self.error = None
            return True
        except Exception as exc:  # pragma: no cover - loader guarded at runtime
            self.available = False
            self.error = f'{type(exc).__name__}: {exc}'
            return False

    def solve_state(self, state: Sequence[int]) -> tuple[list[str] | None, dict[str, Any]]:
        if not self.load():
            return None, {'available': False, 'error': self.error}
        assert self._torch is not None
        assert self.searcher is not None
        state_tensor = self._torch.tensor(list(int(x) for x in state), dtype=self._torch.int64, device=self.device)
        t0 = time.perf_counter()
        try:
            moves_idx, attempts = self.searcher.get_solution(
                state_tensor,
                B=max(1, int(self.config.beam_width)),
                num_steps=max(1, int(self.config.num_steps)),
                num_attempts=max(1, int(self.config.num_attempts)),
            )
        except Exception as exc:  # pragma: no cover - runtime optional
            return None, {
                'available': True,
                'error': f'{type(exc).__name__}: {exc}',
                'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 3),
            }
        if moves_idx is None:
            return None, {
                'available': True,
                'attempts': int(attempts) + 1 if attempts is not None else None,
                'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 3),
                'solved': False,
            }
        raw_indices = [int(x) for x in moves_idx.tolist()]
        training_moves = [self.training_move_names[idx] for idx in raw_indices]
        official_moves = [self._training_to_official[move] for move in training_moves]
        return official_moves, {
            'available': True,
            'attempts': int(attempts) + 1 if attempts is not None else None,
            'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 3),
            'solved': True,
            'training_length': len(training_moves),
            'official_length': len(official_moves),
        }


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['initial_state_id', 'path'])
        writer.writeheader()
        writer.writerows(rows)


def _row_len(row: Mapping[str, str]) -> int:
    path = str(row.get('path') or '').strip()
    return 0 if not path else len(sm.path_to_moves(path))


def improve_submission_rows(
    *,
    submission_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
    central: Sequence[int],
    generators: Mapping[str, Sequence[int]],
    config: NeighbourModelConfig,
) -> tuple[list[dict[str, str]], dict[str, Any], list[dict[str, Any]]]:
    rows = [dict(row) for row in submission_rows]
    improved_rows = [dict(row) for row in rows]
    ranked = list(range(min(len(rows), len(test_rows))))
    if config.sort_rows_by_length:
        ranked.sort(key=lambda idx: _row_len(rows[idx]), reverse=True)
    limit = len(ranked) if int(config.max_rows) <= 0 else min(len(ranked), int(config.max_rows))
    ranked = ranked[:limit]

    runtime = NeighbourModelRuntime(
        config,
        official_generators=generators,
        official_central=central,
    )
    runtime_ready = runtime.load()
    profiles: list[dict[str, Any]] = []
    total_saved = 0
    solved_rows = 0
    improved_count = 0
    attempted = 0
    started = time.perf_counter()

    for row_rank, row_idx in enumerate(ranked):
        test_row = test_rows[row_idx]
        start_state = [int(x) for x in str(test_row.get('initial_state') or '').split(',') if x != '']
        baseline_moves = sm.path_to_moves(rows[row_idx].get('path'))
        profile: dict[str, Any] = {
            'row_index': int(row_idx),
            'row_rank': int(row_rank),
            'initial_state_id': str(test_row.get('initial_state_id') or ''),
            'baseline_len': len(baseline_moves),
            'runtime_available': runtime_ready,
        }
        attempted += 1
        if not runtime_ready:
            profile['error'] = runtime.error
            profiles.append(profile)
            continue

        candidate_moves, solve_meta = runtime.solve_state(start_state)
        profile.update(solve_meta)
        if not candidate_moves:
            profiles.append(profile)
            continue

        solved_rows += 1
        optimized = sm.optimize_moves(candidate_moves, generators)
        profile['candidate_len'] = len(candidate_moves)
        profile['optimized_len'] = len(optimized)
        if not sm.validate_solution(start_state, optimized, central, generators):
            profile['valid'] = False
            profile['error'] = 'invalid_candidate_after_rewrite'
            profiles.append(profile)
            continue
        profile['valid'] = True
        baseline_len = len(baseline_moves)
        better = len(optimized) < baseline_len if config.only_if_shorter else len(optimized) <= baseline_len
        if better:
            improved_rows[row_idx]['path'] = sm.moves_to_path(optimized)
            saved = baseline_len - len(optimized)
            total_saved += saved
            improved_count += 1
            profile['accepted'] = True
            profile['saved_moves'] = saved
        else:
            profile['accepted'] = False
            profile['saved_moves'] = 0
        profiles.append(profile)

    final_score = sum(_row_len(row) for row in improved_rows)
    baseline_score = sum(_row_len(row) for row in rows)
    stats = {
        'attempted_rows': attempted,
        'runtime_available': runtime_ready,
        'runtime_error': runtime.error,
        'solved_rows': solved_rows,
        'improved_rows': improved_count,
        'total_saved_moves': total_saved,
        'baseline_score': baseline_score,
        'final_score': final_score,
        'score_delta': baseline_score - final_score,
        'elapsed_ms_total': round((time.perf_counter() - started) * 1000.0, 3),
        'repo_dir': str(runtime.repo_dir) if runtime.repo_dir is not None else None,
        'model_id': int(config.model_id),
        'device': str(config.device),
        'beam_width': int(config.beam_width),
        'num_steps': int(config.num_steps),
        'num_attempts': int(config.num_attempts),
        'max_rows': int(config.max_rows),
    }
    return improved_rows, stats, profiles


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Megaminx neighbour-model lane for agents_4_puzzles hybrid workflows')
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--test-csv', type=Path, default=_HERE / 'data' / 'test.csv')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stats-out', type=Path, default=None)
    parser.add_argument('--profile-out', type=Path, default=None)
    parser.add_argument('--repo-dir', type=Path, default=None)
    parser.add_argument('--model-id', type=int, default=DEFAULT_MODEL_ID)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--eval-batch-size', type=int, default=2048)
    parser.add_argument('--beam-width', type=int, default=128)
    parser.add_argument('--num-steps', type=int, default=18)
    parser.add_argument('--num-attempts', type=int, default=4)
    parser.add_argument('--max-rows', type=int, default=32)
    parser.add_argument('--allow-equal', action='store_true')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    central, generators = sm.load_puzzle_bundle()
    submission_rows = _load_rows(args.submission)
    test_rows = _load_rows(args.test_csv)
    config = NeighbourModelConfig(
        repo_dir=args.repo_dir,
        model_id=int(args.model_id),
        device=str(args.device),
        eval_batch_size=int(args.eval_batch_size),
        beam_width=int(args.beam_width),
        num_steps=int(args.num_steps),
        num_attempts=int(args.num_attempts),
        max_rows=int(args.max_rows),
        only_if_shorter=not bool(args.allow_equal),
    )
    rows, stats, profiles = improve_submission_rows(
        submission_rows=submission_rows,
        test_rows=test_rows,
        central=central,
        generators=generators,
        config=config,
    )
    _write_rows(args.out, rows)
    stats_out = args.stats_out or args.out.with_suffix('.neighbour_model.stats.json')
    profile_out = args.profile_out or args.out.with_suffix('.neighbour_model.profiles.json')
    Path(stats_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(profile_out).write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'stats_out': str(stats_out), 'profile_out': str(profile_out), **stats}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
