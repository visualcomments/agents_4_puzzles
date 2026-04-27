from __future__ import annotations

import heapq
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import sys
import importlib.util

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _path in [
    _REPO_ROOT / 'third_party' / 'cayleypy-main' / 'cayleypy',
    _REPO_ROOT / 'third_party' / 'cayleypy-main',
    _HERE,
]:
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load_local_module(name: str, filename: str):
    module_path = _HERE / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {filename} from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


sm = _load_local_module('megaminx_adapter_solve_module', 'solve_module.py')

try:  # optional dependency
    import cayleypy  # type: ignore
    from cayleypy.algo import BeamSearchAlgorithm  # type: ignore
except Exception:  # pragma: no cover - optional dependency absent in many envs
    cayleypy = None
    BeamSearchAlgorithm = None


@dataclass
class SearchOutcome:
    path: list[str] | None
    profile: dict[str, Any]
    backend: str


class MegaminxSearchAdapter:
    """Hybrid adapter with cayleypy-first search and deterministic fallback.

    The adapter exposes one stable API used by `search_improver_v3` and other runners.
    When cayleypy is available it prefers:
    1) exact short-path recovery via a BFS neighborhood around the target state;
    2) beam search on a CayleyGraph whose central state is the requested target;
    3) internal lightweight beam/MITM fallback.
    """

    def __init__(self, central: Sequence[int], generators: Dict[str, List[int]], *, prefer_cayleypy: bool = True) -> None:
        self.central = list(int(x) for x in central)
        self.generators = {str(k): [int(v) for v in vals] for k, vals in generators.items()}
        self.inverse = sm.inverse_move_map(self.generators)
        self.move_names = list(self.generators)
        self.forward_faces = sm.forward_faces(self.generators)
        self.perms_bytes = {name: bytes(vals) for name, vals in self.generators.items()}
        self.prefer_cayleypy = prefer_cayleypy and cayleypy is not None and BeamSearchAlgorithm is not None
        self._graph = None
        self._beam = None
        self._central_key = sm.state_to_bytes(self.central)
        self._max_cached_target_graphs = max(0, int(__import__('os').environ.get('MEGAMINX_MAX_CACHED_TARGET_GRAPHS', '1')))
        self._max_cached_bfs_entries = max(0, int(__import__('os').environ.get('MEGAMINX_MAX_CACHED_BFS_ENTRIES', '2')))
        self._graph_cache: 'OrderedDict[bytes, Any]' = OrderedDict()
        self._beam_cache: 'OrderedDict[bytes, Any]' = OrderedDict()
        self._bfs_cache: 'OrderedDict[tuple[bytes, int], Any]' = OrderedDict()




    def _is_central_key(self, key: bytes) -> bool:
        return key == self._central_key

    def _trim_graph_caches(self) -> None:
        custom_keys = [key for key in self._graph_cache.keys() if not self._is_central_key(key)]
        while len(custom_keys) > self._max_cached_target_graphs:
            drop_key = custom_keys.pop(0)
            self._graph_cache.pop(drop_key, None)
            self._beam_cache.pop(drop_key, None)
            stale_bfs = [bk for bk in self._bfs_cache.keys() if bk[0] == drop_key]
            for bfs_key in stale_bfs:
                self._bfs_cache.pop(bfs_key, None)

    def _trim_bfs_cache(self) -> None:
        custom_keys = [key for key in self._bfs_cache.keys() if not self._is_central_key(key[0])]
        while len(custom_keys) > self._max_cached_bfs_entries:
            drop_key = custom_keys.pop(0)
            self._bfs_cache.pop(drop_key, None)

    def clear_caches(self, *, keep_central: bool = True) -> None:
        if not keep_central:
            self._graph_cache.clear()
            self._beam_cache.clear()
            self._bfs_cache.clear()
            self._graph = None
            self._beam = None
            return
        central_graph = self._graph_cache.get(self._central_key)
        central_beam = self._beam_cache.get(self._central_key)
        central_bfs = {key: value for key, value in self._bfs_cache.items() if key[0] == self._central_key}
        self._graph_cache.clear()
        self._beam_cache.clear()
        self._bfs_cache.clear()
        if central_graph is not None:
            self._graph_cache[self._central_key] = central_graph
        if central_beam is not None:
            self._beam_cache[self._central_key] = central_beam
        for key, value in central_bfs.items():
            self._bfs_cache[key] = value
        self._graph = central_graph
        self._beam = central_beam

    def backend_name(self) -> str:
        return 'cayleypy' if self.prefer_cayleypy else 'internal'

    def _apply_bytes(self, state: bytes, move: str) -> bytes:
        perm = self.perms_bytes[move]
        return bytes(state[j] for j in perm)

    def _hamming(self, state: bytes, target: bytes) -> int:
        return sum(1 for a, b in zip(state, target) if a != b)

    def _target_neighborhood(self, target_state: Sequence[int] | bytes, *, depth: int, profile: dict[str, Any]) -> dict[bytes, tuple[str, ...]]:
        target = target_state if isinstance(target_state, bytes) else sm.state_to_bytes(target_state)
        seen: dict[bytes, tuple[str, ...]] = {target: ()}
        frontier: deque[tuple[bytes, tuple[str, ...]]] = deque([(target, ())])
        nodes = 1
        while frontier:
            state, suffix = frontier.popleft()
            if len(suffix) >= depth:
                continue
            for move in self.move_names:
                predecessor = self._apply_bytes(state, self.inverse[move])
                if predecessor in seen:
                    continue
                new_suffix = (move,) + suffix
                seen[predecessor] = new_suffix
                frontier.append((predecessor, new_suffix))
                nodes += 1
        profile['mitm_nodes'] = nodes
        profile['mitm_depth'] = depth
        return seen

    def _build_graph(self, target_state: Sequence[int]) -> Any:
        if not self.prefer_cayleypy:
            return None
        target_key = sm.state_to_bytes(target_state)
        graph = self._graph_cache.get(target_key)
        if graph is not None:
            self._graph_cache.move_to_end(target_key)
            if target_key in self._beam_cache:
                self._beam_cache.move_to_end(target_key)
            return graph
        try:  # pragma: no cover - depends on optional dependency
            definition = cayleypy.CayleyGraphDef.create(
                generators=[self.generators[name] for name in self.move_names],
                generator_names=self.move_names,
                central_state=list(int(x) for x in target_state),
                name='megaminx_custom_v4',
            )
            graph = cayleypy.CayleyGraph(definition)
            beam = BeamSearchAlgorithm(graph)
            self._graph_cache[target_key] = graph
            self._beam_cache[target_key] = beam
            self._trim_graph_caches()
            return graph
        except Exception:
            self.prefer_cayleypy = False
            self.clear_caches(keep_central=False)
            return None

    def _ensure_cayleypy(self) -> None:
        if not self.prefer_cayleypy:
            return
        if self._beam is not None and self._graph is not None:
            return
        graph = self._build_graph(self.central)
        if graph is None:
            return
        self._graph = graph
        self._beam = self._beam_cache.get(sm.state_to_bytes(self.central))

    def _get_beam(self, target_state: Sequence[int]) -> tuple[Any | None, Any | None]:
        graph = self._build_graph(target_state)
        if graph is None:
            return None, None
        key = sm.state_to_bytes(target_state)
        return graph, self._beam_cache.get(key)

    def _target_bfs(self, target_state: Sequence[int], depth: int) -> Any | None:
        if not self.prefer_cayleypy or depth <= 0:
            return None
        key = (sm.state_to_bytes(target_state), int(depth))
        if key in self._bfs_cache:
            self._bfs_cache.move_to_end(key)
            return self._bfs_cache[key]
        graph = self._build_graph(target_state)
        if graph is None:
            return None
        try:  # pragma: no cover - depends on optional dependency
            bfs_result = graph.bfs(
                max_layer_size_to_store=0,
                max_layer_size_to_explore=10**7,
                max_diameter=int(depth),
                return_all_hashes=True,
            )
            self._bfs_cache[key] = bfs_result
            self._trim_bfs_cache()
            return bfs_result
        except Exception:
            return None

    def _exact_path_cayleypy(
        self,
        *,
        start_state: Sequence[int],
        target_state: Sequence[int],
        mitm_depth: int,
        max_total_path_len: int | None,
        profile: dict[str, Any],
    ) -> list[str] | None:
        if not self.prefer_cayleypy or mitm_depth <= 0:
            return None
        graph = self._build_graph(target_state)
        bfs_result = self._target_bfs(target_state, mitm_depth)
        if graph is None or bfs_result is None:
            return None
        try:  # pragma: no cover - depends on optional dependency
            path_ids = graph.find_path_from(list(int(x) for x in start_state), bfs_result)
            if path_ids is None:
                profile['cayleypy_exact_hit'] = False
                return None
            path = [self.move_names[int(idx)] for idx in list(path_ids)]
            if max_total_path_len is not None and len(path) > max_total_path_len:
                profile['cayleypy_exact_hit'] = False
                profile['cayleypy_exact_len'] = len(path)
                return None
            profile['cayleypy_exact_hit'] = True
            profile['cayleypy_exact_len'] = len(path)
            return path
        except Exception as exc:
            profile['cayleypy_exact_error'] = f'{type(exc).__name__}: {exc}'
            return None

    def _search_cayleypy(
        self,
        *,
        start_state: Sequence[int],
        target_state: Sequence[int] | None,
        beam_mode: str,
        beam_width: int,
        max_steps: int,
        history_depth: int,
        mitm_depth: int,
        return_path: bool,
        verbose: int,
        max_total_path_len: int | None,
        profile: dict[str, Any],
    ) -> SearchOutcome | None:
        self._ensure_cayleypy()
        if not self.prefer_cayleypy:
            return None

        dest = self.central if target_state is None else list(int(x) for x in target_state)
        exact_path = self._exact_path_cayleypy(
            start_state=start_state,
            target_state=dest,
            mitm_depth=mitm_depth,
            max_total_path_len=max_total_path_len,
            profile=profile,
        )
        if exact_path is not None:
            profile.update({
                'cayleypy_path_found': True,
                'cayleypy_path_length': len(exact_path),
                'cayleypy_method': 'exact_bfs_mitm',
            })
            return SearchOutcome(path=exact_path, profile=profile, backend='cayleypy')

        graph, beam = self._get_beam(dest)
        if graph is None or beam is None:
            return None
        try:  # pragma: no cover - optional backend
            kwargs: dict[str, Any] = {
                'start_state': list(int(x) for x in start_state),
                'beam_width': beam_width,
                'max_steps': max_steps,
                'return_path': return_path,
                'verbose': verbose,
            }
            bfs_result_for_mitm = self._target_bfs(dest, mitm_depth)
            if target_state is None:
                kwargs['beam_mode'] = beam_mode
                kwargs['history_depth'] = history_depth
                if bfs_result_for_mitm is not None:
                    kwargs['bfs_result_for_mitm'] = bfs_result_for_mitm
            else:
                # For a custom target, we search on a graph where that target is the central state.
                kwargs['beam_mode'] = 'simple'
                if bfs_result_for_mitm is not None:
                    kwargs['bfs_result_for_mitm'] = bfs_result_for_mitm
                profile['requested_beam_mode'] = beam_mode
                profile['effective_beam_mode'] = 'simple(target-centered)'
            result = beam.search(**kwargs)
            path = None
            if getattr(result, 'path_found', False) and getattr(result, 'path', None) is not None:
                path = [self.move_names[int(idx)] for idx in list(result.path)]
                if max_total_path_len is not None and len(path) > max_total_path_len:
                    path = None
            profile.update({
                'cayleypy_path_found': bool(path),
                'cayleypy_path_length': (len(path) if path is not None else None),
                'cayleypy_method': 'beam_search',
            })
            return SearchOutcome(path=path, profile=profile, backend='cayleypy')
        except Exception as exc:
            profile['cayleypy_error'] = f'{type(exc).__name__}: {exc}'
            self.prefer_cayleypy = False
            self._beam = None
            self._graph = None
            self.clear_caches(keep_central=False)
            return None

    def search(
        self,
        *,
        start_state: Sequence[int],
        target_state: Sequence[int] | None = None,
        beam_mode: str = 'simple',
        beam_width: int = 128,
        max_steps: int = 12,
        history_depth: int = 0,
        mitm_depth: int = 3,
        time_budget_s: float | None = None,
        verbose: int = 0,
        max_total_path_len: int | None = None,
    ) -> SearchOutcome:
        profile: dict[str, Any] = {
            'beam_mode': beam_mode,
            'beam_width': beam_width,
            'max_steps': max_steps,
            'history_depth': history_depth,
            'max_total_path_len': max_total_path_len,
            'target_kind': 'center' if target_state is None else 'custom',
        }
        cayley_result = self._search_cayleypy(
            start_state=start_state,
            target_state=target_state,
            beam_mode=beam_mode,
            beam_width=beam_width,
            max_steps=max_steps,
            history_depth=history_depth,
            mitm_depth=mitm_depth,
            return_path=True,
            verbose=verbose,
            max_total_path_len=max_total_path_len,
            profile=dict(profile),
        )
        if cayley_result is not None:
            return cayley_result
        return self._search_internal(
            start_state=start_state,
            target_state=self.central if target_state is None else list(target_state),
            beam_width=beam_width,
            max_steps=max_steps,
            history_depth=history_depth,
            mitm_depth=mitm_depth,
            time_budget_s=time_budget_s,
            beam_mode=beam_mode,
            max_total_path_len=max_total_path_len,
            profile=profile,
        )

    def _search_internal(
        self,
        *,
        start_state: Sequence[int],
        target_state: Sequence[int],
        beam_width: int,
        max_steps: int,
        history_depth: int,
        mitm_depth: int,
        time_budget_s: float | None,
        beam_mode: str,
        max_total_path_len: int | None,
        profile: dict[str, Any],
    ) -> SearchOutcome:
        t0 = time.perf_counter()
        start_b = sm.state_to_bytes(start_state)
        target_b = sm.state_to_bytes(target_state)
        neighborhood = self._target_neighborhood(target_b, depth=max(0, mitm_depth), profile=profile)
        if start_b in neighborhood:
            suffix = list(neighborhood[start_b])
            return SearchOutcome(path=suffix, profile={**profile, 'beam_layers': 0, 'beam_hits': 1}, backend='internal')

        beam: list[tuple[bytes, tuple[str, ...], str | None]] = [(start_b, (), None)]
        history: deque[set[bytes]] = deque(maxlen=max(0, history_depth))
        best_path: list[str] | None = None
        expansions = 0
        candidates = 0
        hits = 0
        layers = 0

        for depth in range(1, max_steps + 1):
            if time_budget_s is not None and (time.perf_counter() - t0) >= time_budget_s:
                profile['timed_out'] = True
                break
            frontier_scores: list[tuple[int, int, bytes, tuple[str, ...], str | None]] = []
            seen_next: dict[bytes, tuple[str, ...]] = {}
            banned: set[bytes] = set().union(*history) if history else set()
            local_counter = 0
            for state, path, last_move in beam:
                for move in self.move_names:
                    if last_move is not None and move == self.inverse[last_move]:
                        continue
                    nxt = self._apply_bytes(state, move)
                    expansions += 1
                    if nxt in banned:
                        continue
                    candidate_path = path + (move,)
                    suffix = neighborhood.get(nxt)
                    if suffix is not None:
                        full_path = list(candidate_path + suffix)
                        if max_total_path_len is None or len(full_path) <= max_total_path_len:
                            hits += 1
                            if best_path is None or len(full_path) < len(best_path):
                                best_path = full_path
                                if len(full_path) <= depth + mitm_depth:
                                    profile['exact_or_mitm_hit'] = True
                    if max_total_path_len is not None and len(candidate_path) >= max_total_path_len:
                        continue
                    if nxt in seen_next and len(candidate_path) >= len(seen_next[nxt]):
                        continue
                    seen_next[nxt] = candidate_path
                    score = self._hamming(nxt, target_b)
                    if beam_mode == 'advanced' and last_move is not None:
                        score -= 1
                    frontier_scores.append((score, local_counter, nxt, candidate_path, move))
                    local_counter += 1
                    candidates += 1
            if best_path is not None and max_total_path_len is not None and len(best_path) <= max(1, depth):
                break
            if not frontier_scores:
                break
            beam = []
            for _score, _idx, nxt, path, move in heapq.nsmallest(beam_width, frontier_scores):
                beam.append((nxt, path, move))
            if history_depth > 0:
                history.append({state for state, _path, _last in beam})
            layers = depth

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        profile.update({
            'beam_layers': layers,
            'beam_candidates': candidates,
            'beam_expanded_states': expansions,
            'beam_hits': hits,
            'elapsed_ms_search': round(elapsed_ms, 3),
        })
        return SearchOutcome(path=best_path, profile=profile, backend='internal')
