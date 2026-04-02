from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import solve_module as sm

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

    def _ensure_cayleypy(self) -> None:
        if not self.prefer_cayleypy:
            return
        if self._beam is not None:
            return
        # Best effort adapter: if graph creation fails, fall back to internal search.
        try:  # pragma: no cover - exercised only when cayleypy is installed
            definition = cayleypy.CayleyGraphDef(
                generators_type='permutations',
                generators=[self.generators[name] for name in self.move_names],
                central_state=self.central,
                name='megaminx_custom_v3',
            )
            self._graph = cayleypy.CayleyGraph(definition)
            self._beam = BeamSearchAlgorithm(self._graph)
        except Exception:
            self.prefer_cayleypy = False
            self._graph = None
            self._beam = None

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
        if not self.prefer_cayleypy or self._beam is None:
            return None
        try:  # pragma: no cover - optional backend
            kwargs: dict[str, Any] = {
                'start_state': list(int(x) for x in start_state),
                'beam_mode': beam_mode,
                'beam_width': beam_width,
                'max_steps': max_steps,
                'history_depth': history_depth,
                'return_path': return_path,
                'verbose': verbose,
            }
            if target_state is not None:
                kwargs['destination_state'] = list(int(x) for x in target_state)
            if mitm_depth > 0 and target_state is None and hasattr(cayleypy.algo, 'bfs_numpy'):
                # Keep this light. For huge graphs full BFS is not realistic; user may later replace
                # this with a graph-specific precomputed neighborhood.
                kwargs['bfs_result_for_mitm'] = None
            result = self._beam.search(**kwargs)
            path = None
            if getattr(result, 'path_found', False) and getattr(result, 'path', None) is not None:
                path = [self.move_names[int(idx)] for idx in list(result.path)]
                if max_total_path_len is not None and len(path) > max_total_path_len:
                    path = None
            profile.update({
                'cayleypy_path_found': bool(path),
                'cayleypy_path_length': (len(path) if path is not None else None),
            })
            return SearchOutcome(path=path, profile=profile, backend='cayleypy')
        except Exception as exc:
            profile['cayleypy_error'] = f'{type(exc).__name__}: {exc}'
            self.prefer_cayleypy = False
            self._beam = None
            self._graph = None
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
