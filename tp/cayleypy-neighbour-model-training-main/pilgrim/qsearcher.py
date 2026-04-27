import torch

from .model import batch_process
from .searcher import Searcher
from .utils import state2hash


class QSearcher(Searcher):
    @staticmethod
    def _sorted_isin(values, sorted_reference):
        """Memory-light membership test assuming sorted_reference is unique and sorted."""
        if sorted_reference.numel() == 0:
            return torch.zeros(values.size(0), dtype=torch.bool, device=values.device)

        pos = torch.searchsorted(sorted_reference, values)
        valid = pos < sorted_reference.numel()
        if not valid.any():
            return valid

        check_pos = pos.clamp_max(sorted_reference.numel() - 1)
        matched = sorted_reference.index_select(0, check_pos) == values
        return valid & matched

    def pred_q(self, states):
        """Predict per-action child-state costs for states using the model."""
        pred = batch_process(self.model, states, self.device, self.batch_size)
        if pred.ndim != 2 or pred.size(1) != self.n_gens:
            raise ValueError(
                f"QSearcher expects model output shape [batch, {self.n_gens}], got {tuple(pred.shape)}"
            )
        return pred

    def _valid_action_mask(self, states, last_moves):
        mask = torch.ones((states.size(0), self.n_gens), dtype=torch.bool, device=self.device)
        if last_moves is not None and self.inverse_moves is not None:
            valid_last = last_moves >= 0
            if valid_last.any():
                rows = torch.nonzero(valid_last, as_tuple=False).view(-1)
                mask[rows, self.inverse_moves[last_moves.index_select(0, rows)]] = False
        return mask

    def _find_solved_child(self, states, valid_mask, q_values):
        if self.solved_predecessors is None or self.solved_predecessor_hashes is None:
            return None

        parent_hashes = state2hash(states, self.hash_vec, self.batch_size)
        solved_mask = (parent_hashes.unsqueeze(1) == self.solved_predecessor_hashes.unsqueeze(0)) & valid_mask
        if not solved_mask.any():
            return None

        flat_matches = torch.nonzero(solved_mask.view(-1), as_tuple=False).view(-1)
        parents = torch.div(flat_matches, self.n_gens, rounding_mode="floor")
        moves = torch.remainder(flat_matches, self.n_gens)
        exact_mask = (
            states.index_select(0, parents) == self.solved_predecessors.index_select(0, moves)
        ).all(dim=1)
        if not exact_mask.any():
            return None

        solved_flat = flat_matches[torch.nonzero(exact_mask, as_tuple=False).view(-1)[0]]
        solved_parent = torch.div(solved_flat, self.n_gens, rounding_mode="floor").view(1)
        solved_move = torch.remainder(solved_flat, self.n_gens).view(1)
        solved_value = q_values[solved_parent, solved_move]
        return (
            self.V0.unsqueeze(0),
            self.V0_hash,
            solved_value.view(1),
            solved_move,
            solved_parent,
        )

    def _candidate_hashes(self, states, flat_indices):
        hashes = torch.empty(flat_indices.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, flat_indices.size(0), self.batch_size):
            batch_flat = flat_indices[i : i + self.batch_size]
            batch_parent = torch.div(batch_flat, self.n_gens, rounding_mode="floor")
            batch_moves = torch.remainder(batch_flat, self.n_gens)
            parent_states = states.index_select(0, batch_parent)
            hashes[i : i + self.batch_size] = self.hash_after_move(parent_states, batch_moves)
        return hashes

    def _first_unique_positions(self, hashes):
        positions = torch.arange(hashes.size(0), dtype=torch.int64, device=self.device)
        unique_hashes, inverse = torch.unique(hashes, return_inverse=True)
        first_pos = torch.full((unique_hashes.size(0),), hashes.size(0), dtype=torch.int64, device=self.device)
        first_pos.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
        return positions[first_pos.index_select(0, inverse) == positions]

    def _select_unique_candidates(self, states, q_flat, states_bad_hashed, B, total_valid):
        if total_valid <= 0:
            return None, None, None

        current_k = min(int(total_valid), max(int(B) * 2, 16384))
        final_unique_count = 0

        while True:
            candidate_values, candidate_indices = torch.topk(
                q_flat,
                k=current_k,
                largest=False,
                sorted=True,
            )
            del candidate_values
            candidate_hashes = self._candidate_hashes(states, candidate_indices)
            if states_bad_hashed.numel() > 0:
                bad_mask = self._sorted_isin(candidate_hashes, states_bad_hashed)
                candidate_indices = candidate_indices[~bad_mask]
                candidate_hashes = candidate_hashes[~bad_mask]

            if candidate_indices.numel() == 0:
                if current_k == total_valid:
                    return None, None, None
            else:
                unique_pos = self._first_unique_positions(candidate_hashes)
                final_unique_count = int(unique_pos.numel())
                if final_unique_count >= B or current_k == total_valid:
                    selected_pos = unique_pos[:B]
                    return (
                        candidate_indices.index_select(0, selected_pos),
                        candidate_hashes.index_select(0, selected_pos),
                        final_unique_count,
                    )

            if current_k == total_valid:
                return None, None, None
            current_k = min(int(total_valid), current_k * 2)

    def do_greedy_step(self, states, states_bad_hashed, last_moves=None, B=1000):
        """Perform one beam step using parent-state Q outputs instead of child-state V inference."""
        q_values = self.pred_q(states)
        valid_mask = self._valid_action_mask(states, last_moves)
        solved_result = self._find_solved_child(states, valid_mask, q_values)
        if solved_result is not None:
            self.counter[0, 0] += int(valid_mask.sum().item())
            self.counter[0, 1] += 1
            self.counter[1, 0] += 1
            self.counter[1, 1] += 1
            self.counter[2, 0] += 1
            self.counter[2, 1] += 1
            return solved_result

        q_flat = q_values.masked_fill(~valid_mask, float("inf")).reshape(-1)
        total_valid = int(valid_mask.sum().item())
        self.counter[0, 0] += total_valid
        self.counter[0, 1] += 1

        if total_valid == 0:
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_moves = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float32, device=self.device)
            return empty_states, empty_hashes, empty_values, empty_moves, empty_moves

        keep, keep_hashes, unique_count = self._select_unique_candidates(states, q_flat, states_bad_hashed, B, total_valid)
        self.counter[1, 0] += 0 if unique_count is None else unique_count
        self.counter[1, 1] += 1

        if keep is None or keep.numel() == 0:
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_moves = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float32, device=self.device)
            return empty_states, empty_hashes, empty_values, empty_moves, empty_moves

        keep_parent = torch.div(keep, self.n_gens, rounding_mode="floor")
        keep_moves = torch.remainder(keep, self.n_gens)
        selected_states = torch.empty((keep.size(0), self.state_size), dtype=states.dtype, device=self.device)
        for i in range(0, keep.size(0), self.batch_size):
            batch_keep = keep[i : i + self.batch_size]
            batch_parent = keep_parent[i : i + self.batch_size]
            batch_moves = keep_moves[i : i + self.batch_size]
            selected_states[i : i + self.batch_size] = self.apply_move(
                states.index_select(0, batch_parent),
                batch_moves,
            )

        self.counter[2, 0] += keep.size(0)
        self.counter[2, 1] += 1

        return (
            selected_states,
            keep_hashes,
            q_flat[keep],
            keep_moves,
            keep_parent,
        )
