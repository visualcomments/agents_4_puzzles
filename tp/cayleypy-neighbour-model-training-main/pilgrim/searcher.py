import torch
import time
from collections import deque
from tqdm import tqdm
from .utils import state2hash
from .model import batch_process


class Searcher:
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

    def __init__(
        self,
        model,
        all_moves,
        V0,
        device=None,
        verbose=0,
        move_names=None,
        inverse_moves=None,
        normalize_path=True,
        batch_size=None,
        hash_seed=0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.all_moves = all_moves
        self.V0 = V0
        if batch_size is None:
            batch_size = 2**14
        self.batch_size = int(batch_size)
        self.n_gens = all_moves.size(0)
        self.state_size = all_moves.size(1)
        hash_generator = torch.Generator(device="cpu")
        hash_generator.manual_seed(int(hash_seed))
        self.hash_vec = torch.randint(0, int(1e15), (self.state_size,), generator=hash_generator, dtype=torch.int64).to(self.device)
        self.verbose = verbose
        self.counter = torch.zeros((3, 2), dtype=torch.int64)
        self.move_names = list(move_names) if move_names is not None else None
        self.inverse_moves = None if inverse_moves is None else torch.as_tensor(inverse_moves, dtype=torch.int64, device=self.device)
        self.normalize_path = bool(normalize_path)
        self.move_to_idx = None if self.move_names is None else {name: idx for idx, name in enumerate(self.move_names)}
        self.inverse_all_moves = torch.empty_like(self.all_moves)
        move_rows = torch.arange(self.n_gens, device=self.device).unsqueeze(1).expand(-1, self.state_size)
        state_cols = torch.arange(self.state_size, device=self.device).unsqueeze(0).expand(self.n_gens, -1)
        self.inverse_all_moves[move_rows, self.all_moves] = state_cols
        self.hash_after_move_weights = self.hash_vec.index_select(0, self.inverse_all_moves.reshape(-1)).view(self.n_gens, self.state_size)
        self.V0_hash = state2hash(self.V0.unsqueeze(0), self.hash_vec, 1)
        self.solved_predecessors = None
        self.solved_predecessor_hashes = None
        if self.inverse_moves is not None:
            solved_batch = self.V0.unsqueeze(0).expand(self.n_gens, -1)
            self.solved_predecessors = torch.gather(solved_batch, 1, self.all_moves[self.inverse_moves])
            self.solved_predecessor_hashes = state2hash(self.solved_predecessors, self.hash_vec, self.n_gens)
    
    def get_unique_states(self, states, states_bad_hashed):
        """Filter unique states by removing duplicates based on hash."""
        idx1 = torch.arange(states.size(0), dtype=torch.int64, device=states.device)
        hashed = state2hash(states, self.hash_vec, self.batch_size)
        mask1  = ~self._sorted_isin(hashed, states_bad_hashed)
        if not mask1.any():
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=states.device)
            empty_idx = torch.empty((0,), dtype=torch.int64, device=states.device)
            return empty_states, empty_idx
        hashed = hashed[mask1]
        hashed_sorted, idx2 = torch.sort(hashed)
        mask2 = torch.concat((torch.tensor([True], device=states.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        return states[mask1][idx2[mask2]], idx1[mask1][idx2[mask2]] 
    
    def get_unique_hashed_states_idx(self, hashed, states_bad_hashed):
        """Filter unique hashed states by removing duplicates."""
        idx1 = torch.arange(hashed.size(0), dtype=torch.int64, device=hashed.device)
        mask1 = ~self._sorted_isin(hashed, states_bad_hashed)
        if not mask1.any():
            return torch.empty((0,), dtype=torch.int64, device=hashed.device)
        hashed = hashed[mask1]
        hashed_sorted, idx2 = torch.sort(hashed)
        mask2 = torch.concat((torch.tensor([True], device=hashed.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        return idx1[mask1][idx2[mask2]]
    
    def get_neighbors(self, states):
        """Return neighboring states for each state in the batch."""
        neighbors = torch.empty(states.size(0), self.n_gens, self.state_size, device=self.device, dtype=states.dtype)
        for i in range(0, states.size(0), self.batch_size):
            batch_states = states[i:i + self.batch_size]
            neighbors[i:i + self.batch_size] = torch.gather(
                batch_states.unsqueeze(1).expand(batch_states.size(0), self.n_gens, self.state_size), 
                2, 
                self.all_moves.unsqueeze(0).expand(batch_states.size(0), self.n_gens, self.state_size)
            )
        return neighbors
    
    def apply_move(self, states, moves):
        moved_states = torch.empty(states.size(0), self.state_size, device=self.device, dtype=states.dtype)
        for i in range(0, states.size(0), self.batch_size):
            moved_states[i:i+self.batch_size] = torch.gather(states[i:i+self.batch_size], 1, self.all_moves[moves[i:i+self.batch_size]])
        return moved_states

    def hash_after_move(self, states, moves):
        hashed = torch.empty(states.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, states.size(0), self.batch_size):
            batch_states = states[i:i+self.batch_size].to(torch.int64)
            batch_moves = moves[i:i+self.batch_size]
            batch_weights = self.hash_after_move_weights.index_select(0, batch_moves)
            hashed[i:i+self.batch_size] = torch.sum(batch_states * batch_weights, dim=1)
        return hashed
    
    def do_greedy_step(self, states, states_bad_hashed, last_moves=None, B=1000):
        """Perform a greedy step to find the best neighbors."""
        idx0 = torch.arange(states.size(0), device=self.device).repeat_interleave(self.n_gens)
        moves = torch.arange(self.n_gens, device=self.device).repeat(states.size(0))

        if last_moves is not None and self.inverse_moves is not None:
            parent_last_moves = last_moves[idx0]
            valid_mask = (parent_last_moves < 0) | (moves != self.inverse_moves[parent_last_moves])
            idx0 = idx0[valid_mask]
            moves = moves[valid_mask]

        self.counter[0, 0] += moves.size(0); self.counter[0, 1] += 1;

        if moves.numel() == 0:
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_moves = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float16, device=self.device)
            return empty_states, empty_hashes, empty_values, empty_moves, empty_moves

        neighbors_hashed = torch.empty(moves.size(0), dtype=torch.int64, device=self.device)
        for i in range(0, moves.size(0), self.batch_size):
            batch_idx = idx0[i:i+self.batch_size]
            batch_moves = moves[i:i+self.batch_size]
            neighbors = self.apply_move(states[batch_idx], batch_moves)
            neighbors_hashed[i:i+self.batch_size] = state2hash(neighbors, self.hash_vec, self.batch_size)
        idx1 = self.get_unique_hashed_states_idx(neighbors_hashed, states_bad_hashed)
        self.counter[1, 0] += idx1.size(0); self.counter[1, 1] += 1;

        if idx1.numel() == 0:
            empty_states = torch.empty((0, self.state_size), dtype=states.dtype, device=self.device)
            empty_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_moves = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_values = torch.empty((0,), dtype=torch.float16, device=self.device)
            return empty_states, empty_hashes, empty_values, empty_moves, empty_moves
        
        value = torch.empty(idx1.size(0), dtype=torch.float16, device=self.device)
        for i in range(0, idx1.size(0), self.batch_size):
            batch_states = self.apply_move(states[idx0[idx1[i:i+self.batch_size]]], moves[idx1[i:i+self.batch_size]])
            value[i:i+self.batch_size] = self.pred_d(batch_states)[0]
        idx2 = torch.argsort(value)[:B]
        self.counter[2, 0] += idx2.size(0); self.counter[2, 1] += 1;

        next_states = torch.empty(idx2.size(0), self.state_size, dtype=states.dtype, device=self.device)
        for i in range(0, idx2.size(0), self.batch_size):
            next_states[i:i+self.batch_size] = self.apply_move(
                states[idx0[idx1[idx2[i:i+self.batch_size]]]],
                moves[idx1[idx2[i:i+self.batch_size]]],
            )

        return (
            next_states,
            neighbors_hashed[idx1[idx2]],
            value[idx2],
            moves[idx1[idx2]],
            idx0[idx1[idx2]],
        )
    
    def check_stagnation(self, states_log):
        """Check if the process is in a stagnation state."""
        return torch.isin(torch.concat(list(states_log)[2:]), torch.concat(list(states_log)[:2])).all().item()

    def normalize_moves_seq(self, moves_seq):
        """Normalize consecutive same-face turns to a shorter equivalent sequence."""
        if not self.normalize_path or self.move_names is None or moves_seq.numel() == 0:
            return moves_seq

        normalized = []
        pending_face = None
        pending_turns = 0

        def flush(face, turns):
            if face is None:
                return
            turns %= 5
            if turns == 0:
                return
            if turns == 1:
                normalized.append(face)
            elif turns == 2:
                normalized.extend([face, face])
            elif turns == 3:
                normalized.extend([f"{face}'", f"{face}'"])
            elif turns == 4:
                normalized.append(f"{face}'")

        for move_idx in moves_seq.tolist():
            move_name = self.move_names[move_idx]
            if move_name.endswith("'"):
                face = move_name[:-1]
                delta = -1
            else:
                face = move_name
                delta = 1

            if pending_face is not None and face != pending_face:
                flush(pending_face, pending_turns)
                pending_turns = 0

            pending_face = face
            pending_turns += delta

        flush(pending_face, pending_turns)

        if not normalized:
            return torch.empty((0,), dtype=torch.int64)
        return torch.tensor([self.move_to_idx[name] for name in normalized], dtype=torch.int64)

    
    def get_solution(self, state, B=2**12, num_steps=200, num_attempts=10):
        """Main solution-finding loop that attempts to solve the cube."""
        if torch.equal(state, self.V0):
            empty_moves = torch.empty((0,), dtype=torch.int64)
            return empty_moves, 0

        states_bad_hashed = torch.tensor([], dtype=torch.int64, device=self.device)
        for J in range(num_attempts):
            states = state.unsqueeze(0).clone()
            tree_move = -torch.ones((num_steps, B), dtype=torch.int64)
            tree_idx = -torch.ones((num_steps, B), dtype=torch.int64)
            states_hash_log = deque(maxlen=4)
            last_moves = -torch.ones((1,), dtype=torch.int64, device=self.device)
            visited_hashed = torch.concat((states_bad_hashed, state2hash(states, self.hash_vec)))
            visited_hashed = torch.unique(visited_hashed)
            
            if self.verbose:
                pbar = tqdm(range(num_steps))
            else:
                pbar = range(num_steps)
            for j in pbar:
                states, states_hashed, y_pred, moves, idx = self.do_greedy_step(states, visited_hashed, last_moves, B)
                if states.size(0) == 0:
                    break
                if self.verbose:
                    pbar.set_description(
                        f"  y_min = {y_pred.min().item():.1f}, y_mean = {y_pred.mean().item():.1f}, y_max = {y_pred.max().item():.1f}"
                    )
                last_moves = moves
                states_hash_log.append(states_hashed)
                visited_hashed = torch.unique(torch.concat((visited_hashed, states_hashed)))
                leaves_num = states.size(0)
                tree_move[j, :leaves_num] = moves
                tree_idx[j, :leaves_num] = idx

                if (states == self.V0).all(dim=1).any():
                    break
                elif (j > 3 and self.check_stagnation(states_hash_log)):
                    states_bad_hashed = torch.concat((states_bad_hashed, torch.concat(list(states_hash_log))))
                    states_bad_hashed = torch.unique(states_bad_hashed)
                    break
            else:
                states_bad_hashed = visited_hashed

            if (states == self.V0).all(dim=1).any():
                break
            states_bad_hashed = torch.unique(torch.concat((states_bad_hashed, visited_hashed)))
        
        if not (states == self.V0).all(dim=1).any():
            return None, J

        V0_pos = torch.nonzero((states == self.V0).all(dim=1), as_tuple=True)[0].item()

        tree_idx, tree_move = tree_idx[:j+1].flip((0,)), tree_move[:j+1].flip((0,))
        path = [tree_idx[0, V0_pos].item()]
        for k in range(1, j+1):
            path.append(tree_idx[k, path[-1]].item())

        moves_seq = torch.tensor([tree_move[k, path[k-1]] if k > 0 else tree_move[k, V0_pos] for k in range(j+1)], dtype=torch.int64)
        moves_seq = self.normalize_moves_seq(moves_seq)
        return moves_seq.flip((0,)), J
    
    def pred_d(self, states):
        """Predict values for states using the model."""
        pred = batch_process(self.model, states, self.device, self.batch_size)
#         pred[(states == self.V0).all(dim=-1)] = 0
        return pred.unsqueeze(0)
