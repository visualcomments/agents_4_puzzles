import argparse
import json
import os
import time
from pathlib import Path

import torch

from pilgrim import QSearcher, Searcher, build_model_from_info, generate_inverse_moves, generate_random_walk_states, parse_generator_spec
from pilgrim.parallel import maybe_wrap_dataparallel, resolve_device


def resolve_model_info_path(log_dir, group_id, target_id, model_id):
    base = f"model_p{int(group_id):03d}-t{int(target_id):03d}"
    candidates = sorted(Path(log_dir).glob(f"{base}*_{int(model_id)}.json"))
    if not candidates:
        raise FileNotFoundError(f"metadata for model_id={model_id} not found under {log_dir}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple metadata files found for model_id={model_id}: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def is_q_model(info):
    return str(info.get("model_name", "")).endswith("-q") or str(info.get("training_mode", "")).startswith("q_")


def print_args(args):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] Evaluation config:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key:<15} {value}")


def main():
    parser = argparse.ArgumentParser(description="Run beam-search evaluation for a Megaminx value or Q model.")
    parser.add_argument("--group_id", type=int, required=True, help="Puzzle group id.")
    parser.add_argument("--target_id", type=int, default=0, help="Target id.")
    parser.add_argument("--rnd_depth", type=int, help="Generate fixed-depth random scrambles on the fly.")
    parser.add_argument("--rnd_seed", type=int, default=0, help="Seed for on-the-fly random scrambles.")
    parser.add_argument("--search_seed", type=int, default=0, help="Seed for deterministic search hashing.")
    parser.add_argument("--model_id", type=int, required=True, help="Model id.")
    parser.add_argument("--epoch", type=int, help="Epoch checkpoint to load unless --best is set.")
    parser.add_argument("--best", action="store_true", help="Load the validation-selected checkpoint.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for single-GPU inference.")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead", help="torch.compile mode.")
    parser.add_argument(
        "--compile_skip_dynamic_cudagraphs",
        action="store_true",
        help="Disable dynamic-shape CUDAGraph capture for torch.compile.",
    )
    parser.add_argument("--B", type=int, default=2**18, help="Beam width.")
    parser.add_argument("--num_attempts", type=int, default=2, help="Number of search restarts.")
    parser.add_argument("--num_steps", type=int, default=200, help="Maximum search steps.")
    parser.add_argument("--tests_num", type=int, default=10, help="Number of scrambles to evaluate.")
    parser.add_argument("--gpu_ids", type=str, help="Comma-separated CUDA ids, e.g. '0,3,5,6'. Defaults to GPU 0.")
    parser.add_argument("--eval_batch_size", type=int, default=2**14, help="Batch size for expansion and scoring.")
    parser.add_argument("--verbose", type=int, default=0, help="Use tqdm if verbose > 0.")
    args = parser.parse_args()
    print_args(args)

    if not args.best and args.epoch is None:
        parser.error("--epoch is required unless --best is set")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    info_path = resolve_model_info_path(log_dir, args.group_id, args.target_id, args.model_id)
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    device, gpu_ids = resolve_device(args.gpu_ids)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] Start testing with device: {device}.")

    with open(f"generators/p{int(args.group_id):03d}.json", "r", encoding="utf-8") as handle:
        data = json.load(handle)
    moves, move_names = parse_generator_spec(data)
    all_moves = torch.tensor(moves, dtype=torch.int64, device=device)

    V0 = torch.load(
        f"targets/p{int(args.group_id):03d}-t{int(args.target_id):03d}.pt",
        weights_only=True,
        map_location=device,
    )

    n_gens = all_moves.size(0)
    state_size = all_moves.size(1)
    num_classes = torch.unique(V0).numel()

    print("Group info:")
    print(f"  # generators   {n_gens}")
    print(f"  # classes      {num_classes}")
    print(f"  state size     {state_size}")
    print(f"  search seed    {args.search_seed}")
    if gpu_ids:
        print(f"  cuda devices   {gpu_ids}")
    if args.compile:
        print(f"  compile mode   {args.compile_mode}")

    inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)
    q_model = is_q_model(info)
    model = build_model_from_info(
        info,
        num_classes=num_classes,
        state_size=state_size,
        output_dim=n_gens if q_model else 1,
    )

    epoch_label = "best" if args.best else str(args.epoch)
    model_name = info.get("model_name", f"p{int(args.group_id):03d}-t{int(args.target_id):03d}")
    if args.best:
        weights_path = f"weights/{model_name}_{args.model_id}_best.pth"
    else:
        weights_path = f"weights/{model_name}_{args.model_id}_e{args.epoch:05d}.pth"
    state = torch.load(weights_path, weights_only=False, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    if device.type == "cuda":
        model.half()
        model.dtype = torch.float16
    else:
        model.dtype = torch.float32
    if V0.min() < 0:
        model.z_add = -V0.min().item()

    model.to(device)
    if len(gpu_ids) > 1 and args.compile:
        raise RuntimeError("torch.compile is only supported with single-GPU inference")
    if len(gpu_ids) > 1:
        model = maybe_wrap_dataparallel(model, gpu_ids)
    elif args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        if args.compile_skip_dynamic_cudagraphs:
            triton_cfg = getattr(getattr(torch, "_inductor", None), "config", None)
            triton_cfg = getattr(triton_cfg, "triton", None)
            if triton_cfg is not None and hasattr(triton_cfg, "cudagraph_skip_dynamic_graphs"):
                triton_cfg.cudagraph_skip_dynamic_graphs = True
                print("Enabled torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True")
        model = torch.compile(model, mode=args.compile_mode)

    if args.rnd_depth is not None:
        tests = generate_random_walk_states(
            V0=V0,
            all_moves=all_moves,
            inverse_moves=inverse_moves,
            num_states=args.tests_num,
            depth=args.rnd_depth,
            device=device,
            seed=args.rnd_seed,
        )
        dataset_label = f"rnd-k{args.rnd_depth}-s{args.rnd_seed}"
        print(f"Generated rnd dataset on the fly: depth={args.rnd_depth}, seed={args.rnd_seed}")
    else:
        tests = torch.load(
            f"datasets/p{int(args.group_id):03d}-t{int(args.target_id):03d}-rnd.pt",
            weights_only=False,
            map_location=device,
        )[: args.tests_num]
        dataset_label = "rnd"
    args.tests_num = tests.size(0)
    print(f"Test dataset size: {args.tests_num}")

    searcher_cls = QSearcher if q_model else Searcher
    searcher = searcher_cls(
        model=model,
        all_moves=all_moves,
        V0=V0,
        device=device,
        verbose=args.verbose,
        move_names=move_names,
        inverse_moves=inverse_moves,
        normalize_path=True,
        batch_size=args.eval_batch_size,
        hash_seed=args.search_seed,
    )

    log_suffix = "_compile" if args.compile else ""
    log_file = f"{log_dir}/test_{model_name}-{dataset_label}_{args.model_id}_{epoch_label}_B{args.B}{log_suffix}.json"

    results = []
    total_length = 0
    started = time.time()

    for test_num, state in enumerate(tests):
        solve_started = time.time()
        moves, attempts = searcher.get_solution(
            state,
            B=args.B,
            num_steps=args.num_steps,
            num_attempts=args.num_attempts,
        )
        solve_time = time.time() - solve_started
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        vertex_num = searcher.counter[:, 0] / searcher.counter[:, 1]
        searcher.counter = torch.zeros((3, 2), dtype=torch.int64)

        if moves is not None:
            solution_length = len(moves)
            total_length += solution_length
            entry = {
                "test_num": test_num,
                "solution_length": solution_length,
                "attempts": attempts + 1,
                "time": round(solve_time, 2),
                "moves": moves.tolist(),
                "vertex_num": f"[{vertex_num[0]:.2e}, {vertex_num[1]:.2e}, {vertex_num[2]:.2e}]",
            }
            print(f"[{timestamp}] Solution {test_num}: Length = {solution_length}")
        else:
            entry = {
                "test_num": test_num,
                "solution_length": None,
                "attempts": None,
                "time": round(solve_time, 2),
                "moves": None,
                "vertex_num": f"[{vertex_num[0]:.2e}, {vertex_num[1]:.2e}, {vertex_num[2]:.2e}]",
            }
            print(f"[{timestamp}] Solution {test_num} not found")

        results.append(entry)
        with open(log_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=4)

    finished = time.time()
    solved_results = [entry for entry in results if entry["solution_length"] is not None]
    avg_length = total_length / len(solved_results) if solved_results else 0.0

    print(f"Test completed in {(finished - started):.2f}s.")
    print(f"Average solution length: {avg_length:.2f}.")
    print(f"Solved {len(solved_results)}/{args.tests_num} scrambles.")
    print(f"Results saved to {log_file}.")


if __name__ == "__main__":
    main()
