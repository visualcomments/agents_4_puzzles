import argparse
import json
import os
import time

import torch

from pilgrim import Trainer, build_model, count_parameters, generate_inverse_moves, parse_generator_spec
from pilgrim.parallel import maybe_wrap_dataparallel, resolve_device


def infer_mode(hd2, nrd):
    if hd2 == 0 and nrd == 0:
        return "MLP1"
    if hd2 > 0 and nrd == 0:
        return "MLP2"
    if hd2 > 0 and nrd > 0:
        return "MLP2RB"
    raise ValueError("invalid combination of hd2 and nrd")


def print_args(args):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] Value training config:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key:<12} {value}")


def main():
    parser = argparse.ArgumentParser(description="Train the Megaminx value model.")
    parser.add_argument("--group_id", type=int, required=True, help="Puzzle group id.")
    parser.add_argument("--target_id", type=int, default=0, help="Target id.")
    parser.add_argument("--epochs", type=int, default=512, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=10000, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--K_min", type=int, default=1, help="Minimum random-walk depth.")
    parser.add_argument("--K_max", type=int, default=50, help="Maximum random-walk depth.")
    parser.add_argument(
        "--train_walkers",
        type=int,
        default=0,
        help="Training walkers per depth. 0 uses the default schedule.",
    )
    parser.add_argument(
        "--val_walkers",
        type=int,
        default=0,
        help="Validation walkers per depth. 0 uses the default schedule; negative disables validation.",
    )
    parser.add_argument("--weights", type=str, default="", help="Optional init weights name in weights/ or explicit path.")
    parser.add_argument("--gpu_ids", type=str, help="Comma-separated CUDA ids, e.g. '0,3,5,6'. Defaults to GPU 0.")
    parser.add_argument("--hd1", type=int, default=1024, help="First hidden layer size.")
    parser.add_argument("--hd2", type=int, default=256, help="Second hidden layer size. 0 disables it.")
    parser.add_argument("--nrd", type=int, default=4, help="Number of residual blocks. 0 disables them.")
    args = parser.parse_args()
    print_args(args)

    device, gpu_ids = resolve_device(args.gpu_ids)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Start training with device: {device}.")
    if gpu_ids:
        print(f"Using CUDA devices: {gpu_ids}")

    with open(f"generators/p{int(args.group_id):03d}.json", "r", encoding="utf-8") as handle:
        data = json.load(handle)
    all_moves, move_names = parse_generator_spec(data)
    all_moves = torch.tensor(all_moves, dtype=torch.int64, device=device)

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

    inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)
    mode = infer_mode(args.hd2, args.nrd)
    name = f"p{int(args.group_id):03d}-t{int(args.target_id):03d}"

    model = build_model(
        num_classes=num_classes,
        state_size=state_size,
        output_dim=1,
        dropout_rate=args.dropout,
        hd1=args.hd1,
        hd2=args.hd2,
        nrd=args.nrd,
    ).to(device)

    if V0.min() < 0:
        model.z_add = -V0.min().item()

    if args.weights:
        weights_path = args.weights if os.path.isabs(args.weights) or "/" in args.weights else f"weights/{args.weights}.pth"
        state = torch.load(weights_path, weights_only=True, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"Weights '{weights_path}' loaded.")

    if device.type == "cuda" and len(gpu_ids) > 1:
        model = maybe_wrap_dataparallel(model, gpu_ids)

    num_parameters = count_parameters(model)
    if args.train_walkers < 0:
        raise ValueError("train_walkers must be >= 0")
    if args.val_walkers == 0:
        args.val_walkers = max(100_000 // max(args.K_max, 1), 1)

    trainer = Trainer(
        net=model,
        num_epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        name=name,
        K_min=args.K_min,
        K_max=args.K_max,
        all_moves=all_moves,
        inverse_moves=inverse_moves,
        V0=V0,
        train_walkers_num=args.train_walkers,
        val_walkers_num=max(args.val_walkers, 0),
    )

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict.update(
        {
            "model_name": name,
            "model_mode": mode,
            "model_id": trainer.id,
            "num_parameters": num_parameters,
            "training_mode": "value",
            "gpu_ids": gpu_ids,
            "parallel_mode": "data_parallel" if len(gpu_ids) > 1 else "none",
        }
    )

    meta_path = f"{log_dir}/model_{name}_{trainer.id}.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(args_dict, handle, indent=4)

    print("Model info:")
    print(f"  mode          {mode}")
    print(f"  name          {name}")
    print(f"  id            {trainer.id}")
    print(f"  # parameters  {num_parameters:_}")
    print(f"  train walkers {trainer.walkers_num} per depth")
    if args.val_walkers >= 0:
        print(f"  val walkers   {args.val_walkers} per depth")

    summary = trainer.run()
    args_dict.update(summary)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(args_dict, handle, indent=4)


if __name__ == "__main__":
    main()
