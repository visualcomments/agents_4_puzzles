import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from pilgrim import build_model, build_model_from_info, count_parameters, generate_inverse_moves, parse_generator_spec
from pilgrim.model import batch_process
from pilgrim.parallel import maybe_wrap_dataparallel, model_state_dict, resolve_device


def append_csv_row(path, row):
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def infer_mode(hd2, nrd):
    if hd2 == 0 and nrd == 0:
        return "QMLP1"
    if hd2 > 0 and nrd == 0:
        return "QMLP2"
    if hd2 > 0 and nrd > 0:
        return "QMLP2RB"
    raise ValueError("invalid combination of hd2 and nrd")


def print_args(args):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] Q training config:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key:<18} {value}")


def resolve_weights_path(weights_spec):
    raw = Path(weights_spec)
    candidates = [raw]
    if not raw.is_absolute() and len(raw.parts) == 1:
        candidates.append(Path("weights") / raw)

    checked = []
    for candidate in candidates:
        checked.append(candidate)
        if candidate.suffix != ".pth":
            checked.append(Path(f"{candidate}.pth"))

    for candidate in checked:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("weights file not found; tried: " + ", ".join(str(candidate) for candidate in checked))


def resolve_teacher_info_path(group_id, target_id, teacher_model_id):
    exact = Path("logs") / f"model_p{int(group_id):03d}-t{int(target_id):03d}_{int(teacher_model_id)}.json"
    if exact.exists():
        return exact

    base = f"model_p{int(group_id):03d}-t{int(target_id):03d}"
    candidates = [
        path
        for path in Path("logs").glob(f"{base}*_{int(teacher_model_id)}.json")
        if f"{base}-q_" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(f"teacher metadata for model_id={teacher_model_id} not found")
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple teacher metadata files found for model_id={teacher_model_id}: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def load_generator_data(group_id, device):
    with open(f"generators/p{int(group_id):03d}.json", "r", encoding="utf-8") as handle:
        data = json.load(handle)
    moves, move_names = parse_generator_spec(data)
    return torch.tensor(moves, dtype=torch.int64, device=device), move_names


def load_target_state(group_id, target_id, device):
    return torch.load(
        f"targets/p{int(group_id):03d}-t{int(target_id):03d}.pt",
        weights_only=True,
        map_location=device,
    )


def load_compatible_weights(model, weights_path, device):
    payload = torch.load(weights_path, weights_only=False, map_location=device)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"unsupported weights payload type: {type(payload)!r}")

    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state or model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible[key] = value

    incompatible = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_keys": sorted(compatible.keys()),
        "skipped_keys": skipped,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def make_rng(device, seed):
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed))
    return gen


def sample_random_walk_batch(batch_size, K_min, K_max, V0, all_moves, inverse_moves, rng):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if K_min < 0 or K_max < K_min:
        raise ValueError("invalid depth range")

    device = V0.device
    n_gens = all_moves.size(0)
    depths = torch.randint(K_min, K_max + 1, (batch_size,), generator=rng, dtype=torch.int64, device=device)
    states = V0.unsqueeze(0).expand(batch_size, -1).clone()
    last_moves = torch.full((batch_size,), -1, dtype=torch.int64, device=device)

    for step in range(K_max):
        active = torch.nonzero(depths > step, as_tuple=False).view(-1)
        if active.numel() == 0:
            break

        prev_moves = last_moves.index_select(0, active)
        next_moves = torch.randint(n_gens, (active.numel(),), generator=rng, dtype=torch.int64, device=device)
        valid_prev = prev_moves >= 0
        if valid_prev.any():
            inverse_prev = inverse_moves[prev_moves.clamp_min(0)]
            invalid = valid_prev & (next_moves == inverse_prev)
            while invalid.any():
                next_moves[invalid] = torch.randint(
                    n_gens,
                    (int(invalid.sum().item()),),
                    generator=rng,
                    dtype=torch.int64,
                    device=device,
                )
                invalid = valid_prev & (next_moves == inverse_prev)

        active_states = states.index_select(0, active)
        states[active] = torch.gather(active_states, 1, all_moves[next_moves])
        last_moves[active] = next_moves

    return states, last_moves


def build_action_mask(last_moves, inverse_moves, n_gens):
    mask = torch.ones((last_moves.size(0), n_gens), dtype=torch.bool, device=last_moves.device)
    valid_last = last_moves >= 0
    if valid_last.any():
        rows = torch.nonzero(valid_last, as_tuple=False).view(-1)
        mask[rows, inverse_moves[last_moves.index_select(0, rows)]] = False
    return mask


def apply_all_moves(states, all_moves):
    if states.numel() == 0:
        return torch.empty((0, all_moves.size(0), all_moves.size(1)), dtype=states.dtype, device=states.device)
    count = states.size(0)
    move_index = all_moves.unsqueeze(0).expand(count, -1, -1)
    state_view = states.unsqueeze(1).expand(-1, all_moves.size(0), -1)
    return torch.gather(state_view, 2, move_index)


def build_teacher_targets(states, all_moves, V0, teacher, teacher_batch_size):
    batch_size = states.size(0)
    n_gens = all_moves.size(0)
    state_size = all_moves.size(1)
    solved_cube = V0.view(1, 1, -1)

    children = apply_all_moves(states, all_moves)
    flat_children = children.reshape(batch_size * n_gens, state_size)
    with torch.no_grad():
        targets = batch_process(teacher, flat_children, states.device, teacher_batch_size)
    targets = targets.reshape(batch_size, n_gens).to(torch.float32)

    solved_children = (children == solved_cube).all(dim=2)
    if solved_children.any():
        targets[solved_children] = 0.0
    return targets


def masked_top1(preds, targets, mask):
    valid_rows = mask.any(dim=1)
    if not valid_rows.any():
        return preds.new_tensor(0.0)
    rows = torch.nonzero(valid_rows, as_tuple=False).view(-1)
    masked_preds = preds.index_select(0, rows).masked_fill(~mask.index_select(0, rows), float("inf"))
    masked_targets = targets.index_select(0, rows).masked_fill(~mask.index_select(0, rows), float("inf"))
    pred_idx = masked_preds.argmin(dim=1)
    target_idx = masked_targets.argmin(dim=1)
    return (pred_idx == target_idx).float().mean()


def masked_mse_loss(preds, targets):
    return (preds - targets).pow(2).mean()


def evaluate(model, states, targets, action_mask, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_items = 0

    with torch.no_grad():
        for start in range(0, states.size(0), batch_size):
            batch_states = states[start : start + batch_size].to(device)
            batch_targets = targets[start : start + batch_size].to(device)
            batch_action_mask = action_mask[start : start + batch_size].to(device)

            preds = model(batch_states)
            loss = masked_mse_loss(preds, batch_targets)
            top1 = masked_top1(preds, batch_targets, batch_action_mask)

            items = batch_states.size(0)
            total_loss += float(loss.item()) * items
            total_top1 += float(top1.item()) * items
            total_items += items

    denom = max(total_items, 1)
    return {
        "loss": total_loss / denom,
        "top1": total_top1 / denom,
    }


def build_validation_set(num_samples, batch_size, K_min, K_max, V0, all_moves, inverse_moves, rng, teacher, teacher_batch_size):
    states_parts = []
    targets_parts = []
    action_mask_parts = []
    remaining = int(num_samples)
    while remaining > 0:
        current = min(batch_size, remaining)
        states, last_moves = sample_random_walk_batch(
            batch_size=current,
            K_min=K_min,
            K_max=K_max,
            V0=V0,
            all_moves=all_moves,
            inverse_moves=inverse_moves,
            rng=rng,
        )
        targets = build_teacher_targets(
            states=states,
            all_moves=all_moves,
            V0=V0,
            teacher=teacher,
            teacher_batch_size=teacher_batch_size,
        )
        action_mask = build_action_mask(last_moves, inverse_moves, all_moves.size(0))
        states_parts.append(states.cpu())
        targets_parts.append(targets.cpu())
        action_mask_parts.append(action_mask.cpu())
        remaining -= current

    return (
        torch.cat(states_parts, dim=0),
        torch.cat(targets_parts, dim=0),
        torch.cat(action_mask_parts, dim=0),
    )


def load_teacher_model(group_id, target_id, teacher_model_id, num_classes, state_size, device, V0, gpu_ids):
    info_path = resolve_teacher_info_path(group_id, target_id, teacher_model_id)
    with info_path.open("r", encoding="utf-8") as handle:
        teacher_info = json.load(handle)

    teacher = build_model_from_info(
        teacher_info,
        num_classes=num_classes,
        state_size=state_size,
        output_dim=1,
    )
    if V0.min() < 0:
        teacher.z_add = -V0.min().item()

    teacher_name = teacher_info.get("model_name", f"p{int(group_id):03d}-t{int(target_id):03d}")
    weights_spec = (
        teacher_info.get("best_weights_file")
        or teacher_info.get("final_weights_file")
        or f"{teacher_name}_{int(teacher_model_id)}_best"
    )
    weights_path = resolve_weights_path(weights_spec)
    state = torch.load(weights_path, weights_only=False, map_location="cpu")
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    teacher.to(device)
    if device.type == "cuda":
        teacher.half()
        teacher.dtype = torch.float16
    else:
        teacher.dtype = torch.float32
    for param in teacher.parameters():
        param.requires_grad_(False)
    if device.type == "cuda" and len(gpu_ids) > 1:
        teacher = maybe_wrap_dataparallel(teacher, gpu_ids)
    return teacher, teacher_info, str(info_path), str(weights_path)


def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()


def main():
    parser = argparse.ArgumentParser(description="Train the Megaminx Q model with teacher supervision.")
    parser.add_argument("--group_id", type=int, required=True, help="Puzzle group id.")
    parser.add_argument("--target_id", type=int, default=0, help="Target id.")
    parser.add_argument("--epochs", type=int, default=128, help="Number of training epochs.")
    parser.add_argument("--steps_per_epoch", type=int, default=256, help="Optimizer steps per epoch.")
    parser.add_argument("--batch_size", type=int, default=1024, help="States per optimization step.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm. 0 disables it.")
    parser.add_argument("--freeze_bn_after", type=int, default=0, help="Freeze BatchNorm running stats after this epoch.")
    parser.add_argument("--K_min", type=int, default=1, help="Minimum random-walk depth.")
    parser.add_argument("--K_max", type=int, default=50, help="Maximum random-walk depth.")
    parser.add_argument("--teacher_model_id", type=int, required=True, help="Value-model id used as scalar teacher.")
    parser.add_argument("--teacher_batch_size", type=int, default=65536, help="Teacher child-eval batch size.")
    parser.add_argument("--val_size", type=int, default=16384, help="Number of fixed validation states. 0 disables validation.")
    parser.add_argument("--val_batch_size", type=int, default=2048, help="Validation batch size.")
    parser.add_argument("--weights", type=str, default="", help="Optional init weights path or name.")
    parser.add_argument("--gpu_ids", type=str, help="Comma-separated CUDA ids, e.g. '0,3,5,6'. Defaults to GPU 0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hd1", type=int, default=1536, help="First hidden layer size.")
    parser.add_argument("--hd2", type=int, default=512, help="Second hidden layer size. 0 disables it.")
    parser.add_argument("--nrd", type=int, default=2, help="Number of residual blocks. 0 disables them.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    args = parser.parse_args()
    print_args(args)

    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.val_size < 0:
        raise ValueError("val_size must be >= 0")
    if args.val_batch_size <= 0:
        raise ValueError("val_batch_size must be > 0")
    if args.lr <= 0:
        raise ValueError("lr must be > 0")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be >= 0")
    if args.freeze_bn_after < 0:
        raise ValueError("freeze_bn_after must be >= 0")
    if args.teacher_batch_size <= 0:
        raise ValueError("teacher_batch_size must be > 0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device, gpu_ids = resolve_device(args.gpu_ids)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_rng = make_rng(device, args.seed)
    val_rng = make_rng(device, args.seed + 1_000_000)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] Start Q-training with device: {device}.")
    if gpu_ids:
        print(f"Using CUDA devices: {gpu_ids}")

    all_moves, move_names = load_generator_data(args.group_id, device)
    V0 = load_target_state(args.group_id, args.target_id, device)
    inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)

    n_gens = all_moves.size(0)
    state_size = all_moves.size(1)
    num_classes = torch.unique(V0).numel()

    print("Group info:")
    print(f"  # generators   {n_gens}")
    print(f"  # classes      {num_classes}")
    print(f"  state size     {state_size}")

    mode = infer_mode(args.hd2, args.nrd)
    name = f"p{int(args.group_id):03d}-t{int(args.target_id):03d}-q"

    model = build_model(
        num_classes=num_classes,
        state_size=state_size,
        output_dim=n_gens,
        dropout_rate=args.dropout,
        hd1=args.hd1,
        hd2=args.hd2,
        nrd=args.nrd,
    ).to(device)
    if V0.min() < 0:
        model.z_add = -V0.min().item()

    init_info = None
    if args.weights:
        weights_path = resolve_weights_path(args.weights)
        init_info = load_compatible_weights(model, weights_path, device)
        print(
            f"Loaded compatible init weights from '{weights_path}'. "
            f"matched={len(init_info['loaded_keys'])} skipped={len(init_info['skipped_keys'])}"
        )

    if device.type == "cuda" and len(gpu_ids) > 1:
        model = maybe_wrap_dataparallel(model, gpu_ids)

    teacher, teacher_info, teacher_info_path, teacher_weights_path = load_teacher_model(
        group_id=args.group_id,
        target_id=args.target_id,
        teacher_model_id=args.teacher_model_id,
        num_classes=num_classes,
        state_size=state_size,
        device=device,
        V0=V0,
        gpu_ids=gpu_ids,
    )
    print(f"Loaded teacher model_id={args.teacher_model_id} from '{teacher_weights_path}'.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_id = int(time.time())
    log_dir = Path("logs")
    weights_dir = Path("weights")
    log_dir.mkdir(exist_ok=True)
    weights_dir.mkdir(exist_ok=True)

    log_path = log_dir / f"train_{name}_{model_id}.csv"
    meta_path = log_dir / f"model_{name}_{model_id}.json"
    best_weights_path = weights_dir / f"{name}_{model_id}_best.pth"

    args_dict = vars(args).copy()
    args_dict.update(
        {
            "model_name": name,
            "model_mode": mode,
            "model_id": model_id,
            "num_parameters": count_parameters(model),
            "n_gens": n_gens,
            "training_mode": "q_teacher",
            "gpu_ids": gpu_ids,
            "parallel_mode": "data_parallel" if len(gpu_ids) > 1 else "none",
            "teacher_model_name": teacher_info.get("model_name"),
            "teacher_metadata_path": teacher_info_path,
            "teacher_weights_file": teacher_weights_path,
        }
    )
    if init_info is not None:
        args_dict["init_weights_loaded_keys"] = len(init_info["loaded_keys"])
        args_dict["init_weights_skipped_keys"] = len(init_info["skipped_keys"])
        args_dict["init_weights_missing_keys"] = init_info["missing_keys"]

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(args_dict, handle, indent=4)

    val_states = None
    val_targets = None
    val_action_mask = None
    val_build_time = 0.0
    if args.val_size > 0:
        val_started = time.time()
        val_states, val_targets, val_action_mask = build_validation_set(
            num_samples=args.val_size,
            batch_size=args.val_batch_size,
            K_min=args.K_min,
            K_max=args.K_max,
            V0=V0,
            all_moves=all_moves,
            inverse_moves=inverse_moves,
            rng=val_rng,
            teacher=teacher,
            teacher_batch_size=args.teacher_batch_size,
        )
        val_build_time = time.time() - val_started
        print(f"Validation set built in {val_build_time:.2f}s with {args.val_size} states.")

    best_val_loss = math.inf
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        model.train(True)
        if args.freeze_bn_after > 0 and epoch >= args.freeze_bn_after:
            freeze_batchnorm_stats(model)

        train_loss_total = 0.0
        train_top1_total = 0.0
        train_items = 0
        label_time_total = 0.0
        optim_time_total = 0.0

        for _ in range(args.steps_per_epoch):
            label_started = time.time()
            states, last_moves = sample_random_walk_batch(
                batch_size=args.batch_size,
                K_min=args.K_min,
                K_max=args.K_max,
                V0=V0,
                all_moves=all_moves,
                inverse_moves=inverse_moves,
                rng=train_rng,
            )
            action_mask = build_action_mask(last_moves, inverse_moves, n_gens)
            targets = build_teacher_targets(
                states=states,
                all_moves=all_moves,
                V0=V0,
                teacher=teacher,
                teacher_batch_size=args.teacher_batch_size,
            )
            label_time_total += time.time() - label_started

            optim_started = time.time()
            preds = model(states)
            loss = masked_mse_loss(preds, targets)
            top1 = masked_top1(preds, targets, action_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            optim_time_total += time.time() - optim_started

            batch_items = states.size(0)
            train_loss_total += float(loss.item()) * batch_items
            train_top1_total += float(top1.item()) * batch_items
            train_items += batch_items

        train_metrics = {
            "loss": train_loss_total / max(train_items, 1),
            "top1": train_top1_total / max(train_items, 1),
        }

        val_metrics = None
        if val_states is not None and val_targets is not None and val_action_mask is not None:
            val_metrics = evaluate(
                model=model,
                states=val_states,
                targets=val_targets,
                action_mask=val_action_mask,
                batch_size=args.val_batch_size,
                device=device,
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save(model_state_dict(model), best_weights_path)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "vertices_seen": train_items,
            "label_time": label_time_total,
            "optim_time": optim_time_total,
            "epoch_time": time.time() - epoch_started,
        }
        if val_metrics is not None:
            row["val_loss"] = val_metrics["loss"]
            row["val_top1"] = val_metrics["top1"]
            row["best_val_loss"] = min(best_val_loss, val_metrics["loss"])
        append_csv_row(log_path, row)

        if (epoch & (epoch - 1)) == 0:
            weights_path = weights_dir / f"{name}_{model_id}_e{epoch:05d}.pth"
            torch.save(model_state_dict(model), weights_path)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if val_metrics is None:
            print(
                f"[{timestamp}] epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_top1={train_metrics['top1']:.4f} "
                f"label_time={label_time_total:.1f}s optim_time={optim_time_total:.1f}s"
            )
        else:
            print(
                f"[{timestamp}] epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_top1={train_metrics['top1']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_top1={val_metrics['top1']:.4f} "
                f"best_val={best_val_loss:.4f}@{best_epoch:04d}"
            )

    final_weights_path = weights_dir / f"{name}_{model_id}_final.pth"
    torch.save(model_state_dict(model), final_weights_path)

    args_dict.update(
        {
            "final_weights_file": str(final_weights_path),
            "best_weights_file": str(best_weights_path) if best_epoch > 0 else None,
            "best_epoch": best_epoch,
            "best_val_loss": None if math.isinf(best_val_loss) else best_val_loss,
            "val_build_time": val_build_time,
        }
    )
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(args_dict, handle, indent=4)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if best_epoch > 0:
        print(
            f"[{timestamp}] Finished. Saved final weights to {final_weights_path}. "
            f"Best val loss {best_val_loss:.4f} at epoch {best_epoch}."
        )
    else:
        print(f"[{timestamp}] Finished. Saved final weights to {final_weights_path}.")


if __name__ == "__main__":
    main()
