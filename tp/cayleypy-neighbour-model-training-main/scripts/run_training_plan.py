import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pilgrim.parallel import gpu_ids_to_cli_args

LOG_DIR = PACKAGE_ROOT / "logs"
STAGE_ORDER = ["value_stage1", "value_stage2", "q_stage1", "q_stage2", "q_stage3"]


def sanitize_cli_token(token):
    text = str(token)
    if "/" in text or "\\" in text:
        return Path(text).name
    return text


def sanitize_command(command):
    return [sanitize_cli_token(token) if idx == 0 else str(token) for idx, token in enumerate(command)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full value-to-neighbour training plan for the Megaminx release pipeline."
    )
    parser.add_argument("--group_id", type=int, default=900, help="Puzzle group id.")
    parser.add_argument("--target_id", type=int, default=0, help="Target id.")
    parser.add_argument("--gpu_ids", type=str, help="Comma-separated CUDA ids, e.g. '0,1'.")
    parser.add_argument("--python", type=str, default=sys.executable or "python3", help="Python executable.")
    parser.add_argument("--start_stage", type=str, choices=STAGE_ORDER, default=STAGE_ORDER[0], help="Start the pipeline from this stage.")
    parser.add_argument("--report_json", type=Path, help="Optional pipeline report JSON path.")

    parser.add_argument("--value_stage1_id", type=int, help="Existing model_id for value stage 1.")
    parser.add_argument("--value_stage2_id", type=int, help="Existing model_id for value stage 2.")
    parser.add_argument("--q_stage1_id", type=int, help="Existing model_id for Q stage 1.")
    parser.add_argument("--q_stage2_id", type=int, help="Existing model_id for Q stage 2.")
    parser.add_argument("--q_stage3_id", type=int, help="Existing model_id for Q stage 3.")

    parser.add_argument("--value_stage1_epochs", type=int, default=2048, help="Epochs for value stage 1.")
    parser.add_argument("--value_stage2_epochs", type=int, default=256, help="Epochs for value stage 2.")
    parser.add_argument("--q_stage1_epochs", type=int, default=4096, help="Epochs for Q stage 1.")
    parser.add_argument("--q_stage2_epochs", type=int, default=4096, help="Epochs for Q stage 2.")
    parser.add_argument("--q_stage3_epochs", type=int, default=4096, help="Epochs for Q stage 3.")

    parser.add_argument("--value_stage1_batch_size", type=int, default=10000, help="Batch size for value stage 1.")
    parser.add_argument("--value_stage2_batch_size", type=int, default=20000, help="Batch size for value stage 2.")
    parser.add_argument("--value_stage1_train_walkers", type=int, default=0, help="Training walkers per depth for value stage 1. 0 uses the default schedule.")
    parser.add_argument("--value_stage2_train_walkers", type=int, default=0, help="Training walkers per depth for value stage 2. 0 uses the default schedule.")
    parser.add_argument("--value_stage1_val_walkers", type=int, default=512, help="Validation walkers per depth for value stage 1.")
    parser.add_argument("--value_stage2_val_walkers", type=int, default=512, help="Validation walkers per depth for value stage 2.")
    parser.add_argument("--value_stage1_lr", type=float, default=1e-3, help="Learning rate for value stage 1.")
    parser.add_argument("--value_stage2_lr", type=float, default=2e-5, help="Learning rate for value stage 2.")
    parser.add_argument("--value_stage1_k_min", type=int, default=0, help="Minimum random-walk depth for value stage 1.")
    parser.add_argument("--value_stage1_k_max", type=int, default=50, help="Maximum random-walk depth for value stage 1.")
    parser.add_argument("--value_stage2_k_min", type=int, default=0, help="Minimum random-walk depth for value stage 2.")
    parser.add_argument("--value_stage2_k_max", type=int, default=65, help="Maximum random-walk depth for value stage 2.")

    parser.add_argument("--q_steps_per_epoch", type=int, default=64, help="Optimizer steps per epoch for all Q stages.")
    parser.add_argument("--q_batch_size", type=int, default=2048, help="States per optimizer step for all Q stages.")
    parser.add_argument("--q_teacher_batch_size", type=int, default=65536, help="Teacher batch size for all Q stages.")
    parser.add_argument("--q_val_size", type=int, default=65536, help="Validation set size for all Q stages.")
    parser.add_argument("--q_val_batch_size", type=int, default=2048, help="Validation batch size for all Q stages.")
    parser.add_argument("--q_stage1_lr", type=float, default=1e-4, help="Learning rate for Q stage 1.")
    parser.add_argument("--q_stage2_lr", type=float, default=5e-5, help="Learning rate for Q stage 2.")
    parser.add_argument("--q_stage3_lr", type=float, default=2e-5, help="Learning rate for Q stage 3.")
    parser.add_argument("--q_stage1_k_min", type=int, default=1, help="Minimum random-walk depth for Q stage 1.")
    parser.add_argument("--q_stage1_k_max", type=int, default=50, help="Maximum random-walk depth for Q stage 1.")
    parser.add_argument("--q_stage2_k_min", type=int, default=1, help="Minimum random-walk depth for Q stage 2.")
    parser.add_argument("--q_stage2_k_max", type=int, default=50, help="Maximum random-walk depth for Q stage 2.")
    parser.add_argument("--q_stage3_k_min", type=int, default=1, help="Minimum random-walk depth for Q stage 3.")
    parser.add_argument("--q_stage3_k_max", type=int, default=65, help="Maximum random-walk depth for Q stage 3.")

    return parser.parse_args()


def print_args(args):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] [run_training_plan] pipeline config:", flush=True)
    for key, value in sorted(vars(args).items()):
        print(f"  {key:<28} {value}", flush=True)


def default_report_path():
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return LOG_DIR / f"training_plan_{timestamp}.json"


def stage_index(stage_name):
    return STAGE_ORDER.index(stage_name)


def metadata_regex(q_model):
    if q_model:
        return re.compile(r"^model_p\d{3}-t\d{3}-q_(\d+)\.json$")
    return re.compile(r"^model_p\d{3}-t\d{3}_(\d+)\.json$")


def snapshot_metadata(group_id, target_id, q_model):
    base = f"model_p{int(group_id):03d}-t{int(target_id):03d}"
    if q_model:
        candidates = sorted(LOG_DIR.glob(f"{base}-q_*.json"))
    else:
        candidates = sorted(path for path in LOG_DIR.glob(f"{base}_*.json") if f"{base}-q_" not in path.name)

    pattern = metadata_regex(q_model=q_model)
    result = {}
    for path in candidates:
        match = pattern.match(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def resolve_existing_metadata(group_id, target_id, model_id, q_model):
    snapshot = snapshot_metadata(group_id=group_id, target_id=target_id, q_model=q_model)
    if int(model_id) not in snapshot:
        kind = "Q" if q_model else "value"
        raise FileNotFoundError(f"{kind} metadata for model_id={model_id} not found in logs/")
    return snapshot[int(model_id)]


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_summary_from_metadata(path, stage_name, status):
    info = load_json(path)
    return {
        "stage": stage_name,
        "status": status,
        "model_id": int(info["model_id"]),
        "model_name": info.get("model_name"),
        "training_mode": info.get("training_mode"),
        "metadata_path": str(path.relative_to(PACKAGE_ROOT)),
        "best_weights_file": info.get("best_weights_file"),
        "final_weights_file": info.get("final_weights_file"),
        "best_epoch": info.get("best_epoch"),
        "best_val_loss": info.get("best_val_loss"),
    }


def maybe_load_existing(stage_name, model_id, group_id, target_id, q_model):
    if model_id is None:
        return None
    meta_path = resolve_existing_metadata(group_id=group_id, target_id=target_id, model_id=model_id, q_model=q_model)
    return model_summary_from_metadata(meta_path, stage_name=stage_name, status="existing")


def jsonable_config(args):
    result = {}
    for key, value in vars(args).items():
        if key == "python":
            result[key] = sanitize_cli_token(value)
        else:
            result[key] = str(value) if isinstance(value, Path) else value
    return result


def checkpoint_name_from_stage(stage):
    weights_file = stage.get("best_weights_file") or stage.get("final_weights_file")
    if not weights_file:
        raise ValueError("stage is missing both best_weights_file and final_weights_file")
    return Path(weights_file).stem


def command_options(command):
    options = []
    i = 2
    while i < len(command):
        token = str(command[i])
        if token.startswith("--"):
            key = token[2:]
            if i + 1 < len(command) and not str(command[i + 1]).startswith("--"):
                options.append((key, command[i + 1]))
                i += 2
            else:
                options.append((key, True))
                i += 1
        else:
            options.append((f"arg_{i}", command[i]))
            i += 1
    return options


def print_stage_start(stage_name, command):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] [run_training_plan] starting {stage_name}", flush=True)
    for key, value in command_options(command):
        print(f"  {key:<18} {value}", flush=True)


def run_stage(stage_name, command, group_id, target_id, q_model):
    before = snapshot_metadata(group_id=group_id, target_id=target_id, q_model=q_model)
    started_at = time.time()
    print_stage_start(stage_name, command)
    subprocess.run(command, cwd=PACKAGE_ROOT, check=True)
    after = snapshot_metadata(group_id=group_id, target_id=target_id, q_model=q_model)

    new_ids = sorted(set(after) - set(before))
    if len(new_ids) != 1:
        raise RuntimeError(f"expected exactly one new metadata file after {stage_name}, got {new_ids}")

    meta_path = after[new_ids[0]]
    result = model_summary_from_metadata(meta_path, stage_name=stage_name, status="trained")
    result["command"] = sanitize_command(command)
    result["duration_sec"] = round(time.time() - started_at, 2)
    return result


def ensure_required_ids(args):
    required = {
        "value_stage2": ["value_stage1_id"],
        "q_stage1": ["value_stage2_id"],
        "q_stage2": ["value_stage2_id", "q_stage1_id"],
        "q_stage3": ["value_stage2_id", "q_stage2_id"],
    }
    missing = [name for name in required.get(args.start_stage, []) if getattr(args, name) is None]
    if missing:
        raise ValueError(f"{args.start_stage} requires: {', '.join('--' + name for name in missing)}")


def value_command(args, epochs, batch_size, train_walkers, val_walkers, lr, k_min, k_max, weights=None):
    command = [
        args.python,
        "train.py",
        "--group_id",
        str(args.group_id),
        "--target_id",
        str(args.target_id),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--dropout",
        "0.0",
        "--K_min",
        str(k_min),
        "--K_max",
        str(k_max),
        "--train_walkers",
        str(train_walkers),
        "--val_walkers",
        str(val_walkers),
        *gpu_ids_to_cli_args(args.gpu_ids),
        "--hd1",
        "1536",
        "--hd2",
        "512",
        "--nrd",
        "2",
    ]
    if weights is not None:
        command.extend(["--weights", str(weights)])
    return command


def q_command(args, teacher_model_id, epochs, lr, k_min, k_max, init_weights):
    return [
        args.python,
        "train_q.py",
        "--group_id",
        str(args.group_id),
        "--target_id",
        str(args.target_id),
        "--epochs",
        str(epochs),
        "--steps_per_epoch",
        str(args.q_steps_per_epoch),
        "--batch_size",
        str(args.q_batch_size),
        "--lr",
        str(lr),
        "--weight_decay",
        "0.01",
        "--grad_clip",
        "1.0",
        "--freeze_bn_after",
        "1",
        "--K_min",
        str(k_min),
        "--K_max",
        str(k_max),
        "--teacher_model_id",
        str(teacher_model_id),
        "--teacher_batch_size",
        str(args.q_teacher_batch_size),
        "--val_size",
        str(args.q_val_size),
        "--val_batch_size",
        str(args.q_val_batch_size),
        "--weights",
        str(init_weights),
        *gpu_ids_to_cli_args(args.gpu_ids),
        "--seed",
        "42",
        "--hd1",
        "2048",
        "--hd2",
        "768",
        "--nrd",
        "4",
        "--dropout",
        "0.0",
    ]


def main():
    args = parse_args()
    ensure_required_ids(args)
    print_args(args)

    report_path = args.report_json.expanduser().resolve() if args.report_json else default_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    stages = {
        "value_stage1": maybe_load_existing("value_stage1", args.value_stage1_id, args.group_id, args.target_id, False),
        "value_stage2": maybe_load_existing("value_stage2", args.value_stage2_id, args.group_id, args.target_id, False),
        "q_stage1": maybe_load_existing("q_stage1", args.q_stage1_id, args.group_id, args.target_id, True),
        "q_stage2": maybe_load_existing("q_stage2", args.q_stage2_id, args.group_id, args.target_id, True),
        "q_stage3": maybe_load_existing("q_stage3", args.q_stage3_id, args.group_id, args.target_id, True),
    }

    started_at = time.time()
    start_idx = stage_index(args.start_stage)

    if start_idx <= stage_index("value_stage1"):
        command = value_command(
            args=args,
            epochs=args.value_stage1_epochs,
            batch_size=args.value_stage1_batch_size,
            train_walkers=args.value_stage1_train_walkers,
            val_walkers=args.value_stage1_val_walkers,
            lr=args.value_stage1_lr,
            k_min=args.value_stage1_k_min,
            k_max=args.value_stage1_k_max,
        )
        stages["value_stage1"] = run_stage("value_stage1", command, args.group_id, args.target_id, False)

    if start_idx <= stage_index("value_stage2"):
        parent = stages["value_stage1"]
        if parent is None:
            raise RuntimeError("value_stage1 result is unavailable")
        command = value_command(
            args=args,
            epochs=args.value_stage2_epochs,
            batch_size=args.value_stage2_batch_size,
            train_walkers=args.value_stage2_train_walkers,
            val_walkers=args.value_stage2_val_walkers,
            lr=args.value_stage2_lr,
            k_min=args.value_stage2_k_min,
            k_max=args.value_stage2_k_max,
            weights=checkpoint_name_from_stage(parent),
        )
        stages["value_stage2"] = run_stage("value_stage2", command, args.group_id, args.target_id, False)

    if start_idx <= stage_index("q_stage1"):
        teacher = stages["value_stage2"]
        if teacher is None:
            raise RuntimeError("value_stage2 result is unavailable")
        command = q_command(
            args=args,
            teacher_model_id=int(teacher["model_id"]),
            epochs=args.q_stage1_epochs,
            lr=args.q_stage1_lr,
            k_min=args.q_stage1_k_min,
            k_max=args.q_stage1_k_max,
            init_weights=checkpoint_name_from_stage(teacher),
        )
        stages["q_stage1"] = run_stage("q_stage1", command, args.group_id, args.target_id, True)

    if start_idx <= stage_index("q_stage2"):
        teacher = stages["value_stage2"]
        parent = stages["q_stage1"]
        if teacher is None or parent is None:
            raise RuntimeError("q_stage2 dependencies are unavailable")
        command = q_command(
            args=args,
            teacher_model_id=int(teacher["model_id"]),
            epochs=args.q_stage2_epochs,
            lr=args.q_stage2_lr,
            k_min=args.q_stage2_k_min,
            k_max=args.q_stage2_k_max,
            init_weights=checkpoint_name_from_stage(parent),
        )
        stages["q_stage2"] = run_stage("q_stage2", command, args.group_id, args.target_id, True)

    if start_idx <= stage_index("q_stage3"):
        teacher = stages["value_stage2"]
        parent = stages["q_stage2"]
        if teacher is None or parent is None:
            raise RuntimeError("q_stage3 dependencies are unavailable")
        command = q_command(
            args=args,
            teacher_model_id=int(teacher["model_id"]),
            epochs=args.q_stage3_epochs,
            lr=args.q_stage3_lr,
            k_min=args.q_stage3_k_min,
            k_max=args.q_stage3_k_max,
            init_weights=checkpoint_name_from_stage(parent),
        )
        stages["q_stage3"] = run_stage("q_stage3", command, args.group_id, args.target_id, True)

    report = {
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "duration_sec": round(time.time() - started_at, 2),
        "config": jsonable_config(args),
        "stages": stages,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[run_training_plan] wrote report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
