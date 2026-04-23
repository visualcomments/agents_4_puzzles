# `cayleypy-neighbour-model-training`

Public snapshot for Megaminx neighbour-model training around the released Q checkpoint `1776581286`.

This snapshot keeps:

- value-model training and inference code
- Q-model training and inference code
- the full public training/evaluation log set for the released lineage
- the canonical orchestration and comparison scripts
- only the two released checkpoints needed for inference and for reproducing the final Q lineage from the shipped teacher

This snapshot does not keep:

- epoch-by-epoch checkpoints such as `weights/*_e*.pth`
- unreleased stage checkpoints
- TensorBoard and cloudflared helpers
- unrelated experiments and archive docs

## Environment

- Python `3.10+`
- PyTorch installed separately for your exact CPU/CUDA environment

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# install torch first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

Optional notebook extras:

```bash
pip install -r requirements-notebook.txt
```

## Repository Layout

- `train.py`: value-model training
- `train_q.py`: Q-model training
- `test.py`: beam-search inference and evaluation
- `scripts/run_training_plan.py`: full value-to-Q release pipeline
- `scripts/compare_models.py`: fixed value-vs-Q benchmark runner and aggregator
- `notebooks/compare_models.ipynb`: notebook for plotting benchmark logs from `test.py`
- `pilgrim/`: models, searchers, and training helpers
- `generators/p900.json`: Megaminx move table
- `targets/p900-t000.pt`: solved target state
- `datasets/p900-t000-rnd.pt`: fixed random-walk benchmark dataset
- `logs/`: metadata, train curves, training-plan reports, benchmark runs, and comparison reports for the released lineage
- `weights/p900-t000_1776548012_best.pth`: released value-model teacher checkpoint
- `weights/p900-t000-q_1776581286_best.pth`: released Q-model checkpoint

The public snapshot intentionally ships logs for the full released lineage, but only the final released teacher and Q checkpoints.

## Released Lineage

The released chain documented in `logs/` is:

```text
1776528010_best -> 1776548012_best -> 1776553010_best -> 1776568728_best -> 1776581286_best
```

Included lineage logs:

- `logs/model_p900-t000_1776528010.json`
- `logs/model_p900-t000_1776548012.json`
- `logs/model_p900-t000-q_1776553010.json`
- `logs/model_p900-t000-q_1776568728.json`
- `logs/model_p900-t000-q_1776581286.json`
- matching `logs/train_*.csv`
- `logs/training_plan_20260418-190005.json`

Included benchmark logs:

- `logs/test_p900-t000-rnd-k1000-s42_1776548012_best_B*_compile.json`
- `logs/test_p900-t000-q-rnd-k1000-s42_1776581286_best_B*_compile.json`
- `logs/compare_p_20260420-163526_900-t000_rnd-k1000-s42_n30_Eb16384_1776548012_vs_1776581286.json`

## Reproduce The Released Teacher

The public snapshot ships the released teacher checkpoint `weights/p900-t000_1776548012_best.pth`. If you want to retrain a compatible teacher from scratch, reproduce the two value stages below.

Stage 1:

```bash
python3 train.py \
  --group_id 900 \
  --target_id 0 \
  --epochs 2048 \
  --batch_size 10000 \
  --lr 1e-3 \
  --K_min 0 \
  --K_max 50 \
  --val_walkers 512 \
  --gpu_ids 0 \
  --hd1 1536 \
  --hd2 512 \
  --nrd 2
```

Stage 2:

```bash
python3 train.py \
  --group_id 900 \
  --target_id 0 \
  --epochs 1024 \
  --batch_size 20000 \
  --lr 2e-5 \
  --K_min 0 \
  --K_max 65 \
  --val_walkers 512 \
  --weights p900-t000_<STAGE1_ID>_best \
  --gpu_ids 0 \
  --hd1 1536 \
  --hd2 512 \
  --nrd 2
```

For the released Q lineage in this repo, you do not need to retrain the teacher; you can start directly from the shipped `p900-t000_1776548012_best.pth`.

## Reproduce Training For `1776581286`

Canonical full-cycle entrypoint:

```bash
python3 scripts/run_training_plan.py --gpu_ids 0
```

Tiny end-to-end smoke run:

```bash
python3 scripts/run_training_plan.py \
  --gpu_ids 0 \
  --value_stage1_epochs 1 \
  --value_stage2_epochs 1 \
  --q_stage1_epochs 1 \
  --q_stage2_epochs 1 \
  --q_stage3_epochs 1 \
  --value_stage1_train_walkers 16 \
  --value_stage2_train_walkers 16 \
  --value_stage1_val_walkers -1 \
  --value_stage2_val_walkers -1 \
  --q_steps_per_epoch 1 \
  --q_batch_size 2 \
  --q_teacher_batch_size 32 \
  --q_val_size 0 \
  --q_val_batch_size 2
```

Q stage 1:

```bash
python3 train_q.py \
  --group_id 900 \
  --target_id 0 \
  --epochs 4096 \
  --steps_per_epoch 16 \
  --batch_size 2048 \
  --lr 1e-4 \
  --K_min 1 \
  --K_max 50 \
  --teacher_model_id 1776548012 \
  --teacher_batch_size 65536 \
  --weights p900-t000_1776548012_best \
  --val_size 65536 \
  --val_batch_size 2048 \
  --gpu_ids 0 \
  --hd1 2048 \
  --hd2 768 \
  --nrd 4
```

Q stage 2:

```bash
python3 train_q.py \
  --group_id 900 \
  --target_id 0 \
  --epochs 4096 \
  --steps_per_epoch 16 \
  --batch_size 2048 \
  --lr 5e-5 \
  --K_min 1 \
  --K_max 50 \
  --teacher_model_id 1776548012 \
  --teacher_batch_size 65536 \
  --weights p900-t000-q_<STAGE1_ID>_best \
  --val_size 65536 \
  --val_batch_size 2048 \
  --gpu_ids 0 \
  --hd1 2048 \
  --hd2 768 \
  --nrd 4
```

Q stage 3:

```bash
python3 train_q.py \
  --group_id 900 \
  --target_id 0 \
  --epochs 4096 \
  --steps_per_epoch 16 \
  --batch_size 2048 \
  --lr 2e-5 \
  --K_min 1 \
  --K_max 65 \
  --teacher_model_id 1776548012 \
  --teacher_batch_size 65536 \
  --weights p900-t000-q_<STAGE2_ID>_best \
  --val_size 65536 \
  --val_batch_size 2048 \
  --gpu_ids 0 \
  --hd1 2048 \
  --hd2 768 \
  --nrd 4
```

Notes:

- `--gpu_ids` accepts comma-separated CUDA ids such as `0` or `0,1`
- `train_q.py` keeps only the released teacher-distillation path
- each training stage writes metadata to `logs/model_*.json` and curves to `logs/train_*.csv`

## Inference

Run evaluation for the released Q checkpoint:

```bash
python3 test.py \
  --group_id 900 \
  --target_id 0 \
  --rnd_depth 1000 \
  --rnd_seed 42 \
  --tests_num 10 \
  --model_id 1776581286 \
  --best \
  --B 65536 \
  --eval_batch_size 65536 \
  --gpu_ids 0
```

Single-GPU `torch.compile` inference:

```bash
python3 test.py \
  --group_id 900 \
  --target_id 0 \
  --rnd_depth 1000 \
  --rnd_seed 42 \
  --tests_num 10 \
  --model_id 1776581286 \
  --best \
  --B 65536 \
  --eval_batch_size 65536 \
  --gpu_ids 0 \
  --compile
```

Run the released value-vs-Q benchmark sweep:

```bash
python3 scripts/compare_models.py \
  --value_model_id 1776548012 \
  --q_model_id 1776581286 \
  --tests_num 30 \
  --rnd_depth 1000 \
  --rnd_seed 42 \
  --eval_batch_size 16384 \
  --gpu_ids 0 \
  --compile
```
