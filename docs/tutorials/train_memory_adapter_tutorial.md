# Tutorial: Train a Memory Adapter from Scratch

This tutorial walks through the full two-stage training pipeline on a small debug
dataset so you can verify your setup before launching full-scale training.

## Prerequisites

```bash
conda activate embench
pip install -e ".[qlora]"   # installs bitsandbytes
```

## Step 1 — Generate a Debug Dataset

```bash
python embodiedbench/scripts/build_preference_dataset.py \
    --episodes_dir data/episodes/eb_alfred \
    --output_dir   data/debug_training \
    --max_episodes 50
```

If you do not have real episode data yet, generate synthetic debug data:

```bash
python embodiedbench/scripts/build_preference_dataset.py \
    --synthetic \
    --num_examples 200 \
    --output_dir   data/debug_training
```

## Step 2 — SFT with Debug Config

```bash
python -m embodiedbench.memory_adapter_training.trainer \
    --config     embodiedbench/configs/memory_adapter_training/debug_tiny.yaml \
    --train_data data/debug_training/train.jsonl \
    --val_data   data/debug_training/val.jsonl \
    --output_dir /tmp/memadapt_sft_debug
```

Expected output:
```
Epoch 1/1: loss=1.42 → 0.81
Saved checkpoint to /tmp/memadapt_sft_debug/checkpoint-final
```

## Step 3 — GRPO with Debug Config

```bash
python embodiedbench/scripts/train_memory_adapter_grpo.py \
    --config          embodiedbench/configs/memory_adapter_rl/debug_grpo_tiny.yaml \
    --sft_checkpoint  /tmp/memadapt_sft_debug/checkpoint-final \
    --train_data      data/debug_training/train.jsonl \
    --output_dir      /tmp/memadapt_grpo_debug
```

Expected output:
```
GRPO iter 1/2: mean_reward=0.31 → 0.44
Saved checkpoint to /tmp/memadapt_grpo_debug/checkpoint-final
```

## Step 4 — Quick Inference Check

```python
from embodiedbench.memory_adapter import MemoryAdapter
from embodiedbench.memory_adapter.config import MemoryAdapterConfig

cfg = MemoryAdapterConfig(
    model_name_or_path="/tmp/memadapt_grpo_debug/checkpoint-final",
    load_in_4bit=False,   # CPU inference
)
adapter = MemoryAdapter(cfg)

from embodiedbench.memory_adapter import MemoryAdapterInput
adapter_input = MemoryAdapterInput(
    task_instruction="Put the apple in the fridge.",
    memory_context=memory_manager.retrieve("apple fridge"),
)
output = adapter.adapt(adapter_input)
print(output.foresight_plan)
```

## Full-Scale Training

Once you have verified the debug run, switch to the full configs:

```bash
# SFT
python -m embodiedbench.memory_adapter_training.train_sft \
    --config embodiedbench/configs/memory_adapter_training/qwen3_14b.yaml \
    --output_dir outputs/memory_adapter_training/qwen3_14b

# GRPO
python embodiedbench/scripts/train_memory_adapter_grpo.py \
    --config embodiedbench/configs/memory_adapter_rl/qwen_grpo.yaml \
    --sft_checkpoint outputs/memory_adapter_training/qwen3_14b/checkpoint-final \
    --output_dir outputs/memory_adapter_rl/grpo_qwen7b
```

> **Single-GPU?** Use the Unsloth backend instead — see the *Unsloth Backend* section in
> [docs/sft_training.md](../sft_training.md).

