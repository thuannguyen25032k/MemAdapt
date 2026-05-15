# SFT Training

## Overview

Stage 1 of MemAdapt training fine-tunes Qwen2.5-7B-Instruct on the hindsight-annotated
dataset using QLoRA (4-bit NF4 base + LoRA adapters).  This teaches the model to
produce well-formed XML adapter outputs and baseline staleness-reasoning capability
before GRPO refinement in Stage 2.

Stage 1 SFT uses the `memory_adapter_training` module with configs under
`embodiedbench/configs/memory_adapter_training/`.  The `train_memory_adapter_dpo.py`
script handles preference-optimisation variants (DPO / ORPO / SimPO) and is **not**
involved in Stage 1.

## Quick Start

```bash
python -c "
from embodiedbench.memory_adapter_training.config import MemoryAdapterTrainingConfig
from embodiedbench.memory_adapter_training.modeling import build_model_and_tokenizer
from embodiedbench.memory_adapter_training.dataset import load_sft_dataset
from embodiedbench.memory_adapter_training.trainer import MemoryAdapterTrainer

cfg = MemoryAdapterTrainingConfig.from_yaml(
    'embodiedbench/configs/memory_adapter_training/qwen_qlora.yaml'
)
model, tokenizer = build_model_and_tokenizer(cfg)
train_ds = load_sft_dataset(cfg.dataset.train_path, tokenizer, cfg)
val_ds   = load_sft_dataset(cfg.dataset.val_path,   tokenizer, cfg)
trainer  = MemoryAdapterTrainer(cfg, model, tokenizer, train_ds, val_ds)
trainer.train()
"
```

Or run via the convenience launcher:

```bash
python -m embodiedbench.memory_adapter_training.trainer \
    --config embodiedbench/configs/memory_adapter_training/qwen_qlora.yaml \
    --output_dir outputs/memory_adapter_training/qwen_qlora
```

## Config Reference

```yaml
# embodiedbench/configs/memory_adapter_training/qwen_qlora.yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
  trust_remote_code: true
  use_flash_attention: true
  load_in_4bit: true
  torch_dtype: bfloat16

dataset:
  train_path: "data/memory_adapter_training/train.jsonl"
  val_path:   "data/memory_adapter_training/val.jsonl"
  max_seq_length: 2048
  dataset_format: hf_sft
  text_field: text

training:
  output_dir: "outputs/memory_adapter_training/qwen_qlora"
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8   # effective batch = 8
  learning_rate: 2.0e-4
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  bf16: true
  gradient_checkpointing: true
  seed: 42

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  task_type: CAUSAL_LM
  target_modules: [q_proj, k_proj, v_proj, o_proj]
```

## Debug Run (no GPU required)

```bash
python -m embodiedbench.memory_adapter_training.trainer \
    --config embodiedbench/configs/memory_adapter_training/debug_tiny.yaml \
    --output_dir /tmp/memadapt_sft_debug
```

## Hardware Requirements

| Config | GPU | VRAM | Time |
|---|---|---|---|
| `qwen_qlora.yaml` | A100 80 GB | ~16 GB (4-bit) | ~4 h |
| `debug_tiny.yaml` | CPU | — | ~2 min |

## Checkpoint Layout

```
outputs/memory_adapter_training/qwen_qlora/
├── checkpoint-epoch-1/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── trainer_state.json
├── checkpoint-final/    ← use this for GRPO
└── training_log.jsonl
```
