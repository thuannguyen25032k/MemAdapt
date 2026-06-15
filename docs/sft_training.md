# SFT Training

## Overview

Stage 1 of MemAdapt training fine-tunes **Qwen3-14B** on the filtered adapter-target
dataset using QLoRA (4-bit NF4 base + LoRA adapters). This teaches the model to
produce well-formed XML adapter outputs (`FORESIGHT_PLAN`, `FEASIBILITY_CRITERIA`,
`FALLBACK_STRATEGY`) before GRPO refinement in Stage 2.

The training prompt is split into the **same** system + user chat messages used at
inference (`build_adapter_messages` / `build_adapter_user_content`): the fixed
adapter instructions form the `system` turn, and the task instruction plus retrieved
memory form the `user` turn. The data collator applies the model's chat template
(`apply_chat_template(..., add_generation_prompt=True)`), so training and deployment
are byte-for-byte aligned. Loss is computed only over the assistant response.

Stage 1 SFT uses the `memory_adapter_training` module with configs under
`embodiedbench/configs/memory_adapter_training/`.

## Quick Start

```bash
python -m embodiedbench.memory_adapter_training.train_sft \
    --config embodiedbench/configs/memory_adapter_training/qwen3_14b.yaml \
    --train_path memory_adapter_dataset/alfred_memory_logs/sft_filtered/sft_targets_filtered.jsonl \
                 memory_adapter_dataset/habitat_memory_logs/sft_filtered/sft_targets_filtered.jsonl \
    --output_dir outputs/memory_adapter_training/qwen3_14b
```

The `--train_path` files are produced by `filter_sft_targets.py`, which removes any
plan that degrades the task progress of either the expert (38B) or novice (14B)
planner. When `dataset.val_path` is empty, `dataset.val_ratio` triggers an automatic
hold-out split.

## Config Reference

```yaml
# embodiedbench/configs/memory_adapter_training/qwen3_14b.yaml
model:
  model_name_or_path: "embodiedbench/memory_adapter/models/Qwen3-14B"
  trust_remote_code: true
  use_flash_attention: true
  load_in_4bit: false         # set true for QLoRA (halves VRAM)
  torch_dtype: bfloat16
  enable_thinking: false      # must match inference

dataset:
  train_path:
    - "memory_adapter_dataset/alfred_memory_logs/sft_filtered/sft_targets_filtered.jsonl"
    - "memory_adapter_dataset/habitat_memory_logs/sft_filtered/sft_targets_filtered.jsonl"
  val_path:   ""
  val_ratio:  0.1             # auto split when val_path is empty
  max_seq_length: 8192

training:
  output_dir: "outputs/memory_adapter_training/qwen3_14b"
  num_train_epochs: 6
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-4
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
  bias: none
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

## Checkpoint Layout

```
outputs/memory_adapter_training/qwen3_14b/
├── checkpoint-50/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── trainer_state.json
├── checkpoint-100/
└── training_config.yaml
```

The final adapter is also saved directly under `output_dir` by `trainer.save()`.

