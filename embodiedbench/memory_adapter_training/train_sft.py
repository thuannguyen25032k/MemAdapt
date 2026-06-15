#!/usr/bin/env python
"""
memory_adapter_training/train_sft.py

Entry point for supervised fine-tuning (SFT) of the Memory Adapter.

Pipeline
--------
1. Load MemoryAdapterTrainingConfig from YAML (with optional CLI overrides).
2. Load the base causal-LM + tokenizer and wrap with a LoRA / QLoRA adapter.
3. Load the curated SFT dataset(s) (filter_sft_targets.py output) as
   {"prompt", "response"} pairs, with an optional automatic train/val split.
4. Train with the HF Trainer wrapper (loss only over the assistant response).
5. Save the LoRA adapter; optionally run generation-based evaluation.

Usage
-----
python -m embodiedbench.memory_adapter_training.train_sft \
    --config embodiedbench/configs/memory_adapter_training/qwen3_14b.yaml \
    --train_path memory_adapter_dataset/alfred_memory_logs/sft_filtered/sft_targets_filtered.jsonl \
                 memory_adapter_dataset/habitat_memory_logs/sft_filtered/sft_targets_filtered.jsonl
"""

from __future__ import annotations

import argparse
import os
from embodiedbench.memory_adapter_training.config import MemoryAdapterTrainingConfig
from embodiedbench.memory_adapter_training.dataset import (
    load_sft_records,
    make_hf_dataset,
    split_train_val,
)
from embodiedbench.memory_adapter_training.modeling import get_trainable_model
from embodiedbench.memory_adapter_training.trainer import MemoryAdapterTrainer
from embodiedbench.memory_adapter_training.utils import (
    count_parameters,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT training for the Memory Adapter.")
    p.add_argument("--config", required=True, help="Path to training-config YAML.")
    p.add_argument(
        "--train_path", nargs="+", default=None,
        help="One or more curated SFT JSONL files (overrides config.dataset.train_path).",
    )
    p.add_argument(
        "--val_path", default=None,
        help="Validation JSONL file (overrides config.dataset.val_path).",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Override config.training.output_dir.",
    )
    return p.parse_args()


def _load_datasets(cfg, train_paths, val_path):
    """Build (train_ds, val_ds) HF datasets from the resolved paths."""
    train_records = []
    for path in train_paths:
        train_records.extend(load_sft_records(path))

    val_records = []
    if val_path:
        val_records = load_sft_records(val_path)
    elif cfg.dataset.val_ratio > 0.0:
        train_records, val_records = split_train_val(
            train_records, cfg.dataset.val_ratio, cfg.training.seed
        )

    train_ds = make_hf_dataset(train_records)
    val_ds = make_hf_dataset(val_records) if val_records else None
    return train_ds, val_ds


def main() -> None:
    args = parse_args()
    cfg = MemoryAdapterTrainingConfig.from_yaml(args.config)

    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.val_path:
        cfg.dataset.val_path = args.val_path

    setup_logging(cfg.logging.log_level)
    set_seed(cfg.training.seed)

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    cfg.save_yaml(os.path.join(cfg.training.output_dir, "training_config.yaml"))

    # Resolve dataset paths (CLI > config).
    train_paths = args.train_path or [cfg.dataset.train_path]
    val_path = cfg.dataset.val_path or None
    train_ds, val_ds = _load_datasets(cfg, train_paths, val_path)
    # Model + tokenizer (+ LoRA).
    model, tokenizer = get_trainable_model(cfg)
    total, trainable = count_parameters(model)
    print(
        f"[train_sft] Parameters: total={total:,} trainable={trainable:,} "
        f"({100 * trainable / max(total, 1):.3f}%)"
    )
    print(f"[train_sft] Train samples: {len(train_ds)} | "
          f"Val samples: {len(val_ds) if val_ds is not None else 0}")

    # Train + save.
    trainer = MemoryAdapterTrainer(cfg, model, tokenizer, train_ds, val_ds)
    trainer.train()
    trainer.save(cfg.training.output_dir)
    print(f"[train_sft] Done. Adapter saved to {cfg.training.output_dir}")


if __name__ == "__main__":
    main()
