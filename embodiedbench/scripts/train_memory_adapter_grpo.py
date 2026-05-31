#!/usr/bin/env python
"""
scripts/train_memory_adapter_grpo.py

GRPO refinement training script for the Memory Adapter.

Usage
-----
python scripts/train_memory_adapter_grpo.py \\
    --config configs/memory_adapter_rl/qwen_grpo.yaml \\
    --sft_checkpoint outputs/memory_adapter_training/qwen_qlora/checkpoint-final \\
    --output_dir outputs/memory_adapter_rl/grpo_qwen7b

The script:
1. Loads RLConfig from YAML (algorithm must be "grpo").
2. Loads model + tokenizer (with optional QLoRA / 4-bit support).
3. Loads SFT adapter checkpoint if provided.
4. Loads prompt dataset from train_data_path (JSONL list of prompt strings).
5. Runs MemoryAdapterGRPOTrainer (TRL GRPOTrainer or custom loop fallback).
6. Saves GRPO adapter to output_dir.
7. Writes training metrics to output_dir/grpo_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Make sure the repo root is importable when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from embodiedbench.memory_adapter_rl.config import RLConfig
from embodiedbench.memory_adapter_rl.trainer import MemoryAdapterGRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_grpo")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO refinement training for the MemAdapt Memory Adapter."
    )
    p.add_argument(
        "--config", required=True,
        help="Path to RLConfig YAML (algorithm must be 'grpo').",
    )
    p.add_argument(
        "--sft_checkpoint", default=None,
        help="Path to SFT adapter checkpoint to initialise from.",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Override output_dir from config.",
    )
    p.add_argument(
        "--train_data_path", default=None,
        help="Override train_data_path from config.",
    )
    p.add_argument(
        "--num_train_epochs", type=int, default=None,
        help="Override num_train_epochs.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Build trainer and exit without training (for config validation).",
    )
    # W&B overrides
    p.add_argument("--wandb_enabled", action="store_true", default=None,
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", default=None, help="W&B project name.")
    p.add_argument("--wandb_entity", default=None, help="W&B entity (user/team).")
    p.add_argument("--wandb_group", default=None, help="W&B run group.")
    p.add_argument("--wandb_tags", default=None,
                   help="Comma-separated W&B tags, e.g. 'grpo,qwen'.")
    p.add_argument("--wandb_mode", default=None,
                   help="W&B mode: online | offline | disabled.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> list[str]:
    """
    Load prompt strings from a JSONL file.

    Each line should be either:
    - A JSON string: "prompt text here"
    - A JSON object with a 'prompt' key: {"prompt": "...", ...}
    """
    prompts = []
    if not path or not os.path.exists(path):
        logger.warning("train_data_path '%s' not found; using empty prompt list.", path)
        return prompts

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                prompts.append(obj)
            elif isinstance(obj, dict):
                prompts.append(obj.get("prompt", obj.get("text", str(obj))))
            else:
                prompts.append(str(obj))
    logger.info("Loaded %d prompts from %s", len(prompts), path)
    return prompts


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: RLConfig):
    """Load causal-LM + tokenizer; apply 4-bit quantization if requested."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
        import torch  # type: ignore

        logger.info("Loading tokenizer: %s", cfg.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = None
        if cfg.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info("Loading model: %s", cfg.model_name_or_path)
        dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            device_map="auto",
        )
        return model, tokenizer

    except ImportError as exc:
        logger.error("transformers / torch not available: %s", exc)
        raise


def apply_sft_checkpoint(model, sft_checkpoint: str):
    """Load SFT LoRA weights and merge them into the base model.

    The SFT adapter is baked into the base weights (merge_and_unload) so a fresh,
    trainable GRPO LoRA adapter can be stacked on top in ``wrap_with_lora``.
    """
    if not sft_checkpoint or not os.path.exists(sft_checkpoint):
        logger.info("No SFT checkpoint; starting from base model weights.")
        return model
    try:
        from peft import PeftModel  # type: ignore
        logger.info("Loading SFT adapter from %s", sft_checkpoint)
        model = PeftModel.from_pretrained(model, sft_checkpoint)
        model = model.merge_and_unload()
        logger.info("SFT adapter merged into base weights.")
        return model
    except Exception as exc:
        logger.warning("Failed to load SFT adapter (%s); continuing with base model.", exc)
        return model


def wrap_with_lora(model, cfg: RLConfig):
    """Attach a fresh, trainable LoRA adapter for GRPO refinement."""
    try:
        from peft import get_peft_model  # type: ignore
        from embodiedbench.memory_adapter_rl.trainer import build_lora_config

        # Required so gradients flow through a gradient-checkpointed base model.
        if cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        model = get_peft_model(model, build_lora_config(cfg))
        model.print_trainable_parameters()
        return model
    except ImportError as exc:
        logger.error("peft not available: %s", exc)
        raise



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Load config ----
    logger.info("Loading config: %s", args.config)
    cfg = RLConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.sft_checkpoint:
        cfg.sft_checkpoint = args.sft_checkpoint
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.train_data_path:
        cfg.train_data_path = args.train_data_path
    if args.num_train_epochs is not None:
        cfg.num_train_epochs = args.num_train_epochs

    # Apply W&B CLI overrides
    if args.wandb_enabled:
        cfg.wandb.enabled = True
    if args.wandb_project:
        cfg.wandb.project = args.wandb_project
    if args.wandb_entity:
        cfg.wandb.entity = args.wandb_entity
    if args.wandb_group:
        cfg.wandb.group = args.wandb_group
    if args.wandb_tags:
        cfg.wandb.tags = [t.strip() for t in args.wandb_tags.split(",")]
    if args.wandb_mode:
        cfg.wandb.mode = args.wandb_mode

    if cfg.algorithm.lower() != "grpo":
        raise ValueError(
            f"Config algorithm is '{cfg.algorithm}', expected 'grpo'."
        )

    logger.info("Run: %s | Algorithm: GRPO | Output: %s", cfg.run_name, cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- Load model ----
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ---- Apply SFT checkpoint ----
    if cfg.sft_checkpoint:
        model = apply_sft_checkpoint(model, cfg.sft_checkpoint)

    # ---- Attach a trainable GRPO LoRA adapter ----
    model = wrap_with_lora(model, cfg)

    # ---- Load prompts ----
    train_prompts = load_prompts(cfg.train_data_path)
    eval_prompts = load_prompts(cfg.val_data_path) if cfg.val_data_path else None

    if not train_prompts:
        logger.warning(
            "No training prompts found. Running with empty list "
            "(custom loop will produce zero-sample metrics)."
        )

    # ---- Build trainer ----
    trainer = MemoryAdapterGRPOTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_prompts=train_prompts,
        eval_prompts=eval_prompts,
    )
    trainer.build()

    if args.dry_run:
        logger.info("--dry_run: trainer built successfully. Exiting.")
        return

    # ---- Train ----
    logger.info("Starting GRPO training …")
    train_metrics = trainer.train()
    logger.info("Training complete. Metrics: %s", train_metrics)

    # ---- Evaluate ----
    eval_metrics: dict = {}
    if eval_prompts or cfg.eval_format_validity:
        logger.info("Running evaluation …")
        eval_metrics = trainer.evaluate()
        logger.info("Eval metrics: %s", eval_metrics)

    # ---- Save ----
    save_path = trainer.save()
    logger.info("GRPO adapter saved to %s", save_path)

    # ---- Write metrics ----
    all_metrics = {**train_metrics, **eval_metrics}
    metrics_path = os.path.join(cfg.output_dir, "grpo_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info("Metrics written to %s", metrics_path)


if __name__ == "__main__":
    main()
