#!/usr/bin/env python
"""
scripts/merge_adapter.py

Merge a LoRA adapter into its base model and save the full merged weights.

The merged model can be loaded as a plain HuggingFace ``AutoModelForCausalLM``
(no PEFT required) and is ready for lmdeploy / vLLM serving or further
export (GGUF, AWQ, etc.).

Usage
-----
# Standard HuggingFace / PEFT merge
python embodiedbench/scripts/merge_adapter.py \
    --base_model  embodiedbench/memory_adapter/models/Qwen3-14B \
    --adapter_dir outputs/memory_adapter_training/qwen3_14b/checkpoint-final \
    --output_dir  outputs/merged/qwen3_14b_merged

# Then serve the merged model with lmdeploy (no --adapters needed)
lmdeploy serve api_server \
    outputs/merged/qwen3_14b_merged \
    --model-name qwen3-14b-adapter \
    --server-port 8000 \
    --tp 1

Optional flags
--------------
--dtype        bfloat16 | float16 | float32   (default: bfloat16)
--device       cpu | cuda | auto               (default: auto — shards across all GPUs)
--push_to_hub  HF_REPO_ID                      (optional: push merged model to the Hub)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("merge_adapter")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base model and save merged weights."
    )
    p.add_argument(
        "--base_model", required=True,
        help="Path or HuggingFace Hub ID of the base model.",
    )
    p.add_argument(
        "--adapter_dir", required=True,
        help="Path to the saved LoRA adapter directory (from save_lora_adapter / train_sft).",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory where the merged model will be written.",
    )
    p.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for loading the base model (default: bfloat16).",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device map for loading ('cpu' avoids OOM on large models; use 'auto' for GPU).",
    )
    p.add_argument(
        "--push_to_hub", default=None, metavar="REPO_ID",
        help="If set, push the merged model to this HuggingFace Hub repository.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _merge_peft(args)
    
def _merge_peft(args) -> None:
    """Standard HuggingFace + PEFT merge path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # 1. Load base model
    logger.info(f"Loading base model from: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
    )

    # 2. Load tokenizer
    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # 3. Attach LoRA adapter
    logger.info(f"Attaching LoRA adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir)

    # 4. Merge weights and discard the PEFT wrapper
    logger.info("Merging LoRA weights into base model …")
    model = model.merge_and_unload()
    logger.info("Merge complete.")

    # 5. Save merged model + tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")

    _maybe_push(model, tokenizer, args.push_to_hub)


def _maybe_push(model, tokenizer, repo_id) -> None:
    if not repo_id:
        return
    logger.info(f"Pushing merged model to Hub: {repo_id}")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    logger.info("Push complete.")


if __name__ == "__main__":
    main()
