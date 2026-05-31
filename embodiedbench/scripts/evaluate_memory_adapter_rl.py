#!/usr/bin/env python
"""
scripts/evaluate_memory_adapter_rl.py

Evaluate a trained GRPO adapter on response quality: structural validity and the
per-section quality of FORESIGHT_PLAN / FEASIBILITY_CRITERIA / FALLBACK_STRATEGY,
plus the composite reward.

Usage
-----
    python embodiedbench/scripts/evaluate_memory_adapter_rl.py \\
        --checkpoint outputs/memory_adapter_rl/grpo_qwen7b/checkpoint-final \\
        --prompts    data/memory_adapter_rl/grpo_prompts_val.jsonl \\
        --output_dir outputs/rl_eval/grpo_qwen7b \\
        --generate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from embodiedbench.memory_adapter_rl.evaluation import RLEvaluator
from embodiedbench.memory_adapter_rl.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Memory Adapter GRPO checkpoint."
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to GRPO adapter checkpoint directory.")
    p.add_argument("--prompts", required=True,
                   help="JSONL of prompt strings or {'prompt': ...} objects.")
    p.add_argument("--output_dir", required=True,
                   help="Where to save evaluation metrics.")
    p.add_argument("--num_samples", type=int, default=50,
                   help="Max prompts to evaluate.")
    p.add_argument("--generate", action="store_true",
                   help="Generate responses from the checkpoint model (needs GPU + weights).")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def load_prompts(path: str, limit: int) -> list:
    prompts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                prompts.append(obj)
            elif isinstance(obj, dict):
                prompts.append(obj.get("prompt", obj.get("text", "")))
    return prompts[:limit]


def generate_responses(checkpoint: str, prompts: list, max_new_tokens: int) -> list:
    """Generate one response per prompt from the checkpoint model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from peft import PeftModel  # type: ignore
    import torch  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    meta_path = Path(checkpoint) / "rl_checkpoint_meta.json"
    base_name = "Qwen/Qwen2.5-7B-Instruct"
    if meta_path.exists():
        base_name = json.loads(meta_path.read_text()).get("base_model", base_name)

    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        responses.append(
            tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        )
    return responses


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("EB_logger")

    prompts = load_prompts(args.prompts, args.num_samples)
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompts)

    if args.generate:
        try:
            responses = generate_responses(args.checkpoint, prompts, args.max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.error("Generation failed (%s); cannot evaluate response quality.", exc)
            responses = []
    else:
        logger.warning("--generate not set; nothing to score. Pass --generate to "
                       "produce responses from the checkpoint model.")
        responses = []

    evaluator = RLEvaluator()
    metrics = evaluator.evaluate_all(responses=responses)

    logger.info("Evaluation metrics:")
    for k, v in sorted(metrics.items()):
        logger.info("  %s: %s", k, v)

    path = evaluator.save_metrics(metrics, args.output_dir)
    logger.info("Saved metrics -> %s", path)


if __name__ == "__main__":
    main()
