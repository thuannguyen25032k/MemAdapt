"""
memory_adapter_training/evaluation.py

Post-training evaluation utilities.

Generates completions on a small held-out sample set and computes lightweight
rule-based metrics to check whether the adapter produces well-formed Memory
Adapter outputs (matching the inference format).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from embodiedbench.memory_adapter_training.formatting import (
    format_sample,
    parse_target_text,
    to_chat_messages,
)

logger = logging.getLogger("EB_logger")

# Required XML section tags in the model output format (matches inference).
_REQUIRED_SECTIONS = [
    "<FORESIGHT_PLAN>",
    "<FEASIBILITY_CRITERIA>",
    "<FALLBACK_STRATEGY>",
]


def _generate_one(model, tokenizer, prompt: str, gen_cfg, enable_thinking: bool) -> str:
    """Generate a single completion using the same chat template as inference."""
    import torch

    messages = to_chat_messages(prompt)
    if getattr(tokenizer, "chat_template", None):
        try:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    else:
        chat_text = "\n\n".join(m["content"] for m in messages)

    inputs = tokenizer(chat_text, return_tensors="pt", truncation=False)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature if gen_cfg.do_sample else None,
            top_p=gen_cfg.top_p if gen_cfg.do_sample else None,
            repetition_penalty=gen_cfg.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_ids = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].lstrip("\n")
    return text


def _compute_metrics(generated: List[str]) -> Dict[str, float]:
    n = len(generated)
    if n == 0:
        return {}

    total_len = malformed = fallback_present = feasibility_present = foresight_present = 0
    for gen in generated:
        total_len += len(gen.split())
        if not any(sec in gen for sec in _REQUIRED_SECTIONS):
            malformed += 1
        if "<FALLBACK_STRATEGY>" in gen:
            fallback_present += 1
        if "<FEASIBILITY_CRITERIA>" in gen:
            feasibility_present += 1
        if parse_target_text(gen)["foresight_plan"]:
            foresight_present += 1

    return {
        "generation_length_avg": total_len / n,
        "malformed_section_rate": malformed / n,
        "foresight_presence_rate": foresight_present / n,
        "feasibility_presence_rate": feasibility_present / n,
        "fallback_presence_rate": fallback_present / n,
    }


def evaluate_memory_adapter_generations(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    cfg,
) -> Dict[str, Any]:
    """Generate completions for *samples* and compute quality metrics."""
    eval_cfg = cfg.evaluation
    gen_cfg = cfg.generation
    enable_thinking = getattr(cfg.model, "enable_thinking", False)

    n_eval = min(eval_cfg.num_eval_generations, len(samples))
    eval_samples = samples[:n_eval]

    model.eval()
    generated: List[str] = []
    eval_records: List[Dict[str, Any]] = []

    for i, sample in enumerate(eval_samples):
        prompt = format_sample(sample)["prompt"]
        try:
            gen = _generate_one(model, tokenizer, prompt, gen_cfg, enable_thinking)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Evaluation] Generation {i} failed: {exc}")
            gen = ""
        generated.append(gen)
        if eval_cfg.save_eval_samples:
            eval_records.append(
                {"index": i, "generated": gen, "parsed": parse_target_text(gen)}
            )

    metrics = _compute_metrics(generated)
    logger.info(f"[Evaluation] Metrics over {n_eval} samples: {metrics}")

    if eval_cfg.save_eval_samples and eval_cfg.eval_output_file:
        os.makedirs(os.path.dirname(os.path.abspath(eval_cfg.eval_output_file)), exist_ok=True)
        with open(eval_cfg.eval_output_file, "w", encoding="utf-8") as fh:
            json.dump({"metrics": metrics, "samples": eval_records}, fh, indent=2)
        logger.info(f"[Evaluation] Eval samples saved to {eval_cfg.eval_output_file}")

    result = dict(metrics)
    if eval_cfg.save_eval_samples:
        result["eval_samples"] = eval_records
    return result
