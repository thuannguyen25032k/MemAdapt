"""
memory_adapter_training/checkpoints.py

Helpers for saving, loading, merging and exporting LoRA adapters.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("EB_logger")


def save_lora_adapter(model, output_dir: str) -> None:
    """
    Save the LoRA adapter weights to *output_dir*.

    For PEFT models this calls ``model.save_pretrained(output_dir)``.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info(f"[Checkpoints] LoRA adapter saved to {output_dir}")


def load_lora_adapter(base_model, adapter_dir: str):
    """
    Load a saved LoRA adapter on top of an already-loaded *base_model*.

    Parameters
    ----------
    base_model  : PreTrainedModel (not yet wrapped with PEFT).
    adapter_dir : directory produced by ``save_lora_adapter``.

    Returns
    -------
    PeftModel
    """
    from peft import PeftModel  # type: ignore

    logger.info(f"[Checkpoints] Loading LoRA adapter from {adapter_dir}")
    return PeftModel.from_pretrained(base_model, adapter_dir)


def merge_and_unload(peft_model):
    """
    Merge LoRA weights into the base model and return the merged model.

    The resulting model has no PEFT wrappers and can be used like a normal
    ``PreTrainedModel``.
    """
    logger.info("[Checkpoints] Merging LoRA weights into base model …")
    merged = peft_model.merge_and_unload()
    logger.info("[Checkpoints] Merge complete.")
    return merged


def export_merged_model(peft_model, output_dir: str) -> None:
    """
    Merge LoRA weights and save the full model to *output_dir*.

    Parameters
    ----------
    peft_model : PeftModel with LoRA adapter attached.
    output_dir : destination directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    merged = merge_and_unload(peft_model)
    merged.save_pretrained(output_dir)
    logger.info(f"[Checkpoints] Merged model saved to {output_dir}")
