"""
memory_adapter_rl/checkpoints.py

Save, load, merge, and export RL adapter checkpoints.

Supports
--------
- Save/load RL PEFT adapters (LoRA weights only)
- Merge RL adapter with SFT adapter (sequential composition)
- Merge RL adapter into base model (full merge)
- Export merged checkpoint for inference

All heavy imports are lazy and guarded so this module can be imported
without GPU / torch available.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .config import RLConfig

logger = logging.getLogger("EB_logger")

_CHECKPOINT_META_FILE = "rl_checkpoint_meta.json"


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manage RL adapter checkpoints on disk.

    Parameters
    ----------
    output_dir : root directory for all checkpoints.
    cfg        : RLConfig (used for metadata).
    """

    def __init__(self, output_dir: str, cfg: Optional[RLConfig] = None) -> None:
        self.output_dir = Path(output_dir)
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: Any,
        tokenizer: Any = None,
        step: Optional[int] = None,
        tag: str = "best",
    ) -> str:
        """
        Save a PEFT adapter (LoRA weights) to disk.

        Parameters
        ----------
        model     : PEFT-wrapped model or plain HF model.
        tokenizer : HF tokenizer (optional).
        step      : training step (used in subdirectory name).
        tag       : checkpoint tag ("best", "final", "step-N").

        Returns
        -------
        Path of the saved checkpoint directory.
        """
        subdir = f"checkpoint-{tag}" if step is None else f"checkpoint-step{step}"
        ckpt_dir = self.output_dir / subdir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(ckpt_dir))
        else:
            logger.warning("Model does not have save_pretrained; skipping weight save.")

        # Save tokenizer
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(ckpt_dir))

        # Write metadata
        meta = {
            "tag": tag,
            "step": step,
            "output_dir": str(ckpt_dir),
            "algorithm": self.cfg.algorithm if self.cfg else "unknown",
            "run_name": self.cfg.run_name if self.cfg else "unknown",
        }
        (ckpt_dir / _CHECKPOINT_META_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("Saved RL checkpoint → %s", ckpt_dir)
        return str(ckpt_dir)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint: returns the model loaded with PEFT adapter.

        Parameters
        ----------
        checkpoint_path : path to the checkpoint directory.

        Returns
        -------
        dict with keys: "model", "tokenizer" (if saved), "metadata".
        """
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        meta_path = ckpt / _CHECKPOINT_META_FILE
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        result: Dict[str, Any] = {"metadata": meta}

        try:
            from peft import PeftModel  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            base_name = meta.get("base_model", self.cfg.model_name_or_path
                                  if self.cfg else "")
            if base_name:
                base = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True)
                model = PeftModel.from_pretrained(base, str(ckpt))
                result["model"] = model

            tok_path = ckpt / "tokenizer_config.json"
            if tok_path.exists():
                result["tokenizer"] = AutoTokenizer.from_pretrained(
                    str(ckpt), trust_remote_code=True
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load model weights: %s", exc)

        logger.info("Loaded RL checkpoint from %s", checkpoint_path)
        return result

    # ------------------------------------------------------------------
    # Merge adapters
    # ------------------------------------------------------------------

    def merge_adapter_into_base(self, model: Any, output_path: str) -> str:
        """
        Merge LoRA weights into the base model and save a full-parameter checkpoint.
        Requires peft ≥ 0.6.

        Parameters
        ----------
        model       : PEFT-wrapped model.
        output_path : where to save the merged full model.

        Returns
        -------
        output_path
        """
        try:
            merged = model.merge_and_unload()
            os.makedirs(output_path, exist_ok=True)
            merged.save_pretrained(output_path)
            logger.info("Merged RL adapter → full model at %s", output_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Merge failed: %s", exc)
            raise
        return output_path

    def merge_sft_and_rl_adapters(
        self,
        base_model: Any,
        sft_adapter_path: str,
        rl_adapter_path: str,
        output_path: str,
    ) -> str:
        """
        Load SFT + RL adapters sequentially and merge both into the base model.

        Parameters
        ----------
        base_model       : raw HF causal-LM.
        sft_adapter_path : path to SFT PEFT adapter.
        rl_adapter_path  : path to RL PEFT adapter.
        output_path      : where to save the merged model.
        """
        try:
            from peft import PeftModel  # type: ignore

            # Stack SFT, then RL
            model = PeftModel.from_pretrained(base_model, sft_adapter_path,
                                              adapter_name="sft")
            model.load_adapter(rl_adapter_path, adapter_name="rl")
            # Merge both adapters: SFT + RL
            model = model.merge_and_unload()
            os.makedirs(output_path, exist_ok=True)
            model.save_pretrained(output_path)
            logger.info("Merged SFT+RL adapters → %s", output_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("SFT+RL merge failed: %s", exc)
            raise
        return output_path

    # ------------------------------------------------------------------
    # List checkpoints
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> list:
        """Return sorted list of checkpoint subdirectories."""
        if not self.output_dir.exists():
            return []
        ckpts = sorted(
            [d for d in self.output_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: d.stat().st_mtime,
        )
        return [str(c) for c in ckpts]

    def latest_checkpoint(self) -> Optional[str]:
        """Return the most recently modified checkpoint path, or None."""
        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def save_rl_adapter(model: Any, output_dir: str, tokenizer: Any = None) -> str:
    """Shortcut: save a PEFT model + optional tokenizer."""
    mgr = CheckpointManager(output_dir)
    return mgr.save(model, tokenizer=tokenizer, tag="final")


def load_rl_adapter(checkpoint_path: str, cfg: Optional[RLConfig] = None) -> Dict[str, Any]:
    """Shortcut: load a checkpoint."""
    mgr = CheckpointManager(os.path.dirname(checkpoint_path), cfg=cfg)
    return mgr.load(checkpoint_path)
