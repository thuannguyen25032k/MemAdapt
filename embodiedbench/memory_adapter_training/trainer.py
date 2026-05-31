"""
memory_adapter_training/trainer.py

Thin wrapper around ``transformers.Trainer`` for Memory Adapter SFT.

Responsibilities
----------------
* Build ``TrainingArguments`` from ``MemoryAdapterTrainingConfig``.
* Create the HF ``Trainer`` with the right data collator and datasets.
* Expose ``train()`` and ``evaluate()`` helpers.
* Support resuming from a checkpoint.
* Integrate W&B via SFTWandbCallback (no-op when disabled).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from embodiedbench.wandb_utils import wandb_run
from embodiedbench.wandb_utils.artifact_utils import (
    log_checkpoint_artifact,
    log_config_artifact,
)
from embodiedbench.wandb_utils.callbacks import SFTWandbCallback

logger = logging.getLogger("EB_logger")


class MemoryAdapterTrainer:
    """
    Wrapper around ``transformers.Trainer`` for LoRA / QLoRA SFT runs.

    Parameters
    ----------
    cfg          : MemoryAdapterTrainingConfig
    model        : (peft-wrapped) causal-LM model
    tokenizer    : tokenizer
    train_dataset: HF Dataset with "text" column
    eval_dataset : HF Dataset with "text" column (optional)
    """

    def __init__(
        self,
        cfg,  # MemoryAdapterTrainingConfig
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._trainer = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _make_training_args(self):
        import os
        from transformers import TrainingArguments  # type: ignore

        t = self.cfg.training
        log = self.cfg.logging

        # Propagate wandb_project via env var (TrainingArguments has no wandb_project param).
        if log.report_to and "wandb" in log.report_to and log.wandb_project:
            os.environ.setdefault("WANDB_PROJECT", log.wandb_project)

        eval_strategy = "steps" if self.eval_dataset is not None else "no"

        # Unsloth applies its own gradient checkpointing inside get_peft_model,
        # so HF's must be disabled to avoid double-wrapping the model.
        gradient_checkpointing = t.gradient_checkpointing
        if getattr(self.cfg.model, "use_unsloth", False):
            gradient_checkpointing = False

        return TrainingArguments(
            output_dir=t.output_dir,
            num_train_epochs=t.num_train_epochs,
            per_device_train_batch_size=t.per_device_train_batch_size,
            per_device_eval_batch_size=t.per_device_eval_batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            learning_rate=t.learning_rate,
            weight_decay=t.weight_decay,
            warmup_ratio=t.warmup_ratio,
            lr_scheduler_type=t.lr_scheduler_type,
            bf16=t.bf16,
            fp16=t.fp16,
            gradient_checkpointing=gradient_checkpointing,
            logging_steps=t.logging_steps,
            eval_steps=t.eval_steps,
            save_steps=t.save_steps,
            save_total_limit=t.save_total_limit,
            save_strategy="steps",
            eval_strategy=eval_strategy,
            seed=t.seed,
            report_to=log.report_to,
            run_name=log.run_name or None,
            dataloader_num_workers=t.dataloader_num_workers,
            remove_unused_columns=False,  # we handle columns ourselves
            resume_from_checkpoint=t.resume_from_checkpoint,
        )

    def _make_collator(self):
        from embodiedbench.memory_adapter_training.collator import MemoryAdapterDataCollator

        return MemoryAdapterDataCollator(
            tokenizer=self.tokenizer,
            max_seq_length=self.cfg.dataset.max_seq_length,
            enable_thinking=self.cfg.model.enable_thinking,
        )

    def build(self) -> None:
        """Instantiate the underlying HF Trainer (lazy, call once)."""
        from transformers import Trainer  # type: ignore

        args = self._make_training_args()
        collator = self._make_collator()

        # ---- W&B callback ----
        wb_cfg = getattr(self.cfg, "wandb", None)
        dataset_stats: dict = {}
        if self.train_dataset is not None:
            try:
                dataset_stats["train_samples"] = len(self.train_dataset)
            except Exception:
                pass
        if self.eval_dataset is not None:
            try:
                dataset_stats["val_samples"] = len(self.eval_dataset)
            except Exception:
                pass
        wb_callback = SFTWandbCallback(
            run_cfg_dict=self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else {},
            dataset_stats=dataset_stats,
        )
        if wb_cfg is not None and wb_cfg.enabled:
            wandb_run.init(
                wb_cfg,
                run_name=self.cfg.logging.run_name or "sft_run",
                config_dict=self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else {},
                extra_tags=["sft", self.cfg.model.model_name_or_path.split("/")[-1]],
                extra_group=wb_cfg.group or "sft",
            )
            # Upload config as artifact after init
            log_config_artifact(
                self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else {},
                run_name=self.cfg.logging.run_name or "sft_run",
            )

        self._trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,  # replaces deprecated `tokenizer=` (transformers ≥ 4.46)
            data_collator=collator,
            callbacks=[wb_callback],
        )
        logger.info("[Trainer] HF Trainer built successfully.")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def train(self):
        """Run training, returning a ``TrainOutput`` from HF Trainer."""
        if self._trainer is None:
            self.build()

        resume = self.cfg.training.resume_from_checkpoint
        if resume:
            if not os.path.isdir(str(resume)):
                logger.warning(
                    f"[Trainer] resume_from_checkpoint path not found: {resume}. "
                    "Starting from scratch."
                )
                resume = None

        logger.info("[Trainer] Starting training …")
        output = self._trainer.train(resume_from_checkpoint=resume or None)
        logger.info(
            f"[Trainer] Training finished. "
            f"global_step={output.global_step}, "
            f"train_loss={output.training_loss:.4f}"
        )
        return output

    def evaluate(self) -> dict:
        """Run evaluation on eval_dataset, return metrics dict."""
        if self._trainer is None:
            self.build()
        if self.eval_dataset is None:
            logger.warning("[Trainer] No eval_dataset provided; skipping evaluation.")
            return {}
        return self._trainer.evaluate()

    def save(self, output_dir: Optional[str] = None) -> None:
        """Save the model (LoRA adapter) and tokenizer; upload W&B artifact if enabled."""
        if self._trainer is None:
            self.build()
        out = output_dir or self.cfg.training.output_dir
        self._trainer.save_model(out)
        self.tokenizer.save_pretrained(out)
        logger.info(f"[Trainer] Model saved to {out}")

        # Upload checkpoint as W&B model artifact
        wb_cfg = getattr(self.cfg, "wandb", None)
        if wb_cfg is not None and wb_cfg.enabled and wb_cfg.log_model:
            run_name = self.cfg.logging.run_name or "sft_run"
            log_checkpoint_artifact(out, run_name=run_name)
        wandb_run.finish()
