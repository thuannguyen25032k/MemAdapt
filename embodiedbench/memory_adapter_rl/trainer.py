"""
memory_adapter_rl/trainer.py

MemoryAdapterGRPOTrainer — wrapper around TRL's GRPOTrainer for GRPO refinement
of the Memory Adapter.

Design principles
-----------------
- Does NOT modify planner or critic inference code.
- All heavy imports (torch, transformers, trl, peft) are lazy and guarded.
- Compatible with LoRA / QLoRA and resume-from-checkpoint.
- Falls back to a CPU-safe custom rollout loop (rollout scoring only, no
  gradient updates) when TRL is unavailable, so the pipeline is importable and
  testable everywhere.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .config import RLConfig
from embodiedbench.wandb_utils import wandb_run
from embodiedbench.wandb_utils.callbacks import GRPOStepLogger
from embodiedbench.wandb_utils.artifact_utils import (
    log_checkpoint_artifact,
    log_config_artifact,
)

logger = logging.getLogger("EB_logger")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_lora_config(cfg: RLConfig):
    """Build a PEFT LoraConfig from RLConfig fields."""
    from peft import LoraConfig, TaskType  # type: ignore

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.lora_target_modules or "all-linear",
    )


def _build_training_args(cfg: RLConfig):
    from transformers import TrainingArguments  # type: ignore

    # TrainingArguments has no wandb_project param; propagate via env var.
    if cfg.report_to and "wandb" in cfg.report_to and cfg.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        seed=cfg.seed,
    )


# ---------------------------------------------------------------------------
# MemoryAdapterGRPOTrainer
# ---------------------------------------------------------------------------

class MemoryAdapterGRPOTrainer:
    """
    GRPO trainer for the Memory Adapter.

    Two execution paths:
      1. TRL GRPOTrainer — used when ``trl.GRPOTrainer`` is importable (real
         gradient updates).
      2. Custom lightweight loop — CPU-safe fallback that only scores rollouts
         (no gradient updates); used for tests and dependency-free environments.

    Parameters
    ----------
    cfg           : RLConfig (cfg.algorithm must be "grpo")
    model         : causal-LM (PEFT-wrapped or raw)
    tokenizer     : HF tokenizer
    train_prompts : list of prompt strings OR an HF Dataset with a 'prompt' column
    eval_prompts  : optional evaluation prompts
    ref_model     : optional reference model for KL regularisation (TRL < 0.24)
    """

    def __init__(
        self,
        cfg: RLConfig,
        model: Any,
        tokenizer: Any,
        train_prompts: Any,
        eval_prompts: Any = None,
        ref_model: Any = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_prompts = train_prompts
        self.eval_prompts = eval_prompts
        self.ref_model = ref_model
        self._trainer = None
        self._use_trl = False
        self._built = False
        self._grpo_step_logger: Optional[GRPOStepLogger] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Instantiate the GRPO trainer (TRL path, else custom loop)."""
        if self._built:
            return
        training_args = _build_training_args(self.cfg)
        try:
            self._trainer = self._build_grpo_trainer(training_args)
            self._use_trl = True
            logger.info("Built TRL GRPOTrainer.")
        except ImportError as exc:
            logger.warning(
                "TRL not installed (%s); using custom GRPO loop "
                "(rollout scoring only — install trl>=0.9 for real training).", exc,
            )
            self._use_trl = False
        except (AttributeError, TypeError) as exc:
            logger.error(
                "TRL GRPOTrainer construction failed (%s: %s) — this is a "
                "CONFIGURATION error, not a missing dependency. Falling back to "
                "the no-gradient custom loop.", type(exc).__name__, exc,
            )
            self._use_trl = False
        self._built = True

        # W&B run + step logger
        wb_cfg = getattr(self.cfg, "wandb", None)
        if wb_cfg is not None and wb_cfg.enabled:
            wandb_run.init(
                wb_cfg,
                run_name=self.cfg.run_name,
                config_dict=self.cfg.to_dict(),
                extra_tags=["grpo"],
            )
            log_config_artifact(self.cfg.to_dict(), run_name=self.cfg.run_name or "grpo_run")
        self._grpo_step_logger = GRPOStepLogger()

    def _build_grpo_trainer(self, training_args):
        import inspect

        from trl import GRPOConfig as TRLGRPOConfig  # type: ignore
        from trl import GRPOTrainer  # type: ignore

        from .grpo import make_trl_reward_fn

        grpo_cfg = self.cfg.grpo
        reward_fn = make_trl_reward_fn(weights=self.cfg.reward_weights)

        # ---- Probe TRL API at runtime (version-safe) ----
        trl_cfg_params = set(inspect.signature(TRLGRPOConfig.__init__).parameters)
        trl_trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters)

        # TRL >= 0.9 renamed max_new_tokens -> max_completion_length.
        completion_len_kwarg = (
            "max_completion_length"
            if "max_completion_length" in trl_cfg_params
            else "max_new_tokens"
        )

        # TRL 0.24+ requires generation_batch_size % num_generations == 0.
        extra_gen_kwargs: Dict[str, Any] = {}
        if "generation_batch_size" in trl_cfg_params:
            extra_gen_kwargs["generation_batch_size"] = (
                self.cfg.per_device_train_batch_size * grpo_cfg.num_generations
            )

        trl_cfg = TRLGRPOConfig(
            num_generations=grpo_cfg.num_generations,
            temperature=grpo_cfg.temperature,
            top_p=grpo_cfg.top_p,
            beta=grpo_cfg.kl_beta,
            **{completion_len_kwarg: grpo_cfg.max_new_tokens},
            **extra_gen_kwargs,
            **training_args.to_dict(),
        )

        trainer_kwargs: Dict[str, Any] = {
            "model": self.model,
            "reward_funcs": [reward_fn],
            "args": trl_cfg,
            "train_dataset": self.train_prompts,
            "eval_dataset": self.eval_prompts,
        }
        # ref_model present in TRL < 0.24, removed afterwards.
        if "ref_model" in trl_trainer_params:
            trainer_kwargs["ref_model"] = self.ref_model
        # processing_class (TRL >= 0.9) vs tokenizer (TRL < 0.9).
        if "processing_class" in trl_trainer_params:
            trainer_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in trl_trainer_params:
            trainer_kwargs["tokenizer"] = self.tokenizer

        return GRPOTrainer(**trainer_kwargs)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run GRPO training (TRL path) or the custom rollout loop (fallback)."""
        if not self._built:
            self.build()

        if self._use_trl and self._trainer is not None:
            result = self._trainer.train(
                resume_from_checkpoint=self.cfg.resume_from_checkpoint or None
            )
            return result.metrics if hasattr(result, "metrics") else {}

        return self._custom_train_loop()

    def _custom_train_loop(self) -> Dict[str, Any]:
        """
        CPU-safe rollout-scoring loop (NO gradient updates).

        Use the TRL path for real GRPO training. This loop only computes and logs
        rollout statistics, which is useful for testing the reward + rollout
        plumbing without TRL/torch.
        """
        logger.warning(
            "Custom GRPO loop: rollout statistics only — NO gradient updates. "
            "Install trl>=0.9 and use the TRL GRPOTrainer path for real training."
        )
        from .grpo import generate_candidate_group

        grpo_cfg = self.cfg.grpo
        prompts = list(self.train_prompts) if self.train_prompts is not None else []

        total_reward = 0.0
        total_xml = 0.0
        n_groups = 0
        for prompt in prompts:
            group = generate_candidate_group(
                prompt=prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                num_generations=grpo_cfg.num_generations,
                temperature=grpo_cfg.temperature,
                top_p=grpo_cfg.top_p,
                max_new_tokens=grpo_cfg.max_new_tokens,
                weights=self.cfg.reward_weights,
                advantage_epsilon=grpo_cfg.advantage_epsilon,
            )
            total_reward += group.mean_reward
            total_xml += group.xml_validity_rate
            n_groups += 1
            if self._grpo_step_logger is not None:
                self._grpo_step_logger.log_rollout_group(group, step=n_groups)
                self._grpo_step_logger.maybe_log_example_table(group, step=n_groups)

        n = max(n_groups, 1)
        metrics = {
            "train/mean_reward": round(total_reward / n, 4),
            "train/xml_validity_rate": round(total_xml / n, 4),
            "train/num_groups": n_groups,
        }
        logger.info("Custom GRPO loop complete: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate over eval_prompts (or train_prompts when none given)."""
        if self._use_trl and self._trainer is not None:
            return self._trainer.evaluate()

        from .grpo import generate_candidate_group

        grpo_cfg = self.cfg.grpo
        prompts = self.eval_prompts if self.eval_prompts is not None else self.train_prompts
        if prompts is None:
            return {}
        prompts = list(prompts)[: self.cfg.num_eval_samples]

        total_reward = 0.0
        total_xml = 0.0
        for prompt in prompts:
            group = generate_candidate_group(
                prompt=prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                num_generations=grpo_cfg.num_generations,
                temperature=grpo_cfg.temperature,
                top_p=grpo_cfg.top_p,
                max_new_tokens=grpo_cfg.max_new_tokens,
                weights=self.cfg.reward_weights,
                advantage_epsilon=grpo_cfg.advantage_epsilon,
            )
            total_reward += group.mean_reward
            total_xml += group.xml_validity_rate

        n = max(len(prompts), 1)
        return {
            "eval/mean_reward": round(total_reward / n, 4),
            "eval/xml_validity_rate": round(total_xml / n, 4),
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Save the trained adapter; upload a W&B artifact when configured."""
        path = path or self.cfg.output_dir
        os.makedirs(path, exist_ok=True)
        if self._use_trl and self._trainer is not None:
            self._trainer.save_model(path)
        elif hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        logger.info("Saved GRPO adapter -> %s", path)

        wb_cfg = getattr(self.cfg, "wandb", None)
        if wb_cfg is not None and wb_cfg.enabled and wb_cfg.log_model:
            log_checkpoint_artifact(path, run_name=self.cfg.run_name or "grpo_run")
        wandb_run.finish()
        return path

    @property
    def inner_trainer(self):
        return self._trainer
