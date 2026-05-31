"""
memory_adapter_rl/config.py

Configuration dataclasses for the GRPO refinement pipeline.
All fields have safe defaults; load from YAML with RLConfig.from_yaml().
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from embodiedbench.wandb_utils.config import WandbConfig


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------

@dataclass
class RLRewardWeights:
    """
    Scalar weights for each component of the composite GRPO reward.

        R = w_success     * task_success
          + w_progress    * task_progress
          + w_format      * format_validity
          + w_foresight   * foresight_quality
          + w_feasibility * feasibility_quality
          + w_fallback    * fallback_quality
          - w_replan      * replan_count
          - w_invalid     * invalid_action_count
          - w_repetition  * repetition_penalty

    The three guidance sections (foresight / feasibility / fallback) carry the
    largest quality weights because they are exactly the signals the planner and
    critic consume to raise task success.
    """
    # Task outcome (available only when reward comes from environment rollouts)
    w_success:     float = 1.0
    w_progress:    float = 0.5

    # Structural validity of the three required XML sections
    w_format:      float = 0.5

    # Per-section content quality (the optimisation target)
    w_foresight:   float = 0.6
    w_feasibility: float = 0.6
    w_fallback:    float = 0.6

    # Penalties
    w_replan:      float = 0.1
    w_invalid:     float = 0.1
    w_repetition:  float = 0.3

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RLRewardWeights":
        valid = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# GRPO algorithm config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """
    GRPO (Group Relative Policy Optimisation) hyper-parameters.

    GRPO samples K candidate completions per prompt, scores each with the
    composite reward, normalises the rewards within the group to form
    advantages, and optimises with a KL-regularised policy gradient.
    """
    num_generations: int = 8           # K completions sampled per prompt
    kl_beta: float = 0.04              # KL penalty against the reference policy
    temperature: float = 0.9           # rollout sampling temperature
    top_p: float = 0.95                # nucleus sampling top-p
    max_new_tokens: int = 512          # max tokens per completion
    rollout_batch_size: int = 4        # prompts per rollout batch (custom loop)
    advantage_epsilon: float = 1e-8    # numerical stability in advantage normalisation


# ---------------------------------------------------------------------------
# Top-level RL config
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    """Complete configuration for one GRPO refinement run."""

    # ---- Identity ----
    run_name: str = "memadapt_grpo"
    algorithm: str = "grpo"
    seed: int = 42

    # ---- Model ----
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    sft_checkpoint: Optional[str] = None   # SFT adapter to initialise from
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = False

    # ---- LoRA ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # ---- Data (JSONL of prompt strings or {"prompt": ...} objects) ----
    train_data_path: str = ""
    val_data_path: str = ""
    max_prompt_length: int = 1024
    max_length: int = 2048

    # ---- Training ----
    output_dir: str = "./outputs/memory_adapter_rl"
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    dataloader_num_workers: int = 0

    # ---- Reward weights ----
    reward_weights: RLRewardWeights = field(default_factory=RLRewardWeights)

    # ---- GRPO hyper-parameters ----
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    # ---- Logging ----
    report_to: str = "none"          # "wandb" | "tensorboard" | "none"
    wandb_project: str = "memadapt-rl"
    log_level: str = "INFO"

    # ---- W&B experiment tracking ----
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # ---- Evaluation ----
    eval_format_validity: bool = True
    num_eval_samples: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "RLConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "RLConfig":
        d = dict(d)
        if "reward_weights" in d and isinstance(d["reward_weights"], dict):
            d["reward_weights"] = RLRewardWeights.from_dict(d["reward_weights"])
        if "grpo" in d and isinstance(d["grpo"], dict):
            d["grpo"] = GRPOConfig(**{k: v for k, v in d["grpo"].items()
                                      if k in GRPOConfig.__dataclass_fields__})
        if "wandb" in d and isinstance(d["wandb"], dict):
            d["wandb"] = WandbConfig.from_mapping(d["wandb"])
        valid = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid})
