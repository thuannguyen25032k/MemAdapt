"""
memory_adapter_training/config.py

MemoryAdapterTrainingConfig — YAML-backed dataclass for reproducible SFT runs.

Usage
-----
cfg = MemoryAdapterTrainingConfig.from_yaml("configs/memory_adapter_training/qwen_qlora.yaml")
cfg.save_yaml(output_dir / "training_config.yaml")
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml

from embodiedbench.wandb_utils.config import WandbConfig


# ---------------------------------------------------------------------------
# Sub-configs (nested dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer_name_or_path: Optional[str] = None   # defaults to model_name_or_path
    trust_remote_code: bool = True
    use_flash_attention: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "bfloat16"                  # "float16" | "bfloat16" | "float32"
    attn_implementation: Optional[str] = None      # "flash_attention_2" | None
    # Qwen3-style thinking mode; must match the inference setting so the
    # training chat template aligns with deployment.
    enable_thinking: bool = False


@dataclass
class DatasetConfig:
    train_path: Union[str, List[str]] = ""   # single path or list of paths to merge
    val_path: str = ""
    val_ratio: float = 0.0                         # auto-split when val_path empty
    max_seq_length: int = 2048


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs/memory_adapter_training"
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"                             # "none" | "all" | "lora_only"
    # None → PEFT "all-linear" (targets every linear layer, architecture-agnostic)
    target_modules: Optional[List[str]] = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class LoggingConfig:
    report_to: str = "none"                        # "wandb" | "tensorboard" | "none"
    wandb_project: str = "memadapt-sft"
    run_name: str = "memory_adapter_lora"
    log_level: str = "INFO"


@dataclass
class EvaluationConfig:
    num_eval_generations: int = 20
    save_eval_samples: bool = True
    eval_output_file: str = "eval_generations.json"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class MemoryAdapterTrainingConfig:
    """
    Complete training configuration for one SFT run.

    All sub-configs are nested dataclasses; the whole object is serialisable
    to/from YAML and JSON.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryAdapterTrainingConfig":
        return cls(
            model=ModelConfig(**{k: v for k, v in d.get("model", {}).items()
                                 if k in ModelConfig.__dataclass_fields__}),
            dataset=DatasetConfig(**{k: v for k, v in d.get("dataset", {}).items()
                                     if k in DatasetConfig.__dataclass_fields__}),
            training=TrainingConfig(**{k: v for k, v in d.get("training", {}).items()
                                       if k in TrainingConfig.__dataclass_fields__}),
            lora=LoraConfig(**{k: v for k, v in d.get("lora", {}).items()
                               if k in LoraConfig.__dataclass_fields__}),
            generation=GenerationConfig(**{k: v for k, v in d.get("generation", {}).items()
                                           if k in GenerationConfig.__dataclass_fields__}),
            logging=LoggingConfig(**{k: v for k, v in d.get("logging", {}).items()
                                     if k in LoggingConfig.__dataclass_fields__}),
            evaluation=EvaluationConfig(**{k: v for k, v in d.get("evaluation", {}).items()
                                           if k in EvaluationConfig.__dataclass_fields__}),
            wandb=WandbConfig.from_mapping(d.get("wandb")),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "MemoryAdapterTrainingConfig":
        with open(path, encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        return cls.from_dict(d)

    @classmethod
    def from_yaml_str(cls, yaml_str: str) -> "MemoryAdapterTrainingConfig":
        d = yaml.safe_load(yaml_str) or {}
        return cls.from_dict(d)

    # ------------------------------------------------------------------
    def resolve_tokenizer_path(self) -> str:
        return self.model.tokenizer_name_or_path or self.model.model_name_or_path

    def is_qlora(self) -> bool:
        return self.lora.enabled and (self.model.load_in_4bit or self.model.load_in_8bit)

    def __repr__(self) -> str:
        return (
            f"MemoryAdapterTrainingConfig("
            f"model={self.model.model_name_or_path!r}, "
            f"lora_r={self.lora.r}, "
            f"qlora={self.is_qlora()}, "
            f"epochs={self.training.num_train_epochs})"
        )
