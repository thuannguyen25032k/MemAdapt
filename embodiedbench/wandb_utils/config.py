"""
wandb_utils/config.py

WandbConfig dataclass — the single config object that controls all W&B behaviour
in MemAdapt.  Supported by every trainer and evaluator config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WandbConfig:
    """
    Complete configuration for W&B experiment tracking.

    Fields
    ------
    enabled      : master switch — False means complete no-op everywhere.
    project      : W&B project name.
    entity       : W&B team / user name (None = personal account).
    group        : run group (for grouping by experiment condition).
    tags         : list of tags attached to the run.
    log_model    : upload final checkpoint as a W&B model artifact.
    log_examples : log qualitative generation examples as W&B tables.
    mode         : "online" | "offline" | "disabled".
    save_code    : upload source code snapshot to W&B.
    resume       : "allow" | "must" | "never" | "auto".
    run_id       : W&B run ID to resume (None = new run).
    """

    enabled: bool = False
    project: str = "memadapt"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_model: bool = True
    log_examples: bool = True
    mode: str = "online"        # "online" | "offline" | "disabled"
    save_code: bool = True
    resume: str = "allow"
    run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Any) -> "WandbConfig":
        """
        Build from any dict-like object (plain dict, OmegaConf DictConfig,
        dataclass, or None).  Unknown keys are silently ignored.
        """
        if mapping is None:
            return cls()
        if hasattr(mapping, "items"):
            raw: Dict[str, Any] = {k: v for k, v in mapping.items()}
        elif hasattr(mapping, "__dict__"):
            raw = dict(vars(mapping))
        else:
            raw = {}
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in valid})
