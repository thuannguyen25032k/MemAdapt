"""
wandb_utils — centralized, gracefully-degrading W&B integration for MemAdapt.

Public surface
--------------
from embodiedbench.wandb_utils import wandb_run, WandbConfig
from embodiedbench.wandb_utils.callbacks import SFTWandbCallback, RLWandbCallback, GRPOStepLogger
from embodiedbench.wandb_utils.eval_logger import EvalWandbLogger
from embodiedbench.wandb_utils.artifact_utils import (
    log_config_artifact, log_checkpoint_artifact,
    log_results_artifact, log_summary_artifact,
)
"""

from .config import WandbConfig
from .run import wandb_run

__all__ = ["WandbConfig", "wandb_run"]
