"""
wandb_utils/artifact_utils.py

Helpers for uploading checkpoints, configs, and result files as W&B Artifacts.

All functions are safe no-ops when W&B is disabled.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import yaml

from .run import wandb_run

logger = logging.getLogger("EB_logger")


def log_config_artifact(cfg_dict: Dict[str, Any], run_name: str) -> None:
    """
    Serialize *cfg_dict* as YAML and upload as a 'config' W&B Artifact.

    Parameters
    ----------
    cfg_dict  : complete run configuration (any nesting level).
    run_name  : used as artifact name prefix.
    """
    try:
        content = yaml.dump(
            _json_safe(cfg_dict), default_flow_style=False, sort_keys=False
        )
        wandb_run.log_artifact_from_string(
            content=content,
            filename="config.yaml",
            artifact_name=f"{run_name}_config",
            artifact_type="config",
            metadata={"run_name": run_name},
        )
    except Exception as exc:
        logger.debug("[W&B] log_config_artifact failed: %s", exc)


def log_checkpoint_artifact(
    checkpoint_dir: str,
    run_name: str,
    step: Optional[int] = None,
) -> None:
    """
    Upload a checkpoint directory as a 'model' W&B Artifact.

    Parameters
    ----------
    checkpoint_dir : local path to the checkpoint directory.
    run_name       : artifact name prefix.
    step           : training step (used in artifact name).
    """
    if not os.path.isdir(checkpoint_dir):
        logger.debug("[W&B] Checkpoint dir not found, skipping: %s", checkpoint_dir)
        return
    suffix = f"_step{step}" if step is not None else "_final"
    wandb_run.log_artifact(
        path=checkpoint_dir,
        name=f"{run_name}_checkpoint{suffix}",
        artifact_type="model",
        metadata={"run_name": run_name, "step": step},
    )


def log_results_artifact(results_dir: str, run_name: str) -> None:
    """
    Upload an evaluation results directory (JSON files) as an 'evaluation' Artifact.
    """
    if not os.path.isdir(results_dir):
        logger.debug("[W&B] Results dir not found, skipping: %s", results_dir)
        return
    wandb_run.log_artifact(
        path=results_dir,
        name=f"{run_name}_eval_results",
        artifact_type="evaluation",
        metadata={"run_name": run_name},
    )


def log_summary_artifact(
    summary_dict: Dict[str, Any],
    run_name: str,
    extra_md: str = "",
) -> None:
    """
    Build a Markdown summary from *summary_dict* and upload as a 'report' Artifact.

    Parameters
    ----------
    summary_dict : aggregated evaluation metrics.
    run_name     : artifact name prefix.
    extra_md     : optional additional Markdown appended at the end.
    """
    lines = [f"# Experiment Summary: `{run_name}`\n"]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in sorted(summary_dict.items()):
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    if extra_md:
        lines.append("\n---\n")
        lines.append(extra_md)
    content = "\n".join(lines)
    wandb_run.log_artifact_from_string(
        content=content,
        filename="summary.md",
        artifact_name=f"{run_name}_summary",
        artifact_type="report",
        metadata={"run_name": run_name},
    )


def log_json_artifact(
    data: Any,
    filename: str,
    artifact_name: str,
    artifact_type: str = "result",
    run_name: str = "",
) -> None:
    """Serialize *data* as JSON and upload as a W&B Artifact."""
    try:
        content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        wandb_run.log_artifact_from_string(
            content=content,
            filename=filename,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            metadata={"run_name": run_name},
        )
    except Exception as exc:
        logger.debug("[W&B] log_json_artifact failed: %s", exc)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serialisable form."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)
