"""
wandb_utils/run.py

WandbRun — a thread-safe singleton that wraps the wandb API.

Design goals
------------
- Every public method is a guaranteed no-op when W&B is disabled or unavailable.
- A single global instance ``wandb_run`` is used across the entire process.
- ``init()`` is idempotent: calling it a second time (e.g. per eval-set) is safe.
- No import of ``wandb`` at module load; lazy import only inside methods.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("EB_logger")

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


class WandbRun:
    """
    Safe, gracefully-degrading wrapper around wandb.

    Usage
    -----
    from embodiedbench.wandb_utils import wandb_run, WandbConfig

    cfg = WandbConfig(enabled=True, project="memadapt", tags=["grpo"])
    wandb_run.init(cfg, run_name="grpo_qwen7b_seed42", config_dict=my_cfg)

    wandb_run.log({"train/loss": 0.34}, step=100)
    wandb_run.log_table("eval/examples", ["prompt", "response"], rows)
    wandb_run.log_artifact("outputs/checkpoint", "my_run_ckpt", "model")

    wandb_run.finish()
    """

    def __init__(self) -> None:
        self._run: Optional[Any] = None
        self._enabled: bool = False

    # ------------------------------------------------------------------
    # Init / Finish
    # ------------------------------------------------------------------

    def init(
        self,
        cfg: Any,                               # WandbConfig
        run_name: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        extra_tags: Optional[List[str]] = None,
        extra_group: Optional[str] = None,
    ) -> None:
        """
        Initialize a W&B run.

        Parameters
        ----------
        cfg         : WandbConfig
        run_name    : display name for the run (overrides cfg.group-derived name)
        config_dict : arbitrary key-value config to record in the run
        extra_tags  : additional tags appended to cfg.tags
        extra_group : override / extend cfg.group
        """
        if not cfg.enabled:
            logger.debug("[W&B] Disabled (cfg.enabled=False).")
            return

        if not _WANDB_AVAILABLE:
            logger.warning(
                "[W&B] wandb is not installed. "
                "Run `pip install wandb` to enable experiment tracking."
            )
            return

        mode = str(cfg.mode or "online").lower()
        if mode == "disabled":
            self._enabled = False
            return

        tags: List[str] = list(cfg.tags or []) + list(extra_tags or [])
        group: Optional[str] = extra_group or cfg.group or None

        try:
            self._run = _wandb.init(
                project=cfg.project,
                entity=cfg.entity or None,
                name=run_name or None,
                group=group or None,
                tags=tags or None,
                config=config_dict or {},
                mode=mode,
                resume=cfg.resume,
                id=cfg.run_id or None,
                save_code=cfg.save_code,
                reinit="finish_previous",  # replaces deprecated `reinit=True` (wandb ≥ 0.18)
            )
            self._enabled = True
            url = getattr(self._run, "url", "(offline/unknown)")
            logger.info("[W&B] Run initialized: %s", url)
        except Exception as exc:
            logger.warning(
                "[W&B] Failed to initialize run (%s). "
                "Continuing without experiment tracking.",
                exc,
            )
            self._enabled = False
            self._run = None

    def finish(self, exit_code: int = 0) -> None:
        """Mark the run as finished. Safe no-op if not initialized."""
        if not self._enabled or self._run is None:
            return
        try:
            self._run.finish(exit_code=exit_code)
            logger.info("[W&B] Run finished.")
        except Exception as exc:
            logger.debug("[W&B] finish() failed: %s", exc)
        finally:
            self._enabled = False
            self._run = None

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log a dict of scalar metrics. Safe no-op when disabled."""
        if not self._enabled or self._run is None:
            return
        try:
            kwargs: Dict[str, Any] = {"commit": commit}
            if step is not None:
                kwargs["step"] = step
            self._run.log(metrics, **kwargs)
        except Exception as exc:
            logger.debug("[W&B] log() failed: %s", exc)

    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
    ) -> None:
        """Log a W&B Table. Safe no-op when disabled."""
        if not self._enabled or self._run is None:
            return
        if not data:
            return
        try:
            table = _wandb.Table(columns=columns, data=data)
            self._run.log({key: table})
        except Exception as exc:
            logger.debug("[W&B] log_table() failed (key=%s): %s", key, exc)

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """Write values directly into run.summary (shown on project page)."""
        if not self._enabled or self._run is None:
            return
        try:
            for k, v in metrics.items():
                self._run.summary[k] = v
        except Exception as exc:
            logger.debug("[W&B] log_summary() failed: %s", exc)

    def log_config(self, config_dict: Dict[str, Any]) -> None:
        """Update the run config (e.g. after a mid-run override)."""
        if not self._enabled or self._run is None:
            return
        try:
            self._run.config.update(config_dict, allow_val_change=True)
        except Exception as exc:
            logger.debug("[W&B] log_config() failed: %s", exc)

    # ------------------------------------------------------------------
    # Artifact helpers
    # ------------------------------------------------------------------

    def log_artifact(
        self,
        path: str,
        name: str,
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upload a file or directory as a W&B Artifact."""
        if not self._enabled or self._run is None:
            return
        if not os.path.exists(path):
            logger.debug("[W&B] Artifact path not found, skipping: %s", path)
            return
        try:
            artifact = _wandb.Artifact(
                name=name, type=artifact_type, metadata=metadata or {}
            )
            if os.path.isdir(path):
                artifact.add_dir(path)
            else:
                artifact.add_file(path)
            self._run.log_artifact(artifact)
            logger.debug("[W&B] Artifact '%s' logged from '%s'.", name, path)
        except Exception as exc:
            logger.warning("[W&B] log_artifact() failed (%s): %s", name, exc)

    def log_artifact_from_string(
        self,
        content: str,
        filename: str,
        artifact_name: str,
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write *content* to a temp file and upload as a W&B Artifact."""
        if not self._enabled or self._run is None:
            return
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=f"_{filename}",
                delete=False,
                encoding="utf-8",
            ) as fh:
                fh.write(content)
                tmp_path = fh.name
            self.log_artifact(tmp_path, artifact_name, artifact_type, metadata)
            os.unlink(tmp_path)
        except Exception as exc:
            logger.debug("[W&B] log_artifact_from_string() failed: %s", exc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run(self) -> Optional[Any]:
        """The underlying wandb.Run object (None when disabled)."""
        return self._run

    @property
    def run_url(self) -> Optional[str]:
        if self._run is not None:
            return getattr(self._run, "url", None)
        return None


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
wandb_run = WandbRun()
