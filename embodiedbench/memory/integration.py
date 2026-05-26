"""
memory/integration.py

Helper functions for wiring MemoryManager into evaluators.

These are intentionally thin wrappers so evaluator code stays minimal.
All functions are safe no-ops when memory is disabled or unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger("EB_logger")

# ---------------------------------------------------------------------------
# Lazy imports — never crash the evaluator if memory package is missing.
# ---------------------------------------------------------------------------
try:
    from embodiedbench.memory.manager import MemoryManager, MemoryConfig
    _MEMORY_AVAILABLE = True
except ImportError as e:
    _MEMORY_AVAILABLE = False
    MemoryManager = None  # type: ignore
    MemoryConfig = None   # type: ignore
    logger.warning(f"[Memory] Package unavailable: {e}")

# ---------------------------------------------------------------------------
# Optional memory adapter imports
# ---------------------------------------------------------------------------
try:
    from embodiedbench.memory_adapter.adapter import MemoryAdapter
    from embodiedbench.memory_adapter.config import MemoryAdapterConfig
    _ADAPTER_AVAILABLE = True
except ImportError:
    _ADAPTER_AVAILABLE = False
    MemoryAdapter = None      # type: ignore
    MemoryAdapterConfig = None  # type: ignore

# ---------------------------------------------------------------------------
# Shared config helpers (centralised in utils.py)
# ---------------------------------------------------------------------------
from embodiedbench.memory.utils import _get_cfg_key, _cfg_enabled, _cfg_flag  # noqa: E402


# ---------------------------------------------------------------------------
# Internal shortcuts — keep calls terse inside this module
# ---------------------------------------------------------------------------

def _get_memory_cfg(cfg: Any) -> Optional[Any]:
    return _get_cfg_key(cfg, "memory")


def _get_adapter_cfg(cfg: Any) -> Optional[Any]:
    return _get_cfg_key(cfg, "memory_adapter")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _make_vlm_call(planner: Any):
    """
    Extract a ``vlm_call`` callable from a planner instance.

    The planner is expected to have a ``model`` attribute (a ``RemoteModel``)
    with a ``respond(message_history)`` method that accepts a list of message
    dicts and returns a plain-text string.

    Returns ``None`` if the planner or its model is unavailable.
    """
    if planner is None:
        return None
    model = getattr(planner, "model", None)
    if model is None:
        return None
    respond = getattr(model, "respond", None)
    if respond is None:
        return None

    def vlm_call(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return respond(messages)

    return vlm_call


def create_memory_manager_from_config(cfg: Any, planner: Any = None) -> Optional["MemoryManager"]:
    """
    Create a ``MemoryManager`` from the top-level config object.

    Returns ``None`` when the package is unavailable, the ``memory`` key is
    absent, or ``memory.enabled`` is false.
    """
    if not _MEMORY_AVAILABLE:
        return None
    memory_cfg = _get_memory_cfg(cfg)
    if not _cfg_enabled(memory_cfg):
        return None
    try:
        mem_config = MemoryConfig.from_mapping(memory_cfg)
        vlm_call = _make_vlm_call(planner)
        manager = MemoryManager(config=mem_config, vlm_call=vlm_call)
        if _cfg_flag(memory_cfg, "load_on_start", default=True):
            try:
                manager.load()
            except Exception as e:
                logger.warning(f"[Memory] load() failed (non-fatal): {e}")
        logger.info(f"[Memory] MemoryManager created. storage_dir={mem_config.storage_dir}")
        return manager
    except Exception as e:
        logger.warning(f"[Memory] Failed to create MemoryManager: {e}")
        return None


def attach_memory_to_planner(planner: Any, memory_manager: Optional["MemoryManager"]) -> None:
    """
    Attach memory_manager to planner via set_memory_manager() if it exists.
    Safe no-op if planner doesn't support memory.
    """
    if memory_manager is None:
        return
    if hasattr(planner, "set_memory_manager"):
        planner.set_memory_manager(memory_manager)
        logger.info("[Memory] MemoryManager attached to planner.")
    else:
        logger.debug("[Memory] Planner does not support set_memory_manager(); skipping.")


def attach_memory_to_critic(critic: Any, memory_manager: Optional["MemoryManager"]) -> None:
    """
    Attach memory_manager to a critic (VLMCritic or DualCritic) via set_memory_manager().
    Safe no-op when critic is None, memory_manager is None, or critic lacks the method.
    """
    if critic is None or memory_manager is None:
        return
    if hasattr(critic, "set_memory_manager"):
        critic.set_memory_manager(memory_manager)
        logger.info("[Memory] MemoryManager attached to critic.")
    else:
        logger.debug("[Memory] Critic does not support set_memory_manager(); skipping.")


# ---------------------------------------------------------------------------
# MemoryAdapter lifecycle helpers
# ---------------------------------------------------------------------------

def create_memory_adapter_from_config(cfg: Any) -> Optional["MemoryAdapter"]:
    """
    Create a ``MemoryAdapter`` from the top-level config object.

    Returns ``None`` when the package is unavailable, the key is absent, or
    ``enabled`` is false.  Raises ``ValueError`` if ``enabled=True`` but
    neither ``model_name_or_path`` nor ``openai_model`` is set.
    """
    if not _ADAPTER_AVAILABLE:
        return None
    adapter_cfg = _get_adapter_cfg(cfg)
    if not _cfg_enabled(adapter_cfg):
        return None

    model_name   = _get_cfg_key(adapter_cfg, "model_name_or_path", None)
    openai_model = _get_cfg_key(adapter_cfg, "openai_model", None)

    if not model_name and not openai_model:
        raise ValueError(
            "[MemoryAdapter] memory_adapter.enabled=True but neither "
            "memory_adapter.model_name_or_path nor memory_adapter.openai_model is set."
        )

    try:
        adapter_config = MemoryAdapterConfig.from_mapping(adapter_cfg)
        adapter = MemoryAdapter(adapter_config)
        backend = adapter_config.openai_model or adapter_config.model_name_or_path
        logger.info(f"[MemoryAdapter] Created. model={backend}")
        return adapter
    except Exception as e:
        logger.warning(f"[MemoryAdapter] Failed to create MemoryAdapter: {e}")
        return None


def attach_memory_adapter_to_planner(planner: Any, memory_adapter: Optional["MemoryAdapter"]) -> None:
    """
    Attach memory_adapter to planner via set_memory_adapter() if it exists.
    Safe no-op if planner is None, adapter is None, or planner lacks the method.
    """
    if planner is None or memory_adapter is None:
        return
    if hasattr(planner, "set_memory_adapter"):
        planner.set_memory_adapter(memory_adapter)
        logger.info("[MemoryAdapter] Attached to planner.")
    else:
        logger.debug("[MemoryAdapter] Planner does not support set_memory_adapter(); skipping.")


def attach_memory_adapter_to_critic(critic: Any, memory_adapter: Optional["MemoryAdapter"]) -> None:
    """
    Attach memory_adapter to critic (VLMCritic or DualCritic) via set_memory_adapter().
    Safe no-op if critic is None, adapter is None, or critic lacks the method.
    """
    if critic is None or memory_adapter is None:
        return
    if hasattr(critic, "set_memory_adapter"):
        critic.set_memory_adapter(memory_adapter)
        logger.info("[MemoryAdapter] Attached to critic.")
    else:
        logger.debug("[MemoryAdapter] Critic does not support set_memory_adapter(); skipping.")


def unload_memory_adapter(memory_adapter: Optional["MemoryAdapter"]) -> None:
    """
    Call memory_adapter.unload() to free GPU/CPU resources at run end.

    Safe no-op if memory_adapter is None or lacks an unload() method.
    Never raises — logged exceptions only.
    """
    if memory_adapter is None:
        return
    if hasattr(memory_adapter, "unload"):
        try:
            memory_adapter.unload()
            logger.info("[MemoryAdapter] Unloaded model and freed resources.")
        except Exception as e:
            logger.warning(f"[MemoryAdapter] unload() failed (non-fatal): {e}")
    else:
        logger.debug("[MemoryAdapter] Adapter has no unload() method; skipping.")


# ---------------------------------------------------------------------------
# Experiment mode helpers
# ---------------------------------------------------------------------------

_VALID_EXPERIMENT_MODES = frozenset({
    "none",
    "raw_planner",
    "raw_planner_critic",
    "adapted_planner",
    "adapted_planner_critic",
})


def _get_experiment_mode(cfg: Any) -> Optional[str]:
    """
    Safely extract cfg.memory_experiment.mode.

    Returns None if the key is absent (backward-compat: caller uses old path).
    Raises ValueError for unknown modes.
    """
    exp = _get_cfg_key(cfg, "memory_experiment")
    if exp is None:
        return None

    mode = _get_cfg_key(exp, "mode", "none")
    if mode is None:
        return None

    mode = str(mode).strip().lower()
    if mode not in _VALID_EXPERIMENT_MODES:
        raise ValueError(
            f"[MemoryExperiment] Unknown mode {mode!r}. "
            f"Valid modes: {sorted(_VALID_EXPERIMENT_MODES)}"
        )
    return mode


def setup_memory_experiment(
    cfg: Any,
    planner: Any = None,
    critic: Any = None,
) -> tuple:
    """
    Create and attach MemoryManager and MemoryAdapter per ``cfg.memory_experiment.mode``.

    Modes: ``none`` | ``raw_planner`` | ``raw_planner_critic`` |
    ``adapted_planner`` | ``adapted_planner_critic``.

    Falls back to the legacy ``create_memory_manager_from_config`` path when
    ``cfg.memory_experiment`` is absent.  Returns ``(memory_manager, memory_adapter)``.
    """
    mode = _get_experiment_mode(cfg)

    # ------------------------------------------------------------------ #
    # Backward-compat path: no memory_experiment key in config            #
    # ------------------------------------------------------------------ #
    if mode is None:
        mm = create_memory_manager_from_config(cfg, planner=planner)
        attach_memory_to_planner(planner, mm)
        ma = create_memory_adapter_from_config(cfg)
        attach_memory_adapter_to_planner(planner, ma)
        attach_memory_to_critic(critic, mm)
        attach_memory_adapter_to_critic(critic, ma)
        return mm, ma

    logger.info(f"[MemoryExperiment] Mode: {mode!r}")

    # ------------------------------------------------------------------ #
    # none                                                                #
    # ------------------------------------------------------------------ #
    if mode == "none":
        return None, None

    # ------------------------------------------------------------------ #
    # raw_* modes — MemoryManager only, no adapter                       #
    # ------------------------------------------------------------------ #
    if mode in ("raw_planner", "raw_planner_critic"):
        mm = create_memory_manager_from_config(cfg, planner=planner)
        attach_memory_to_planner(planner, mm)
        if mode == "raw_planner_critic":
            attach_memory_to_critic(critic, mm)
        return mm, None

    # ------------------------------------------------------------------ #
    # adapted_* modes — MemoryManager + MemoryAdapter                    #
    # ------------------------------------------------------------------ #
    if mode in ("adapted_planner", "adapted_planner_critic"):
        mm = create_memory_manager_from_config(cfg, planner=planner)
        ma = create_memory_adapter_from_config(cfg)
        attach_memory_to_planner(planner, mm)
        attach_memory_adapter_to_planner(planner, ma)
        if mode == "adapted_planner_critic":
            attach_memory_to_critic(critic, mm)
            attach_memory_adapter_to_critic(critic, ma)
        return mm, ma

    # Should never reach here — _get_experiment_mode validates modes
    return None, None  # pragma: no cover


def compute_final_status(info: dict) -> str:
    """
    Map episode info to a final_status string for episodic memory.
      task_success=True  → "success"
      task_progress>0    → "partial"
      otherwise          → "failure"
    """
    if not isinstance(info, dict):
        return "unknown"
    task_success = info.get("task_success", 0)
    task_progress = info.get("task_progress", 0)
    if task_success:
        return "success"
    if task_progress and float(task_progress) > 0:
        return "partial"
    return "failure"


def _sanitize_metadata(d: dict) -> dict:
    """Keep only JSON-safe scalar values from a metadata dict."""
    safe = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) and v is not None:
            safe[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                pass
    return safe


def finalize_memory_episode(
    memory_manager: Optional["MemoryManager"],
    planner: Any,
    task_instruction: str,
    info: dict,
    env_name: str = "",
    scene_id: str = "",
    episode_idx: Optional[int] = None,
    extra_metadata: Optional[dict] = None,
) -> None:
    """
    Call memory_manager.finalize_episode() at the end of an episode.

    Safe no-op if memory_manager is None or memory is disabled.
    Never raises — logged exceptions only.
    """
    if memory_manager is None:
        return
    if not memory_manager.is_enabled():
        return
    try:
        final_status = compute_final_status(info)

        memory_manager.finalize_episode(
            task_instruction=task_instruction,
            final_status=final_status,
            env_name=env_name,
            scene_id=scene_id,
        )
        logger.info(
            f"[Memory] Episode finalized. status={final_status}, "
            f"instruction={task_instruction[:60]!r}"
        )
    except Exception as e:
        logger.warning(f"[Memory] finalize_memory_episode failed (non-fatal): {e}")


def save_memory_if_configured(
    memory_manager: Optional["MemoryManager"],
    cfg: Any,
    *,
    on_episode_end: bool = False,
    on_run_end: bool = False,
) -> None:
    """
    Call memory_manager.save() when the config flags say to.

    on_episode_end=True  → save if cfg.memory.save_on_episode_end
    on_run_end=True      → save if cfg.memory.save_on_end
    """
    if memory_manager is None:
        return
    memory_cfg = _get_memory_cfg(cfg)
    should_save = False
    if on_episode_end and _cfg_flag(memory_cfg, "save_on_episode_end", default=True):
        should_save = True
    if on_run_end and _cfg_flag(memory_cfg, "save_on_end", default=True):
        should_save = True
    if should_save:
        try:
            memory_manager.save()
            logger.debug("[Memory] Saved.")
        except Exception as e:
            logger.warning(f"[Memory] save() failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def create_metrics_from_config(cfg: Any) -> "MemoryExperimentMetrics":
    """
    Create a MemoryExperimentMetrics object for the current experiment mode.
    Safe to call even when the memory_experiment key is absent.
    """
    from embodiedbench.memory.metrics import MemoryExperimentMetrics
    mode = _get_experiment_mode(cfg) or "none"
    return MemoryExperimentMetrics(mode=mode)


def attach_metrics_to_planner(planner: Any, metrics: Any) -> None:
    """Inject a metrics object into the planner via set_metrics()."""
    if planner is not None and hasattr(planner, "set_metrics"):
        planner.set_metrics(metrics)


def attach_metrics_to_critic(critic: Any, metrics: Any) -> None:
    """Inject a metrics object into the critic via set_metrics()."""
    if critic is not None and hasattr(critic, "set_metrics"):
        critic.set_metrics(metrics)


def collect_episode_metrics(metrics: Any, episode_info: dict) -> None:
    """
    Copy episode-level outcome data from episode_info into a
    MemoryExperimentMetrics object.  Safe no-op when metrics is None.
    """
    if metrics is None or not isinstance(episode_info, dict):
        return
    success = episode_info.get("task_success")
    if success is not None:
        metrics.task_success = bool(success)
    progress = episode_info.get("task_progress")
    if progress is not None:
        metrics.task_progress = float(progress)
    if "num_steps" in episode_info:
        metrics.env_steps = int(episode_info["num_steps"])
    if "num_replans" in episode_info:
        metrics.replans = int(episode_info["num_replans"])
    if "num_invalid_actions" in episode_info:
        metrics.invalid_actions = int(episode_info["num_invalid_actions"])
    if "critic_total_rejections" in episode_info:
        metrics.critic_rejections = int(episode_info["critic_total_rejections"])


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def create_logger_from_config(cfg: Any) -> "MemoryExperimentLogger":
    """
    Build a ``MemoryExperimentLogger`` from ``cfg.memory_experiment``.

    Reads ``log_dir``, ``log_memory_outputs``, ``log_adapter_outputs``, and
    ``save_training_records``; returns a disabled logger when the key is absent.
    """
    from embodiedbench.memory.logging import MemoryExperimentLogger

    exp = _get_cfg_key(cfg, "memory_experiment")
    if exp is None:
        return MemoryExperimentLogger(log_dir="./memory_logs", enabled=False)

    enabled  = _cfg_flag(exp, "log_memory_outputs", True) or _cfg_flag(exp, "log_adapter_outputs", True)
    log_dir  = str(_get_cfg_key(exp, "log_dir", "./memory_logs"))
    save_tr  = _cfg_flag(exp, "save_training_records", True)

    return MemoryExperimentLogger(log_dir=log_dir, enabled=enabled, save_training_records=save_tr)
