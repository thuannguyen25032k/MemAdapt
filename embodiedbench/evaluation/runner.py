"""
evaluation/runner.py

Experiment runner. ``run_experiment(config)`` orchestrates adapter loading,
evaluator invocation, per-episode collection, summary aggregation, and JSON
persistence. The runner patches the existing evaluator config rather than
implementing simulator logic; tests inject episodes via the ``episode_fn`` seam.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from .schemas import (
    ExperimentConfig,
    ExperimentResult,
    VALID_MODES,
    VALID_BENCHMARKS,
)
from .metrics import compute_aggregate_metrics, episode_result_from_evaluator_dict

logger = logging.getLogger("EB_logger")

# ---------------------------------------------------------------------------
# Mode → memory_experiment.mode mapping
# ---------------------------------------------------------------------------

_MODE_TO_EXPERIMENT_MODE: Dict[str, str] = {
    "baseline": "none",
    "raw_memory": "raw_memory",
    "adapted_memory": "adapted_memory",
    "adapted_memory_planner_only": "adapted_memory_planner_only",
    "adapted_memory_critic_only": "adapted_memory_critic_only",
    "adapted_memory_planner_critic": "adapted_memory_planner_critic",
}


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

def _maybe_load_adapter(config: ExperimentConfig) -> Optional[Any]:
    """
    Load a LoRA adapter from ``config.adapter_checkpoint`` when the mode
    requires it. Returns the loaded PeftModel or None. Heavy imports are
    deferred so tests can run without peft / transformers installed.
    """
    adapter_modes = {
        "adapted_memory",
        "adapted_memory_planner_only",
        "adapted_memory_critic_only",
        "adapted_memory_planner_critic",
    }
    if config.mode not in adapter_modes:
        return None
    if not config.adapter_checkpoint:
        logger.warning(
            f"[Runner] Mode '{config.mode}' needs adapter_checkpoint but none provided."
        )
        return None
    if not os.path.isdir(config.adapter_checkpoint):
        raise FileNotFoundError(
            f"adapter_checkpoint not found: {config.adapter_checkpoint}"
        )
    logger.info(f"[Runner] Loading adapter from {config.adapter_checkpoint} …")
    try:
        from embodiedbench.memory_adapter_training.checkpoints import load_lora_adapter
        from embodiedbench.memory_adapter_training.modeling import load_base_model
        from embodiedbench.memory_adapter_training.config import MemoryAdapterTrainingConfig

        # Try to find an adjacent training_config.yaml
        cfg_path = os.path.join(config.adapter_checkpoint, "training_config.yaml")
        if os.path.isfile(cfg_path):
            train_cfg = MemoryAdapterTrainingConfig.from_yaml(cfg_path)
        else:
            train_cfg = MemoryAdapterTrainingConfig()
            train_cfg.model.model_name_or_path = config.model_name

        base = load_base_model(train_cfg)
        adapter = load_lora_adapter(base, config.adapter_checkpoint)
        logger.info("[Runner] Adapter loaded successfully.")
        return adapter
    except Exception as exc:
        logger.error(f"[Runner] Adapter load failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------

def _build_evaluator_config(exp_cfg: ExperimentConfig) -> Dict[str, Any]:
    """Build the ``config`` dict expected by existing EB evaluators."""
    ecfg: Dict[str, Any] = dict(exp_cfg.extra_config)
    ecfg.setdefault("model_name", exp_cfg.model_name)
    ecfg.setdefault("model_type", "api")
    ecfg.setdefault("eval_sets", exp_cfg.eval_sets)
    ecfg.setdefault("n_shots", 3)
    ecfg.setdefault("multistep", True)
    ecfg.setdefault("chat_history", False)
    ecfg.setdefault("language_only", False)
    ecfg.setdefault("multiview", False)
    ecfg.setdefault("visual_icl", False)
    ecfg.setdefault("resolution", "high")
    ecfg.setdefault("num_episodes", exp_cfg.num_episodes)

    # Memory experiment settings
    ecfg["memory_experiment"] = {
        "mode": _MODE_TO_EXPERIMENT_MODE.get(exp_cfg.mode, "none"),
        "log_memory_outputs": True,
        "log_adapter_outputs": True,
        "log_dir": os.path.join(exp_cfg.output_dir, "memory_logs"),
        "save_training_records": False,
        "adapter_checkpoint": exp_cfg.adapter_checkpoint or "",
    }
    return ecfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    episode_fn: Optional[Callable[[ExperimentConfig], List[Dict[str, Any]]]] = None,
) -> ExperimentResult:
    """
    Run a benchmark evaluation experiment.

    When *episode_fn* is provided (tests/CI), it is called with *config* and
    must return a list of raw episode-info dicts; otherwise the real evaluator
    is used. Returns an ExperimentResult with all episodes and summary set.
    """
    if config.benchmark not in VALID_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark {config.benchmark!r}. "
            f"Valid: {sorted(VALID_BENCHMARKS)}"
        )
    if config.mode not in VALID_MODES:
        raise ValueError(
            f"Unknown mode {config.mode!r}. Valid: {sorted(VALID_MODES)}"
        )

    os.makedirs(config.output_dir, exist_ok=True)
    started = time.time()
    started_ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    result = ExperimentResult(config=config, started_at=started_ts)

    # ------------------------------------------------------------------ #
    # 1. Load adapter (if needed)
    # ------------------------------------------------------------------ #
    if episode_fn is None:
        _maybe_load_adapter(config)

    # ------------------------------------------------------------------ #
    # 2. Collect raw episode dicts
    # ------------------------------------------------------------------ #
    if episode_fn is not None:
        raw_episodes = episode_fn(config)
    else:
        raw_episodes = _run_real_evaluator(config)

    # ------------------------------------------------------------------ #
    # 3. Convert to EpisodeResult
    # ------------------------------------------------------------------ #
    for idx, ep_dict in enumerate(raw_episodes):
        ep_result = episode_result_from_evaluator_dict(
            ep_dict,
            benchmark=config.benchmark,
            mode=config.mode,
            episode_id=ep_dict.get("episode_id", f"ep_{idx:04d}"),
        )
        result.episodes.append(ep_result)

    # ------------------------------------------------------------------ #
    # 4. Aggregate summary
    # ------------------------------------------------------------------ #
    agg = compute_aggregate_metrics(
        result.episodes,
        label=f"{config.mode}/{config.benchmark}",
        benchmark=config.benchmark,
        mode=config.mode,
    )
    result.summary = agg.to_dict()

    # ------------------------------------------------------------------ #
    # 5. Persist
    # ------------------------------------------------------------------ #
    finished = time.time()
    result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    result.total_runtime_seconds = finished - started

    if config.save_episode_jsons:
        out_path = os.path.join(
            config.output_dir, f"{config.experiment_id or config.mode}_result.json"
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(result.to_json())
        logger.info(f"[Runner] Results saved → {out_path}")

    logger.info(
        f"[Runner] Experiment done. "
        f"success_rate={agg.success_rate:.3f}, "
        f"stale_recovery={agg.stale_memory_recovery_rate:.3f}, "
        f"n_episodes={len(result.episodes)}"
    )
    return result


# ---------------------------------------------------------------------------
# Real evaluator dispatch (only called in live runs)
# ---------------------------------------------------------------------------

def _run_real_evaluator(config: ExperimentConfig) -> List[Dict[str, Any]]:
    """
    Build the evaluator config, instantiate the right EB evaluator, run it,
    then collect the saved per-episode JSON files into a list of dicts.
    """
    ev_cfg = _build_evaluator_config(config)

    bench = config.benchmark
    logger.info(f"[Runner] Launching real evaluator: {bench}")

    if bench == "eb_alfred":
        from embodiedbench.evaluator.eb_alfred_evaluator import EB_AlfredEvaluator
        evaluator = EB_AlfredEvaluator(ev_cfg)
    elif bench == "eb_habitat":
        from embodiedbench.evaluator.eb_habitat_evaluator import EB_HabitatEvaluator
        evaluator = EB_HabitatEvaluator(ev_cfg)
    elif bench == "eb_navigation":
        from embodiedbench.evaluator.eb_navigation_evaluator import EB_NavigationEvaluator
        evaluator = EB_NavigationEvaluator(ev_cfg)
    elif bench == "eb_manipulation":
        from embodiedbench.evaluator.eb_manipulation_evaluator import EB_ManipulationEvaluator
        evaluator = EB_ManipulationEvaluator(ev_cfg)
    else:
        raise ValueError(f"No evaluator for benchmark: {bench}")

    evaluator.evaluate_main()

    # Collect episode JSONs from the evaluator's log directory
    results_dir = _find_results_dir(config)
    return _load_episode_jsons(results_dir)


def _find_results_dir(config: ExperimentConfig) -> str:
    """Best-effort discovery of the results/ folder produced by the evaluator."""
    # Evaluators typically write to:  <log_root>/<model>/<eval_set>/results/
    # Fall back to config.output_dir if nothing found.
    candidate = os.path.join(config.output_dir, "results")
    if os.path.isdir(candidate):
        return candidate
    return config.output_dir


def _load_episode_jsons(directory: str) -> List[Dict[str, Any]]:
    """Load all episode_*.json files from ``directory`` into a list of dicts."""
    if not os.path.isdir(directory):
        logger.warning(f"[Runner] Results directory not found: {directory}")
        return []
    episodes = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("episode_") and fname.endswith(".json"):
            path = os.path.join(directory, fname)
            try:
                with open(path, encoding="utf-8") as fh:
                    episodes.append(json.load(fh))
            except Exception as exc:
                logger.warning(f"[Runner] Could not load {path}: {exc}")
    logger.info(f"[Runner] Loaded {len(episodes)} episode JSONs from {directory}")
    return episodes
