"""
evaluation/metrics.py

Embodied-task metric functions: stale-memory recovery rate, per-episode
conversion from raw evaluator dicts, and aggregate roll-ups.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .schemas import AggregateMetrics, EpisodeResult

logger = logging.getLogger("EB_logger")


# ---------------------------------------------------------------------------
# Individual scalar helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(booleans: List[bool]) -> float:
    return sum(1 for b in booleans if b) / len(booleans) if booleans else 0.0


# ---------------------------------------------------------------------------
# Step 29D  —  Stale-memory recovery rate
# ---------------------------------------------------------------------------

def compute_stale_memory_recovery_rate(episodes: List[EpisodeResult]) -> float:
    """
    Fraction of stale-memory episodes in which the agent recovered.

    An episode counts as recovery when stale/conflicting memory was detected
    AND the agent still succeeded (``task_success`` or ``task_progress > 0.5``):

        rate = |{stale_detected AND recovered}| / |{stale_detected}|

    This is the core MemAdapt novelty metric. Returns 0.0 when no stale
    episodes are present.
    """
    stale_episodes = [e for e in episodes if e.stale_memory_detected]
    if not stale_episodes:
        return 0.0
    recovered = [
        e for e in stale_episodes
        if e.stale_memory_recovered or e.task_success or e.task_progress > 0.5
    ]
    rate = len(recovered) / len(stale_episodes)
    logger.debug(
        f"[Metrics] stale_episodes={len(stale_episodes)}, "
        f"recovered={len(recovered)}, rate={rate:.3f}"
    )
    return rate


# ---------------------------------------------------------------------------
# Episode-result → EpisodeResult mapping from raw evaluator dicts
# ---------------------------------------------------------------------------

def episode_result_from_evaluator_dict(
    d: Dict[str, Any],
    benchmark: str,
    mode: str,
    episode_id: str = "",
) -> EpisodeResult:
    """
    Convert a raw episode_info dict (as saved by existing evaluators) into an
    ``EpisodeResult``. All keys are optional with safe defaults.
    """
    mem = d.get("memory_metrics", {}) or {}

    stale_detected = bool(
        mem.get("stale_memory_detected", 0)
        or mem.get("stale_warnings_issued", 0)
        or d.get("stale_memory_detected", False)
    )
    stale_recovered = bool(
        mem.get("stale_memory_recovered", 0)
        or d.get("stale_memory_recovered", False)
    )

    return EpisodeResult(
        episode_id=episode_id or d.get("episode_id", ""),
        benchmark=benchmark,
        mode=mode,
        task_success=bool(d.get("task_success", 0)),
        task_progress=float(d.get("task_progress", d.get("avg_reward", 0.0))),
        num_steps=int(d.get("num_steps", d.get("env_step", 0))),
        num_replans=int(d.get("planner_steps", mem.get("replans", 0))),
        num_invalid_actions=int(d.get("num_invalid_actions", 0)),
        trajectory_length=int(d.get("trajectory_length", d.get("num_steps", 0))),
        runtime_seconds=float(d.get("episode_elapsed_seconds", 0.0)),
        planner_memory_usage=bool(mem.get("planner_calls", 0)),
        critic_memory_usage=bool(mem.get("critic_calls", 0)),
        adapter_used=bool(mem.get("adapter_calls", 0)),
        adapter_fallback=bool(mem.get("adapter_fallbacks", 0)),
        stale_memory_detected=stale_detected,
        stale_memory_recovered=stale_recovered,
        extra={k: v for k, v in d.items() if k not in {
            "task_success", "task_progress", "num_steps", "episode_elapsed_seconds",
            "memory_metrics",
        }},
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_aggregate_metrics(
    episodes: List[EpisodeResult],
    label: str = "",
    benchmark: str = "",
    mode: str = "",
) -> AggregateMetrics:
    """
    Roll up a list of EpisodeResult objects into AggregateMetrics.
    Returns an empty AggregateMetrics when *episodes* is empty.
    """
    n = len(episodes)
    if n == 0:
        return AggregateMetrics(label=label, benchmark=benchmark, mode=mode)

    return AggregateMetrics(
        label=label,
        benchmark=benchmark,
        mode=mode,
        num_episodes=n,
        success_rate=_rate([e.task_success for e in episodes]),
        avg_task_progress=_safe_mean([e.task_progress for e in episodes]),
        avg_steps=_safe_mean([float(e.num_steps) for e in episodes]),
        avg_replans=_safe_mean([float(e.num_replans) for e in episodes]),
        avg_invalid_actions=_safe_mean([float(e.num_invalid_actions) for e in episodes]),
        avg_trajectory_length=_safe_mean([float(e.trajectory_length) for e in episodes]),
        avg_runtime_seconds=_safe_mean([e.runtime_seconds for e in episodes]),
        planner_memory_usage_rate=_rate([e.planner_memory_usage for e in episodes]),
        critic_memory_usage_rate=_rate([e.critic_memory_usage for e in episodes]),
        adapter_usage_rate=_rate([e.adapter_used for e in episodes]),
        adapter_fallback_rate=_rate([e.adapter_fallback for e in episodes]),
        stale_detection_rate=_rate([e.stale_memory_detected for e in episodes]),
        stale_memory_recovery_rate=compute_stale_memory_recovery_rate(episodes),
    )
