"""
memory/metrics.py

Lightweight per-episode metrics for MemAdapt ablation experiments.

Usage
-----
    from embodiedbench.memory.metrics import MemoryExperimentMetrics

    m = MemoryExperimentMetrics(mode="adapted_planner_critic")
    # planner/critic increment counters via set_metrics() injection
    episode_info["memory_metrics"] = m.to_dict()
    m.reset_episode()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryExperimentMetrics:
    """Accumulates per-episode memory/adaptation counters for ablation analysis."""

    # Experiment configuration
    mode: str = "none"

    # --- Memory retrieval ---
    memory_retrieval_calls: int = 0       # times MemoryManager.retrieve() was called
    planner_memory_injections: int = 0    # times a non-empty memory prompt reached the planner
    critic_memory_injections: int = 0     # times a non-empty memory prompt reached the critic

    # --- Adapter usage ---
    adapter_calls: int = 0                # total MemoryAdapter.adapt() calls (planner + critic)
    adapter_planner_calls: int = 0        # adapt() calls from the planner path
    adapter_critic_calls: int = 0         # adapt() calls from the critic path
    adapter_fallbacks: int = 0            # times adapter failed / returned empty → raw fallback

    # --- Prompt/context lengths (chars) ---
    planner_memory_prompt_chars: int = 0  # total chars of memory prompts given to planner
    critic_memory_prompt_chars: int = 0   # total chars of memory prompts given to critic
    adapted_planner_prompt_chars: int = 0 # total chars of adapter-produced planner context
    adapted_critic_prompt_chars: int = 0  # total chars of adapter-produced critic context

    # --- Planning hints ---
    planning_hint_count: int = 0          # planning hints surfaced across the episode

    # --- Episode outcome ---
    env_steps: int = 0                    # number of environment steps taken
    task_success: Optional[bool] = None   # whether the task was completed successfully
    task_progress: Optional[float] = None # fractional task progress [0, 1]
    critic_rejections: int = 0            # number of critic rejections
    replans: int = 0                      # number of replanning events
    invalid_actions: int = 0              # number of invalid actions issued

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-safe dict of all fields."""
        d: dict = {
            "mode": self.mode,
            "memory_retrieval_calls": self.memory_retrieval_calls,
            "planner_memory_injections": self.planner_memory_injections,
            "critic_memory_injections": self.critic_memory_injections,
            "adapter_calls": self.adapter_calls,
            "adapter_planner_calls": self.adapter_planner_calls,
            "adapter_critic_calls": self.adapter_critic_calls,
            "adapter_fallbacks": self.adapter_fallbacks,
            "planning_hint_count": self.planning_hint_count,
            "planner_memory_prompt_chars": self.planner_memory_prompt_chars,
            "critic_memory_prompt_chars": self.critic_memory_prompt_chars,
            "adapted_planner_prompt_chars": self.adapted_planner_prompt_chars,
            "adapted_critic_prompt_chars": self.adapted_critic_prompt_chars,
            "critic_rejections": self.critic_rejections,
            "replans": self.replans,
            "invalid_actions": self.invalid_actions,
            "env_steps": self.env_steps,
            "task_success": self.task_success,
            "task_progress": self.task_progress,
        }
        return d

    def reset_episode(self) -> None:
        """
        Reset all per-episode counters back to zero/None.
        Preserves ``mode`` so the object can be reused across episodes.
        """
        self.memory_retrieval_calls = 0
        self.planner_memory_injections = 0
        self.critic_memory_injections = 0
        self.adapter_calls = 0
        self.adapter_planner_calls = 0
        self.adapter_critic_calls = 0
        self.adapter_fallbacks = 0
        self.planning_hint_count = 0
        self.planner_memory_prompt_chars = 0
        self.critic_memory_prompt_chars = 0
        self.adapted_planner_prompt_chars = 0
        self.adapted_critic_prompt_chars = 0
        self.critic_rejections = 0
        self.replans = 0
        self.invalid_actions = 0
        self.env_steps = 0
        self.task_success = None
        self.task_progress = None
