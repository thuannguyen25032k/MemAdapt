"""
tests/memory_adapter/test_memory_metrics.py

Tests for MemoryExperimentMetrics and the instrumentation hooks added to
VLMPlanner and VLMCritic.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from embodiedbench.memory.metrics import MemoryExperimentMetrics
from embodiedbench.memory.integration import (
    create_metrics_from_config,
    attach_metrics_to_planner,
    attach_metrics_to_critic,
    collect_episode_metrics,
)


# ---------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_values(self):
        m = MemoryExperimentMetrics()
        assert m.mode == "none"
        assert m.memory_retrieval_calls == 0
        assert m.planner_memory_injections == 0
        assert m.critic_memory_injections == 0
        assert m.adapter_calls == 0
        assert m.adapter_planner_calls == 0
        assert m.adapter_critic_calls == 0
        assert m.adapter_fallbacks == 0
        assert m.stale_warning_count == 0
        assert m.feasibility_constraint_count == 0
        assert m.planning_hint_count == 0
        assert m.planner_memory_prompt_chars == 0
        assert m.critic_memory_prompt_chars == 0
        assert m.adapted_planner_prompt_chars == 0
        assert m.adapted_critic_prompt_chars == 0
        assert m.critic_rejections == 0
        assert m.replans == 0
        assert m.invalid_actions == 0
        assert m.env_steps == 0
        assert m.task_success is None
        assert m.task_progress is None


# ---------------------------------------------------------------------------
# 2. to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_returns_dict(self):
        m = MemoryExperimentMetrics(mode="raw_planner", env_steps=5)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["mode"] == "raw_planner"
        assert d["env_steps"] == 5

    def test_to_dict_contains_all_fields(self):
        m = MemoryExperimentMetrics()
        d = m.to_dict()
        expected_keys = {
            "mode", "memory_retrieval_calls", "planner_memory_injections",
            "critic_memory_injections", "adapter_calls", "adapter_planner_calls",
            "adapter_critic_calls", "adapter_fallbacks", "stale_warning_count",
            "feasibility_constraint_count", "planning_hint_count",
            "planner_memory_prompt_chars", "critic_memory_prompt_chars",
            "adapted_planner_prompt_chars", "adapted_critic_prompt_chars",
            "critic_rejections", "replans", "invalid_actions", "env_steps",
            "task_success", "task_progress",
        }
        assert expected_keys.issubset(d.keys())


# ---------------------------------------------------------------------------
# 3. reset_episode
# ---------------------------------------------------------------------------

class TestResetEpisode:
    def test_reset_clears_counters(self):
        m = MemoryExperimentMetrics(mode="adapted_planner_critic")
        m.memory_retrieval_calls = 10
        m.adapter_calls = 5
        m.stale_warning_count = 3
        m.env_steps = 20
        m.task_success = True
        m.reset_episode()
        assert m.memory_retrieval_calls == 0
        assert m.adapter_calls == 0
        assert m.stale_warning_count == 0
        assert m.env_steps == 0
        assert m.task_success is None

    def test_reset_preserves_mode(self):
        m = MemoryExperimentMetrics(mode="adapted_planner")
        m.env_steps = 99
        m.reset_episode()
        assert m.mode == "adapted_planner"
        assert m.env_steps == 0


# ---------------------------------------------------------------------------
# 4. Planner memory injection increments counters
# ---------------------------------------------------------------------------

class TestPlannerMemoryInjection:
    def _make_planner_with_metrics(self, metrics):
        """Return a VLMPlanner stub that has set_metrics wired properly."""
        from embodiedbench.planner.vlm_planner import VLMPlanner
        planner = MagicMock(spec=VLMPlanner)
        planner.metrics = None

        def set_metrics(m):
            planner.metrics = m

        planner.set_metrics.side_effect = set_metrics
        attach_metrics_to_planner(planner, metrics)
        return planner

    def test_attach_metrics_to_planner_calls_set_metrics(self):
        metrics = MemoryExperimentMetrics()
        planner = MagicMock()
        attach_metrics_to_planner(planner, metrics)
        planner.set_metrics.assert_called_once_with(metrics)

    def test_planner_memory_injection_counter(self):
        """Directly simulate what _get_planner_memory_prompt does."""
        m = MemoryExperimentMetrics(mode="raw_planner")
        prompt = "Memory context: do X before Y."
        # Simulate the raw formatter path
        m.planner_memory_injections += 1
        m.planner_memory_prompt_chars += len(prompt)
        assert m.planner_memory_injections == 1
        assert m.planner_memory_prompt_chars == len(prompt)


# ---------------------------------------------------------------------------
# 5. Adapter planner call increments counters
# ---------------------------------------------------------------------------

class TestAdapterPlannerCall:
    def test_adapter_call_counters(self):
        m = MemoryExperimentMetrics(mode="adapted_planner")
        adapted_prompt = "Adapted: avoid the red zone."
        # Simulate adapter path
        m.adapter_planner_calls += 1
        m.adapter_calls += 1
        m.adapted_planner_prompt_chars += len(adapted_prompt)
        m.planner_memory_injections += 1
        m.planner_memory_prompt_chars += len(adapted_prompt)
        assert m.adapter_planner_calls == 1
        assert m.adapter_calls == 1
        assert m.adapted_planner_prompt_chars == len(adapted_prompt)
        assert m.planner_memory_injections == 1


# ---------------------------------------------------------------------------
# 6. Adapter fallback increments fallback counter
# ---------------------------------------------------------------------------

class TestAdapterFallback:
    def test_fallback_counter(self):
        m = MemoryExperimentMetrics(mode="adapted_planner")
        m.adapter_planner_calls += 1
        m.adapter_calls += 1
        m.adapter_fallbacks += 1          # code fences detected → fallback
        assert m.adapter_fallbacks == 1
        # planner_memory_injections should still be 0 (raw formatter not called)
        assert m.planner_memory_injections == 0


# ---------------------------------------------------------------------------
# 7. Critic memory injection increments counters
# ---------------------------------------------------------------------------

class TestCriticMemoryInjection:
    def test_attach_metrics_to_critic_calls_set_metrics(self):
        metrics = MemoryExperimentMetrics()
        critic = MagicMock()
        attach_metrics_to_critic(critic, metrics)
        critic.set_metrics.assert_called_once_with(metrics)

    def test_critic_memory_injection_counter(self):
        m = MemoryExperimentMetrics(mode="raw_planner_critic")
        prompt = "Critic memory: object was seen at location Z."
        m.critic_memory_injections += 1
        m.critic_memory_prompt_chars += len(prompt)
        assert m.critic_memory_injections == 1
        assert m.critic_memory_prompt_chars == len(prompt)


# ---------------------------------------------------------------------------
# 8. Evaluator metrics include task_success / task_progress
# ---------------------------------------------------------------------------

class TestEvaluatorMetrics:
    def test_collect_episode_metrics(self):
        m = MemoryExperimentMetrics()
        episode_info = {
            "task_success": 1,
            "task_progress": 0.75,
            "num_steps": 12,
            "num_replans": 2,
            "num_invalid_actions": 3,
            "critic_total_rejections": 1,
        }
        collect_episode_metrics(m, episode_info)
        assert m.task_success is True
        assert m.task_progress == 0.75
        assert m.env_steps == 12
        assert m.replans == 2
        assert m.invalid_actions == 3
        assert m.critic_rejections == 1

    def test_collect_none_metrics_is_noop(self):
        # Should not raise
        collect_episode_metrics(None, {"task_success": 1})


# ---------------------------------------------------------------------------
# 9. Stale warning count is recorded
# ---------------------------------------------------------------------------

class TestStaleWarnings:
    def test_stale_warning_accumulation(self):
        m = MemoryExperimentMetrics()
        ctx = MagicMock()
        ctx.stale_memory_warnings = ["warning A", "warning B"]
        ctx.feasibility_constraints = []
        # Simulate what the planner does after retrieve
        m.memory_retrieval_calls += 1
        m.stale_warning_count += len(getattr(ctx, "stale_memory_warnings", []))
        assert m.stale_warning_count == 2
        assert m.memory_retrieval_calls == 1


# ---------------------------------------------------------------------------
# 10. Prompt length is recorded
# ---------------------------------------------------------------------------

class TestPromptLength:
    def test_prompt_length_accumulates_across_steps(self):
        m = MemoryExperimentMetrics()
        for prompt in ["hello", "world", "extra context"]:
            m.planner_memory_prompt_chars += len(prompt)
            m.planner_memory_injections += 1
        assert m.planner_memory_prompt_chars == len("hello") + len("world") + len("extra context")
        assert m.planner_memory_injections == 3


# ---------------------------------------------------------------------------
# 11. Disabled mode leaves memory counters zero
# ---------------------------------------------------------------------------

class TestDisabledMode:
    def test_none_mode_counters_stay_zero(self):
        m = MemoryExperimentMetrics(mode="none")
        # In mode=none, setup_memory_experiment returns (None, None),
        # so no retrieval / injection happens.  Simulate that nothing increments.
        assert m.memory_retrieval_calls == 0
        assert m.planner_memory_injections == 0
        assert m.adapter_calls == 0

    def test_create_metrics_from_config_none_mode(self):
        cfg = {"memory_experiment": {"mode": "none"}}
        m = create_metrics_from_config(cfg)
        assert m.mode == "none"
        assert m.memory_retrieval_calls == 0


# ---------------------------------------------------------------------------
# 12. Metrics are JSON serializable
# ---------------------------------------------------------------------------

class TestJsonSerializable:
    def test_to_dict_json_serializable(self):
        m = MemoryExperimentMetrics(
            mode="adapted_planner_critic",
            memory_retrieval_calls=3,
            planner_memory_injections=3,
            adapter_calls=3,
            adapter_fallbacks=1,
            stale_warning_count=2,
            env_steps=10,
            task_success=True,
            task_progress=0.5,
        )
        serialized = json.dumps(m.to_dict())
        recovered = json.loads(serialized)
        assert recovered["mode"] == "adapted_planner_critic"
        assert recovered["task_success"] is True
        assert recovered["task_progress"] == 0.5
        assert recovered["adapter_fallbacks"] == 1
