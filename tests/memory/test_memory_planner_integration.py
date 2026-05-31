"""
tests/memory/test_memory_planner_integration.py

Integration tests for VLMPlanner + MemoryManager.

All heavy planner dependencies are mocked via sys.modules so no real API
calls or GPU initialisation occur.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out every heavy dependency BEFORE any planner import happens.
# ---------------------------------------------------------------------------
for _mod in [
    "google", "google.generativeai", "openai", "anthropic", "lmdeploy",
    "pydantic", "typing_extensions", "cv2",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Stub planner sub-modules that do top-level heavy imports
import types as _types

_pu = _types.ModuleType("embodiedbench.planner.planner_utils")
_pu.local_image_to_data_url = MagicMock(return_value="data:image/png;base64,abc")
_pu.template = ""
_pu.template_lang = ""
_pu.fix_json = lambda x: x
sys.modules["embodiedbench.planner.planner_utils"] = _pu

_rm = _types.ModuleType("embodiedbench.planner.remote_model")
_rm.RemoteModel = MagicMock()
sys.modules["embodiedbench.planner.remote_model"] = _rm

_cm = _types.ModuleType("embodiedbench.planner.custom_model")
_cm.CustomModel = MagicMock()
sys.modules["embodiedbench.planner.custom_model"] = _cm

_gg = _types.ModuleType("embodiedbench.planner.planner_config.generation_guide")
_gg.llm_generation_guide = ""
_gg.vlm_generation_guide = ""
sys.modules["embodiedbench.planner.planner_config"] = _types.ModuleType("embodiedbench.planner.planner_config")
sys.modules["embodiedbench.planner.planner_config.generation_guide"] = _gg

import pytest
from embodiedbench.memory.manager import MemoryManager, MemoryConfig
from embodiedbench.memory.prompt_formatter import MemoryPromptFormatter
from embodiedbench.memory.base import MemoryContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_planner():
    from embodiedbench.planner.vlm_planner import VLMPlanner

    with patch("embodiedbench.planner.vlm_planner.RemoteModel") as MockRemote, \
         patch("embodiedbench.planner.vlm_planner.CustomModel"):
        mock_model = MagicMock()
        mock_model.respond.return_value = (
            '{"visual_state": "apple on table", "reasoning": "pick up apple", '
            '"plan": "find then pick", '
            '"executable_plan": [{"action_id": 1, "action_name": "pick up the apple"}]}'
        )
        MockRemote.return_value = mock_model

        actions = ["find a apple", "pick up the apple", "put down apple", "done"]
        planner = VLMPlanner(
            model_name="gpt-4o",
            model_type="openai",
            actions=actions,
            system_prompt="System {0} {1} {2}",
            examples=[],
            n_shot=0,
            language_only=True,
        )
        planner.model = mock_model
    return planner


def _make_memory_manager() -> MemoryManager:
    return MemoryManager(config=MemoryConfig(enabled=True, top_k_per_memory=3))


# ---------------------------------------------------------------------------
# Tests: default (no memory) behavior
# ---------------------------------------------------------------------------

class TestVLMPlannerWithoutMemory:
    def test_construction_without_memory_has_safe_defaults(self):
        p = _make_stub_planner()
        assert p.memory_manager is None
        assert p.last_memory_context is None
        assert p.last_memory_prompt == ""
        assert p.current_instruction == ""

    def test_memory_enabled_returns_false_by_default(self):
        assert _make_stub_planner()._memory_enabled() is False

    def test_get_planner_memory_prompt_returns_empty_when_disabled(self):
        assert _make_stub_planner()._get_planner_memory_prompt("pick up apple") == ""

    def test_update_info_works_without_memory(self):
        p = _make_stub_planner()
        p.reset()
        p.update_info({"action_id": 0, "env_feedback": "ok", "last_action_success": True, "env_step": 1})
        assert len(p.episode_act_feedback) == 1

    def test_reset_works_without_memory(self):
        p = _make_stub_planner()
        p.reset()
        assert p.episode_act_feedback == []
        assert p.last_memory_context is None
        assert p.last_memory_prompt == ""


# ---------------------------------------------------------------------------
# Tests: set_memory_manager / _memory_enabled
# ---------------------------------------------------------------------------

class TestVLMPlannerSetMemoryManager:
    def test_set_memory_manager_attaches_manager(self):
        p = _make_stub_planner()
        mm = _make_memory_manager()
        p.set_memory_manager(mm)
        assert p.memory_manager is mm

    def test_memory_enabled_true_after_set_manager(self):
        p = _make_stub_planner()
        p.set_memory_manager(_make_memory_manager())
        assert p._memory_enabled() is True

    def test_set_memory_manager_none_disables(self):
        p = _make_stub_planner()
        p.set_memory_manager(_make_memory_manager())
        p.set_memory_manager(None)
        assert p._memory_enabled() is False


# ---------------------------------------------------------------------------
# Tests: _get_planner_memory_prompt
# ---------------------------------------------------------------------------

class TestVLMPlannerMemoryPrompt:
    def test_returns_string_when_enabled_but_empty_memory(self):
        p = _make_stub_planner()
        p.set_memory_manager(_make_memory_manager())
        assert isinstance(p._get_planner_memory_prompt("pick up apple"), str)

    def test_returns_formatted_text_after_memory_populated(self):
        p = _make_stub_planner()
        mm = _make_memory_manager()
        p.set_memory_manager(mm)
        p.reset()
        for i in range(3):
            mm.update(task_instruction="pick up apple", action_text="find a apple",
                      env_feedback="ok", step_id=i)
        result = p._get_planner_memory_prompt("pick up apple")
        assert "[Retrieved Memory for Planning]" in result

    def test_memory_prompt_has_no_code_fences(self):
        p = _make_stub_planner()
        mm = _make_memory_manager()
        p.set_memory_manager(mm)
        mm.update(task_instruction="pick apple", action_text="find a apple", step_id=0)
        assert "```" not in p._get_planner_memory_prompt("pick apple")

    def test_memory_prompt_has_no_raw_json_action_output(self):
        p = _make_stub_planner()
        mm = _make_memory_manager()
        p.set_memory_manager(mm)
        mm.update(task_instruction="pick apple", action_text="find a apple", step_id=0)
        result = p._get_planner_memory_prompt("pick apple")
        assert '"executable_plan"' not in result


# ---------------------------------------------------------------------------
# Tests: update_info with memory
# ---------------------------------------------------------------------------

class TestVLMPlannerUpdateInfoWithMemory:
    def test_update_info_increments_temporal_memory(self):
        p = _make_stub_planner()
        mm = _make_memory_manager()
        p.set_memory_manager(mm)
        p.reset()
        p.current_instruction = "pick up apple"
        p.update_info({"action_id": 0, "env_feedback": "moved", "last_action_success": True, "env_step": 1})
        assert len(mm.temporal) == 1

    def test_update_info_does_not_inject_critic_sentinel(self):
        p = _make_stub_planner()
        mm = _make_memory_manager()
        p.set_memory_manager(mm)
        p.reset()
        p.current_instruction = "pick up apple"
        p.update_info({"action_id": 1, "env_feedback": "picked", "last_action_success": True, "env_step": 2})
        for fb in p.episode_act_feedback:
            assert fb[0] != -3, "Memory must not inject critic sentinel -3"


# ---------------------------------------------------------------------------
# Tests: reset with memory
# ---------------------------------------------------------------------------

class TestVLMPlannerResetWithMemory:
    def test_reset_clears_temporal_but_preserves_episodic(self):
        p = _make_stub_planner()
        mm = _make_memory_manager()
        p.set_memory_manager(mm)
        p.reset()
        p.current_instruction = "task"
        mm.episodic.add_episode_from_trajectory(task_instruction="old task", env_name="alfred", final_status="success")
        mm.update(task_instruction="task", action_text="find a apple", step_id=0)
        assert len(mm.temporal) == 1
        assert len(mm.episodic.episodes) == 1

        p.reset()
        assert len(mm.temporal) == 0
        assert len(mm.episodic.episodes) == 1

    def test_reset_clears_last_memory_state(self):
        p = _make_stub_planner()
        mm = _make_memory_manager()
        p.set_memory_manager(mm)
        p.last_memory_context = MemoryContext()
        p.last_memory_prompt = "old"
        p.reset()
        assert p.last_memory_context is None
        assert p.last_memory_prompt == ""
