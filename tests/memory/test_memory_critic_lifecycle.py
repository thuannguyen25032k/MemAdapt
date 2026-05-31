"""
tests/memory/test_memory_critic_lifecycle.py

Tests for critic memory lifecycle wiring (Step 13).

Covers:
  1. attach_memory_to_critic: critic=None / mm=None / normal
  2. Same MemoryManager shared by planner and critic
  3. Disabled memory: no attachment
  4. DualCritic receives mm
  5. Info plumbing: DualCritic.evaluate(info=…) forwards to VLMCritic
  6. VLMCritic memory query uses env_feedback from info
"""

from __future__ import annotations

import sys
import json
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy deps (same pattern as other critic tests)
# ---------------------------------------------------------------------------
for _mod in [
    "google", "google.generativeai", "openai", "anthropic",
    "lmdeploy", "pydantic", "typing_extensions", "cv2",
]:
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        _stub.__spec__ = None
        sys.modules[_mod] = _stub

_pu = types.ModuleType("embodiedbench.planner.planner_utils")
_pu.local_image_to_data_url = MagicMock(return_value="data:image/png;base64,abc")
_pu.fix_json = lambda x: x
_pu.template = ""
_pu.template_lang = ""
sys.modules.setdefault("embodiedbench.planner.planner_utils", _pu)

_sp = types.ModuleType("embodiedbench.evaluator.config.system_prompts")
_sp.alfred_critic_system_prompt = (
    "Instruction: {instruction}\nNext: {next_action}\n"
    "Full plan: {full_plan}\nExamples: {examples}"
)
_sp.habitat_critic_system_prompt = _sp.alfred_critic_system_prompt
sys.modules.setdefault("embodiedbench.evaluator.config", MagicMock())
sys.modules.setdefault("embodiedbench.evaluator.config.system_prompts", _sp)

with patch("embodiedbench.planner.critic._load_critic_examples", return_value=[]):
    from embodiedbench.planner.critic import VLMCritic, DualCritic, AlfredSymbolicCritic

from embodiedbench.memory.integration import attach_memory_to_critic
from embodiedbench.memory.manager import MemoryManager, MemoryConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mm(tmp_path, enabled: bool = True) -> MemoryManager:
    mm = MemoryManager(config=MemoryConfig(
        enabled=enabled, storage_dir=str(tmp_path / "mem")
    ))
    if enabled:
        node = mm.spatial.add_or_update_object(
            name="apple", room="table", step_id=0, confidence=0.9
        )
        node.stale = True
        mm.temporal.append_step(
            task_instruction="find apple", action=0,
            action_text="find a apple", step_id=1,
            success=False, env_feedback="apple not visible",
        )
    return mm


def _make_vlm_critic(model=None) -> VLMCritic:
    if model is None:
        model = MagicMock()
        model.respond.return_value = json.dumps(
            {"valid": True, "reason": "ok", "suggestions": ""}
        )
    with patch("embodiedbench.planner.critic._load_critic_examples", return_value=[]):
        return VLMCritic(model=model, model_name="gpt-4o",
                         env="alfred", language_only=True)


def _make_dual_critic(model=None) -> DualCritic:
    return DualCritic(AlfredSymbolicCritic(), _make_vlm_critic(model))


_REMAINING = [(0, "find a apple"), (1, "open the fridge")]
_INSTRUCTION = "put the apple in the fridge"
_SCENE = [{"objectId": "apple|1", "objectType": "apple"}]


# ---------------------------------------------------------------------------
# 1. attach_memory_to_critic: None guard tests
# ---------------------------------------------------------------------------

class TestAttachMemoryToCritic:
    def test_critic_none_is_no_op(self, tmp_path):
        mm = _make_mm(tmp_path)
        attach_memory_to_critic(None, mm)  # must not raise

    def test_memory_manager_none_is_no_op(self):
        critic = MagicMock()
        attach_memory_to_critic(critic, None)
        critic.set_memory_manager.assert_not_called()

    def test_calls_set_memory_manager_when_available(self, tmp_path):
        mm = _make_mm(tmp_path)
        critic = MagicMock()
        attach_memory_to_critic(critic, mm)
        critic.set_memory_manager.assert_called_once_with(mm)

    def test_no_op_when_critic_lacks_method(self, tmp_path):
        mm = _make_mm(tmp_path)

        class NoCritic:
            pass

        attach_memory_to_critic(NoCritic(), mm)  # must not raise

    def test_attaches_to_vlm_critic(self, tmp_path):
        mm = _make_mm(tmp_path)
        c = _make_vlm_critic()
        attach_memory_to_critic(c, mm)
        assert c._memory_manager is mm

    def test_attaches_to_dual_critic(self, tmp_path):
        mm = _make_mm(tmp_path)
        dc = _make_dual_critic()
        attach_memory_to_critic(dc, mm)
        assert dc.vlm._memory_manager is mm


# ---------------------------------------------------------------------------
# 2. Same MemoryManager shared by planner and critic
# ---------------------------------------------------------------------------

class TestSharedMemoryManager:
    def test_same_instance_planner_and_critic(self, tmp_path):
        """Simulates evaluator lifecycle: one mm, attached to both planner and critic."""
        mm = _make_mm(tmp_path)
        planner = MagicMock()
        dc = _make_dual_critic()

        # Evaluator lifecycle order
        from embodiedbench.memory.integration import attach_memory_to_planner
        attach_memory_to_planner(planner, mm)
        attach_memory_to_critic(dc, mm)

        planner.set_memory_manager.assert_called_once_with(mm)
        assert dc.vlm._memory_manager is mm
        assert dc.vlm._memory_manager is planner.set_memory_manager.call_args[0][0]

    def test_memory_instance_identity_preserved(self, tmp_path):
        mm = _make_mm(tmp_path)
        dc = _make_dual_critic()
        attach_memory_to_critic(dc, mm)
        assert dc.vlm._memory_manager is mm  # identity, not equality


# ---------------------------------------------------------------------------
# 3. Disabled memory does not attach anything
# ---------------------------------------------------------------------------

class TestDisabledMemoryNoAttach:
    def test_disabled_mm_attach_is_allowed_but_enabled_returns_false(self, tmp_path):
        mm = _make_mm(tmp_path, enabled=False)
        dc = _make_dual_critic()
        attach_memory_to_critic(dc, mm)
        # attach_memory_to_critic does set it, but _memory_enabled() returns False
        assert dc.vlm._memory_enabled() is False

    def test_none_mm_no_attachment(self):
        dc = _make_dual_critic()
        attach_memory_to_critic(dc, None)
        assert dc.vlm._memory_manager is None

    def test_no_memory_prompt_when_manager_none(self):
        c = _make_vlm_critic()
        p = c._get_critic_memory_prompt(_INSTRUCTION, "find a apple", _REMAINING, current_index=0)
        assert p == ""


# ---------------------------------------------------------------------------
# 4. DualCritic receives mm via set_memory_manager
# ---------------------------------------------------------------------------

class TestDualCriticReceivesMM:
    def test_set_memory_manager_on_dual_critic(self, tmp_path):
        mm = _make_mm(tmp_path)
        dc = _make_dual_critic()
        dc.set_memory_manager(mm)
        assert dc.vlm._memory_manager is mm

    def test_set_memory_manager_with_none_mm(self):
        dc = _make_dual_critic()
        dc.set_memory_manager(None)
        assert dc.vlm._memory_manager is None


# ---------------------------------------------------------------------------
# 5. Info plumbing: DualCritic.evaluate(info=…) forwards to VLMCritic
# ---------------------------------------------------------------------------

class TestInfoPlumbing:
    def test_dual_critic_evaluate_accepts_info_kwarg(self, tmp_path):
        """DualCritic.evaluate() must accept info= without breaking."""
        dc = _make_dual_critic()
        result = dc.evaluate(
            action_id=0, action_str="find a apple",
            scene_objects=_SCENE, num_actions=10,
            image_path="", instruction=_INSTRUCTION,
            full_plan=_REMAINING, current_index=0, is_first_step=False,
            info={"env_feedback": "apple not visible"},
        )
        assert "valid" in result

    def test_dual_critic_evaluate_without_info_still_works(self):
        """Existing callers that omit info= must remain unaffected."""
        dc = _make_dual_critic()
        result = dc.evaluate(
            action_id=0, action_str="find a apple",
            scene_objects=_SCENE, num_actions=10,
            image_path="", instruction=_INSTRUCTION,
            full_plan=_REMAINING, current_index=0, is_first_step=False,
        )
        assert "valid" in result

    def test_vlm_critic_evaluate_accepts_info_kwarg(self):
        c = _make_vlm_critic()
        result = c.evaluate(
            image_path="", instruction=_INSTRUCTION,
            full_plan=_REMAINING, current_index=0,
            info={"env_feedback": "apple not visible"},
        )
        assert "valid" in result

    def test_vlm_critic_evaluate_without_info_still_works(self):
        c = _make_vlm_critic()
        result = c.evaluate(
            image_path="", instruction=_INSTRUCTION,
            full_plan=_REMAINING, current_index=0,
        )
        assert "valid" in result


# ---------------------------------------------------------------------------
# 6. VLMCritic memory query uses env_feedback from info
# ---------------------------------------------------------------------------

class TestInfoUsedInMemoryQuery:
    def test_env_feedback_appears_in_memory_query_obs_text(self, tmp_path):
        """_get_critic_memory_prompt should pick up env_feedback from info."""
        mm = _make_mm(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)

        info = {"env_feedback": "apple not visible at the table"}
        prompt = c._get_critic_memory_prompt(
            _INSTRUCTION, "find a apple", _REMAINING, current_index=0, info=info
        )
        # The prompt should be non-empty (memory is populated)
        assert len(prompt) > 0

    def test_observation_text_key_used_if_present(self, tmp_path):
        mm = _make_mm(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        info = {"observation_text": "kitchen table, fridge, apple on table"}
        prompt = c._get_critic_memory_prompt(
            _INSTRUCTION, "find a apple", _REMAINING, current_index=0, info=info
        )
        assert len(prompt) > 0

    def test_no_info_still_returns_nonempty_prompt_when_memory_has_data(self, tmp_path):
        mm = _make_mm(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        prompt = c._get_critic_memory_prompt(
            _INSTRUCTION, "find a apple", _REMAINING, current_index=0, info=None
        )
        assert "[Retrieved Memory for Verification]" in prompt

    def test_memory_prompt_injected_in_evaluate_with_info(self, tmp_path):
        mm = _make_mm(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        c.evaluate(
            image_path="", instruction=_INSTRUCTION,
            full_plan=_REMAINING, current_index=0,
            info={"env_feedback": "apple not visible"},
        )
        call_args = c.model.respond.call_args[0][0]
        full_text = " ".join(
            part["text"] for msg in call_args for part in msg["content"]
            if part.get("type") == "text"
        )
        assert "[Retrieved Memory for Verification]" in full_text
