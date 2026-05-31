"""
tests/memory/test_memory_critic_integration.py

Tests for critic-side memory injection (Step 12).
No real model APIs are called — all model interactions are mocked.
"""

from __future__ import annotations

import sys
import json
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies so critic.py can be imported without real services
# ---------------------------------------------------------------------------
for _mod in [
    "google", "google.generativeai", "openai", "anthropic",
    "lmdeploy", "pydantic", "typing_extensions", "cv2",
]:
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        _stub.__spec__ = None  # prevent pytest collection errors
        sys.modules[_mod] = _stub

# planner_utils — only the symbols used by critic are needed
_pu = types.ModuleType("embodiedbench.planner.planner_utils")
_pu.local_image_to_data_url = MagicMock(return_value="data:image/png;base64,abc")
_pu.fix_json = lambda x: x
_pu.template = ""
_pu.template_lang = ""
sys.modules.setdefault("embodiedbench.planner.planner_utils", _pu)

# critic imports system_prompts and critic_examples via _load_* helpers,
# but those are called lazily — stub the config module to avoid IO
_sp = types.ModuleType("embodiedbench.evaluator.config.system_prompts")
_sp.alfred_critic_system_prompt = (
    "Instruction: {instruction}\n"
    "Next action: {next_action}\n"
    "Full plan: {full_plan}\n"
    "Examples: {examples}"
)
_sp.habitat_critic_system_prompt = _sp.alfred_critic_system_prompt
sys.modules.setdefault("embodiedbench.evaluator.config", MagicMock())
sys.modules.setdefault("embodiedbench.evaluator.config.system_prompts", _sp)

# Patch _load_critic_examples so it never touches the filesystem
import unittest.mock as _um


# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------
with patch("embodiedbench.planner.critic._load_critic_examples", return_value=[]):
    from embodiedbench.planner.critic import VLMCritic, DualCritic, AlfredSymbolicCritic

# memory modules are always available in this workspace
from embodiedbench.memory.manager import MemoryManager, MemoryConfig
from embodiedbench.memory.base import MemoryQuery


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_model(response: dict = None):
    """Return a mock RemoteModel whose respond() returns a JSON string."""
    if response is None:
        response = {"valid": True, "reason": "looks good", "suggestions": ""}
    m = MagicMock()
    m.respond.return_value = json.dumps(response)
    return m


def _make_vlm_critic(**kwargs) -> VLMCritic:
    with patch("embodiedbench.planner.critic._load_critic_examples", return_value=[]):
        c = VLMCritic(
            model=_make_model(),
            model_name="gpt-4o",
            env="alfred",
            language_only=True,
            **kwargs,
        )
    return c


def _make_dual_critic(vlm_response: dict = None) -> DualCritic:
    sym = AlfredSymbolicCritic()
    vlm = _make_vlm_critic()
    if vlm_response:
        vlm.model.respond.return_value = json.dumps(vlm_response)
    return DualCritic(sym, vlm)


def _build_memory_manager(tmp_path) -> MemoryManager:
    mm = MemoryManager(config=MemoryConfig(
        enabled=True,
        storage_dir=str(tmp_path / "mem"),
    ))
    # Add some data so retrieval is non-empty
    node = mm.spatial.add_or_update_object(
        name="apple", room="kitchen table", step_id=0, confidence=0.9
    )
    node.stale = True  # force stale so warnings appear

    mm.temporal.append_step(
        task_instruction="put the apple in the fridge",
        action=0, action_text="find a apple",
        env_feedback="action failed: apple not visible",
        success=False, step_id=1,
    )
    mm.semantic.add_fact(
        content="Objects must be visible before they can be picked up.",
        category="precondition", confidence=0.99,
    )
    return mm


_REMAINING = [(0, "find a apple"), (1, "open the fridge")]
_INSTRUCTION = "put the apple in the fridge"


# ---------------------------------------------------------------------------
# 1. Critic without memory behaves unchanged
# ---------------------------------------------------------------------------

class TestCriticWithoutMemory:
    def test_evaluate_returns_valid_when_model_says_valid(self):
        c = _make_vlm_critic()
        result = c.evaluate(image_path="", instruction=_INSTRUCTION,
                            full_plan=_REMAINING, current_index=0)
        assert result["valid"] is True

    def test_evaluate_returns_invalid_when_model_says_invalid(self):
        c = _make_vlm_critic()
        c.model.respond.return_value = json.dumps(
            {"valid": False, "reason": "object not visible", "suggestions": "find it first"}
        )
        result = c.evaluate(image_path="", instruction=_INSTRUCTION,
                            full_plan=_REMAINING, current_index=0)
        assert result["valid"] is False
        assert "object not visible" in result["reason"]

    def test_evaluate_no_remaining_actions_valid(self):
        c = _make_vlm_critic()
        result = c.evaluate(image_path="", instruction=_INSTRUCTION,
                            full_plan=[], current_index=0)
        assert result["valid"] is True

    def test_memory_manager_is_none_by_default(self):
        c = _make_vlm_critic()
        assert c._memory_manager is None

    def test_memory_enabled_false_by_default(self):
        c = _make_vlm_critic()
        assert c._memory_enabled() is False


# ---------------------------------------------------------------------------
# 2. Critic accepts set_memory_manager()
# ---------------------------------------------------------------------------

class TestSetMemoryManager:
    def test_set_memory_manager_attaches_manager(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        assert c._memory_manager is mm

    def test_set_memory_manager_none_is_no_op(self):
        c = _make_vlm_critic()
        c.set_memory_manager(None)
        assert c._memory_manager is None

    def test_memory_enabled_true_after_attach(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        assert c._memory_enabled() is True

    def test_dual_critic_set_memory_manager_delegates_to_vlm(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        dc = _make_dual_critic()
        dc.set_memory_manager(mm)
        assert dc.vlm._memory_manager is mm


# ---------------------------------------------------------------------------
# 3. Critic memory prompt is empty when disabled
# ---------------------------------------------------------------------------

class TestMemoryPromptDisabled:
    def test_prompt_empty_when_no_manager(self):
        c = _make_vlm_critic()
        p = c._get_critic_memory_prompt(_INSTRUCTION, "find a apple", _REMAINING, current_index=0)
        assert p == ""

    def test_prompt_empty_when_manager_disabled(self, tmp_path):
        mm = MemoryManager(config=MemoryConfig(enabled=False,
                                               storage_dir=str(tmp_path)))
        c = _make_vlm_critic()
        c._memory_manager = mm  # bypass set_memory_manager guard
        assert c._memory_enabled() is False
        p = c._get_critic_memory_prompt(_INSTRUCTION, "find a apple", _REMAINING, current_index=0)
        assert p == ""

    def test_model_called_with_base_prompt_when_memory_disabled(self):
        """When memory is off, the prompt sent to the model should not contain memory header."""
        c = _make_vlm_critic()
        c.evaluate(image_path="", instruction=_INSTRUCTION, full_plan=_REMAINING, current_index=0)
        call_args = c.model.respond.call_args[0][0]  # first positional arg = messages list
        full_text = " ".join(
            part["text"] for msg in call_args for part in msg["content"]
            if part.get("type") == "text"
        )
        assert "[Retrieved Memory for Verification]" not in full_text


# ---------------------------------------------------------------------------
# 4. Critic memory prompt appears when enabled and data exists
# ---------------------------------------------------------------------------

class TestMemoryPromptEnabled:
    def test_get_critic_memory_prompt_returns_nonempty(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        p = c._get_critic_memory_prompt(_INSTRUCTION, "find a apple", _REMAINING, current_index=0)
        assert "[Retrieved Memory for Verification]" in p

    def test_evaluate_prepends_memory_to_prompt(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        c.evaluate(image_path="", instruction=_INSTRUCTION, full_plan=_REMAINING, current_index=0)
        call_args = c.model.respond.call_args[0][0]
        full_text = " ".join(
            part["text"] for msg in call_args for part in msg["content"]
            if part.get("type") == "text"
        )
        assert "[Retrieved Memory for Verification]" in full_text

    def test_evaluate_output_schema_unchanged_when_memory_enabled(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        result = c.evaluate(image_path="", instruction=_INSTRUCTION,
                            full_plan=_REMAINING, current_index=0)
        assert "valid" in result
        assert "reason" in result
        assert "suggestions" in result


# ---------------------------------------------------------------------------
# 5. Stale spatial warning appears in critic prompt
# ---------------------------------------------------------------------------

class TestStaleWarningInCriticPrompt:
    def test_stale_warning_present(self, tmp_path):
        mm = _build_memory_manager(tmp_path)  # apple node is stale
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        p = c._get_critic_memory_prompt(_INSTRUCTION, "find a apple", _REMAINING, current_index=0)
        # The stale warning section or stale keyword should be present
        assert (
            "stale" in p.lower()
            or "Stale Memory Warnings" in p
            or "override" in p.lower()
        )


# ---------------------------------------------------------------------------
# 6. Feasibility constraints appear in critic prompt
# ---------------------------------------------------------------------------

class TestFeasibilityConstraints:
    def test_feasibility_section_present(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        p = c._get_critic_memory_prompt(_INSTRUCTION, "find a apple", _REMAINING, current_index=0)
        assert "Feasibility" in p or "feasib" in p.lower() or "Constraints" in p

    def test_reject_plan_hint_present(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        p = c._get_critic_memory_prompt(_INSTRUCTION, "find a apple", _REMAINING, current_index=0)
        # At least one rejection-related keyword should be in the constraints
        assert any(kw in p.lower() for kw in ("reject", "manipulat", "precondition", "stale"))


# ---------------------------------------------------------------------------
# 7. Critic prompt contains no code fences or raw JSON examples
# ---------------------------------------------------------------------------

class TestCriticPromptFormat:
    def test_no_code_fences_in_memory_prompt(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        p = c._get_critic_memory_prompt(_INSTRUCTION, "find a apple", _REMAINING, current_index=0)
        assert "```" not in p

    def test_no_raw_json_in_memory_prompt(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        c = _make_vlm_critic()
        c.set_memory_manager(mm)
        p = c._get_critic_memory_prompt(_INSTRUCTION, "find a apple", _REMAINING, current_index=0)
        # Should not contain raw JSON structure braces at top level
        assert '{"valid"' not in p


# ---------------------------------------------------------------------------
# 8. DualCritic passes memory_manager to VLMCritic
# ---------------------------------------------------------------------------

class TestDualCriticMemory:
    def test_dual_critic_set_memory_manager(self, tmp_path):
        mm = _build_memory_manager(tmp_path)
        dc = _make_dual_critic()
        dc.set_memory_manager(mm)
        assert dc.vlm._memory_manager is mm

    def test_dual_critic_evaluate_uses_memory(self, tmp_path):
        """Memory prompt should be injected when DualCritic calls VLMCritic."""
        mm = _build_memory_manager(tmp_path)
        dc = _make_dual_critic()
        dc.set_memory_manager(mm)

        # Provide a valid scene so symbolic critic passes
        scene = [{"objectId": "apple|1", "objectType": "apple"}]
        result = dc.evaluate(
            action_id=0, action_str="find a apple",
            scene_objects=scene, num_actions=10,
            image_path="", instruction=_INSTRUCTION,
            full_plan=_REMAINING, current_index=0,
            is_first_step=False,
        )
        # Verify VLM was called and model received memory-augmented prompt
        assert dc.vlm.model.respond.called
        call_args = dc.vlm.model.respond.call_args[0][0]
        full_text = " ".join(
            part["text"] for msg in call_args for part in msg["content"]
            if part.get("type") == "text"
        )
        assert "[Retrieved Memory for Verification]" in full_text

    def test_dual_critic_first_step_skips_vlm_memory(self, tmp_path):
        """When is_first_step=True, VLM (and its memory injection) is skipped."""
        mm = _build_memory_manager(tmp_path)
        dc = _make_dual_critic()
        dc.set_memory_manager(mm)

        scene = [{"objectId": "apple|1", "objectType": "apple"}]
        result = dc.evaluate(
            action_id=0, action_str="find a apple",
            scene_objects=scene, num_actions=10,
            image_path="", instruction=_INSTRUCTION,
            full_plan=_REMAINING, current_index=0,
            is_first_step=True,
        )
        assert result["valid"] is True
        assert result["vlm_result"] is None
        # VLM model should NOT have been called
        dc.vlm.model.respond.assert_not_called()

    def test_dual_critic_disabled_memory_no_change(self):
        """With no memory manager attached, DualCritic behavior is identical to baseline."""
        dc = _make_dual_critic(vlm_response={"valid": True, "reason": "ok", "suggestions": ""})
        scene = [{"objectId": "apple|1", "objectType": "apple"}]
        result = dc.evaluate(
            action_id=0, action_str="find a apple",
            scene_objects=scene, num_actions=10,
            image_path="", instruction=_INSTRUCTION,
            full_plan=_REMAINING, current_index=0,
            is_first_step=False,
        )
        assert result["valid"] is True
