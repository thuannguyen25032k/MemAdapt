"""
tests/memory_adapter/test_memory_adapter_core.py

Unit tests for the Memory Adapter core module (Step 15).
No real HuggingFace model is downloaded or loaded.
A MockMemoryAdapter overrides generate() to return canned text.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from embodiedbench.memory_adapter.config import MemoryAdapterConfig
from embodiedbench.memory_adapter.schemas import MemoryAdapterInput, MemoryAdapterOutput
from embodiedbench.memory_adapter.prompts import build_adapter_prompt
from embodiedbench.memory_adapter.parsing import parse_adapter_output
from embodiedbench.memory_adapter.adapter import MemoryAdapter


# ---------------------------------------------------------------------------
# Canonical mock output — all five sections present and well-formed
# ---------------------------------------------------------------------------

_MOCK_OUTPUT = """\
<ADAPTED_CONTEXT>
The apple was last seen on the kitchen table at step 0, but has not been verified recently.
The fridge is in the kitchen with high confidence.
</ADAPTED_CONTEXT>

<FORESIGHT_PLAN>
- Navigate to the kitchen table area.
- Verify apple location with current observation.
- If apple is on table, pick it up.
- Open the fridge.
- Place apple inside the fridge.
</FORESIGHT_PLAN>

<FEASIBILITY_CRITERIA>
- Apple must be visible before pick-up action is valid.
- Fridge must be open before placing apple inside.
- Robot must not already be holding another object.
</FEASIBILITY_CRITERIA>

<STALE_MEMORY_ASSESSMENT>
- Apple location (kitchen table, step 0) is uncertain — verify before acting.
</STALE_MEMORY_ASSESSMENT>

<CONFIDENCE>
0.82
</CONFIDENCE>
"""

_MALFORMED_OUTPUT = "The robot should probably find the apple. Also the fridge is nearby."


# ---------------------------------------------------------------------------
# MockMemoryAdapter — overrides generate() to avoid any HF loading
# ---------------------------------------------------------------------------

class MockMemoryAdapter(MemoryAdapter):
    """Test double: overrides generate() with a canned response."""

    def __init__(self, mock_response: str = _MOCK_OUTPUT, **kwargs):
        # Do not call super().__init__() to avoid loading a real model.
        # Manually set required attributes.
        self._mock_response = mock_response
        self.config    = MemoryAdapterConfig(enabled=True, model_name_or_path="mock")
        self.tokenizer = None
        self.model     = None
        self.device    = "cpu"

    def generate(self, prompt: str) -> str:
        return self._mock_response


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_input(**kwargs) -> MemoryAdapterInput:
    defaults = dict(
        task_instruction="put the apple in the fridge",
        observation_text="kitchen table, fridge visible",
    )
    defaults.update(kwargs)
    return MemoryAdapterInput(**defaults)


def _make_mock_memory_context() -> MagicMock:
    ctx = MagicMock()
    ctx.combined_context = (
        "[Spatial Memory]\napple at kitchen table, step 0\n"
        "[Temporal Memory]\nfailed find at step 1\n"
    )
    ctx.is_empty.return_value = False
    return ctx


# ---------------------------------------------------------------------------
# 1. MemoryAdapterConfig.from_mapping
# ---------------------------------------------------------------------------

class TestConfig:
    def test_from_mapping_plain_dict(self):
        cfg = MemoryAdapterConfig.from_mapping({
            "model_name_or_path": "meta-llama/Llama-2-7b-hf",
            "max_new_tokens": 256,
            "device": "cpu",
        })
        assert cfg.model_name_or_path == "meta-llama/Llama-2-7b-hf"
        assert cfg.max_new_tokens == 256
        assert cfg.device == "cpu"

    def test_from_mapping_ignores_unknown_keys(self):
        cfg = MemoryAdapterConfig.from_mapping({"unknown_key": "val", "enabled": False})
        assert cfg.enabled is False

    def test_from_mapping_none_returns_defaults(self):
        cfg = MemoryAdapterConfig.from_mapping(None)
        assert cfg.enabled is True
        assert cfg.temperature == 0.0

    def test_to_dict_round_trip(self):
        cfg = MemoryAdapterConfig(model_name_or_path="test/model", max_new_tokens=128)
        d = cfg.to_dict()
        cfg2 = MemoryAdapterConfig.from_mapping(d)
        assert cfg2.model_name_or_path == "test/model"
        assert cfg2.max_new_tokens == 128

    def test_defaults(self):
        cfg = MemoryAdapterConfig()
        assert cfg.device == "auto"
        assert cfg.do_sample is False
        assert cfg.trust_remote_code is True


# ---------------------------------------------------------------------------
# 2. MemoryAdapterInput serialization
# ---------------------------------------------------------------------------

class TestInputSerialization:
    def test_to_dict_contains_all_fields(self):
        ai = _make_input(proposed_plan="find apple, pick up apple")
        d = ai.to_dict()
        assert "task_instruction" in d
        assert "observation_text" in d
        assert "proposed_plan" in d
        assert d["proposed_plan"] == "find apple, pick up apple"

    def test_from_dict_round_trip(self):
        ai = _make_input(mode="planner", proposed_action="find a apple")
        d = ai.to_dict()
        ai2 = MemoryAdapterInput.from_dict(d)
        assert ai2.task_instruction == ai.task_instruction
        assert ai2.mode == "planner"

    def test_memory_context_serialised_as_string(self):
        ctx = _make_mock_memory_context()
        ai = _make_input(memory_context=ctx)
        d = ai.to_dict()
        assert isinstance(d["memory_context"], str)
        assert "Spatial" in d["memory_context"]

    def test_none_memory_context_serialised_as_none(self):
        ai = _make_input()
        d = ai.to_dict()
        assert d["memory_context"] is None


# ---------------------------------------------------------------------------
# 3. MemoryAdapterOutput serialization
# ---------------------------------------------------------------------------

class TestOutputSerialization:
    def test_to_dict_contains_all_fields(self):
        out = MemoryAdapterOutput(
            adapted_context="ctx",
            foresight_plan=["step 1"],
            feasibility_criteria=["crit 1"],
            confidence=0.9,
        )
        d = out.to_dict()
        assert d["adapted_context"] == "ctx"
        assert d["foresight_plan"] == ["step 1"]
        assert d["confidence"] == 0.9

    def test_from_dict_round_trip(self):
        out = MemoryAdapterOutput(foresight_plan=["a", "b"], confidence=0.5)
        d = out.to_dict()
        out2 = MemoryAdapterOutput.from_dict(d)
        assert out2.foresight_plan == ["a", "b"]
        assert out2.confidence == 0.5

    def test_is_empty_true_for_blank(self):
        assert MemoryAdapterOutput().is_empty()

    def test_is_empty_false_when_context_set(self):
        out = MemoryAdapterOutput(adapted_context="something")
        assert not out.is_empty()


# ---------------------------------------------------------------------------
# 4 & 5. build_adapter_prompt
# ---------------------------------------------------------------------------

class TestBuildAdapterPrompt:
    def test_prompt_contains_task_instruction(self):
        ai  = _make_input()
        cfg = MemoryAdapterConfig()
        p   = build_adapter_prompt(ai, cfg)
        assert "put the apple in the fridge" in p

    def test_prompt_contains_memory_context(self):
        ctx = _make_mock_memory_context()
        ai  = _make_input(memory_context=ctx)
        cfg = MemoryAdapterConfig()
        p   = build_adapter_prompt(ai, cfg)
        assert "Spatial" in p or "apple" in p.lower()

    def test_prompt_contains_observation(self):
        ai  = _make_input(observation_text="fridge is open")
        cfg = MemoryAdapterConfig()
        p   = build_adapter_prompt(ai, cfg)
        assert "fridge is open" in p

    def test_prompt_no_memory_shows_placeholder(self):
        ai  = _make_input(memory_context=None)
        cfg = MemoryAdapterConfig()
        p   = build_adapter_prompt(ai, cfg)
        assert "no memory available" in p.lower()

    def test_prompt_respects_max_input_chars(self):
        ai  = _make_input(observation_text="x" * 10000)
        cfg = MemoryAdapterConfig(max_input_chars=500)
        p   = build_adapter_prompt(ai, cfg)
        # max_input_chars + len("\n... [prompt truncated]") = 500 + 22 = 522
        assert len(p) <= 530

    def test_proposed_plan_included(self):
        ai  = _make_input(proposed_plan="open fridge, place apple")
        cfg = MemoryAdapterConfig()
        p   = build_adapter_prompt(ai, cfg)
        assert "open fridge" in p


# ---------------------------------------------------------------------------
# 6–11. parse_adapter_output
# ---------------------------------------------------------------------------

class TestParseAdapterOutput:
    def test_parses_adapted_context(self):
        out = parse_adapter_output(_MOCK_OUTPUT)
        assert "apple" in out.adapted_context.lower()

    def test_parses_foresight_plan_bullets(self):
        out = parse_adapter_output(_MOCK_OUTPUT)
        assert len(out.foresight_plan) >= 3
        assert any("apple" in s.lower() for s in out.foresight_plan)

    def test_parses_feasibility_criteria_bullets(self):
        out = parse_adapter_output(_MOCK_OUTPUT)
        assert len(out.feasibility_criteria) >= 2
        assert any("visible" in c.lower() for c in out.feasibility_criteria)

    def test_parses_stale_warnings(self):
        out = parse_adapter_output(_MOCK_OUTPUT)
        assert len(out.stale_memory_assessment) >= 1
        assert any("uncertain" in w.lower() or "stale" in w.lower() or "verify" in w.lower()
                   for w in out.stale_memory_assessment)

    def test_parses_confidence(self):
        out = parse_adapter_output(_MOCK_OUTPUT)
        assert 0.0 <= out.confidence <= 1.0
        assert out.confidence == pytest.approx(0.82)

    def test_malformed_output_sets_parse_error(self):
        out = parse_adapter_output(_MALFORMED_OUTPUT)
        assert out.parse_error is not None
        assert len(out.parse_error) > 0

    def test_malformed_output_fallback_adapted_context(self):
        out = parse_adapter_output(_MALFORMED_OUTPUT)
        # On total parse failure the full text is used as adapted_context fallback
        assert len(out.adapted_context) > 0

    def test_empty_string_returns_parse_error(self):
        out = parse_adapter_output("")
        assert out.parse_error is not None

    def test_raw_output_preserved(self):
        out = parse_adapter_output(_MOCK_OUTPUT)
        assert out.raw_output == _MOCK_OUTPUT


# ---------------------------------------------------------------------------
# 12–15. MockMemoryAdapter.adapt + adapt_for_planner/critic
# ---------------------------------------------------------------------------

class TestMockAdapter:
    def test_adapt_returns_memory_adapter_output(self):
        adapter = MockMemoryAdapter()
        ai = _make_input()
        out = adapter.adapt(ai)
        assert isinstance(out, MemoryAdapterOutput)

    def test_adapt_output_not_empty(self):
        adapter = MockMemoryAdapter()
        ai = _make_input()
        out = adapter.adapt(ai)
        assert not out.is_empty()

    def test_adapt_for_planner_returns_string(self):
        adapter = MockMemoryAdapter()
        ctx = adapter.adapt_for_planner(
            task_instruction="put the apple in the fridge",
            observation_text="kitchen table visible",
        )
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_adapt_for_planner_contains_header(self):
        adapter = MockMemoryAdapter()
        ctx = adapter.adapt_for_planner(task_instruction="put the apple in the fridge")
        assert "[Adapted Memory for Planning]" in ctx

    def test_adapt_for_critic_returns_string(self):
        adapter = MockMemoryAdapter()
        ctx = adapter.adapt_for_critic(
            task_instruction="put the apple in the fridge",
            proposed_action="find a apple",
        )
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_adapt_for_critic_contains_header(self):
        adapter = MockMemoryAdapter()
        ctx = adapter.adapt_for_critic(task_instruction="put the apple in the fridge")
        assert "[Adapted Memory for Verification]" in ctx

    def test_empty_memory_context_still_works(self):
        adapter = MockMemoryAdapter()
        ai = MemoryAdapterInput(
            task_instruction="navigate to the kitchen",
            memory_context=None,
        )
        out = adapter.adapt(ai)
        assert isinstance(out, MemoryAdapterOutput)

    def test_disabled_adapter_returns_empty_output(self):
        adapter = MockMemoryAdapter()
        adapter.config.enabled = False
        out = adapter.adapt(_make_input())
        assert out.is_empty()
        assert out.parse_error is not None


# ---------------------------------------------------------------------------
# 16. No code fences in output contexts
# ---------------------------------------------------------------------------

class TestNoCodeFences:
    def test_planner_context_no_code_fences(self):
        adapter = MockMemoryAdapter()
        ctx = adapter.adapt_for_planner(task_instruction="test")
        assert "```" not in ctx

    def test_critic_context_no_code_fences(self):
        adapter = MockMemoryAdapter()
        ctx = adapter.adapt_for_critic(task_instruction="test")
        assert "```" not in ctx

    def test_adapted_context_no_code_fences(self):
        out = parse_adapter_output(_MOCK_OUTPUT)
        assert "```" not in out.adapted_context
