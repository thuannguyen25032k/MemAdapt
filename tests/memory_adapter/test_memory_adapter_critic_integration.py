"""
Tests for MemoryAdapter integration into VLMCritic._get_critic_memory_prompt().

Strategy
--------
* Stub all heavy imports before importing critic.py.
* MockMemoryAdapter bypasses _load_model(); adapt() is directly overridable.
* All tests use minimal VLMCritic instances built without a real model.
"""

import sys
import types
import importlib.machinery
import pytest


# ---------------------------------------------------------------------------
# Stubs for heavy dependencies
# ---------------------------------------------------------------------------

def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return mod


for _pkg in [
    "openai", "anthropic", "google", "google.generativeai",
    "lmdeploy", "lmdeploy.serve", "lmdeploy.serve.openai",
    "lmdeploy.serve.openai.api_client",
    "cv2", "transformers", "torch", "torch.cuda",
    "bitsandbytes",
]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_stub(_pkg)

sys.modules["torch"].cuda = sys.modules["torch.cuda"]
sys.modules["torch.cuda"].is_available = lambda: False

# ---------------------------------------------------------------------------
# Project imports (safe after stubs)
# ---------------------------------------------------------------------------

from embodiedbench.memory_adapter.schemas import MemoryAdapterInput, MemoryAdapterOutput
from embodiedbench.memory_adapter.config import MemoryAdapterConfig
from embodiedbench.memory_adapter.adapter import MemoryAdapter

# ---------------------------------------------------------------------------
# MockMemoryAdapter
# ---------------------------------------------------------------------------

_MOCK_CRITIC_OUTPUT = (
    "[Adapted Context]\n"
    "Object is reachable from current position.\n"
    "[Foresight Plan]\n"
    "- approach table\n"
    "- grasp mug\n"
    "[Feasibility Criteria]\n"
    "- mug is not obstructed\n"
    "[Stale Memory Assessment]\n"
    "- object position may have shifted\n"
    "[Confidence]\n"
    "0.85\n"
)


class MockMemoryAdapter(MemoryAdapter):
    def __init__(self, config: MemoryAdapterConfig = None):
        if config is None:
            config = MemoryAdapterConfig(model_name_or_path="mock")
        self.config = config
        self.model = None
        self.tokenizer = None
        self._adapt_calls: list = []

    def generate(self, prompt: str) -> str:
        return _MOCK_CRITIC_OUTPUT

    def adapt(self, adapter_input: MemoryAdapterInput) -> MemoryAdapterOutput:
        self._adapt_calls.append(adapter_input)
        return super().adapt(adapter_input)


class ErrorMemoryAdapter(MockMemoryAdapter):
    def adapt(self, adapter_input):
        raise RuntimeError("simulated adapter error")


class EmptyOutputAdapter(MockMemoryAdapter):
    def adapt(self, adapter_input):
        return MemoryAdapterOutput(adapted_context="", critic_context="")


class CodeFenceAdapter(MockMemoryAdapter):
    def adapt(self, adapter_input):
        return MemoryAdapterOutput(
            adapted_context="stuff",
            critic_context="```json\n{\"action_id\": 3}\n```",
        )


class JsonSchemaAdapter(MockMemoryAdapter):
    def adapt(self, adapter_input):
        return MemoryAdapterOutput(
            adapted_context="stuff",
            critic_context='Some text with "action_id" embedded in it.',
        )


# ---------------------------------------------------------------------------
# Fake memory infrastructure
# ---------------------------------------------------------------------------

class FakeMemoryContext:
    def __init__(self, empty=False):
        self._empty = empty
        self.episodes = []

    def is_empty(self):
        return self._empty


class FakeMemoryQuery:
    observation_text = "robot near table"
    env_name = "test-env"
    scene_id = "s0"
    step_id  = 2


class FakeMemoryManager:
    def __init__(self, return_empty=False):
        self._return_empty = return_empty

    def is_enabled(self):
        return True

    def retrieve(self, query):
        return FakeMemoryContext(empty=self._return_empty)


class FakeFormatter:
    def format_for_critic(self, ctx):
        return "[Raw Critic Memory] raw fallback text"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _make_critic(adapter=None, memory_enabled=True, return_empty_ctx=False):
    from embodiedbench.planner.critic import VLMCritic

    critic = object.__new__(VLMCritic)
    critic._memory_manager   = FakeMemoryManager(return_empty=return_empty_ctx) if memory_enabled else None
    critic._memory_formatter = FakeFormatter()
    critic.memory_adapter             = None
    critic.last_adapted_memory_output = None
    critic.last_adapted_memory_prompt = ""
    critic.metrics = None

    # Avoid loading external prompt files
    critic._system_prompt_template = ""
    critic._examples = []
    critic.model = None
    critic.model_name = "mock"
    critic.language_only = True
    critic.n_shot = 0

    if adapter is not None:
        critic.set_memory_adapter(adapter)

    return critic


_REMAINING = [(3, "pick up the mug"), (4, "place mug on counter")]

# ---------------------------------------------------------------------------
# Tests: API surface
# ---------------------------------------------------------------------------

class TestVLMCriticAdapterAPI:
    def test_has_set_memory_adapter(self):
        from embodiedbench.planner.critic import VLMCritic
        assert hasattr(VLMCritic, "set_memory_adapter")

    def test_adapter_enabled_false_without_adapter(self):
        c = _make_critic()
        assert c._adapter_enabled() is False

    def test_adapter_enabled_true_with_mock(self):
        c = _make_critic(adapter=MockMemoryAdapter())
        assert c._adapter_enabled() is True

    def test_adapter_enabled_false_when_config_disabled(self):
        cfg = MemoryAdapterConfig(model_name_or_path="m", enabled=False)
        c = _make_critic(adapter=MockMemoryAdapter(cfg))
        assert c._adapter_enabled() is False

    def test_set_memory_adapter_stores_instance(self):
        c = _make_critic()
        adapter = MockMemoryAdapter()
        c.set_memory_adapter(adapter)
        assert c.memory_adapter is adapter


# ---------------------------------------------------------------------------
# Tests: raw formatter (no adapter)
# ---------------------------------------------------------------------------

class TestRawFormatterPath:
    def test_no_adapter_uses_raw_formatter(self):
        c = _make_critic()
        result = c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert result == "[Raw Critic Memory] raw fallback text"

    def test_memory_disabled_returns_empty(self):
        c = _make_critic(memory_enabled=False)
        result = c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert result == ""

    def test_empty_context_returns_empty(self):
        c = _make_critic(return_empty_ctx=True)
        result = c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: adapter-enabled path
# ---------------------------------------------------------------------------

class TestAdapterEnabledPath:
    def test_returns_adapted_critic_context(self):
        c = _make_critic(adapter=MockMemoryAdapter())
        result = c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert "[Adapted Memory for Verification]" in result

    def test_adapt_called_with_mode_critic(self):
        adapter = MockMemoryAdapter()
        c = _make_critic(adapter=adapter)
        c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert len(adapter._adapt_calls) == 1
        assert adapter._adapt_calls[0].mode == "critic"

    def test_adapt_receives_proposed_action(self):
        adapter = MockMemoryAdapter()
        c = _make_critic(adapter=adapter)
        c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert adapter._adapt_calls[0].proposed_action == "action id 3, pick up mug"

    def test_adapt_receives_memory_context(self):
        adapter = MockMemoryAdapter()
        c = _make_critic(adapter=adapter)
        c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        ctx = adapter._adapt_calls[0].memory_context
        assert isinstance(ctx, FakeMemoryContext)

    def test_last_adapted_memory_output_stored(self):
        c = _make_critic(adapter=MockMemoryAdapter())
        c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert c.last_adapted_memory_output is not None
        assert isinstance(c.last_adapted_memory_output, MemoryAdapterOutput)

    def test_last_adapted_memory_prompt_stored(self):
        c = _make_critic(adapter=MockMemoryAdapter())
        result = c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert c.last_adapted_memory_prompt == result


# ---------------------------------------------------------------------------
# Tests: fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    def test_adapter_exception_falls_back_to_raw(self):
        c = _make_critic(adapter=ErrorMemoryAdapter())
        result = c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert result == "[Raw Critic Memory] raw fallback text"

    def test_empty_adapter_output_falls_back_to_raw(self):
        c = _make_critic(adapter=EmptyOutputAdapter())
        result = c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert result == "[Raw Critic Memory] raw fallback text"

    def test_code_fence_output_falls_back_to_raw(self):
        c = _make_critic(adapter=CodeFenceAdapter())
        result = c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert result == "[Raw Critic Memory] raw fallback text"

    def test_json_schema_output_falls_back_to_raw(self):
        c = _make_critic(adapter=JsonSchemaAdapter())
        result = c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert result == "[Raw Critic Memory] raw fallback text"

    def test_disabled_adapter_falls_back_to_raw(self):
        cfg = MemoryAdapterConfig(model_name_or_path="m", enabled=False)
        c = _make_critic(adapter=MockMemoryAdapter(cfg))
        result = c._get_critic_memory_prompt("task", "action id 3, pick up mug", _REMAINING)
        assert result == "[Raw Critic Memory] raw fallback text"


# ---------------------------------------------------------------------------
# Tests: DualCritic forwarding
# ---------------------------------------------------------------------------

class TestDualCriticForwarding:
    def _make_dual(self, adapter=None):
        from embodiedbench.planner.critic import DualCritic, AlfredSymbolicCritic
        vlm  = _make_critic()
        dual = object.__new__(DualCritic)
        dual.symbolic = AlfredSymbolicCritic()
        dual.vlm      = vlm
        dual.log_path = None
        dual._episode_critic_records = []
        if adapter is not None:
            dual.set_memory_adapter(adapter)
        return dual

    def test_dual_has_set_memory_adapter(self):
        from embodiedbench.planner.critic import DualCritic
        assert hasattr(DualCritic, "set_memory_adapter")

    def test_dual_forwards_adapter_to_vlm(self):
        adapter = MockMemoryAdapter()
        dual = self._make_dual(adapter=adapter)
        assert dual.vlm.memory_adapter is adapter

    def test_dual_set_memory_adapter_safe_with_no_vlm(self):
        from embodiedbench.planner.critic import DualCritic, AlfredSymbolicCritic
        dual = object.__new__(DualCritic)
        dual.symbolic = AlfredSymbolicCritic()
        dual.vlm      = None
        dual.log_path = None
        dual._episode_critic_records = []
        # Must not raise
        dual.set_memory_adapter(MockMemoryAdapter())
