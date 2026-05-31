"""
Tests for MemoryAdapter integration into VLMPlanner._get_planner_memory_prompt().

Strategy
--------
* Stub all heavy imports before importing VLMPlanner so the test file is
  importable without a GPU or the full embodiedbench runtime.
* Use a MockMemoryAdapter that overrides __init__ (skips _load_model) and
  generate() to return a controlled string.
* Exercise: adapter-enabled path, empty-output fallback, code-fence fallback,
  adapter-exception fallback, disabled adapter, no-memory path, attribute
  persistence, metadata forwarding, set_memory_adapter(), _adapter_enabled().
"""

import sys
import types
import importlib.machinery
import pytest


# ---------------------------------------------------------------------------
# Minimal stubs for heavy modules
# ---------------------------------------------------------------------------

def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return mod


def _ensure_stub(name):
    if name not in sys.modules:
        sys.modules[name] = _make_stub(name)


for _pkg in [
    "openai", "anthropic", "google", "google.generativeai",
    "lmdeploy", "lmdeploy.serve", "lmdeploy.serve.openai",
    "lmdeploy.serve.openai.api_client",
    "cv2", "transformers", "torch", "torch.cuda",
    "bitsandbytes",
]:
    _ensure_stub(_pkg)

# Make torch.cuda.is_available importable
sys.modules["torch"].cuda = sys.modules["torch.cuda"]
sys.modules["torch.cuda"].is_available = lambda: False

# ---------------------------------------------------------------------------
# Now safe to import project code
# ---------------------------------------------------------------------------

from embodiedbench.memory_adapter.schemas import MemoryAdapterInput, MemoryAdapterOutput
from embodiedbench.memory_adapter.config import MemoryAdapterConfig
from embodiedbench.memory_adapter.adapter import MemoryAdapter

# ---------------------------------------------------------------------------
# MockMemoryAdapter – no real model, controllable output
# ---------------------------------------------------------------------------

_MOCK_PLANNER_CONTEXT = (
    "[Adapted Memory for Planning]\n"
    "The robot is near the table.\n"
    "Foresight plan: grasp mug → place on counter\n"
    "Stale warnings: lighting changed since last visit"
)

_MOCK_GENERATE_OUTPUT = (
    "[Adapted Context]\n"
    "The robot is near the table.\n"
    "[Foresight Plan]\n"
    "- grasp mug\n"
    "- place on counter\n"
    "[Feasibility Criteria]\n"
    "- mug is reachable\n"
    "[Stale Memory Assessment]\n"
    "- lighting changed since last visit\n"
    "[Confidence]\n"
    "0.9\n"
)


class MockMemoryAdapter(MemoryAdapter):
    """Bypass _load_model; return _MOCK_GENERATE_OUTPUT from generate()."""

    def __init__(self, config: MemoryAdapterConfig = None):
        if config is None:
            config = MemoryAdapterConfig(model_name_or_path="mock-model")
        self.config = config
        self.model = None
        self.tokenizer = None
        self._generate_calls: list = []

    def generate(self, prompt: str) -> str:  # noqa: D401
        self._generate_calls.append(prompt)
        return _MOCK_GENERATE_OUTPUT


class ErrorMemoryAdapter(MockMemoryAdapter):
    """Always raises inside generate() to test fallback behaviour."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("simulated adapter failure")


class EmptyMemoryAdapter(MockMemoryAdapter):
    """Returns an output whose planner_context is empty string."""

    def adapt(self, adapter_input):
        return MemoryAdapterOutput(adapted_context="", planner_context="")


# ---------------------------------------------------------------------------
# Minimal VLMPlanner-compatible fakes
# ---------------------------------------------------------------------------

class FakeMemoryContext:
    def __init__(self):
        self.episodes = []
        self.metadata = {}


class FakeMemoryQuery:
    def __init__(self):
        self.observation_text = "robot sees mug on table"
        self.env_name = "test-env"
        self.scene_id = "scene-0"
        self.step_id = 1


class FakeMemoryManager:
    def is_enabled(self):
        return True

    def retrieve(self, query):
        return FakeMemoryContext()


class FakeFormatter:
    def format_for_planner(self, ctx):
        return "[Raw Memory] raw fallback text"


# ---------------------------------------------------------------------------
# Helper: build a VLMPlanner with memory + optional adapter attached
# ---------------------------------------------------------------------------

def _make_planner(adapter=None, memory_enabled=True):
    """Import VLMPlanner lazily so stubs are in place first."""
    from embodiedbench.planner.vlm_planner import VLMPlanner

    planner = object.__new__(VLMPlanner)

    # Minimal attrs expected by memory helpers
    planner.memory_manager = FakeMemoryManager() if memory_enabled else None
    planner.memory_formatter = FakeFormatter()
    planner.last_memory_context = None
    planner.last_memory_prompt = ""
    planner.memory_adapter = None
    planner.last_adapted_memory_output = None
    planner.last_adapted_memory_prompt = ""
    planner.episode_act_feedback = []
    planner.metrics = None

    # Patch helpers the method under test calls
    planner._build_memory_query = lambda instruction, obs=None, info=None: FakeMemoryQuery()

    if adapter is not None:
        planner.set_memory_adapter(adapter)

    return planner


# ---------------------------------------------------------------------------
# Tests: _adapter_enabled()
# ---------------------------------------------------------------------------

class TestAdapterEnabled:
    def test_no_adapter_returns_false(self):
        p = _make_planner()
        assert p._adapter_enabled() is False

    def test_adapter_attached_enabled_returns_true(self):
        p = _make_planner(adapter=MockMemoryAdapter())
        assert p._adapter_enabled() is True

    def test_adapter_config_disabled_returns_false(self):
        cfg = MemoryAdapterConfig(model_name_or_path="m", enabled=False)
        p = _make_planner(adapter=MockMemoryAdapter(cfg))
        assert p._adapter_enabled() is False

    def test_set_memory_adapter_stores_instance(self):
        from embodiedbench.planner.vlm_planner import VLMPlanner
        p = _make_planner()
        adapter = MockMemoryAdapter()
        p.set_memory_adapter(adapter)
        assert p.memory_adapter is adapter


# ---------------------------------------------------------------------------
# Tests: adapter-enabled prompt path
# ---------------------------------------------------------------------------

class TestAdapterEnabledPath:
    def test_returns_nonempty_adapted_prompt(self):
        p = _make_planner(adapter=MockMemoryAdapter())
        result = p._get_planner_memory_prompt("pick up the mug")
        assert result.strip()
        assert "[Adapted Memory for Planning]" in result

    def test_adapter_output_stored_on_planner(self):
        p = _make_planner(adapter=MockMemoryAdapter())
        p._get_planner_memory_prompt("pick up the mug")
        assert p.last_adapted_memory_output is not None
        assert isinstance(p.last_adapted_memory_output, MemoryAdapterOutput)

    def test_last_adapted_memory_prompt_set(self):
        p = _make_planner(adapter=MockMemoryAdapter())
        result = p._get_planner_memory_prompt("pick up the mug")
        assert p.last_adapted_memory_prompt == result

    def test_last_memory_prompt_also_set(self):
        p = _make_planner(adapter=MockMemoryAdapter())
        result = p._get_planner_memory_prompt("pick up the mug")
        assert p.last_memory_prompt == result

    def test_generate_called_once(self):
        adapter = MockMemoryAdapter()
        p = _make_planner(adapter=adapter)
        p._get_planner_memory_prompt("task")
        assert len(adapter._generate_calls) == 1


# ---------------------------------------------------------------------------
# Tests: fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    def test_empty_adapter_output_falls_back_to_raw(self):
        p = _make_planner(adapter=EmptyMemoryAdapter())
        result = p._get_planner_memory_prompt("task")
        assert result == "[Raw Memory] raw fallback text"

    def test_adapter_exception_falls_back_to_raw(self):
        p = _make_planner(adapter=ErrorMemoryAdapter())
        result = p._get_planner_memory_prompt("task")
        assert result == "[Raw Memory] raw fallback text"

    def test_code_fence_output_falls_back_to_raw(self):
        """Adapter output containing ``` must fall back to raw formatter."""

        class FenceAdapter(MockMemoryAdapter):
            def adapt(self, adapter_input):
                out = MemoryAdapterOutput(
                    adapted_context="some context",
                    planner_context="```\nsome plan\n```",
                )
                return out

        p = _make_planner(adapter=FenceAdapter())
        result = p._get_planner_memory_prompt("task")
        assert result == "[Raw Memory] raw fallback text"


# ---------------------------------------------------------------------------
# Tests: disabled memory path
# ---------------------------------------------------------------------------

class TestDisabledMemory:
    def test_no_memory_manager_returns_empty(self):
        p = _make_planner(memory_enabled=False)
        result = p._get_planner_memory_prompt("task")
        assert result == ""

    def test_disabled_adapter_uses_raw_formatter(self):
        cfg = MemoryAdapterConfig(model_name_or_path="m", enabled=False)
        p = _make_planner(adapter=MockMemoryAdapter(cfg))
        result = p._get_planner_memory_prompt("task")
        assert result == "[Raw Memory] raw fallback text"

    def test_no_adapter_uses_raw_formatter(self):
        p = _make_planner()
        result = p._get_planner_memory_prompt("task")
        assert result == "[Raw Memory] raw fallback text"
