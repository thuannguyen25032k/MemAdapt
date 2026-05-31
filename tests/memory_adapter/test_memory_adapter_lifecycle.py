"""
Tests for MemoryAdapter evaluator lifecycle (Step 18).

Strategy
--------
* Patch MemoryAdapter.__init__ and _load_model to avoid HF loading.
* Test create_memory_adapter_from_config() + attach helpers.
* Test that the same adapter instance is shared by planner and critic.
* Test disabled config preserves original behaviour.
"""

import sys
import types
import importlib.machinery
from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Stubs (must appear before project imports)
# ---------------------------------------------------------------------------

def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    return mod


for _pkg in [
    "openai", "anthropic", "google", "google.generativeai",
    "lmdeploy", "pydantic", "typing_extensions", "cv2",
    "transformers", "torch", "torch.cuda", "bitsandbytes",
]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_stub(_pkg)

# Attribute patches
sys.modules["openai"].OpenAI = object
sys.modules["openai"].AzureOpenAI = object
sys.modules["pydantic"].BaseModel = object
sys.modules["pydantic"].Field = lambda *a, **kw: None
sys.modules["lmdeploy"].pipeline = lambda *a, **kw: None
sys.modules["lmdeploy"].GenerationConfig = object
sys.modules["lmdeploy"].PytorchEngineConfig = object
sys.modules["torch"].cuda = sys.modules["torch.cuda"]
sys.modules["torch.cuda"].is_available = lambda: False
import typing as _typing
sys.modules["typing_extensions"].TypedDict = _typing.TypedDict

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from embodiedbench.memory_adapter.config import MemoryAdapterConfig
from embodiedbench.memory_adapter.schemas import MemoryAdapterInput, MemoryAdapterOutput
from embodiedbench.memory.integration import (
    create_memory_adapter_from_config,
    attach_memory_adapter_to_planner,
    attach_memory_adapter_to_critic,
    unload_memory_adapter,
)


# ---------------------------------------------------------------------------
# Helpers: build config dicts / objects
# ---------------------------------------------------------------------------

def _cfg(adapter_block=None):
    """Build a minimal top-level config dict."""
    return {"model_name": "mock", "memory_adapter": adapter_block}


def _enabled_cfg(model="mock-model", **extra):
    return _cfg({"enabled": True, "model_name_or_path": model, **extra})


def _disabled_cfg():
    return _cfg({"enabled": False, "model_name_or_path": "some-model"})


# ---------------------------------------------------------------------------
# MockMemoryAdapter — bypasses _load_model
# ---------------------------------------------------------------------------

from embodiedbench.memory_adapter.adapter import MemoryAdapter


class MockMemoryAdapter(MemoryAdapter):
    _instance_count = 0

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        MockMemoryAdapter._instance_count += 1

    def generate(self, prompt):
        return ""


# ---------------------------------------------------------------------------
# Tests: create_memory_adapter_from_config
# ---------------------------------------------------------------------------

class TestCreateMemoryAdapterFromConfig:
    def test_missing_memory_adapter_key_returns_none(self):
        result = create_memory_adapter_from_config({"model_name": "x"})
        assert result is None

    def test_null_memory_adapter_returns_none(self):
        result = create_memory_adapter_from_config(_cfg(None))
        assert result is None

    def test_enabled_false_returns_none(self):
        result = create_memory_adapter_from_config(_disabled_cfg())
        assert result is None

    def test_enabled_true_no_model_raises_value_error(self):
        cfg = _cfg({"enabled": True, "model_name_or_path": None})
        with pytest.raises(ValueError, match="model_name_or_path"):
            create_memory_adapter_from_config(cfg)

    def test_enabled_true_missing_model_key_raises_value_error(self):
        cfg = _cfg({"enabled": True})
        with pytest.raises(ValueError, match="model_name_or_path"):
            create_memory_adapter_from_config(cfg)

    def test_enabled_true_constructs_adapter(self):
        with patch(
            "embodiedbench.memory_adapter.adapter.MemoryAdapter.__init__",
            lambda self, config: setattr(self, "config", config) or setattr(self, "model", None) or setattr(self, "tokenizer", None),
        ):
            result = create_memory_adapter_from_config(_enabled_cfg("hf/model"))
        assert result is not None
        assert result.config.model_name_or_path == "hf/model"

    def test_enabled_true_config_fields_forwarded(self):
        with patch(
            "embodiedbench.memory_adapter.adapter.MemoryAdapter.__init__",
            lambda self, config: setattr(self, "config", config) or setattr(self, "model", None) or setattr(self, "tokenizer", None),
        ):
            result = create_memory_adapter_from_config(
                _cfg({"enabled": True, "model_name_or_path": "m", "max_new_tokens": 256})
            )
        assert result.config.max_new_tokens == 256


# ---------------------------------------------------------------------------
# Tests: attach helpers
# ---------------------------------------------------------------------------

class TestAttachMemoryAdapterToPlanner:
    def test_calls_set_memory_adapter(self):
        planner = MagicMock()
        adapter = MagicMock()
        attach_memory_adapter_to_planner(planner, adapter)
        planner.set_memory_adapter.assert_called_once_with(adapter)

    def test_none_adapter_is_noop(self):
        planner = MagicMock()
        attach_memory_adapter_to_planner(planner, None)
        planner.set_memory_adapter.assert_not_called()

    def test_none_planner_is_noop(self):
        # Must not raise
        attach_memory_adapter_to_planner(None, MagicMock())

    def test_planner_without_method_is_noop(self):
        planner = object()  # no set_memory_adapter
        attach_memory_adapter_to_planner(planner, MagicMock())  # must not raise


class TestAttachMemoryAdapterToCritic:
    def test_calls_set_memory_adapter(self):
        critic = MagicMock()
        adapter = MagicMock()
        attach_memory_adapter_to_critic(critic, adapter)
        critic.set_memory_adapter.assert_called_once_with(adapter)

    def test_none_adapter_is_noop(self):
        critic = MagicMock()
        attach_memory_adapter_to_critic(critic, None)
        critic.set_memory_adapter.assert_not_called()

    def test_none_critic_is_noop(self):
        attach_memory_adapter_to_critic(None, MagicMock())  # must not raise

    def test_critic_without_method_is_noop(self):
        critic = object()
        attach_memory_adapter_to_critic(critic, MagicMock())  # must not raise


# ---------------------------------------------------------------------------
# Tests: shared adapter / single construction
# ---------------------------------------------------------------------------

class TestSharedAdapterLifecycle:
    def test_same_adapter_attached_to_planner_and_critic(self):
        """Single adapter instance is shared by planner and critic."""
        planner = MagicMock()
        critic  = MagicMock()

        with patch(
            "embodiedbench.memory_adapter.adapter.MemoryAdapter.__init__",
            lambda self, config: setattr(self, "config", config) or setattr(self, "model", None) or setattr(self, "tokenizer", None),
        ):
            adapter = create_memory_adapter_from_config(_enabled_cfg("hf/model"))

        attach_memory_adapter_to_planner(planner, adapter)
        attach_memory_adapter_to_critic(critic, adapter)

        planner.set_memory_adapter.assert_called_once_with(adapter)
        critic.set_memory_adapter.assert_called_once_with(adapter)
        # Same object
        assert planner.set_memory_adapter.call_args[0][0] is critic.set_memory_adapter.call_args[0][0]

    def test_adapter_constructed_once_not_per_episode(self):
        """create_memory_adapter_from_config must be called once, not per episode."""
        call_count = []

        original_init = MemoryAdapter.__init__

        def counting_init(self, config):
            call_count.append(1)
            self.config = config
            self.model = None
            self.tokenizer = None

        with patch.object(MemoryAdapter, "__init__", counting_init):
            cfg = _enabled_cfg("hf/model")
            adapter = create_memory_adapter_from_config(cfg)
            # Simulate evaluator: episodes run — adapter is NOT re-created
            for _ in range(5):
                attach_memory_adapter_to_planner(MagicMock(), adapter)

        assert len(call_count) == 1  # constructed exactly once

    def test_disabled_config_leaves_planner_unchanged(self):
        planner = MagicMock()
        adapter = create_memory_adapter_from_config(_disabled_cfg())
        attach_memory_adapter_to_planner(planner, adapter)
        planner.set_memory_adapter.assert_not_called()

    def test_disabled_config_leaves_critic_unchanged(self):
        critic = MagicMock()
        adapter = create_memory_adapter_from_config(_disabled_cfg())
        attach_memory_adapter_to_critic(critic, adapter)
        critic.set_memory_adapter.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: unload lifecycle
# ---------------------------------------------------------------------------

class TestUnloadMemoryAdapter:
    def test_none_is_noop(self):
        unload_memory_adapter(None)  # must not raise

    def test_calls_unload_method(self):
        adapter = MagicMock()
        unload_memory_adapter(adapter)
        adapter.unload.assert_called_once()

    def test_adapter_without_unload_is_noop(self):
        # object() has no unload()
        unload_memory_adapter(object())  # must not raise

    def test_unload_exception_does_not_propagate(self):
        adapter = MagicMock()
        adapter.unload.side_effect = RuntimeError("GPU freed elsewhere")
        unload_memory_adapter(adapter)  # must not raise

    def test_evaluator_lifecycle_create_attach_unload(self):
        """Simulate the full evaluator lifecycle with a mock adapter."""
        planner = MagicMock()
        critic  = MagicMock()

        with patch(
            "embodiedbench.memory_adapter.adapter.MemoryAdapter.__init__",
            lambda self, config: setattr(self, "config", config) or setattr(self, "model", None) or setattr(self, "tokenizer", None),
        ):
            adapter = create_memory_adapter_from_config(_enabled_cfg("hf/model"))

        # Attach (evaluator start)
        attach_memory_adapter_to_planner(planner, adapter)
        attach_memory_adapter_to_critic(critic, adapter)

        # Simulate episodes — no re-creation
        for _ in range(3):
            planner.reset()

        # Unload (evaluator end)
        adapter.unload = MagicMock()
        unload_memory_adapter(adapter)

        planner.set_memory_adapter.assert_called_once()
        critic.set_memory_adapter.assert_called_once()
        adapter.unload.assert_called_once()
