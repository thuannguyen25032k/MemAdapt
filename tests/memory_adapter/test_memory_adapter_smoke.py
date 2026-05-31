"""
Smoke test for real Hugging Face model loading via MemoryAdapter.

SKIPPED BY DEFAULT.  Only runs when the env var is set:

    RUN_HF_ADAPTER_SMOKE=1 pytest tests/memory_adapter/test_memory_adapter_smoke.py -v

Uses sshleifer/tiny-gpt2 (≈ 5 MB) to verify the full load→adapt→unload
cycle without large downloads.  This checks wiring only, NOT output quality.

Set a different model via env var:
    ADAPTER_SMOKE_MODEL=gpt2 RUN_HF_ADAPTER_SMOKE=1 pytest ...
"""

import os
import pytest

# ---------------------------------------------------------------------------
# Skip unless opt-in
# ---------------------------------------------------------------------------
_RUN_SMOKE = os.environ.get("RUN_HF_ADAPTER_SMOKE", "0").strip() not in ("", "0", "false", "no")
pytestmark = pytest.mark.skipif(
    not _RUN_SMOKE,
    reason="Set RUN_HF_ADAPTER_SMOKE=1 to run real HF model smoke tests.",
)

_SMOKE_MODEL = os.environ.get("ADAPTER_SMOKE_MODEL", "sshleifer/tiny-gpt2")

# ---------------------------------------------------------------------------
# Imports (only reached when opt-in — transformers/torch must be installed)
# ---------------------------------------------------------------------------
from embodiedbench.memory_adapter.config import MemoryAdapterConfig
from embodiedbench.memory_adapter.adapter import MemoryAdapter
from embodiedbench.memory_adapter.schemas import MemoryAdapterInput, MemoryAdapterOutput
from embodiedbench.memory.base import MemoryContext


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_adapter():
    """Load the tiny model once for the whole module."""
    cfg = MemoryAdapterConfig(
        model_name_or_path=_SMOKE_MODEL,
        max_new_tokens=64,
        do_sample=False,
        device="cpu",
    )
    adapter = MemoryAdapter(cfg)
    yield adapter
    adapter.unload()


def _fake_input(**kwargs):
    defaults = dict(
        task_instruction="Pick up the mug from the table.",
        observation_text="Robot sees a table with a mug and a bowl.",
        memory_context=MemoryContext(),
        mode="planner",
    )
    defaults.update(kwargs)
    return MemoryAdapterInput(**defaults)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestMemoryAdapterSmoke:
    def test_model_loads(self, tiny_adapter):
        """Model and tokenizer must be loaded after construction."""
        assert tiny_adapter.model is not None
        assert tiny_adapter.tokenizer is not None

    def test_adapt_returns_output(self, tiny_adapter):
        """adapt() must return a MemoryAdapterOutput without raising."""
        out = tiny_adapter.adapt(_fake_input())
        assert isinstance(out, MemoryAdapterOutput)

    def test_raw_output_or_parse_error_handled(self, tiny_adapter):
        """Either raw_output is non-empty or parse_error is set — never both None."""
        out = tiny_adapter.adapt(_fake_input())
        has_content = bool(out.raw_output) or bool(out.parse_error)
        assert has_content, (
            "MemoryAdapterOutput has neither raw_output nor parse_error; "
            "something went wrong silently."
        )

    def test_planner_context_is_string(self, tiny_adapter):
        out = tiny_adapter.adapt(_fake_input(mode="planner"))
        assert isinstance(out.planner_context, str)

    def test_critic_context_is_string(self, tiny_adapter):
        out = tiny_adapter.adapt(_fake_input(mode="critic"))
        assert isinstance(out.critic_context, str)

    def test_adapt_for_planner_convenience(self, tiny_adapter):
        result = tiny_adapter.adapt_for_planner(
            task_instruction="Find the keys.",
            memory_context=MemoryContext(),
        )
        assert isinstance(result, str)

    def test_adapt_for_critic_convenience(self, tiny_adapter):
        result = tiny_adapter.adapt_for_critic(
            task_instruction="Find the keys.",
            memory_context=MemoryContext(),
            proposed_action="action id 2, pick up the keys",
        )
        assert isinstance(result, str)

    def test_unload_does_not_crash(self, tiny_adapter):
        """unload() is called by the fixture teardown; pre-test it safely."""
        # Create a separate adapter just to verify unload doesn't crash
        cfg = MemoryAdapterConfig(
            model_name_or_path=_SMOKE_MODEL,
            max_new_tokens=8,
            device="cpu",
        )
        tmp = MemoryAdapter(cfg)
        tmp.unload()  # must not raise
        assert tmp.model is None
        assert tmp.tokenizer is None
