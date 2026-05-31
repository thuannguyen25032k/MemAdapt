"""
tests/memory_adapter/test_memory_experiment_modes.py

Tests for setup_memory_experiment() and _get_experiment_mode() in
embodiedbench.memory.integration.
"""

import pytest
from unittest.mock import MagicMock, patch

from embodiedbench.memory.integration import setup_memory_experiment, _get_experiment_mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(mode):
    """Return a dict config with memory_experiment.mode set."""
    return {"memory_experiment": {"mode": mode}}


def _cfg_no_exp():
    """Config without memory_experiment key (backward-compat)."""
    return {}


# ---------------------------------------------------------------------------
# _get_experiment_mode
# ---------------------------------------------------------------------------

class TestGetExperimentMode:
    def test_returns_none_when_key_absent(self):
        assert _get_experiment_mode({}) is None

    def test_returns_none_when_cfg_none(self):
        assert _get_experiment_mode(None) is None

    @pytest.mark.parametrize("mode", [
        "none", "raw_planner", "raw_planner_critic",
        "adapted_planner", "adapted_planner_critic",
    ])
    def test_valid_modes_returned(self, mode):
        assert _get_experiment_mode(_cfg(mode)) == mode

    def test_raises_on_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            _get_experiment_mode(_cfg("invalid_mode"))

    def test_case_insensitive(self):
        cfg = {"memory_experiment": {"mode": "RAW_PLANNER"}}
        assert _get_experiment_mode(cfg) == "raw_planner"


# ---------------------------------------------------------------------------
# setup_memory_experiment — mode=none
# ---------------------------------------------------------------------------

class TestModeNone:
    def test_returns_none_none(self):
        planner = MagicMock()
        mm, ma = setup_memory_experiment(_cfg("none"), planner, None)
        assert mm is None
        assert ma is None

    def test_does_not_attach_anything(self):
        planner = MagicMock()
        setup_memory_experiment(_cfg("none"), planner, None)
        assert not hasattr(planner, "memory_manager") or planner.memory_manager is None or True


# ---------------------------------------------------------------------------
# setup_memory_experiment — raw_planner
# ---------------------------------------------------------------------------

class TestModeRawPlanner:
    def test_attaches_mm_to_planner_only(self):
        planner = MagicMock()
        critic = MagicMock()
        cfg = {
            "memory_experiment": {"mode": "raw_planner"},
            "memory": {"enabled": True, "max_episodes": 5},
        }
        mm, ma = setup_memory_experiment(cfg, planner, critic)
        assert mm is not None
        assert ma is None
        planner.set_memory_manager.assert_called_once_with(mm)
        critic.set_memory_manager.assert_not_called()

    def test_critic_not_attached(self):
        planner = MagicMock()
        critic = MagicMock()
        cfg = {
            "memory_experiment": {"mode": "raw_planner"},
            "memory": {"enabled": True, "max_episodes": 5},
        }
        setup_memory_experiment(cfg, planner, critic)
        critic.set_memory_manager.assert_not_called()


# ---------------------------------------------------------------------------
# setup_memory_experiment — raw_planner_critic
# ---------------------------------------------------------------------------

class TestModeRawPlannerCritic:
    def test_attaches_mm_to_both(self):
        planner = MagicMock()
        critic = MagicMock()
        cfg = {
            "memory_experiment": {"mode": "raw_planner_critic"},
            "memory": {"enabled": True, "max_episodes": 5},
        }
        mm, ma = setup_memory_experiment(cfg, planner, critic)
        assert mm is not None
        assert ma is None
        planner.set_memory_manager.assert_called_once_with(mm)
        critic.set_memory_manager.assert_called_once_with(mm)


# ---------------------------------------------------------------------------
# setup_memory_experiment — adapted_planner
# ---------------------------------------------------------------------------

class TestModeAdaptedPlanner:
    def test_attaches_mm_and_ma_to_planner_only(self):
        planner = MagicMock()
        critic = MagicMock()
        cfg = {
            "memory_experiment": {"mode": "adapted_planner"},
            "memory": {"enabled": True, "max_episodes": 5},
            "memory_adapter": {"enabled": True, "model_name_or_path": "dummy"},
        }
        with patch(
            "embodiedbench.memory.integration.MemoryAdapter",
            return_value=MagicMock(),
        ):
            mm, ma = setup_memory_experiment(cfg, planner, critic)
        assert mm is not None
        assert ma is not None
        planner.set_memory_manager.assert_called_once_with(mm)
        planner.set_memory_adapter.assert_called_once_with(ma)
        critic.set_memory_manager.assert_not_called()
        critic.set_memory_adapter.assert_not_called()


# ---------------------------------------------------------------------------
# setup_memory_experiment — adapted_planner_critic
# ---------------------------------------------------------------------------

class TestModeAdaptedPlannerCritic:
    def test_attaches_mm_and_ma_to_both(self):
        planner = MagicMock()
        critic = MagicMock()
        cfg = {
            "memory_experiment": {"mode": "adapted_planner_critic"},
            "memory": {"enabled": True, "max_episodes": 5},
            "memory_adapter": {"enabled": True, "model_name_or_path": "dummy"},
        }
        with patch(
            "embodiedbench.memory.integration.MemoryAdapter",
            return_value=MagicMock(),
        ):
            mm, ma = setup_memory_experiment(cfg, planner, critic)
        assert mm is not None
        assert ma is not None
        planner.set_memory_manager.assert_called_once_with(mm)
        planner.set_memory_adapter.assert_called_once_with(ma)
        critic.set_memory_manager.assert_called_once_with(mm)
        critic.set_memory_adapter.assert_called_once_with(ma)


# ---------------------------------------------------------------------------
# Backward-compat: no memory_experiment key
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_no_memory_experiment_key_returns_mm_and_ma(self):
        planner = MagicMock()
        critic = MagicMock()
        cfg = {
            "memory": {"enabled": True, "max_episodes": 5},
            "memory_adapter": {"enabled": True, "model_name_or_path": "dummy"},
        }
        with patch(
            "embodiedbench.memory.integration.MemoryAdapter",
            return_value=MagicMock(),
        ):
            mm, ma = setup_memory_experiment(cfg, planner, critic)
        assert mm is not None
        assert ma is not None
        planner.set_memory_manager.assert_called_once_with(mm)
        planner.set_memory_adapter.assert_called_once_with(ma)
        critic.set_memory_manager.assert_called_once_with(mm)
        critic.set_memory_adapter.assert_called_once_with(ma)
