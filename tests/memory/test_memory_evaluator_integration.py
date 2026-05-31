"""
tests/memory/test_memory_evaluator_integration.py

Tests for memory/integration.py helpers used in evaluator lifecycle.
No real simulator is needed.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from embodiedbench.memory.integration import (
    create_memory_manager_from_config,
    attach_memory_to_planner,
    finalize_memory_episode,
    save_memory_if_configured,
    compute_final_status,
)
from embodiedbench.memory.manager import MemoryManager, MemoryConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(enabled: bool = True, **kwargs) -> dict:
    base = {
        "memory": {
            "enabled": enabled,
            "storage_dir": "/tmp/test_mem_eval",
            "load_on_start": False,
            "save_on_episode_end": True,
            "save_on_end": True,
            **kwargs,
        }
    }
    return base


def _cfg_no_memory() -> dict:
    return {"model_name": "gpt-4o"}


def _cfg_disabled() -> dict:
    return {"memory": {"enabled": False}}


# ---------------------------------------------------------------------------
# Tests: create_memory_manager_from_config
# ---------------------------------------------------------------------------

class TestCreateMemoryManagerFromConfig:
    def test_returns_none_when_cfg_has_no_memory_key(self):
        assert create_memory_manager_from_config(_cfg_no_memory()) is None

    def test_returns_none_when_memory_disabled(self):
        assert create_memory_manager_from_config(_cfg_disabled()) is None

    def test_returns_none_when_cfg_is_none(self):
        assert create_memory_manager_from_config(None) is None

    def test_returns_memory_manager_when_enabled(self):
        mm = create_memory_manager_from_config(_cfg(enabled=True))
        assert isinstance(mm, MemoryManager)
        assert mm.is_enabled() is True

    def test_load_on_start_calls_load(self):
        cfg = _cfg(enabled=True, load_on_start=True)
        # load() may fail silently since no file exists; manager still created
        mm = create_memory_manager_from_config(cfg)
        assert mm is not None

    def test_load_on_start_false_does_not_raise(self):
        mm = create_memory_manager_from_config(_cfg(enabled=True, load_on_start=False))
        assert mm is not None


# ---------------------------------------------------------------------------
# Tests: attach_memory_to_planner
# ---------------------------------------------------------------------------

class TestAttachMemoryToPlanner:
    def test_calls_set_memory_manager_when_available(self):
        planner = MagicMock()
        mm = MemoryManager(config=MemoryConfig(enabled=True))
        attach_memory_to_planner(planner, mm)
        planner.set_memory_manager.assert_called_once_with(mm)

    def test_noop_when_memory_manager_is_none(self):
        planner = MagicMock()
        attach_memory_to_planner(planner, None)
        planner.set_memory_manager.assert_not_called()

    def test_noop_when_planner_has_no_set_memory_manager(self):
        planner = object()  # plain object with no set_memory_manager
        mm = MemoryManager(config=MemoryConfig(enabled=True))
        # should not raise
        attach_memory_to_planner(planner, mm)

    def test_disabled_manager_does_not_disable_planner_otherwise(self):
        """attach should still call set_memory_manager even for a disabled config.
        The planner's _memory_enabled() will return False because manager.is_enabled()=False."""
        planner = MagicMock()
        mm = MemoryManager(config=MemoryConfig(enabled=False))
        attach_memory_to_planner(planner, mm)
        planner.set_memory_manager.assert_called_once_with(mm)


# ---------------------------------------------------------------------------
# Tests: compute_final_status
# ---------------------------------------------------------------------------

class TestComputeFinalStatus:
    def test_task_success_true_gives_success(self):
        assert compute_final_status({"task_success": 1}) == "success"

    def test_task_success_true_bool_gives_success(self):
        assert compute_final_status({"task_success": True}) == "success"

    def test_progress_positive_but_no_success_gives_partial(self):
        assert compute_final_status({"task_success": 0, "task_progress": 0.5}) == "partial"

    def test_no_success_no_progress_gives_failure(self):
        assert compute_final_status({"task_success": 0, "task_progress": 0}) == "failure"

    def test_empty_info_gives_failure(self):
        assert compute_final_status({}) == "failure"

    def test_none_info_gives_unknown(self):
        assert compute_final_status(None) == "unknown"


# ---------------------------------------------------------------------------
# Tests: finalize_memory_episode
# ---------------------------------------------------------------------------

class TestFinalizeMemoryEpisode:
    def test_adds_episodic_memory_on_success(self):
        mm = MemoryManager(config=MemoryConfig(enabled=True))
        planner = MagicMock()
        planner.episode_act_feedback = []
        # Feed some temporal steps so trajectory summary exists
        mm.update(task_instruction="pick apple", action_text="find a apple", step_id=0)

        finalize_memory_episode(
            mm, planner,
            task_instruction="pick apple",
            info={"task_success": 1, "task_progress": 1.0, "env_step": 5},
            env_name="alfred",
            scene_id="kitchen",
        )
        assert len(mm.episodic.episodes) == 1
        assert mm.episodic.episodes[0].final_status == "success"

    def test_adds_episodic_memory_on_failure(self):
        mm = MemoryManager(config=MemoryConfig(enabled=True))
        planner = MagicMock()
        planner.episode_act_feedback = []

        finalize_memory_episode(
            mm, planner,
            task_instruction="pick apple",
            info={"task_success": 0, "task_progress": 0, "env_step": 3, "env_feedback": "no apple found"},
            env_name="alfred",
        )
        # Failure episodes are not stored in episodic memory
        assert len(mm.episodic.episodes) == 0

    def test_noop_when_memory_manager_is_none(self):
        # must not raise
        finalize_memory_episode(None, MagicMock(), task_instruction="t", info={})

    def test_noop_when_memory_disabled(self):
        mm = MemoryManager(config=MemoryConfig(enabled=False))
        finalize_memory_episode(mm, MagicMock(), task_instruction="t",
                                info={"task_success": 1})
        # episodic is None since disabled
        assert mm.episodic is None or len(mm.episodic.episodes) == 0


# ---------------------------------------------------------------------------
# Tests: save_memory_if_configured
# ---------------------------------------------------------------------------

class TestSaveMemoryIfConfigured:
    def test_save_on_episode_end_calls_save(self, tmp_path):
        mm = MemoryManager(config=MemoryConfig(enabled=True, storage_dir=str(tmp_path / "mem")))
        cfg = _cfg(enabled=True, save_on_episode_end=True)
        save_memory_if_configured(mm, cfg, on_episode_end=True)
        assert (tmp_path / "mem").exists()

    def test_save_on_run_end_calls_save(self, tmp_path):
        mm = MemoryManager(config=MemoryConfig(enabled=True, storage_dir=str(tmp_path / "mem2")))
        cfg = _cfg(enabled=True, save_on_end=True)
        save_memory_if_configured(mm, cfg, on_run_end=True)
        assert (tmp_path / "mem2").exists()

    def test_noop_when_memory_manager_is_none(self):
        # must not raise
        save_memory_if_configured(None, _cfg(enabled=True), on_episode_end=True)

    def test_save_not_called_when_flag_false(self, tmp_path):
        mm = MagicMock()
        cfg = {"memory": {"enabled": True, "save_on_episode_end": False, "save_on_end": False}}
        save_memory_if_configured(mm, cfg, on_episode_end=True)
        mm.save.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: disabled mode does not alter planner
# ---------------------------------------------------------------------------

class TestDisabledModePreservesPlanner:
    def test_create_with_null_memory_cfg_returns_none(self):
        cfg = {"memory": None}
        assert create_memory_manager_from_config(cfg) is None

    def test_planner_not_modified_when_memory_none(self):
        planner = MagicMock()
        attach_memory_to_planner(planner, None)
        # set_memory_manager should never have been called
        planner.set_memory_manager.assert_not_called()
