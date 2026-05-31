"""
tests/memory_adapter/test_memory_logging.py

Tests for MemoryEpisodeLog, MemoryExperimentLogger, and build_episode_log.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from unittest.mock import MagicMock

from embodiedbench.memory.logging import (
    MemoryEpisodeLog,
    MemoryExperimentLogger,
)
from embodiedbench.memory.integration import create_logger_from_config


# ---------------------------------------------------------------------------
# 1. MemoryEpisodeLog serialization
# ---------------------------------------------------------------------------

class TestMemoryEpisodeLog:
    def test_to_dict_returns_dict(self):
        rec = MemoryEpisodeLog(episode_id="ep1", task_instruction="do X")
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert d["episode_id"] == "ep1"
        assert d["task_instruction"] == "do X"

    def test_from_dict_roundtrip(self):
        original = MemoryEpisodeLog(
            episode_id="ep42",
            env_name="alfred",
            task_instruction="pick up the mug",
            mode="adapted_planner",
            foresight_plan=["step 1", "step 2"],
            task_success=True,
            task_progress=1.0,
        )
        recovered = MemoryEpisodeLog.from_dict(original.to_dict())
        assert recovered.episode_id == "ep42"
        assert recovered.foresight_plan == ["step 1", "step 2"]
        assert recovered.task_success is True

    def test_from_dict_ignores_unknown_keys(self):
        d = MemoryEpisodeLog().to_dict()
        d["_unknown_future_field"] = "value"
        # Should not raise
        rec = MemoryEpisodeLog.from_dict(d)
        assert rec.episode_id == ""


# ---------------------------------------------------------------------------
# 2. Logger disabled does nothing
# ---------------------------------------------------------------------------

class TestLoggerDisabled:
    def test_log_episode_returns_empty_when_disabled(self, tmp_path):
        logger = MemoryExperimentLogger(log_dir=str(tmp_path), enabled=False)
        result = logger.log_episode(MemoryEpisodeLog(episode_id="ep1"))
        assert result == ""
        assert not any(tmp_path.iterdir())

    def test_append_training_record_returns_empty_when_disabled(self, tmp_path):
        logger = MemoryExperimentLogger(log_dir=str(tmp_path), enabled=False)
        result = logger.append_training_record(MemoryEpisodeLog(episode_id="ep1"))
        assert result == ""
        assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# 3. log_episode writes JSON
# ---------------------------------------------------------------------------

class TestLogEpisode:
    def test_writes_json_file(self, tmp_path):
        logger = MemoryExperimentLogger(log_dir=str(tmp_path), enabled=True)
        rec = MemoryEpisodeLog(episode_id="ep_test", task_instruction="navigate to kitchen")
        path = logger.log_episode(rec)
        assert path != ""
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert data["episode_id"] == "ep_test"
        assert data["task_instruction"] == "navigate to kitchen"

    def test_episode_file_in_episodes_subdir(self, tmp_path):
        logger = MemoryExperimentLogger(log_dir=str(tmp_path), enabled=True)
        path = logger.log_episode(MemoryEpisodeLog(episode_id="ep_sub"))
        assert "episodes" in path

    def test_auto_generates_episode_id_when_empty(self, tmp_path):
        logger = MemoryExperimentLogger(log_dir=str(tmp_path), enabled=True)
        path = logger.log_episode(MemoryEpisodeLog())
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# 4. append_training_record writes SFT-style JSONL
# ---------------------------------------------------------------------------

class TestAppendTrainingRecord:
    def test_writes_jsonl_row(self, tmp_path):
        logger = MemoryExperimentLogger(
            log_dir=str(tmp_path), enabled=True, save_training_records=True
        )
        rec = MemoryEpisodeLog(
            episode_id="ep_train",
            task_instruction="pick up the apple",
            planner_memory_prompt="Memory says: apple is on counter",
            task_success=True,
            task_progress=1.0,
            metrics={"env_steps": 8, "replans": 1},
        )
        path = logger.append_training_record(rec)
        assert os.path.isfile(path)
        with open(path) as f:
            row = json.loads(f.readline())
        assert row["instruction"] == "pick up the apple"
        assert row["outcome"]["success"] is True

    def test_multiple_rows_appended(self, tmp_path):
        logger = MemoryExperimentLogger(
            log_dir=str(tmp_path), enabled=True, save_training_records=True
        )
        for i in range(3):
            logger.append_training_record(MemoryEpisodeLog(episode_id=f"ep{i}"))
        path = os.path.join(str(tmp_path), MemoryExperimentLogger.TRAINING_FILE)
        with open(path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 3

    def test_no_training_records_when_flag_false(self, tmp_path):
        logger = MemoryExperimentLogger(
            log_dir=str(tmp_path), enabled=True, save_training_records=False
        )
        result = logger.append_training_record(MemoryEpisodeLog(episode_id="ep1"))
        assert result == ""


# ---------------------------------------------------------------------------
# 5. build_episode_log extracts planner last_memory_prompt
# ---------------------------------------------------------------------------

class TestBuildEpisodeLogPlanner:
    def test_extracts_planner_memory_prompt(self):
        planner = MagicMock()
        planner.last_memory_prompt = "Planner memory: avoid red zone"
        planner.last_adapted_memory_prompt = ""
        planner.last_adapted_memory_output = None
        planner.last_memory_context = None
        planner.episode_act_feedback = []
        rec = MemoryExperimentLogger.build_episode_log(
            episode_id="ep1", env_name="alfred", scene_id="s1",
            task_instruction="task", mode="raw_planner", planner=planner,
        )
        assert rec.planner_memory_prompt == "Planner memory: avoid red zone"

    def test_planner_actions_from_feedback(self):
        planner = MagicMock()
        planner.last_memory_prompt = ""
        planner.last_adapted_memory_prompt = ""
        planner.last_adapted_memory_output = None
        planner.last_memory_context = None
        planner.episode_act_feedback = [
            [3, "pick up the mug"],
            [-3, "critic rejected: mug not visible"],
            [5, "place mug on counter"],
        ]
        rec = MemoryExperimentLogger.build_episode_log(
            episode_id="ep2", env_name="alfred", scene_id="s1",
            task_instruction="task", mode="raw_planner", planner=planner,
        )
        assert any("pick up" in a for a in rec.planner_actions)
        assert any("CRITIC_FEEDBACK" in a for a in rec.planner_actions)


# ---------------------------------------------------------------------------
# 6. build_episode_log extracts planner last_adapted_memory_output
# ---------------------------------------------------------------------------

class TestBuildEpisodeLogAdapterOutput:
    def test_extracts_foresight_plan_from_planner(self):
        adapted_out = MagicMock()
        adapted_out.foresight_plan = ["step A", "step B"]
        adapted_out.feasibility_criteria = ["object must be reachable"]
        adapted_out.stale_memory_assessment = ["warning: stale data"]

        planner = MagicMock()
        planner.last_memory_prompt = ""
        planner.last_adapted_memory_prompt = "adapted planner ctx"
        planner.last_adapted_memory_output = adapted_out
        planner.last_memory_context = None
        planner.episode_act_feedback = []

        rec = MemoryExperimentLogger.build_episode_log(
            episode_id="ep3", env_name="alfred", scene_id="s1",
            task_instruction="task", mode="adapted_planner", planner=planner,
        )
        assert rec.foresight_plan == ["step A", "step B"]
        assert rec.feasibility_criteria == ["object must be reachable"]
        assert rec.stale_memory_assessment == ["warning: stale data"]
        assert rec.adapted_planner_context == "adapted planner ctx"


# ---------------------------------------------------------------------------
# 7. build_episode_log extracts critic last_adapted_memory_output
# ---------------------------------------------------------------------------

class TestBuildEpisodeLogCritic:
    def test_extracts_critic_adapted_prompt(self):
        vlm_critic = MagicMock()
        vlm_critic.last_adapted_memory_prompt = "critic adapted ctx"
        vlm_critic.last_adapted_memory_output = None

        dual_critic = MagicMock()
        dual_critic.vlm = vlm_critic
        dual_critic._episode_critic_records = []

        planner = MagicMock()
        planner.last_memory_prompt = ""
        planner.last_adapted_memory_prompt = ""
        planner.last_adapted_memory_output = None
        planner.last_memory_context = None
        planner.episode_act_feedback = []

        rec = MemoryExperimentLogger.build_episode_log(
            episode_id="ep4", env_name="alfred", scene_id="s1",
            task_instruction="task", mode="adapted_planner_critic",
            planner=planner, critic=dual_critic,
        )
        assert rec.critic_memory_prompt == "critic adapted ctx"

    def test_critic_events_from_episode_records(self):
        dual_critic = MagicMock()
        dual_critic.vlm = MagicMock(
            last_adapted_memory_prompt="",
            last_adapted_memory_output=None,
        )
        dual_critic._episode_critic_records = [
            {"env_step": 3, "action_id": 5, "action_str": "pick up mug",
             "is_first_step": False, "final_decision": {"valid": False, "feedback": "not visible"}},
        ]

        planner = MagicMock()
        planner.last_memory_prompt = ""
        planner.last_adapted_memory_prompt = ""
        planner.last_adapted_memory_output = None
        planner.last_memory_context = None
        planner.episode_act_feedback = []

        rec = MemoryExperimentLogger.build_episode_log(
            episode_id="ep5", env_name="alfred", scene_id="s1",
            task_instruction="task", mode="raw_planner_critic",
            planner=planner, critic=dual_critic,
        )
        assert len(rec.critic_events) == 1
        assert rec.critic_events[0]["valid"] is False


# ---------------------------------------------------------------------------
# 8. Metrics included
# ---------------------------------------------------------------------------

class TestMetricsIncluded:
    def test_metrics_dict_in_record(self):
        from embodiedbench.memory.metrics import MemoryExperimentMetrics
        m = MemoryExperimentMetrics(mode="adapted_planner", env_steps=10, replans=2)
        rec = MemoryExperimentLogger.build_episode_log(
            episode_id="ep6", env_name="alfred", scene_id="s1",
            task_instruction="task", mode="adapted_planner", metrics=m,
        )
        assert rec.metrics["env_steps"] == 10
        assert rec.metrics["replans"] == 2
        assert rec.metrics["mode"] == "adapted_planner"


# ---------------------------------------------------------------------------
# 9. Missing adapter output handled safely
# ---------------------------------------------------------------------------

class TestMissingAdapterOutput:
    def test_none_planner_and_critic(self):
        rec = MemoryExperimentLogger.build_episode_log(
            episode_id="ep7", env_name="alfred", scene_id="s1",
            task_instruction="task", mode="none",
        )
        assert rec.planner_memory_prompt == ""
        assert rec.foresight_plan == []
        assert rec.critic_events == []

    def test_planner_with_no_adapted_output(self):
        planner = MagicMock()
        planner.last_memory_prompt = "raw prompt"
        planner.last_adapted_memory_prompt = ""
        planner.last_adapted_memory_output = None
        planner.last_memory_context = None
        planner.episode_act_feedback = []
        rec = MemoryExperimentLogger.build_episode_log(
            episode_id="ep8", env_name="alfred", scene_id="s1",
            task_instruction="task", mode="raw_planner", planner=planner,
        )
        assert rec.foresight_plan == []
        assert rec.planner_memory_prompt == "raw prompt"


# ---------------------------------------------------------------------------
# 10. Logs are JSON serializable
# ---------------------------------------------------------------------------

class TestJsonSerializable:
    def test_record_json_serializable(self):
        from embodiedbench.memory.metrics import MemoryExperimentMetrics
        m = MemoryExperimentMetrics(mode="none", task_success=True, task_progress=0.5)
        rec = MemoryEpisodeLog(
            episode_id="ep9",
            task_instruction="do Y",
            foresight_plan=["a", "b"],
            task_success=True,
            task_progress=1.0,
            metrics=m.to_dict(),
        )
        serialized = json.dumps(rec.to_dict())
        recovered = json.loads(serialized)
        assert recovered["task_success"] is True
        assert recovered["foresight_plan"] == ["a", "b"]


# ---------------------------------------------------------------------------
# 11. Config log_dir respected
# ---------------------------------------------------------------------------

class TestConfigLogDir:
    def test_create_logger_from_config_uses_log_dir(self, tmp_path):
        cfg = {
            "memory_experiment": {
                "mode": "none",
                "log_memory_outputs": True,
                "log_adapter_outputs": True,
                "log_dir": str(tmp_path / "custom_logs"),
                "save_training_records": True,
            }
        }
        mem_logger = create_logger_from_config(cfg)
        assert mem_logger.enabled is True
        assert mem_logger.log_dir == str(tmp_path / "custom_logs")

    def test_create_logger_disabled_when_no_exp_key(self):
        mem_logger = create_logger_from_config({})
        assert mem_logger.enabled is False

    def test_log_writes_to_configured_dir(self, tmp_path):
        log_dir = str(tmp_path / "my_logs")
        mem_logger = MemoryExperimentLogger(
            log_dir=log_dir, enabled=True, save_training_records=True
        )
        rec = MemoryEpisodeLog(episode_id="ep_cfg")
        mem_logger.log_episode(rec)
        assert os.path.isdir(os.path.join(log_dir, "episodes"))
