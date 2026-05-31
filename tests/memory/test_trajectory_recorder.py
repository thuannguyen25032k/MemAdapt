"""
tests/memory/test_trajectory_recorder.py

Tests for TrajectoryRecorder — the one runtime-used component of memory/trajectory.
"""

import json

from embodiedbench.memory.trajectory import (
    TrajectoryRecorder,
    _DEDUP_PROMPT_SENTINEL,
    _MAX_PROMPT_CHARS,
)
from embodiedbench.memory.trajectory_schemas import TrajectoryEpisode


def _make_recorder() -> TrajectoryRecorder:
    return TrajectoryRecorder(
        episode_id="alfred_base_1",
        env_name="alfred",
        scene_id="base",
        task_instruction="Put a cold apple in the fridge.",
    )


class TestRecordStep:
    def test_records_steps(self):
        rec = _make_recorder()
        rec.record_step(0, action="GotoLocation", planner_prompt="P0")
        rec.record_step(1, action="PickupObject", planner_prompt="P1")
        assert rec.num_steps == 2

    def test_dedup_repeated_prompt(self):
        rec = _make_recorder()
        rec.record_step(0, planner_prompt="same prompt")
        rec.record_step(1, planner_prompt="same prompt")
        ep = rec.finalize_episode()
        assert ep.steps[0].planner_prompt == "same prompt"
        assert ep.steps[1].planner_prompt == _DEDUP_PROMPT_SENTINEL

    def test_long_prompt_truncated(self):
        rec = _make_recorder()
        rec.record_step(0, planner_prompt="x" * (_MAX_PROMPT_CHARS + 500))
        ep = rec.finalize_episode()
        assert ep.steps[0].planner_prompt.endswith("…[truncated]")

    def test_list_fields_normalised(self):
        rec = _make_recorder()
        rec.record_step(0, foresight_plan=["step a", "step b"], fallback_strategy="single")
        ep = rec.finalize_episode()
        assert ep.steps[0].foresight_plan == ["step a", "step b"]
        assert ep.steps[0].fallback_strategy == ["single"]


class TestFinalizeEpisode:
    def test_returns_episode_with_outcome(self):
        rec = _make_recorder()
        rec.record_step(0, action="a")
        rec.record_step(1, action="b")
        ep = rec.finalize_episode(
            task_success=True, task_progress=0.5, replans=2, invalid_actions=1
        )
        assert isinstance(ep, TrajectoryEpisode)
        assert ep.task_success is True
        assert ep.task_progress == 0.5
        assert ep.replans == 2
        assert ep.invalid_actions == 1
        assert ep.total_steps == 2

    def test_episode_is_json_serializable(self):
        rec = _make_recorder()
        rec.record_step(0, action="a", foresight_plan=["s1"])
        ep = rec.finalize_episode(task_success=False, task_progress=0.0)
        json.dumps(ep.to_dict())  # must not raise


class TestSave:
    def test_save_requires_finalize(self, tmp_path):
        rec = _make_recorder()
        rec.record_step(0, action="a")
        try:
            rec.save(str(tmp_path))
        except RuntimeError:
            pass
        else:
            raise AssertionError("save() should raise before finalize_episode()")

    def test_save_writes_json(self, tmp_path):
        rec = _make_recorder()
        rec.record_step(0, action="a")
        rec.finalize_episode(task_success=True, task_progress=1.0)
        path = rec.save(str(tmp_path))
        assert path.endswith("alfred_base_1.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["episode_id"] == "alfred_base_1"
        assert data["task_success"] is True
        assert len(data["steps"]) == 1
