"""
embodiedbench/memory/trajectory.py

TrajectoryRecorder — collect per-timestep data during an evaluation episode,
then materialise a ``TrajectoryEpisode`` for persistence.

    recorder = TrajectoryRecorder(episode_id, env_name, scene_id, task_instruction)
    recorder.record_step(step_id, action=..., planner_prompt=..., ...)  # per step
    recorder.finalize_episode(task_success=..., task_progress=...)       # at end
    recorder.save(log_dir)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from embodiedbench.memory.trajectory_schemas import TrajectoryEpisode, TrajectoryStep

logger = logging.getLogger("EB_logger")

# Prompts longer than this are stored truncated to reduce disk use.
_MAX_PROMPT_CHARS = 2000
# Repeated identical planner prompts are deduplicated; only the first is stored.
_DEDUP_PROMPT_SENTINEL = "[same as step 0]"


def _truncate(text: str, max_chars: int = _MAX_PROMPT_CHARS) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if len(text) > max_chars:
        return text[:max_chars] + "…[truncated]"
    return text


def _as_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return str(val)


def _as_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val] if val else []
    return [str(val)]


class TrajectoryRecorder:
    """
    Records per-timestep observations, actions, and memory signals during
    an evaluation episode, then builds a ``TrajectoryEpisode`` at the end.

    All data is kept JSON-serialisable (no raw images or numpy arrays).
    Repeated planner prompts are deduplicated to save space; long prompts
    are truncated to ``_MAX_PROMPT_CHARS`` characters.
    """

    def __init__(
        self,
        episode_id: str,
        env_name: str = "",
        scene_id: str = "",
        task_instruction: str = "",
    ) -> None:
        self.episode_id = episode_id
        self.env_name = env_name
        self.scene_id = scene_id
        self.task_instruction = task_instruction

        self._steps: List[TrajectoryStep] = []
        self._start_time: float = time.time()
        self._first_planner_prompt: str = ""  # for deduplication

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step(
        self,
        step_id: int,
        *,
        action: str = "",
        planner_prompt: str = "",
        planner_output: str = "",
        critic_feedback: str = "",
        env_feedback: str = "",
        retrieved_memory: str = "",
        foresight_plan: Optional[List[str]] = None,
        feasibility_criteria: Optional[List[str]] = None,
        fallback_strategy: Optional[List[str]] = None,
        success: Optional[bool] = None,
        done: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append one timestep record.

        All keyword arguments are optional; pass only what is available at
        that point in the step loop.
        """
        # --- Deduplicate repeated planner prompts ---
        pp = _as_str(planner_prompt)
        if not self._steps:
            # first step: store full prompt and remember for deduplication
            self._first_planner_prompt = pp
            stored_pp = _truncate(pp)
        else:
            if pp and pp == self._first_planner_prompt:
                stored_pp = _DEDUP_PROMPT_SENTINEL
            else:
                stored_pp = _truncate(pp)

        step = TrajectoryStep(
            episode_id=self.episode_id,
            step_id=step_id,
            task_instruction=self.task_instruction,
            action=_as_str(action),
            planner_prompt=stored_pp,
            planner_output=_truncate(_as_str(planner_output)),
            critic_feedback=_as_str(critic_feedback),
            env_feedback=_as_str(env_feedback),
            retrieved_memory=_truncate(_as_str(retrieved_memory)),
            foresight_plan=_as_list(foresight_plan),
            feasibility_criteria=_as_list(feasibility_criteria),
            fallback_strategy=_as_list(fallback_strategy),
            success=success,
            done=done,
            metadata=metadata or {},
        )
        self._steps.append(step)

    def finalize_episode(
        self,
        task_success: float = 0.0,
        task_progress: float = 0.0,
        replans: int = 0,
        invalid_actions: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryEpisode:
        """
        Build and return the completed ``TrajectoryEpisode``.

        The recorder retains a reference to it so ``save()`` can use it
        without receiving it as an argument.
        """
        elapsed = time.time() - self._start_time
        ep_meta: Dict[str, Any] = {
            "env_name": self.env_name,
            "scene_id": self.scene_id,
            "episode_elapsed_seconds": round(elapsed, 2),
            "num_steps": len(self._steps),
        }
        if metadata:
            ep_meta.update(metadata)

        self._episode = TrajectoryEpisode(
            episode_id=self.episode_id,
            env_name=self.env_name,
            scene_id=self.scene_id,
            task_instruction=self.task_instruction,
            steps=list(self._steps),
            task_success=bool(task_success),
            task_progress=float(task_progress),
            replans=replans,
            invalid_actions=invalid_actions,
            total_steps=len(self._steps),
            metadata=ep_meta,
        )
        return self._episode

    def save(self, log_dir: str) -> str:
        """
        Serialise the finalised ``TrajectoryEpisode`` to JSON and return the
        file path.  Creates ``<log_dir>/trajectories/`` if needed.

        Raises ``RuntimeError`` if ``finalize_episode()`` has not been called.
        """
        if not hasattr(self, "_episode"):
            raise RuntimeError(
                "TrajectoryRecorder.save() called before finalize_episode()."
            )
        traj_dir = os.path.join(log_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        filename = f"{self.episode_id}.json"
        path = os.path.join(traj_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._episode.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info(f"[TrajectoryRecorder] Saved trajectory → {path}")
        return path

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return (
            f"TrajectoryRecorder(episode_id={self.episode_id!r}, "
            f"steps={self.num_steps})"
        )
