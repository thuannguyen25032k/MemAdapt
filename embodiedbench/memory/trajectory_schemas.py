"""
embodiedbench/memory/trajectory_schemas.py

Dataclasses for trajectory data collection during evaluation.

  TrajectoryStep    — one timestep in an episode.
  TrajectoryEpisode — full episode with all steps and outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _known_fields(cls) -> set:
    return {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TrajectoryStep
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    """One timestep within an evaluation episode."""

    episode_id: str = ""
    step_id: int = 0

    # Task context
    task_instruction: str = ""

    # Action and feedback
    action: str = ""
    planner_prompt: str = ""          # truncated planner prompt at this step
    planner_output: str = ""          # raw planner response
    critic_feedback: str = ""         # critic decision + reason, if any
    env_feedback: str = ""            # environment feedback string

    # Memory at this step
    retrieved_memory: str = ""        # raw MemoryContext text
    foresight_plan: List[str] = field(default_factory=list)
    feasibility_criteria: List[str] = field(default_factory=list)
    fallback_strategy: List[str] = field(default_factory=list)

    # Step outcome
    success: Optional[bool] = None    # None = unknown at this step
    done: bool = False

    # Free-form metadata (env_name, scene_id, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryStep":
        return cls(**{k: v for k, v in d.items() if k in _known_fields(cls)})


# ---------------------------------------------------------------------------
# TrajectoryEpisode
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryEpisode:
    """Complete episode trajectory with all steps and outcome."""

    episode_id: str = ""
    env_name: str = ""
    scene_id: str = ""
    task_instruction: str = ""

    steps: List[TrajectoryStep] = field(default_factory=list)

    # Outcome
    task_success: bool = False
    task_progress: float = 0.0
    replans: int = 0
    invalid_actions: int = 0
    total_steps: int = 0

    # Free-form metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryEpisode":
        steps_raw = d.pop("steps", [])
        ep = cls(**{k: v for k, v in d.items() if k in _known_fields(cls)})
        ep.steps = [TrajectoryStep.from_dict(s) for s in (steps_raw or [])]
        return ep
