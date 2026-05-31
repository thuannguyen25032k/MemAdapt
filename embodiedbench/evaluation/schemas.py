"""
evaluation/schemas.py

Core dataclasses for the evaluation harness:
ExperimentConfig (one run) → EpisodeResult (per episode) →
ExperimentResult (episodes + summary) → AggregateMetrics (cross-run stats).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Valid benchmark and mode identifiers
# ---------------------------------------------------------------------------

VALID_BENCHMARKS = frozenset(
    ["eb_alfred", "eb_habitat", "eb_navigation", "eb_manipulation"]
)

VALID_MODES = frozenset(
    [
        "baseline",                      # no memory, no adapter
        "raw_memory",                    # memory retrieval only
        "adapted_memory",                # retrieval + fine-tuned adapter
        "adapted_memory_planner_only",   # adapter in planner only
        "adapted_memory_critic_only",    # adapter in critic only
        "adapted_memory_planner_critic", # adapter in both
    ]
)


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Describes one benchmark evaluation run."""

    # ---- Identity ----
    experiment_id: str = ""
    benchmark: str = "eb_alfred"
    mode: str = "baseline"

    # ---- Model / adapter ----
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_checkpoint: Optional[str] = None     # path to LoRA adapter dir

    # ---- Run parameters ----
    num_episodes: int = 10
    seed: int = 42
    eval_sets: List[str] = field(default_factory=lambda: ["valid_seen"])
    max_steps_per_episode: int = 30

    # ---- Output ----
    output_dir: str = "outputs/evaluation"
    save_episode_jsons: bool = True

    # ---- Extra config passed through to existing evaluators ----
    extra_config: Dict[str, Any] = field(default_factory=dict)

    # ---- Bookkeeping ----
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        valid_fields = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    @classmethod
    def from_json(cls, s: str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Per-episode collected metrics from one benchmark episode."""

    episode_id: str = ""
    benchmark: str = ""
    mode: str = ""

    # ---- Core task outcomes ----
    task_success: bool = False
    task_progress: float = 0.0          # [0, 1] partial-credit progress
    num_steps: int = 0
    num_replans: int = 0
    num_invalid_actions: int = 0
    trajectory_length: int = 0
    runtime_seconds: float = 0.0

    # ---- Memory-specific ----
    planner_memory_usage: bool = False   # was memory consulted by planner?
    critic_memory_usage: bool = False    # was memory consulted by critic?
    adapter_used: bool = False
    adapter_fallback: bool = False       # adapter ran but fell back to raw

    # ---- Stale-memory recovery (Step 29D) ----
    stale_memory_detected: bool = False  # adapter / planner detected stale memory
    stale_memory_recovered: bool = False # agent recovered despite stale memory

    # ---- Extra data ----
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeResult":
        valid_fields = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """
    Collected results for one complete experiment run.

    Contains per-episode detail and a rolled-up summary dict.
    """

    config: ExperimentConfig = field(default_factory=ExperimentConfig)
    episodes: List[EpisodeResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    # ---- Timing ----
    started_at: str = ""
    finished_at: str = ""
    total_runtime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "episodes": [e.to_dict() for e in self.episodes],
            "summary": self.summary,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_runtime_seconds": self.total_runtime_seconds,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentResult":
        cfg = ExperimentConfig.from_dict(d.get("config", {}))
        eps = [EpisodeResult.from_dict(e) for e in d.get("episodes", [])]
        return cls(
            config=cfg,
            episodes=eps,
            summary=d.get("summary", {}),
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
            total_runtime_seconds=d.get("total_runtime_seconds", 0.0),
        )

    @classmethod
    def from_json(cls, s: str) -> "ExperimentResult":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# AggregateMetrics
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """
    Summary statistics aggregated across episodes (and optionally seeds / modes).
    """

    # ---- Identity ----
    label: str = ""              # e.g. "adapted_memory / eb_alfred / seed-42"
    benchmark: str = ""
    mode: str = ""
    num_episodes: int = 0

    # ---- Core ----
    success_rate: float = 0.0
    avg_task_progress: float = 0.0
    avg_steps: float = 0.0
    avg_replans: float = 0.0
    avg_invalid_actions: float = 0.0
    avg_trajectory_length: float = 0.0
    avg_runtime_seconds: float = 0.0

    # ---- Memory ----
    planner_memory_usage_rate: float = 0.0
    critic_memory_usage_rate: float = 0.0
    adapter_usage_rate: float = 0.0
    adapter_fallback_rate: float = 0.0

    # ---- Stale-memory recovery (Step 29D) ----
    stale_detection_rate: float = 0.0
    stale_memory_recovery_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AggregateMetrics":
        valid_fields = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid_fields})
