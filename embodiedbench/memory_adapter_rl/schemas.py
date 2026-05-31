"""
memory_adapter_rl/schemas.py

Reward signal schema for the GRPO refinement pipeline.

RewardSignal carries an interpretable breakdown of the composite reward for
one Memory Adapter response. Every component is logged separately alongside the
weighted ``total`` so reward shaping stays transparent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class RewardSignal:
    """
    Interpretable reward breakdown for one Memory Adapter response.

    The Memory Adapter must emit exactly three XML sections that drive the
    downstream planner / critic: FORESIGHT_PLAN, FEASIBILITY_CRITERIA and
    FALLBACK_STRATEGY (see ``memory_adapter.prompts``). The reward therefore
    scores (a) task outcome when an environment rollout is available and
    (b) the structural validity and quality of those three sections.

    Component ranges
    ----------------
    task_success / task_progress       : [0, 1]
    replan_count / invalid_action_count: raw counts (penalised)
    format_validity                    : [0, 1] fraction of the 3 sections present and non-empty
    foresight_quality                  : [0, 1]
    feasibility_quality                : [0, 1]
    fallback_quality                   : [0, 1]
    repetition_penalty                 : [0, 1] degeneracy (penalised)
    total                              : weighted composite (already incorporates weights)
    """

    # ---- Task outcome (0 when no environment rollout is available) ----
    task_success:         float = 0.0
    task_progress:        float = 0.0
    replan_count:         int = 0
    invalid_action_count: int = 0

    # ---- Structural + per-section quality  [0, 1] ----
    format_validity:      float = 0.0
    foresight_quality:    float = 0.0
    feasibility_quality:  float = 0.0
    fallback_quality:     float = 0.0

    # ---- Degeneracy penalty  [0, 1] ----
    repetition_penalty:   float = 0.0

    # ---- Composite ----
    total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardSignal":
        valid = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid})
