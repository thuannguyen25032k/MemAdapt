"""
memory_adapter_rl/

GRPO (Group Relative Policy Optimisation) refinement framework for the Memory
Adapter.

The adapter is refined to emit three guidance sections that the planner and
critic consume — FORESIGHT_PLAN, FEASIBILITY_CRITERIA and FALLBACK_STRATEGY —
optimised by a composite reward that scores task outcome (when available) plus
the structural validity and quality of those sections.

This package is independent of the inference runtime — it does NOT modify
planner or critic behaviour.
"""

from .config import GRPOConfig, RLConfig, RLRewardWeights
from .schemas import RewardSignal
from .rewards import (
    REQUIRED_SECTIONS,
    compute_reward,
    score_fallback_quality,
    score_feasibility_quality,
    score_foresight_quality,
    score_format_validity,
)
from .formatting import build_rl_chat_messages, build_rl_prompt, validate_xml_structure
from .grpo import (
    RolloutGroup,
    generate_candidate_group,
    make_trl_reward_fn,
    normalize_rewards,
    validate_grpo_output,
)
from .trainer import MemoryAdapterGRPOTrainer, build_lora_config
from .evaluation import RLEvaluator
from .checkpoints import CheckpointManager

__all__ = [
    "RLConfig",
    "RLRewardWeights",
    "GRPOConfig",
    "RewardSignal",
    "REQUIRED_SECTIONS",
    "compute_reward",
    "score_format_validity",
    "score_foresight_quality",
    "score_feasibility_quality",
    "score_fallback_quality",
    "build_rl_prompt",
    "build_rl_chat_messages",
    "validate_xml_structure",
    "RolloutGroup",
    "generate_candidate_group",
    "normalize_rewards",
    "validate_grpo_output",
    "make_trl_reward_fn",
    "MemoryAdapterGRPOTrainer",
    "build_lora_config",
    "RLEvaluator",
    "CheckpointManager",
]
