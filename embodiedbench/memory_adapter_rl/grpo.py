"""
memory_adapter_rl/grpo.py

GRPO (Group Relative Policy Optimisation) core logic for the Memory Adapter.

- Generate K candidate responses per prompt (a "rollout group").
- Score each candidate with the composite reward (rewards.compute_reward).
- Normalise rewards within the group to compute advantages.
- Expose a TRL-compatible reward function for trl.GRPOTrainer.

All heavy dependencies (torch, transformers) are lazy-guarded so the module can
be imported on a CPU-only box for the custom loop / tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .config import RLRewardWeights
from .rewards import REQUIRED_SECTIONS, compute_reward, score_format_validity
from .schemas import RewardSignal

logger = logging.getLogger("EB_logger")


# ---------------------------------------------------------------------------
# RolloutGroup — one GRPO group
# ---------------------------------------------------------------------------

@dataclass
class RolloutGroup:
    """
    One group of K candidate responses sampled for a single prompt.

    Attributes
    ----------
    prompt         : the input prompt used for generation.
    responses      : K raw text responses from the model.
    rewards        : per-response composite reward (RewardSignal.total).
    reward_signals : full RewardSignal dicts for detailed logging.
    advantages     : group-normalised advantages (r - mean) / (std + eps).
    xml_valid      : whether each response has all three required sections filled.
    metadata       : arbitrary extra context (episode info, step id, ...).
    """
    prompt: str = ""
    responses: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    reward_signals: List[Dict[str, Any]] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    xml_valid: List[bool] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def best_response(self) -> str:
        """Return the response with the highest reward."""
        if not self.responses or not self.rewards:
            return ""
        best_i = max(range(len(self.rewards)), key=lambda i: self.rewards[i])
        return self.responses[best_i]

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    @property
    def xml_validity_rate(self) -> float:
        return sum(self.xml_valid) / len(self.xml_valid) if self.xml_valid else 0.0


# ---------------------------------------------------------------------------
# Advantage normalisation
# ---------------------------------------------------------------------------

def normalize_rewards(rewards: List[float], epsilon: float = 1e-8) -> List[float]:
    """
    GRPO advantage normalisation: (r - mean(r)) / (std(r) + epsilon).

    Uses Bessel's correction (divide by N-1), consistent with
    ``torch.std(unbiased=True)`` inside TRL's GRPOTrainer. For N=1 the advantage
    is 0.0 (no in-group comparison is possible).
    """
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / (n - 1)
    std = variance ** 0.5
    return [(r - mean) / (std + epsilon) for r in rewards]


def validate_grpo_output(response: str) -> Dict[str, Any]:
    """
    Validate the XML structure of a GRPO completion against the three required
    sections.

    Returns a dict with: ``is_valid``, ``missing_sections``, ``format_score``.
    """
    missing = [s for s in REQUIRED_SECTIONS if f"<{s}>" not in (response or "")]
    return {
        "is_valid": len(missing) == 0,
        "missing_sections": missing,
        "format_score": score_format_validity(response),
    }


# ---------------------------------------------------------------------------
# TRL reward function
# ---------------------------------------------------------------------------

def make_trl_reward_fn(weights: Optional[RLRewardWeights] = None):
    """
    Create a ``trl.GRPOTrainer``-compatible reward function.

    TRL reward_funcs signature:
        fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]

    Environment signals (task_success, task_progress, replan_count,
    invalid_action_count) are read from dataset columns when TRL passes them as
    kwargs; otherwise safe defaults are used so the format and per-section
    quality terms still drive learning.
    """
    weights = weights or RLRewardWeights()

    def _reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        for i, response in enumerate(completions):
            def _col(key: str, default):
                col = kwargs.get(key)
                if col is None:
                    return default
                if isinstance(col, (list, tuple)):
                    return col[i] if i < len(col) else default
                return col

            sig = compute_reward(
                response=response,
                task_success=bool(_col("task_success", False)),
                task_progress=float(_col("task_progress", 0.0)),
                replan_count=int(_col("replan_count", 0)),
                invalid_action_count=int(_col("invalid_action_count", 0)),
                weights=weights,
            )
            rewards.append(sig.total)
        return rewards

    return _reward_fn


# ---------------------------------------------------------------------------
# Rollout generation (custom loop / evaluation)
# ---------------------------------------------------------------------------

def _mock_generate(prompt: str, num_generations: int) -> List[str]:
    """Deterministic mock generation for tests / CPU-only environments."""
    base = (
        "<FORESIGHT_PLAN>\n"
        "- Navigate to the kitchen table.\n"
        "- Pick up the mug.\n"
        "- Navigate to the counter.\n"
        "- Place the mug on the counter.\n"
        "</FORESIGHT_PLAN>\n\n"
        "<FEASIBILITY_CRITERIA>\n"
        '- "pick up the mug": the mug must be within reach and the gripper empty.\n'
        '- "place the mug": the counter surface must be clear.\n'
        "</FEASIBILITY_CRITERIA>\n\n"
        "<FALLBACK_STRATEGY>\n"
        '- If "cannot pick / not near": navigate to table 1, then retry pick.\n'
        '- If "place blocked": navigate to the left counter, then retry place.\n'
        "</FALLBACK_STRATEGY>\n"
    )
    return [base for _ in range(num_generations)]


def _generate_with_model(
    prompt: str,
    model: Any,
    tokenizer: Any,
    num_generations: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: str,
) -> List[str]:
    """Real generation path using a HuggingFace causal-LM."""
    try:
        import torch  # type: ignore

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]
        responses: List[str] = []
        for _ in range(num_generations):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
            new_tokens = outputs[0][input_len:]
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return responses
    except Exception as exc:  # pragma: no cover
        logger.warning("Generation failed (%s); using mock responses.", exc)
        return _mock_generate(prompt, num_generations)


def generate_candidate_group(
    prompt: str,
    model: Any,
    tokenizer: Any,
    num_generations: int = 8,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    device: str = "cpu",
    weights: Optional[RLRewardWeights] = None,
    episode_meta: Optional[Dict[str, Any]] = None,
    advantage_epsilon: float = 1e-8,
) -> RolloutGroup:
    """
    Generate K candidate responses for one prompt and compute GRPO advantages.

    When ``model`` or ``tokenizer`` is None, deterministic mock generation is
    used (CPU-safe, for tests).
    """
    meta = episode_meta or {}

    if model is None or tokenizer is None:
        responses = _mock_generate(prompt, num_generations)
    else:
        responses = _generate_with_model(
            prompt, model, tokenizer, num_generations,
            temperature, top_p, max_new_tokens, device,
        )

    reward_signals: List[RewardSignal] = []
    raw_rewards: List[float] = []
    xml_valid: List[bool] = []
    for resp in responses:
        sig = compute_reward(
            response=resp,
            task_success=bool(meta.get("task_success", False)),
            task_progress=float(meta.get("task_progress", 0.0)),
            replan_count=int(meta.get("replan_count", 0)),
            invalid_action_count=int(meta.get("invalid_action_count", 0)),
            weights=weights,
        )
        reward_signals.append(sig)
        raw_rewards.append(sig.total)
        xml_valid.append(score_format_validity(resp) >= 1.0)

    advantages = normalize_rewards(raw_rewards, epsilon=advantage_epsilon)

    return RolloutGroup(
        prompt=prompt,
        responses=responses,
        rewards=raw_rewards,
        reward_signals=[s.to_dict() for s in reward_signals],
        advantages=advantages,
        xml_valid=xml_valid,
        metadata=dict(meta),
    )
