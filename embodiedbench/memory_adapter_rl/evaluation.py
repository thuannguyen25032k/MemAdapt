"""
memory_adapter_rl/evaluation.py

Evaluation utilities for the GRPO refinement pipeline.

Metrics
-------
- reward_mean / reward_std    : mean and stddev of the composite reward
- format_validity_rate        : fraction of outputs with all 3 sections filled
- avg_foresight_quality
- avg_feasibility_quality
- avg_fallback_quality
- avg_repetition_penalty
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

from .config import RLRewardWeights
from .formatting import validate_xml_structure
from .rewards import (
    compute_reward,
    score_fallback_quality,
    score_feasibility_quality,
    score_foresight_quality,
)

logger = logging.getLogger("EB_logger")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_response_quality(
    responses: List[str],
    weights: Optional[RLRewardWeights] = None,
) -> Dict[str, float]:
    """Evaluate a list of adapter responses on structural + quality metrics."""
    if not responses:
        return {}

    weights = weights or RLRewardWeights()
    xml_valid = 0
    foresight, feasibility, fallback, reward_totals = [], [], [], []

    for resp in responses:
        is_valid, _ = validate_xml_structure(resp)
        if is_valid:
            xml_valid += 1
        foresight.append(score_foresight_quality(resp))
        feasibility.append(score_feasibility_quality(resp))
        fallback.append(score_fallback_quality(resp))
        reward_totals.append(compute_reward(resp, weights=weights).total)

    n = len(responses)
    mean_r = sum(reward_totals) / n
    variance = sum((r - mean_r) ** 2 for r in reward_totals) / n

    return {
        "format_validity_rate": round(xml_valid / n, 4),
        "avg_foresight_quality": round(sum(foresight) / n, 4),
        "avg_feasibility_quality": round(sum(feasibility) / n, 4),
        "avg_fallback_quality": round(sum(fallback) / n, 4),
        "reward_mean": round(mean_r, 4),
        "reward_std": round(math.sqrt(variance), 4),
        "n": n,
    }


def evaluate_reward_trend(reward_history: List[float]) -> Dict[str, Any]:
    """Summarise a reward trend (mean reward per logging step) over training."""
    if not reward_history:
        return {}
    first, last = reward_history[0], reward_history[-1]
    improvement = last - first
    direction = "improving" if improvement > 0 else ("degrading" if improvement < 0 else "flat")
    return {
        "initial_reward": round(first, 4),
        "final_reward": round(last, 4),
        "mean_reward": round(sum(reward_history) / len(reward_history), 4),
        "improvement": round(improvement, 4),
        "trend_direction": direction,
        "n_steps": len(reward_history),
    }


# ---------------------------------------------------------------------------
# RLEvaluator
# ---------------------------------------------------------------------------

class RLEvaluator:
    """Orchestrates GRPO evaluation metrics for one training run."""

    def __init__(self, weights: Optional[RLRewardWeights] = None) -> None:
        self.weights = weights or RLRewardWeights()

    def evaluate_all(
        self,
        responses: Optional[List[str]] = None,
        reward_history: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run all applicable evaluations and return a merged metrics dict."""
        metrics: Dict[str, Any] = {}
        if responses:
            q = evaluate_response_quality(responses, self.weights)
            metrics.update({f"quality/{k}": v for k, v in q.items()})
        if reward_history:
            t = evaluate_reward_trend(reward_history)
            metrics.update({f"trend/{k}": v for k, v in t.items()})
        return metrics

    def save_metrics(self, metrics: Dict[str, Any], output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "rl_eval_metrics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Saved RL eval metrics -> %s", path)
        return path
