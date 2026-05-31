"""
wandb_utils/eval_logger.py

EvalWandbLogger — per-eval-run W&B logging for benchmark evaluations.

Usage
-----
from embodiedbench.wandb_utils.eval_logger import EvalWandbLogger

logger = EvalWandbLogger(benchmark="eb_alfred", eval_set="base", mode="grpo_adapter")
for episode_info in results:
    logger.log_episode(episode_info, episode_idx=i)
    logger.log_trajectory(instruction, env.episode_log, episode_idx=i)
logger.log_summary(summary_dict)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .run import wandb_run

logger = logging.getLogger("EB_logger")


class EvalWandbLogger:
    """
    Accumulates per-episode evaluation data and logs to W&B as:
    - Per-step scalar metrics (task_success, replans, etc.)
    - An episode summary table
    - Trajectory step tables (qualitative)
    - Run-level summary metrics

    All methods are safe no-ops when W&B is disabled.
    """

    # Metric keys to forward from episode_info to W&B
    _EPISODE_SCALAR_KEYS = (
        "task_success",
        "task_progress",
        "subgoal_reward",
        "reward",
        "num_steps",
        "num_invalid_actions",
        "invalid_action_rate",
        "num_replans",
        "replan_rate",
        "planner_steps",
        "episode_elapsed_seconds",
        "planner_json_error_rate",
        # critic
        "critic_total_evaluations",
        "critic_total_rejections",
        "critic_rejection_rate",
        "critic_symbolic_rejections",
        "critic_vlm_rejections",
        "critic_triggered_replans",
        # adapter / memory
        "adapter_calls",
        "memory_injections",
    )

    def __init__(
        self,
        benchmark: str = "",
        eval_set: str = "",
        mode: str = "",
        model_name: str = "",
    ) -> None:
        self.benchmark = benchmark
        self.eval_set = eval_set
        self.mode = mode
        self.model_name = model_name
        self._episode_rows: List[List[Any]] = []
        self._local_idx = 0

    @property
    def _prefix(self) -> str:
        parts = [p for p in [self.benchmark, self.eval_set] if p]
        return "/".join(parts) if parts else "eval"

    # ------------------------------------------------------------------
    # Per-episode logging
    # ------------------------------------------------------------------

    def log_episode(
        self,
        info: Dict[str, Any],
        episode_idx: Optional[int] = None,
    ) -> None:
        """
        Log scalar metrics for one completed episode.

        Parameters
        ----------
        info        : episode_info dict from the evaluator.
        episode_idx : global episode index (uses internal counter if None).
        """
        if not wandb_run.enabled:
            self._local_idx += 1
            return

        idx = episode_idx if episode_idx is not None else self._local_idx

        # Scalar metrics
        metrics: Dict[str, Any] = {}
        for key in self._EPISODE_SCALAR_KEYS:
            val = info.get(key)
            if val is not None and isinstance(val, (int, float)):
                metrics[f"episode/{self._prefix}/{key}"] = val
        if metrics:
            wandb_run.log(metrics, step=idx)

        # Accumulate table row
        self._episode_rows.append(
            [
                idx,
                str(info.get("instruction", ""))[:120],
                float(info.get("task_success", 0)),
                float(info.get("task_progress", 0)),
                int(info.get("num_steps", 0)),
                int(info.get("num_invalid_actions", 0)),
                int(info.get("num_replans", 0)),
                round(float(info.get("replan_rate", 0)), 4),
                round(float(info.get("episode_elapsed_seconds", 0)), 1),
            ]
        )
        self._local_idx += 1

    # ------------------------------------------------------------------
    # Trajectory (qualitative) logging
    # ------------------------------------------------------------------

    def log_trajectory(
        self,
        instruction: str,
        episode_log: List[Dict[str, Any]],
        episode_idx: int,
        max_steps: int = 40,
    ) -> None:
        """
        Log a qualitative trajectory as a W&B Table.

        Parameters
        ----------
        instruction  : task instruction string.
        episode_log  : list of step dicts from env.episode_log.
        episode_idx  : used as the table key suffix.
        max_steps    : truncate long trajectories.
        """
        if not wandb_run.enabled or not episode_log:
            return
        try:
            rows = []
            for step in episode_log[:max_steps]:
                rows.append(
                    [
                        episode_idx,
                        step.get("env_step", ""),
                        str(step.get("action_description", step.get("action_id", ""))),
                        step.get("last_action_success", ""),
                        str(step.get("reasoning", ""))[:400],
                        str(step.get("env_feedback", ""))[:200],
                    ]
                )
            if rows:
                wandb_run.log_table(
                    f"eval/{self._prefix}/trajectory_{episode_idx}",
                    ["episode", "step", "action", "success", "reasoning", "feedback"],
                    rows,
                )
        except Exception as exc:
            logger.debug("[W&B] log_trajectory failed: %s", exc)

    # ------------------------------------------------------------------
    # Planner output logging (adapter-enriched prompts)
    # ------------------------------------------------------------------

    def log_planner_example(
        self,
        episode_idx: int,
        instruction: str,
        adapter_output: str,
        final_action: str,
        task_success: bool,
        step: Optional[int] = None,
    ) -> None:
        """Log one planner call as a W&B table row (qualitative)."""
        if not wandb_run.enabled:
            return
        try:
            wandb_run.log_table(
                f"eval/{self._prefix}/planner_examples",
                [
                    "episode_idx", "instruction",
                    "adapter_output_preview", "final_action", "task_success",
                ],
                [
                    [
                        episode_idx,
                        instruction[:100],
                        adapter_output[:500],
                        final_action[:100],
                        task_success,
                    ]
                ],
            )
        except Exception as exc:
            logger.debug("[W&B] log_planner_example failed: %s", exc)

    # ------------------------------------------------------------------
    # Summary logging
    # ------------------------------------------------------------------

    def log_summary(
        self,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log the episode table and run-level summary metrics.

        Call once at the end of the eval set.

        Parameters
        ----------
        summary : dict from summarize_result (e.g. mean success_rate, etc.)
        """
        if not wandb_run.enabled:
            return

        # Episode table
        if self._episode_rows:
            wandb_run.log_table(
                f"eval/{self._prefix}/episode_table",
                [
                    "episode_idx",
                    "instruction",
                    "task_success",
                    "task_progress",
                    "num_steps",
                    "num_invalid_actions",
                    "num_replans",
                    "replan_rate",
                    "elapsed_sec",
                ],
                self._episode_rows,
            )

        # Aggregated scalars
        if summary:
            scalar_summary = {
                f"summary/{self._prefix}/{k}": v
                for k, v in summary.items()
                if isinstance(v, (int, float))
            }
            if scalar_summary:
                wandb_run.log(scalar_summary)
                wandb_run.log_summary(scalar_summary)
