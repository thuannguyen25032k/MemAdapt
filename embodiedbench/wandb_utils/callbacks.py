"""
wandb_utils/callbacks.py

HuggingFace Trainer callbacks for structured W&B logging in MemAdapt.

Classes
-------
SFTWandbCallback    — for the Memory Adapter SFT trainer.
GRPOStepLogger      — non-HF helper called from MemoryAdapterGRPOTrainer.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .run import wandb_run

logger = logging.getLogger("EB_logger")

# Guard HF import — callbacks degrade gracefully without transformers.
try:
    from transformers import TrainerCallback  # type: ignore
    _HF_AVAILABLE = True
except ImportError:
    TrainerCallback = object  # type: ignore
    _HF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _scalar_metrics(d: Dict[str, Any]) -> Dict[str, float]:
    """Filter a metrics dict to JSON-safe scalar values only."""
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# SFT callback
# ---------------------------------------------------------------------------

class SFTWandbCallback(TrainerCallback):
    """
    HF TrainerCallback for SFT runs.

    Logs
    ----
    - Training loss, eval loss, LR, grad norm (every step via on_log).
    - GPU memory at training start.
    - Config dict at training start.
    - Dataset statistics at training start.
    - Qualitative generation examples at each eval (optional).
    - Checkpoint step in run summary on save.
    """

    def __init__(
        self,
        run_cfg_dict: Optional[Dict[str, Any]] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
        eval_examples_fn: Optional[Callable] = None,
        log_every_n_eval: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        run_cfg_dict     : full config dict to record in the run.
        dataset_stats    : e.g. {"train_samples": 8000, "val_samples": 1000}.
        eval_examples_fn : callable(model, tokenizer) → List[Dict[str, str]]
                           returning qualitative generation examples.
        log_every_n_eval : log qualitative examples every N eval calls.
        """
        self._run_cfg = run_cfg_dict or {}
        self._dataset_stats = dataset_stats or {}
        self._eval_examples_fn = eval_examples_fn
        self._log_every_n_eval = log_every_n_eval
        self._eval_count = 0

    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        if not wandb_run.enabled:
            return
        # Config snapshot
        if self._run_cfg:
            wandb_run.log_config(self._run_cfg)
        # Accumulate all step-0 metrics into a single log call. Logging them in
        # separate commits would advance wandb's internal step past 0 and make
        # the subsequent step-0 call raise a "step less than current" warning.
        init_metrics: Dict[str, Any] = {}
        if self._dataset_stats:
            init_metrics.update(
                {f"data/{k}": v for k, v in self._dataset_stats.items()}
            )
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                init_metrics["system/gpu_name"] = str(props.name)
                init_metrics["system/gpu_memory_total_gb"] = round(
                    props.total_memory / 1e9, 2
                )
        except Exception:
            pass
        if init_metrics:
            wandb_run.log(init_metrics, step=0)

    def on_log(self, args, state, control, logs: Optional[Dict] = None, **kwargs):
        if not wandb_run.enabled or not logs:
            return
        # Rename to explicit namespaces
        metrics: Dict[str, Any] = {}
        for k, v in logs.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("eval_"):
                metrics[f"eval/{k[5:]}"] = v
            elif k in ("loss", "learning_rate", "grad_norm", "epoch"):
                metrics[f"train/{k}"] = v
            else:
                metrics[f"train/{k}"] = v
        if metrics:
            wandb_run.log(metrics, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not wandb_run.enabled:
            return
        self._eval_count += 1
        # Qualitative examples (throttled)
        if (
            self._eval_examples_fn is not None
            and (self._eval_count % self._log_every_n_eval == 0)
        ):
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")
            if model is not None and tokenizer is not None:
                try:
                    examples = self._eval_examples_fn(model, tokenizer)
                    if examples:
                        cols = list(examples[0].keys())
                        rows = [[ex.get(c, "") for c in cols] for ex in examples]
                        wandb_run.log_table(
                            f"eval/generation_examples_step{state.global_step}",
                            cols,
                            rows,
                        )
                except Exception as exc:
                    logger.debug("[W&B] eval_examples_fn failed: %s", exc)

    def on_save(self, args, state, control, **kwargs):
        if not wandb_run.enabled:
            return
        wandb_run.log_summary({"last_checkpoint_step": state.global_step})

    def on_train_end(self, args, state, control, **kwargs):
        if not wandb_run.enabled:
            return
        wandb_run.log_summary(
            {
                "final_global_step": state.global_step,
                "final_epoch": state.epoch,
                "best_metric": state.best_metric,
            }
        )


# ---------------------------------------------------------------------------
# GRPO step logger (not a HF callback — called manually from the trainer)
# ---------------------------------------------------------------------------

class GRPOStepLogger:
    """
    Logs per-step GRPO metrics from RolloutGroup objects.

    This is used both in:
    - The TRL path (added as a HF callback that calls log_rollout_group on_log).
    - The custom loop path (called directly after generate_candidate_group).

    Usage (custom loop)
    -------------------
    step_logger = GRPOStepLogger()
    group = generate_candidate_group(...)
    step_logger.log_rollout_group(group, step=global_step)
    step_logger.maybe_log_example_table(group, step=global_step)
    """

    def __init__(self, log_table_every_n: int = 50) -> None:
        self._log_table_every = log_table_every_n
        self._step_count = 0

    def log_rollout_group(self, group: Any, step: int) -> None:
        """Log scalar metrics from a RolloutGroup."""
        if not wandb_run.enabled:
            return
        try:
            metrics: Dict[str, Any] = {
                "grpo/mean_reward": round(group.mean_reward, 4),
                "grpo/xml_validity_rate": round(group.xml_validity_rate, 4),
                "grpo/group_size": len(group.rewards),
            }

            # Reward component breakdown (from RewardSignal dicts)
            if group.reward_signals:
                for component in [
                    "task_success", "task_progress", "replan_count",
                    "invalid_action_count", "format_validity", "foresight_quality",
                    "feasibility_quality", "fallback_quality", "repetition_penalty",
                    "total",
                ]:
                    vals = [
                        float(s.get(component, 0.0))
                        for s in group.reward_signals
                        if isinstance(s, dict) and component in s
                    ]
                    if vals:
                        metrics[f"grpo/reward_{component}_mean"] = round(
                            sum(vals) / len(vals), 4
                        )

            # Reward distribution
            if group.rewards:
                metrics["grpo/reward_std"] = round(_std(group.rewards), 4)
                metrics["grpo/reward_max"] = round(max(group.rewards), 4)
                metrics["grpo/reward_min"] = round(min(group.rewards), 4)

            # Completion length (proxy for verbosity)
            if group.responses:
                lengths = [len(r.split()) for r in group.responses]
                metrics["grpo/mean_completion_words"] = round(
                    sum(lengths) / len(lengths), 1
                )
                metrics["grpo/max_completion_words"] = max(lengths)

            # Malformed XML rate
            total = len(group.xml_valid)
            if total > 0:
                metrics["grpo/malformed_xml_rate"] = round(
                    sum(1 for v in group.xml_valid if not v) / total, 4
                )

            wandb_run.log(metrics, step=step)

        except Exception as exc:
            logger.debug("[W&B] log_rollout_group failed: %s", exc)

        self._step_count += 1

    def maybe_log_example_table(
        self,
        group: Any,
        step: int,
        max_rows: int = 4,
    ) -> None:
        """Periodically log a W&B table of candidate responses with rewards."""
        if not wandb_run.enabled:
            return
        if self._step_count % self._log_table_every != 0:
            return
        try:
            rows = []
            for i, (resp, reward, xml_ok) in enumerate(
                zip(
                    group.responses[:max_rows],
                    group.rewards[:max_rows],
                    group.xml_valid[:max_rows],
                )
            ):
                rows.append([step, i, round(reward, 4), xml_ok, resp[:600]])
            if rows:
                wandb_run.log_table(
                    "grpo/rollout_examples",
                    ["step", "candidate_idx", "reward", "xml_valid", "response_preview"],
                    rows,
                )
        except Exception as exc:
            logger.debug("[W&B] maybe_log_example_table failed: %s", exc)
