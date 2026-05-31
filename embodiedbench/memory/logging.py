"""
memory/logging.py

Structured episode logging for MemAdapt ablation experiments and future
SFT/RL training data collection.

Two output formats per episode:
  1. Full episode JSON  → <log_dir>/episodes/<episode_id>.json
  2. SFT-ready JSONL    → <log_dir>/training_records.jsonl  (one line per episode)
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from embodiedbench.memory.storage import save_json, append_jsonl
from embodiedbench.memory.utils import safe_attr, safe_str

logger = logging.getLogger("EB_logger")


# ---------------------------------------------------------------------------
# MemoryEpisodeLog  — full structured record for one episode
# ---------------------------------------------------------------------------

@dataclass
class MemoryEpisodeLog:
    """All memory-related data produced during one evaluation episode."""

    episode_id: str = ""
    env_name: str = ""
    scene_id: str = ""
    task_instruction: str = ""
    mode: str = "none"

    # --- Raw / formatted memory prompts ---
    raw_memory_context: str = ""        # MemoryContext.to_text() snapshot
    planner_memory_prompt: str = ""     # last_memory_prompt from planner
    critic_memory_prompt: str = ""      # last_adapted_memory_prompt from critic

    # --- Adapter outputs ---
    foresight_plan: List[str] = field(default_factory=list)
    feasibility_criteria: List[str] = field(default_factory=list)
    fallback_strategy: List[str] = field(default_factory=list)

    # --- Episode trajectory ---
    planner_actions: List[str] = field(default_factory=list)
    critic_events: List[Dict] = field(default_factory=list)

    # --- Outcome ---
    final_status: str = ""              # "success" / "partial" / "failure"
    task_success: Optional[bool] = None
    task_progress: Optional[float] = None

    # --- Metrics and free-form metadata ---
    metrics: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-safe dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# MemoryExperimentLogger
# ---------------------------------------------------------------------------

class MemoryExperimentLogger:
    """
    Writes structured memory logs and SFT-ready training records to disk.

    Parameters
    ----------
    log_dir : str
        Root directory for all output files.
    enabled : bool
        When False every method is a no-op.
    save_training_records : bool
        When True, also append to training_records.jsonl.
    """

    EPISODES_SUBDIR = "episodes"
    TRAINING_FILE   = "training_records.jsonl"

    def __init__(
        self,
        log_dir: str,
        enabled: bool = True,
        save_training_records: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.enabled = enabled
        self.save_training_records = save_training_records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_episode(self, record: MemoryEpisodeLog) -> str:
        """
        Persist a full episode record as <log_dir>/episodes/<episode_id>.json.
        Returns the file path, or "" when disabled.
        """
        if not self.enabled:
            return ""
        try:
            ep_dir = os.path.join(self.log_dir, self.EPISODES_SUBDIR)
            ep_id = record.episode_id or str(uuid.uuid4())
            path = os.path.join(ep_dir, f"{ep_id}.json")
            save_json(path, record.to_dict())
            logger.debug(f"[MemoryLogger] Saved episode log → {path}")
            return path
        except Exception as e:
            logger.warning(f"[MemoryLogger] log_episode failed (non-fatal): {e}")
            return ""

    def append_training_record(self, record: MemoryEpisodeLog) -> str:
        """
        Append a compact SFT-ready row to <log_dir>/training_records.jsonl.
        Returns the file path, or "" when disabled / save_training_records=False.
        """
        if not self.enabled or not self.save_training_records:
            return ""
        try:
            path = os.path.join(self.log_dir, self.TRAINING_FILE)
            append_jsonl(path, self._to_training_row(record))
            logger.debug(f"[MemoryLogger] Appended training record → {path}")
            return path
        except Exception as e:
            logger.warning(f"[MemoryLogger] append_training_record failed (non-fatal): {e}")
            return ""

    # ------------------------------------------------------------------
    # Builder helper
    # ------------------------------------------------------------------

    @staticmethod
    def build_episode_log(
        *,
        episode_id: str,
        env_name: str,
        scene_id: str,
        task_instruction: str,
        mode: str,
        planner: Any = None,
        critic: Any = None,
        episode_info: dict = None,
        metrics: Any = None,
        metadata: dict = None,
    ) -> MemoryEpisodeLog:
        """
        Collect all available data from planner/critic/episode_info into a
        MemoryEpisodeLog.  Every field degrades gracefully to "" / [] / None
        when the corresponding object is absent or lacks the attribute.
        """
        episode_info = episode_info or {}
        metadata = metadata or {}

        # --- Memory prompts from planner ---
        planner_memory_prompt    = safe_str(planner, "last_memory_prompt")

        # --- Raw memory context text ---
        raw_memory_context = ""
        ctx = safe_attr(planner, "last_memory_context")
        if ctx is not None:
            try:
                if hasattr(ctx, "compact"):
                    raw_memory_context = ctx.compact(max_chars=100_000)
                elif hasattr(ctx, "to_text"):
                    raw_memory_context = ctx.to_text()
                else:
                    raw_memory_context = str(ctx)
            except Exception:
                pass

        # --- Planner actions from episode_act_feedback ---
        planner_actions: List[str] = []
        act_fb = safe_attr(planner, "episode_act_feedback") or []
        for entry in act_fb:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                aid, desc = entry[0], entry[1]
                if aid == -3:
                    planner_actions.append(f"[CRITIC_FEEDBACK] {desc}")
                elif aid == -2:
                    planner_actions.append("[EMPTY_PLAN]")
                elif aid == -1:
                    planner_actions.append("[INVALID_ACTION]")
                else:
                    planner_actions.append(f"action_id={aid}: {desc}")
            elif isinstance(entry, str):
                planner_actions.append(entry)

        # --- Critic outputs ---
        critic_memory_prompt   = ""
        critic_events: List[Dict] = []

        # Try DualCritic → vlm sub-critic first, then bare VLMCritic
        vlm_critic = (
            safe_attr(critic, "vlm")
            if critic is not None and hasattr(critic, "vlm")
            else critic
        )
        if vlm_critic is not None:
            critic_memory_prompt   = safe_str(vlm_critic, "last_adapted_memory_prompt") \
                                     or safe_str(vlm_critic, "last_memory_prompt")

        # critic_events from DualCritic._episode_critic_records
        raw_records = safe_attr(critic, "_episode_critic_records") or []
        for r in raw_records:
            if isinstance(r, dict):
                inp = r.get("input", {})
                critic_events.append({
                    "env_step":     r.get("env_step"),
                    "action_id":    inp.get("action_id"),
                    "action_str":   inp.get("action_str", ""),
                    "is_first_step":inp.get("is_first_step", False),
                    "valid":        r.get("final_decision", {}).get("valid", True),
                    "reason":       r.get("final_decision", {}).get("feedback", ""),
                })

        # --- Adapter structured outputs (planner path) ---
        foresight_plan:       List[str] = []
        feasibility_criteria: List[str] = []
        fallback_strategy:    List[str] = []
        adapted_output = safe_attr(planner, "last_adapted_memory_output")
        if adapted_output is None and vlm_critic is not None:
            adapted_output = safe_attr(vlm_critic, "last_adapted_memory_output")
        if adapted_output is not None:
            foresight_plan       = list(getattr(adapted_output, "foresight_plan",       []) or [])
            feasibility_criteria = list(getattr(adapted_output, "feasibility_criteria", []) or [])
            fallback_strategy    = list(getattr(adapted_output, "fallback_strategy",    []) or [])

        # --- Outcome ---
        task_success  = episode_info.get("task_success")
        task_progress = episode_info.get("task_progress")
        if task_success is not None:
            task_success = bool(task_success)
        if task_progress is not None:
            task_progress = float(task_progress)

        final_status = "unknown"
        if task_success:
            final_status = "success"
        elif task_progress and float(task_progress) > 0:
            final_status = "partial"
        elif task_success is not None:
            final_status = "failure"

        # --- Metrics ---
        metrics_dict: dict = {}
        if metrics is not None and hasattr(metrics, "to_dict"):
            try:
                metrics_dict = metrics.to_dict()
            except Exception:
                pass

        return MemoryEpisodeLog(
            episode_id=episode_id,
            env_name=env_name,
            scene_id=scene_id,
            task_instruction=task_instruction,
            mode=mode,
            raw_memory_context=raw_memory_context,
            planner_memory_prompt=planner_memory_prompt,
            critic_memory_prompt=critic_memory_prompt,
            foresight_plan=foresight_plan,
            feasibility_criteria=feasibility_criteria,
            fallback_strategy=fallback_strategy,
            planner_actions=planner_actions,
            critic_events=critic_events,
            final_status=final_status,
            task_success=task_success,
            task_progress=task_progress,
            metrics=metrics_dict,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_training_row(record: MemoryEpisodeLog) -> dict:
        """Compact SFT-ready row."""
        return {
            "instruction": record.task_instruction,
            "retrieved_memory": record.raw_memory_context,
            "planner_prompt": record.planner_memory_prompt,
            "adapter_target": {
                "foresight_plan":       record.foresight_plan,
                "feasibility_criteria": record.feasibility_criteria,
                "fallback_strategy":    record.fallback_strategy,
            },
            "outcome": {
                "success":  record.task_success,
                "progress": record.task_progress,
                "steps":    record.metrics.get("env_steps"),
                "replans":  record.metrics.get("replans"),
            },
        }
