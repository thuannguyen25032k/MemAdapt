"""
embodiedbench/memory_adapter/adapter.py

MemoryAdapter — replay-mode module that loads pre-computed adapter outputs
from a JSONL file (MEMORY_ADAPTER_REPLAY_FILE) and injects them into the
planner / critic as structured guidance.

No LLM is loaded or called.  All outputs come from the replay file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional, Union

from embodiedbench.memory_adapter.config import MemoryAdapterConfig
from embodiedbench.memory_adapter.schemas import MemoryAdapterInput, MemoryAdapterOutput

logger = logging.getLogger("EB_logger")


# ---------------------------------------------------------------------------
# Formatted output builders
# ---------------------------------------------------------------------------

def build_planner_context(output: MemoryAdapterOutput) -> str:
    """Format MemoryAdapterOutput into a planner-injection string.

    Returns an empty string when the output carries no substantive content
    so callers can skip injection entirely.
    """
    if output.is_empty():
        return ""

    lines = []

    if output.foresight_plan:
        lines.append(
            "**Foresight Plan**: This plan may be helpful for you to complete the task."
        )
        for step in output.foresight_plan:
            lines.append(f"- {step}")
        lines.append("")

    if output.fallback_strategy:
        lines.append(
            "**Fallback Strategy**: Follow these fallback strategies when an action fails:"
        )
        for rule in output.fallback_strategy:
            lines.append(f"- {rule}")
    
    lines.append(
        "\nAlways verify these against the live observation. If the image clearly contradicts these, trust the image."
    )
    return "\n".join(lines).strip()


def build_critic_context(output: MemoryAdapterOutput) -> str:
    """Format MemoryAdapterOutput into a critic-injection string."""
    if not output.feasibility_criteria:
        return ""

    lines = []

    lines.append("Feasibility criteria:")
    for c in output.feasibility_criteria:
        lines.append(f"- {c}")
    lines.append("")

    lines.append(
        "Reject or request replanning if the proposed action violates these criteria."
    )
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# MemoryAdapter
# ---------------------------------------------------------------------------

class MemoryAdapter:
    """
    Memory Adapter that serves pre-computed structured guidance from a replay
    JSONL file (``MEMORY_ADAPTER_REPLAY_FILE`` env var or ``replay_file`` arg).

    No LLM model is loaded.  Fully decoupled from ``VLMPlanner``/``VLMCritic``.
    """

    def __init__(
        self,
        config: Union[MemoryAdapterConfig, Any] = None,
        model_name_or_path: Optional[str] = None,
        replay_file: Optional[str] = None,
    ) -> None:
        # Normalise config
        if config is None:
            self.config = MemoryAdapterConfig()
        elif isinstance(config, MemoryAdapterConfig):
            self.config = config
        else:
            self.config = MemoryAdapterConfig.from_mapping(config)

        if model_name_or_path:
            self.config.model_name_or_path = model_name_or_path

        # Cache the last adapt() result so the critic can reuse it without a second inference.
        self.last_output: Optional[MemoryAdapterOutput] = None
        # Set externally (e.g. env.log_path) to enable per-call debug logs.
        self.log_path: Optional[str] = None
        self._episode_index: int = 0   # incremented at the start of each episode
        self._log_call_index: int = 0  # resets to 0 each episode

        # ------------------------------------------------------------------
        # Replay mode: load pre-computed adapter outputs from a JSONL file
        # instead of calling the LLM.  Activated by:
        #   (a) explicit ``replay_file`` argument, or
        #   (b) MEMORY_ADAPTER_REPLAY_FILE environment variable.
        # ------------------------------------------------------------------
        self._replay_data: dict = {}   # normalized_instruction -> MemoryAdapterOutput
        _replay_path = replay_file or os.environ.get("MEMORY_ADAPTER_REPLAY_FILE", "")
        if _replay_path:
            self._load_replay_file(_replay_path)
        else:
            logger.warning(
                "[MemoryAdapter] No replay file specified. "
                "Set MEMORY_ADAPTER_REPLAY_FILE or pass replay_file=. "
                "adapt() will always return empty output."
            )

    # ------------------------------------------------------------------
    # Replay-file loading
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_instruction(text: str) -> str:
        """Normalise a task instruction for lookup (lowercase, collapse whitespace)."""
        return re.sub(r"\s+", " ", (text or "").lower().strip())

    def _load_replay_file(self, path: str) -> None:
        """
        Load pre-computed adapter outputs from *path* (a JSONL file whose rows
        match the training_records format) into ``self._replay_data``.

        Each row must have:
          ``instruction``    — the task instruction (lookup key)
          ``adapter_target`` — dict with foresight_plan / feasibility_criteria /
                               fallback_strategy

        Duplicate instructions keep the **last** seen record.
        """
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            logger.warning(
                f"[MemoryAdapter] replay_file not found: '{abs_path}'. "
                "Falling back to normal LLM generation."
            )
            return

        loaded = 0
        skipped = 0
        with open(abs_path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.debug(f"[MemoryAdapter] replay line {lineno} JSON error: {exc}")
                    skipped += 1
                    continue

                instruction = row.get("instruction", "")
                target = row.get("adapter_target")
                if not instruction or not isinstance(target, dict):
                    skipped += 1
                    continue

                output = MemoryAdapterOutput(
                    foresight_plan=list(target.get("foresight_plan") or []),
                    feasibility_criteria=list(target.get("feasibility_criteria") or []),
                    fallback_strategy=list(target.get("fallback_strategy") or []),
                    raw_output="[replay]",
                )
                self._replay_data[self._norm_instruction(instruction)] = output
                loaded += 1

        logger.info(
            f"[MemoryAdapter] Replay mode ON — loaded {loaded} records "
            f"from '{abs_path}' ({skipped} skipped)."
        )

    # ------------------------------------------------------------------
    # Prompt building (kept for subclass / test compatibility)
    # ------------------------------------------------------------------

    def build_prompt(self, adapter_input: MemoryAdapterInput) -> str:  # pragma: no cover
        raise NotImplementedError("build_prompt() is not used in replay mode.")

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _save_log(
        self,
        adapter_input: MemoryAdapterInput,
        prompt: str,
        raw: str,
        output: MemoryAdapterOutput,
    ) -> None:
        """
        Write one JSON record to ``{log_path}/adapter_logs/call_{N:04d}.json``.
        Silently skipped when ``self.log_path`` is None.
        """
        if not self.log_path:
            return
        try:
            log_dir = os.path.join(
                self.log_path, "adapter_logs",
                f"episode_{self._episode_index:04d}",
            )
            os.makedirs(log_dir, exist_ok=True)
            filename = f"call_{self._log_call_index:04d}.json"
            record = {
                "call_index": self._log_call_index,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "input": adapter_input.to_dict(),
                "prompt": prompt,
                "raw_output": raw,
                "parsed_output": output.to_dict(),
            }
            with open(os.path.join(log_dir, filename), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            self._log_call_index += 1
        except Exception as e:
            logger.debug(f"[MemoryAdapter] _save_log failed: {e}")

    # ------------------------------------------------------------------
    # Main adapt() entry point
    # ------------------------------------------------------------------

    def adapt(self, adapter_input: MemoryAdapterInput) -> MemoryAdapterOutput:
        """
        Transform retrieved memory into structured guidance.

        Returns an empty MemoryAdapterOutput if the adapter is disabled.
        When replay mode is active, looks up the pre-computed output by
        normalised task instruction instead of calling the LLM.
        Never raises — errors are captured into parse_error.
        """
        if not self.config.enabled:
            return MemoryAdapterOutput(
                parse_error="MemoryAdapter is disabled (config.enabled=False)."
            )

        # --- Replay mode: look up pre-computed output ---
        if self._replay_data:
            key = self._norm_instruction(adapter_input.task_instruction)
            if key in self._replay_data:
                output = self._replay_data[key]
                logger.debug(f"[MemoryAdapter] replay hit for: '{adapter_input.task_instruction[:60]}'")
                self._save_log(adapter_input, "[replay]", "[replay]", output)
                self.last_output = output
                return output
            else:
                logger.warning(
                    f"[MemoryAdapter] replay miss for: '{adapter_input.task_instruction[:80]}'. "
                    "No LLM fallback in replay mode — returning empty output."
                )
                empty = MemoryAdapterOutput(
                    raw_output="",
                    parse_error="replay_miss: instruction not found in replay file",
                )
                self.last_output = empty
                return empty

        # No replay data loaded at all — return empty.
        logger.warning("[MemoryAdapter] adapt() called but no replay data is loaded.")
        empty = MemoryAdapterOutput(
            raw_output="",
            parse_error="no_replay_data: MEMORY_ADAPTER_REPLAY_FILE not set or empty",
        )
        self.last_output = empty
        return empty

    def reset_last_output(self) -> None:
        """Clear the cached adapt() result. Call at the start of each episode."""
        self.last_output = None
        self._episode_index += 1
        self._log_call_index = 0

    def unload(self) -> None:
        """No-op in replay mode (no model to unload)."""
        logger.info("[MemoryAdapter] unload() called — no model to unload in replay mode.")
