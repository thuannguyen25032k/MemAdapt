"""
embodiedbench/memory_adapter/schemas.py

Input/output dataclasses for the Memory Adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# MemoryAdapterInput
# ---------------------------------------------------------------------------

@dataclass
class MemoryAdapterInput:
    """
    All information the Memory Adapter needs to produce structured guidance.

    Fields
    ------
    task_instruction        : the natural-language task the robot must complete.
    memory_context          : MemoryContext from MemoryManager.retrieve().
    """

    task_instruction: str
    memory_context: Optional[Any] = None   # MemoryContext — avoid hard import cycle

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        mc = None
        if self.memory_context is not None:
            try:
                mc = self.memory_context.compact(max_chars=100_000)
            except Exception:
                mc = str(self.memory_context)
        return {
            "task_instruction":     self.task_instruction,
            "memory_context":       mc,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryAdapterInput":
        return cls(
            task_instruction=d.get("task_instruction", ""),
            memory_context=None,   # MemoryContext is not serialisable this way
        )


# ---------------------------------------------------------------------------
# MemoryAdapterOutput
# ---------------------------------------------------------------------------

@dataclass
class MemoryAdapterOutput:
    """
    Structured output produced by the Memory Adapter.

    Fields
    ------
    foresight_plan          : high-level plan guidance bullets for the planner.
    feasibility_criteria    : concrete feasibility criteria bullets for the critic.
    fallback_strategy       : per-failure-type recovery rules for the planner.
    raw_output              : original model output before parsing.
    parse_error             : description of any parsing problem, or None.
    """

    foresight_plan: List[str] = field(default_factory=list)
    feasibility_criteria: List[str] = field(default_factory=list)
    fallback_strategy: List[str] = field(default_factory=list)
    raw_output: str = ""
    parse_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """True when no substantive content was produced."""
        return (
            not self.foresight_plan
            and not self.feasibility_criteria
            and not self.fallback_strategy
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "foresight_plan":        list(self.foresight_plan),
            "feasibility_criteria":  list(self.feasibility_criteria),
            "fallback_strategy":     list(self.fallback_strategy),
            "raw_output":            self.raw_output,
            "parse_error":           self.parse_error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryAdapterOutput":
        return cls(
            foresight_plan=list(d.get("foresight_plan", [])),
            feasibility_criteria=list(d.get("feasibility_criteria", [])),
            fallback_strategy=list(d.get("fallback_strategy", [])),
            raw_output=d.get("raw_output", ""),
            parse_error=d.get("parse_error", None),
        )
