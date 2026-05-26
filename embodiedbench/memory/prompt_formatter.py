"""
memory/prompt_formatter.py

MemoryPromptFormatter: converts MemoryContext into prompt-ready text for:
  - Planner injection (format_for_planner)
  - Critic/verifier injection (format_for_critic)
  - Compact debug display (format_compact)

Design constraints:
  - No JSON examples, no code blocks, no "output the following" instructions.
  - Plain bullet text only to avoid breaking planner JSON output parsing.
  - Separate from MemoryManager so planner/critic integration is low-risk.
"""

from __future__ import annotations

from typing import Union

from embodiedbench.memory.base import MemoryContext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLANNER_PREAMBLE = (
    "Retrieved memory is helpful but may be outdated. Current observation and environment feedback have priority."
)

_CRITIC_PREAMBLE = (
    "Use memory only to check feasibility. "
    "Current observation overrides memory.\n"
    "Use memory as auxiliary evidence, not as ground truth."
)


# ---------------------------------------------------------------------------
# MemoryPromptFormatter
# ---------------------------------------------------------------------------

class MemoryPromptFormatter:
    """Converts a ``MemoryContext`` into planner-safe, critic-safe, or compact text."""

    def __init__(
        self,
        include_preamble: bool = True,
        include_empty_sections: bool = False,
    ) -> None:
        self.include_preamble = include_preamble
        self.include_empty_sections = include_empty_sections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_for_planner(self, memory_context: MemoryContext) -> str:
        """Create memory text for VLMPlanner injection (spatial, temporal, episodic, semantic, hints, warnings)."""
        if memory_context is None or memory_context.is_empty():
            return ""

        parts: list[str] = []

        # Header
        parts.append("[Retrieved Memory for Planning]")
        if self.include_preamble:
            parts.append(_PLANNER_PREAMBLE)

        # 1. Spatial
        parts.append(self.format_section("Spatial Memory", memory_context.spatial_context))

        # 2. Temporal
        parts.append(self.format_section("Temporal Memory", memory_context.temporal_context))

        # 3. Episodic
        parts.append(self.format_section("Episodic Memory", memory_context.episodic_context))

        # 4. Semantic
        parts.append(self.format_section("Semantic Memory", memory_context.semantic_context))

        output = "\n\n".join(p for p in parts if p)
        return output

    def format_for_critic(self, memory_context: MemoryContext) -> str:
        """Create memory text for critic/verifier injection (temporal, spatial, semantic, episodic)."""
        if memory_context is None or memory_context.is_empty():
            return ""

        parts: list[str] = []

        # Header
        parts.append("[Retrieved Memory for Verification]")
        if self.include_preamble:
            parts.append(_CRITIC_PREAMBLE)

        # 1. Temporal (recent failures)
        parts.append(self.format_section("Temporal Memory", memory_context.temporal_context))

        # 2. Spatial (object reachability)
        parts.append(self.format_section("Spatial Memory", memory_context.spatial_context))

        # 3. Semantic (preconditions/rules)
        parts.append(self.format_section("Semantic Memory", memory_context.semantic_context))

        # 4. Episodic (past failure patterns)
        parts.append(self.format_section("Episodic Memory", memory_context.episodic_context))

        output = "\n\n".join(p for p in parts if p)
        return output

    def format_compact(self, memory_context: MemoryContext) -> str:
        """
        Short debug/logging representation. Only non-empty sections included.
        """
        if memory_context is None or memory_context.is_empty():
            return ""

        label_map = [
            ("Spatial",  memory_context.spatial_context),
            ("Temporal", memory_context.temporal_context),
            ("Episodic", memory_context.episodic_context),
            ("Semantic", memory_context.semantic_context),
        ]

        parts: list[str] = []
        for label, content in label_map:
            section = self.format_section(label, content)
            if section:
                parts.append(section)

        return "\n".join(parts)

    def format_section(
        self,
        title: str,
        content: Union[str, list, None],
    ) -> str:
        """
        Render a titled section.

        - str content: used as-is (stripped).
        - list content: rendered as bullet lines.
        - Empty content → returns "" unless include_empty_sections=True.
        """
        # Normalise content to string
        if content is None:
            body = ""
        elif isinstance(content, list):
            non_empty = [str(item).strip() for item in content if str(item).strip()]
            body = "\n".join(f"- {item}" for item in non_empty)
        else:
            body = str(content).strip()

        if not body:
            if self.include_empty_sections:
                return f"[{title}]\n(none)"
            return ""

        return f"[{title}]\n{body}"
