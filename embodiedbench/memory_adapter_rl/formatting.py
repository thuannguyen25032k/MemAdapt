"""
memory_adapter_rl/formatting.py

Prompt formatting and structural validation for the GRPO refinement pipeline.

The RL prompt is built from the SAME single source of truth as inference and SFT
training (``memory_adapter.prompts`` + ``memory_adapter_training.formatting``),
so GRPO never introduces format drift: the system turn carries the role/format
instructions and the user turn carries the task instruction + retrieved memory.

The required output schema is the canonical three sections:
    <FORESIGHT_PLAN> / <FEASIBILITY_CRITERIA> / <FALLBACK_STRATEGY>
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from embodiedbench.memory_adapter.prompts import (
    ALL_SECTIONS,
    build_adapter_user_content,
)
from embodiedbench.memory_adapter_training.formatting import to_chat_messages


def build_rl_prompt(task_instruction: str, retrieved_memory: str) -> str:
    """
    Build the user-turn prompt for one RL rollout.

    Identical to the SFT / inference user turn so the policy sees the exact
    distribution it was supervised on. The data collator / chat template
    prepends the shared system prompt at training time.
    """
    return build_adapter_user_content(task_instruction, retrieved_memory)


def build_rl_chat_messages(task_instruction: str, retrieved_memory: str) -> List[Dict[str, str]]:
    """Return ``[system, user]`` chat messages for one RL rollout."""
    return to_chat_messages(build_rl_prompt(task_instruction, retrieved_memory))


def validate_xml_structure(response: str) -> Tuple[bool, List[str]]:
    """
    Check whether a response contains all required section tags.

    Returns ``(is_valid, missing_sections)``.
    """
    missing = [s for s in ALL_SECTIONS if f"<{s}>" not in (response or "")]
    return (len(missing) == 0, missing)
