"""
memory_adapter/utils.py

Shared utility helpers for the MemoryAdapter subsystem.
"""

from __future__ import annotations


def is_unsafe_adapter_output(
    prompt: str,
    *,
    check_action_schema: bool = False,
    max_chars: int = 0,
) -> bool:
    """
    Return True when *prompt* should be rejected and the caller should fall
    back to the raw MemoryPromptFormatter output.

    Parameters
    ----------
    prompt             : the formatted context string from MemoryAdapterOutput.
    check_action_schema: also reject if the output contains ``"action_id"``
                         (a fragment of the environment's JSON action schema).
                         Set True for critic, False for planner.
    max_chars          : if > 0, reject when ``len(prompt) > max_chars``.
    """
    if not prompt or not prompt.strip():
        return True
    if "```" in prompt:
        return True
    if check_action_schema and '"action_id"' in prompt:
        return True
    if max_chars > 0 and len(prompt) > max_chars:
        return True
    return False
