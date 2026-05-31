"""
memory_adapter_training/formatting.py

Deterministic prompt / target formatting for Memory Adapter SFT training.

This module has NO heavy dependencies (no torch, no transformers) and is safe
to import anywhere, including tests.

Train / inference alignment
---------------------------
The prompt is split into a system turn (SYSTEM_PROMPTS) and a user turn
(instruction + retrieved memory), exactly as at inference time
(memory_adapter.adapter.MemoryAdapter). format_sample stores the user-turn
content; the data collator prepends the shared system turn and applies the
model chat template via tokenizer.apply_chat_template, so training matches
deployment.

The target is the structured XML response whose tags (FORESIGHT_PLAN,
FEASIBILITY_CRITERIA, FALLBACK_STRATEGY) match the output format defined in
memory_adapter/prompts.py.
"""

from __future__ import annotations

import re as _re
from typing import Any, Dict, List, Optional

from embodiedbench.memory_adapter.prompts import (
    SYSTEM_PROMPTS,
    build_adapter_user_content,
)


# ---------------------------------------------------------------------------
# Chat-messages helper (shared by collator and evaluation)
# ---------------------------------------------------------------------------

def to_chat_messages(user_content: str) -> List[Dict[str, str]]:
    """Wrap a user-turn string into [system, user] chat messages.

    Uses the same SYSTEM_PROMPTS as inference so training/eval stay aligned.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPTS.rstrip()},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Target (assistant response) builder
# ---------------------------------------------------------------------------

def build_target_text(
    foresight_plan: Optional[List[str]] = None,
    feasibility_criteria: Optional[List[str]] = None,
    fallback_strategy: Optional[List[str]] = None,
) -> str:
    """Build the assistant-side structured target string using XML tags.

    All three sections are always present (using "N/A" when empty) and the tag
    names match the output schema in memory_adapter/prompts.py.
    """
    def _list_block(items: Optional[List[str]], fallback: str = "N/A") -> str:
        cleaned = [i.strip() for i in (items or []) if i and i.strip()]
        return "\n".join(f"- {i}" for i in cleaned) if cleaned else fallback

    blocks = [
        f"<FORESIGHT_PLAN>\n{_list_block(foresight_plan)}\n</FORESIGHT_PLAN>",
        f"<FEASIBILITY_CRITERIA>\n{_list_block(feasibility_criteria)}\n</FEASIBILITY_CRITERIA>",
        f"<FALLBACK_STRATEGY>\n{_list_block(fallback_strategy)}\n</FALLBACK_STRATEGY>",
    ]
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Sample -> (prompt, response)
# ---------------------------------------------------------------------------

def _as_list(value: Any) -> List[str]:
    """Coerce a target field into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def format_sample(sample: Dict[str, Any]) -> Dict[str, str]:
    """Convert a curated record into a {"prompt": ..., "response": ...} pair.

    Supported record shapes:
      * Filtered SFT target (filter_sft_targets.py output) with keys
        "instruction", "retrieved_memory" and a nested "adapter_target" dict
        holding "foresight_plan", "feasibility_criteria", "fallback_strategy".
      * Already-formatted pair {"prompt", "response"} (returned as-is).
    """
    if "prompt" in sample and "response" in sample:
        return {"prompt": str(sample["prompt"]), "response": str(sample["response"])}

    instruction = sample.get("instruction") or sample.get("task_instruction", "")
    memory = sample.get("retrieved_memory") or sample.get("memory_context", "")

    target = sample.get("adapter_target") or {}
    foresight = _as_list(
        target.get("foresight_plan")
        or sample.get("target_foresight_plan")
        or sample.get("teacher_foresight_plan")
    )
    feasibility = _as_list(
        target.get("feasibility_criteria")
        or sample.get("target_feasibility_criteria")
        or sample.get("teacher_feasibility_criteria")
    )
    fallback = _as_list(
        target.get("fallback_strategy")
        or sample.get("target_fallback_strategy")
        or sample.get("teacher_fallback_strategy")
    )

    return {
        "prompt": build_adapter_user_content(str(instruction), str(memory)),
        "response": build_target_text(foresight, feasibility, fallback),
    }


# ---------------------------------------------------------------------------
# Parse a structured response back into fields (used by evaluation)
# ---------------------------------------------------------------------------

_TAG_RE = _re.compile(r"<([A-Z_]+)>(.*?)</\1>", _re.S)


def parse_target_text(text: str) -> Dict[str, List[str]]:
    """Parse a structured assistant response (XML-tag format) into a dict with
    keys foresight_plan, feasibility_criteria and fallback_strategy.
    """
    result: Dict[str, List[str]] = {
        "foresight_plan": [],
        "feasibility_criteria": [],
        "fallback_strategy": [],
    }
    key_by_tag = {
        "FORESIGHT_PLAN": "foresight_plan",
        "FEASIBILITY_CRITERIA": "feasibility_criteria",
        "FALLBACK_STRATEGY": "fallback_strategy",
    }
    for m in _TAG_RE.finditer(text):
        key = key_by_tag.get(m.group(1))
        if key is not None:
            result[key] = _parse_list(m.group(2).strip())
    return result


def _parse_list(text: str) -> List[str]:
    items = []
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        if line and line.lower() not in ("n/a", "none detected", "none"):
            items.append(line)
    return items
