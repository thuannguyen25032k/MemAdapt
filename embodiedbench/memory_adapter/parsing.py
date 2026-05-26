"""
embodiedbench/memory_adapter/parsing.py

Robust section-based parser for Memory Adapter model output.

The adapter model is expected to emit XML-tagged sections in the form:

    <SECTION_NAME>
    - bullet 1
    - bullet 2
    </SECTION_NAME>

This parser:
  - extracts sections via XML tag regex;
  - handles missing sections gracefully;
  - converts bullet-list sections to Python lists;
  - never raises — always returns a MemoryAdapterOutput.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional

from embodiedbench.memory_adapter.schemas import MemoryAdapterOutput
from embodiedbench.memory_adapter.prompts import (
    SECTION_FORESIGHT_PLAN,
    SECTION_FEASIBILITY_CRITERIA,
    SECTION_FALLBACK_STRATEGY,
    ALL_SECTIONS,
)

logger = logging.getLogger("EB_logger")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_into_sections(text: str) -> Dict[str, str]:
    """
    Split raw model output into a dict mapping SECTION_LABEL → raw content.

    Parses XML tags of the form <SECTION_NAME>...</SECTION_NAME> whose names
    match ALL_SECTIONS (FORESIGHT_PLAN, FEASIBILITY_CRITERIA, FALLBACK_STRATEGY).
    """
    sections: Dict[str, str] = {}
    for m in re.finditer(r"<([A-Z_]+)>(.*?)</\1>", text, re.S):
        tag = m.group(1)
        sections[tag] = m.group(2).strip()
    return sections


def _extract_bullets(text: str) -> List[str]:
    """
    Convert a block of bullet text into a list of stripped strings.
    Recognises lines starting with -, *, •, or numbered (1. 2.).
    Filters out placeholder values like "N/A" or "None".
    """
    bullets: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip bullet markers
        cleaned = re.sub(r"^[-*•]\s*", "", line)
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned).strip()
        if cleaned and cleaned.upper() not in ("N/A", "NONE", "NONE."):
            bullets.append(cleaned)
    return bullets


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------

def parse_adapter_output(text: str) -> MemoryAdapterOutput:
    """
    Parse raw model output text into a MemoryAdapterOutput.

    Always returns a valid MemoryAdapterOutput — never raises.
    Sets parse_error if the output is empty or sections are missing.
    """
    if not text or not text.strip():
        return MemoryAdapterOutput(
            raw_output=text or "",
            parse_error="Model returned empty output.",
        )

    sections = _split_into_sections(text)

    # Best-effort extraction even if sections dict is empty
    foresight_raw   = sections.get(SECTION_FORESIGHT_PLAN, "")
    feasibility_raw = sections.get(SECTION_FEASIBILITY_CRITERIA, "")
    fallback_raw    = sections.get(SECTION_FALLBACK_STRATEGY, "")

    foresight_plan       = _extract_bullets(foresight_raw) if foresight_raw else []
    feasibility_criteria = _extract_bullets(feasibility_raw) if feasibility_raw else []
    fallback_strategy    = _extract_bullets(fallback_raw) if fallback_raw else []

    # Detect parse problems
    parse_error: Optional[str] = None
    missing = [s for s in ALL_SECTIONS if s not in sections]
    if missing:
        parse_error = f"Missing sections: {', '.join(missing)}"

    return MemoryAdapterOutput(
        foresight_plan=foresight_plan,
        feasibility_criteria=feasibility_criteria,
        fallback_strategy=fallback_strategy,
        raw_output=text,
        parse_error=parse_error,
    )
