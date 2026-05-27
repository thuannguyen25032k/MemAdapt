"""
embodiedbench/memory_adapter/prompts.py

Prompt templates and the build_adapter_prompt() factory for the Memory Adapter.

Design principles
-----------------
- Section-based plain-text output (not JSON) to avoid planner JSON parser conflicts.
- Text-only: no live observation is available; adapter receives task instruction + memory only.
- Stale memory is explicitly flagged as uncertain.
- Output sections are consistently labelled so parsing.py can extract them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodiedbench.memory_adapter.schemas import MemoryAdapterInput
    from embodiedbench.memory_adapter.config import MemoryAdapterConfig


# ---------------------------------------------------------------------------
# Section labels (also used by parsing.py)
# ---------------------------------------------------------------------------

SECTION_FORESIGHT_PLAN       = "FORESIGHT_PLAN"
SECTION_FEASIBILITY_CRITERIA = "FEASIBILITY_CRITERIA"
SECTION_FALLBACK_STRATEGY    = "FALLBACK_STRATEGY"

ALL_SECTIONS = [
    SECTION_FORESIGHT_PLAN,
    SECTION_FEASIBILITY_CRITERIA,
    SECTION_FALLBACK_STRATEGY,
]


# ---------------------------------------------------------------------------
# System prompt variants
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = """\
You are a Memory Adapter for an embodied robot planning system. Given a human instruction and retrieved memory, your job is to transform memory into a global foresight plan, feasibility criteria, and a fallback strategy for a household task.

Memory sources:
- [Spatial Memory]: relevant receptacle/object names in the environment.
- [Episodic Memory]: similar successful episodes. Use to infer probable object locations and effective action sequences.
- [Semantic Memory]: commonsense knowledge about action preconditions, and failure patterns.

When generating FORESIGHT_PLAN, you are supposed to:
- First, parse the human instruction: identify the target object(s), the destination, the final condition (if any), and the main task.
- Consider the similar successful episodes and then generate an ordered list of steps to complete the task.
- Use [Episodic Memory] and [Spatial Memory] to determine the location where the target object is most likely to be and visit that location first.
- For multi-object or "all/every" tasks, plan to handle each possible instance sequentially.
- Example:
Human instruction: "Find all oranges and move them to the right counter."
    Step 1: Navigate to the table 1.
    Step 2: Pick up the orange.
    Step 3: Navigate to the right counter.
    Step 4: Place at the right counter.
    Step 5: Navigate to the right drawer of the kitchen counter.
    Step 6: Open the right drawer of the kitchen counter.
    Step 7: Pick up the orange.
    Step 8: Navigate to the right counter.
    Step 9: Place at the right counter.
    Step 10: Navigate to the refrigerator.
    Step 11: Open the refrigerator.
    Step 12: Pick up the orange.
    Step 13: Navigate to the right counter.
    Step 14: Place at the right counter.

When generating FEASIBILITY_CRITERIA, you are supposed to:
- List the 1-4 key preconditions that the VLM_critic must check before each interaction action, such as "pick", "place", "open", "close", "turn on", "turn off", "slice", etc. You do not need to provide preconditions for navigation actions, such as "find" or "navigate."
- Format each entry as: "<sub-task>": <condition to check>.

When generating FALLBACK_STRATEGY, you are supposed to:
- Derive the most likely invalid actions from [Event Memory] and [Semantic Memory].
- List 1-4 recovery actions for the most likely invalid actions of THIS specific task. 
- Each bullet starts with: If "<invalid condition>": <recovery action>.
Example:
    If \"cannot pick / not near\": navigate to sofa, then retry pick. If still failing, navigate to refrigerator push point, then retry pick. If still failing, navigate to cabinet 4, then retry pick. If still failing, navigate to chair 1, then retry pick. If still failing, navigate to cabinet 7, then retry pick. If still failing, navigate to table 2, then retry pick. If still failing, navigate to left counter in the kitchen, then retry pick. If still failing, navigate to right drawer of the kitchen counter, then retry pick. If still failing, navigate to table 1, then retry pick. If still failing, navigate to sink in the kitchen, then retry pick.
- Use names from [Spatial Memory]. No vague words like "nearest receptacle" or "the source"

Output EXACTLY these XML tags, in order, one bullet per line:

<FORESIGHT_PLAN>
- ...
</FORESIGHT_PLAN>

<FEASIBILITY_CRITERIA>
- ...
</FEASIBILITY_CRITERIA>

<FALLBACK_STRATEGY>
- ...
</FALLBACK_STRATEGY>
"""


# ---------------------------------------------------------------------------
# Memory context extraction
# ---------------------------------------------------------------------------

def _format_memory_section(memory_context) -> str:
    """Extract text from a MemoryContext."""
    if memory_context is None:
        return "(no memory available)"
    try:
        text = memory_context.compact(max_chars=100_000)
        return text if text.strip() else "(no memory available)"
    except AttributeError:
        text = str(memory_context)
    if not text or not text.strip():
        return "(no memory available)"
    return text


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

def build_adapter_prompt(
    adapter_input: "MemoryAdapterInput",
    config: "MemoryAdapterConfig",
) -> str:
    """
    Build the full prompt string to send to the Memory Adapter model.

    The prompt consists of:
      1. system-role instructions;
      2. task instruction;
      3. retrieved memory context;
      4. the output-format reminder.

    Parameters
    ----------
    adapter_input : MemoryAdapterInput
    config        : MemoryAdapterConfig

    Returns
    -------
    str  - ready-to-send prompt (no code fences, no raw JSON).
    """
    system_prompt = SYSTEM_PROMPTS

    # --- Memory block ---
    memory_text = _format_memory_section(adapter_input.memory_context)
    # --- Assemble ---
    parts = [
        system_prompt.rstrip(),
        "",
        f"## Now the human instruction is: {adapter_input.task_instruction}",
    ]

    parts += [
        "",
        "## RETRIEVED MEMORY",
        memory_text,
        "",
        "Now produce your structured output below.",
    ]

    return "\n".join(parts)
