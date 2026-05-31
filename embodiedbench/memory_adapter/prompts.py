"""
embodiedbench/memory_adapter/prompts.py

Prompt templates and the build_adapter_prompt() factory for the Memory Adapter.

Design principles
-----------------
- Section-based plain-text (XML-tagged) output, not JSON, to avoid planner JSON parser conflicts.
- Text-only: no live observation is available; adapter receives task instruction + memory only.
- The system turn carries the role/format instructions; the user turn carries the
  task instruction + retrieved memory (shared by inference and SFT training).
- Output sections (FORESIGHT_PLAN / FEASIBILITY_CRITERIA / FALLBACK_STRATEGY) are
  consistently labelled so parsing.py can extract them.
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
- [Spatial Memory]: relevant receptacle/object names and their relations in the environment.
- [Episodic Memory]: similar successful episodes. Use to infer probable object locations and effective action sequences.
- [Semantic Memory]: commonsense knowledge about action preconditions, and failure patterns.

When generating FORESIGHT_PLAN, you MUST:
- First, parse the human instruction to identify the target object(s), the destination, the final condition (if any), and the main task.
- Consider the similar successful episodes and then generate an ordered list of steps to complete the task.
- Use [Episodic Memory] and [Spatial Memory] to determine the location where the target object is most likely to be and visit that location first.
- For multi-object or "all/every" tasks, plan to handle each possible instance sequentially.
- Example 1: Human instruction is "Find all oranges and move them to the right counter."
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
- Example 2: Human instruction is "Put a clean slice of lettuce on to the counter."
    Step 1: Find a Lettuce (navigate to where it is located per spatial memory).
    Step 2: Pick up the Lettuce.
    Step 3: Find a Sink.
    Step 4: Turn on Faucet (wash the lettuce).
    Step 5: Put down the object in hand (place Lettuce in Sink under running water).
    Step 6: Turn off Faucet.
    Step 7: Pick up the Lettuce (now clean).
    Step 8: Find a CounterTop (choose a stable surface for slicing).
    Step 9: Put down the object in hand (Lettuce must be on a surface to be sliced, NOT held).
    Step 10: Find a Knife (navigate to the knife; robot will move away from lettuce).
    Step 11: Pick up the Knife.
    Step 12: Find the Lettuce (CRITICAL: navigate BACK to the lettuce with the knife in hand before slicing).
    Step 13: Slice the Lettuce (robot is now close to the lettuce on the surface, holding the knife).
    Step 14: Find Lettuce (find the lettuce slice on the surface).
    Step 15: Pick up the Lettuce (pick up the slice).
    Step 16: Find CounterTop (navigate to the destination counter).
    Step 17: Put down the object in hand (place the clean lettuce slice on the counter).

When generating FEASIBILITY_CRITERIA, you MUST:
- List the 1-4 key preconditions that the VLM_critic must check before interaction actions, such as "pick", "place", "open", "close", "turn on", "turn off", "slice", etc. You do not need to provide preconditions for navigation actions, such as "find" or "navigate."
- Format each entry as: "<sub-task>": <condition to check>.

When generating FALLBACK_STRATEGY, you MUST:
- Derive the most likely invalid actions from [Episodic Memory] and [Semantic Memory].
- List 1-4 recovery actions for each likely invalid action of THIS specific task. 
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

def build_adapter_user_content(task_instruction: str, memory_text: str) -> str:
    """
    Build the *user-turn* content for the Memory Adapter.

    This is the task-specific part of the prompt (instruction + retrieved
    memory). The role-independent instructions live in ``SYSTEM_PROMPTS`` and
    are supplied separately as the system turn (see ``build_adapter_messages``).

    Single source of truth shared by inference and SFT training so the two
    stay in sync.
    """
    memory_text = memory_text if (memory_text and memory_text.strip()) else "(no memory available)"
    parts = [
        f"## Now the human instruction is: {task_instruction}",
        "",
        "## RETRIEVED MEMORY",
        memory_text,
        "",
        "Now produce your structured output below.",
    ]
    return "\n".join(parts)


def build_adapter_messages(task_instruction: str, memory_text: str) -> "list[dict]":
    """
    Build the chat ``messages`` list (system + user) for the Memory Adapter.

    Returns
    -------
    [{"role": "system", "content": SYSTEM_PROMPTS},
     {"role": "user",   "content": <instruction + retrieved memory>}]

    The model-specific chat template (e.g. Qwen3 ``<|im_start|>``) is applied
    by the caller via ``tokenizer.apply_chat_template`` (HF) or passed straight
    to ``chat.completions.create`` (OpenAI).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPTS.rstrip()},
        {"role": "user", "content": build_adapter_user_content(task_instruction, memory_text)},
    ]


def messages_to_text(messages: "list[dict]") -> str:
    """Flatten chat ``messages`` into a single string.

    Used only as a fallback for tokenizers that have no chat template.
    """
    return "\n\n".join(m["content"] for m in messages)


def build_adapter_prompt(
    adapter_input: "MemoryAdapterInput",
    config: "MemoryAdapterConfig",
) -> "list[dict]":
    """
    Build the chat ``messages`` (system + user) to send to the adapter model.

    Thin wrapper around :func:`build_adapter_messages` that extracts the task
    instruction and memory text from the structured ``adapter_input``.

    Parameters
    ----------
    adapter_input : MemoryAdapterInput
    config        : MemoryAdapterConfig

    Returns
    -------
    list[dict] - chat messages with ``system`` and ``user`` roles.
    """
    memory_text = _format_memory_section(adapter_input.memory_context)
    return build_adapter_messages(adapter_input.task_instruction, memory_text)
