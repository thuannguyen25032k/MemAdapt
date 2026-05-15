# Memory Adapter

## Overview

`embodiedbench/memory_adapter/` implements the MemAdapt runtime.  Given retrieved
memories, the current observation, and a task instruction, the adapter reasons about
memory reliability and returns a structured `MemoryAdapterOutput` containing
uncertainty-aware guidance for both the planner and the critic.

## Output Format

The adapter model emits five structured XML sections.  The section-label constants
are defined in `prompts.py` and shared with `parsing.py` to guarantee round-trip
compatibility.  Tags are uppercase; the parser is case-insensitive for robustness.

```xml
<ADAPTED_CONTEXT>
  Uncertainty-aware memory summary. Uncertain entries are prefixed
  with "[POSSIBLY STALE]".
</ADAPTED_CONTEXT>

<FORESIGHT_PLAN>
  - Verify mug is still on table.
  - Navigate to table.
  - Pick up mug.
  - Navigate to shelf.
  - Place mug on shelf.
</FORESIGHT_PLAN>

<FEASIBILITY_CRITERIA>
  - Mug must be reachable (within 1.5 m).
  - Shelf must have space for the mug.
  - No blocking objects between agent and mug.
</FEASIBILITY_CRITERIA>

<STALE_MEMORY_ASSESSMENT>
  - mug.location: possibly stale (last seen 5 steps ago, agent has moved).
  - shelf.occupancy: likely current (seen 1 step ago).
</STALE_MEMORY_ASSESSMENT>

<CONFIDENCE>
  0.72
</CONFIDENCE>
```

The five section labels as Python constants (from `prompts.py`):

| Constant | Tag | Purpose |
|---|---|---|
| `SECTION_ADAPTED_CONTEXT` | `<ADAPTED_CONTEXT>` | Uncertainty-aware memory summary |
| `SECTION_FORESIGHT_PLAN` | `<FORESIGHT_PLAN>` | Memory-grounded step sequence for the planner |
| `SECTION_FEASIBILITY_CRITERIA` | `<FEASIBILITY_CRITERIA>` | Verifiable pass/fail conditions for the critic |
| `SECTION_STALE_ASSESSMENT` | `<STALE_MEMORY_ASSESSMENT>` | Per-entry staleness reasoning |
| `SECTION_CONFIDENCE` | `<CONFIDENCE>` | Self-reported reliability score in [0, 1] |

## MemoryAdapterOutput Fields

| Field | Type | Description |
|---|---|---|
| `adapted_context` | `str` | Uncertainty-aware memory summary text |
| `foresight_plan` | `List[str]` | Bullet steps for the planner |
| `feasibility_criteria` | `List[str]` | Pass/fail conditions for the critic |
| `stale_memory_assessment` | `List[str]` | Per-entry staleness reasoning |
| `planner_context` | `str` | Formatted string ready to inject into planner prompt |
| `critic_context` | `str` | Formatted string ready to inject into critic prompt |
| `confidence` | `float` | Self-reported confidence in [0.0, 1.0] |
| `raw_output` | `str` | Original model output before parsing |
| `parse_error` | `Optional[str]` | Parsing problem description, or `None` |

## Python API

```python
from embodiedbench.memory_adapter import MemoryAdapter, MemoryAdapterInput
from embodiedbench.memory_adapter.config import MemoryAdapterConfig

cfg = MemoryAdapterConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=512,
    temperature=0.1,
    load_in_4bit=True,
)
adapter = MemoryAdapter(cfg)

adapter_input = MemoryAdapterInput(
    task_instruction="Pick up the mug and place it on the shelf.",
    observation_text="I see a table and a shelf. The mug is not visible.",
    memory_context=memory_manager.retrieve("mug shelf"),  # MemoryContext object
    mode="both",   # "planner", "critic", or "both"
)

output = adapter.adapt(adapter_input)

# Inject into planner and critic prompts
print(output.planner_context)
print(output.critic_context)

# Inspect individual components
print(output.adapted_context)
print(output.foresight_plan)           # List[str]
print(output.feasibility_criteria)    # List[str]
print(output.stale_memory_assessment) # List[str]
print(output.confidence)              # float in [0, 1]
print(output.raw_output)              # full raw model output string
```

### Convenience wrappers

```python
# Planner-only adaptation (returns formatted string)
planner_ctx = adapter.adapt_for_planner(
    task_instruction="Pick up the mug.",
    observation_text="I see a table.",
    memory_context=memory_manager.retrieve("mug"),
)

# Critic-only adaptation (returns formatted string)
critic_ctx = adapter.adapt_for_critic(
    task_instruction="Pick up the mug.",
    observation_text="I see a table.",
    memory_context=memory_manager.retrieve("mug"),
    proposed_action="pick mug from table",
)
```

## Configuration Reference

```python
@dataclass
class MemoryAdapterConfig:
    model_name_or_path: str = ""        # HuggingFace model name or local path
    device: str = "auto"
    torch_dtype: str = "auto"           # "auto", "bfloat16", "float16", "float32"
    max_new_tokens: int = 512
    temperature: float = 0.0            # 0.0 = greedy decoding
    top_p: float = 1.0
    do_sample: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False          # 4-bit quantized inference
    trust_remote_code: bool = True
    system_prompt_name: str = "default"
    max_input_chars: int = 6000         # truncate combined input
    max_memory_chars: int = 3500        # truncate memory section
    enabled: bool = True                # False → no-op adapter
```

Instantiate from a dict or OmegaConf config:

```python
cfg = MemoryAdapterConfig.from_mapping({
    "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    "load_in_4bit": True,
})
```

## How the Adapter Reasons About Memory Reliability

The adapter's system prompt instructs it to:

1. **Prioritise the current observation** over any retrieved memory.
2. **Treat stale or conflicting entries as uncertain** — hedge them, do not assert as fact.
3. **Transform** retrieved memories into concise, task-relevant guidance rather than
   copying them verbatim.
4. **Ground feasibility criteria** in what is currently observable, not what was recorded.

Memory entries that have not been observed recently, or that conflict with the current
observation, are explicitly flagged in the adapter's input prompt.  The adapter then
reasons about these flags before producing its `ADAPTED_CONTEXT` and
`STALE_MEMORY_ASSESSMENT` sections.
