# Memory Adapter

## Overview

`embodiedbench/memory_adapter/` implements the MemAdapt runtime.  Given a task
instruction and retrieved memories (spatial, episodic, and semantic), the adapter
returns a structured `MemoryAdapterOutput` containing a foresight plan and fallback
strategy for the planner, plus feasibility criteria for the critic.

## Output Format

The adapter model emits three structured XML sections.  The section-label constants
are defined in `prompts.py` and shared with `parsing.py` to guarantee round-trip
compatibility.  Tags are uppercase; the parser is case-insensitive for robustness.

```xml
<FORESIGHT_PLAN>
  - Navigate to the table.
  - Pick up the mug.
  - Navigate to the shelf.
  - Place the mug on the shelf.
</FORESIGHT_PLAN>

<FEASIBILITY_CRITERIA>
  - "pick mug": mug must be reachable (within ~1.5 m) and visible.
  - "place mug": shelf must have free space for the mug.
</FEASIBILITY_CRITERIA>

<FALLBACK_STRATEGY>
  - If "cannot pick / not near": navigate to table 1, then retry pick.
    If still failing, navigate to the left counter, then retry pick.
</FALLBACK_STRATEGY>
```

The three section labels as Python constants (from `prompts.py`):

| Constant | Tag | Purpose |
|---|---|---|
| `SECTION_FORESIGHT_PLAN` | `<FORESIGHT_PLAN>` | Memory-grounded ordered step sequence for the planner |
| `SECTION_FEASIBILITY_CRITERIA` | `<FEASIBILITY_CRITERIA>` | Per-action preconditions for the critic to verify |
| `SECTION_FALLBACK_STRATEGY` | `<FALLBACK_STRATEGY>` | Per-failure-type recovery rules for the planner |

## MemoryAdapterOutput Fields

| Field | Type | Description |
|---|---|---|
| `foresight_plan` | `List[str]` | Ordered plan-step bullets for the planner |
| `feasibility_criteria` | `List[str]` | Per-action precondition bullets for the critic |
| `fallback_strategy` | `List[str]` | Per-failure recovery rules for the planner |
| `raw_output` | `str` | Original model output before parsing |
| `parse_error` | `Optional[str]` | Parsing problem description, or `None` |

Use `output.is_empty()` to check whether any substantive content was produced.


## Python API

```python
from embodiedbench.memory_adapter import MemoryAdapter, MemoryAdapterInput
from embodiedbench.memory_adapter.config import MemoryAdapterConfig

cfg = MemoryAdapterConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=2048,
    temperature=0.0,
    load_in_4bit=True,
)
adapter = MemoryAdapter(cfg)

adapter_input = MemoryAdapterInput(
    task_instruction="Pick up the mug and place it on the shelf.",
    memory_context=memory_manager.retrieve("mug shelf"),  # MemoryContext object
)

output = adapter.adapt(adapter_input)

# Inspect the parsed components (all List[str])
print(output.foresight_plan)
print(output.feasibility_criteria)
print(output.fallback_strategy)
print(output.raw_output)              # full raw model output string
print(output.parse_error)             # None when parsing succeeded
```

### Building planner / critic injection strings

`build_planner_context` and `build_critic_context` are module-level helpers that
format a `MemoryAdapterOutput` into the strings injected into the planner and critic
prompts.  Each returns an empty string when there is no substantive content, so
callers can skip injection entirely.

```python
from embodiedbench.memory_adapter.adapter import (
    build_planner_context,
    build_critic_context,
)

planner_ctx = build_planner_context(output)   # FORESIGHT_PLAN + FALLBACK_STRATEGY
critic_ctx = build_critic_context(output)      # FEASIBILITY_CRITERIA

# The most recent adapt() result is also cached on the adapter for reuse:
adapter.last_output  # -> MemoryAdapterOutput
```


## Configuration Reference

```python
@dataclass
class MemoryAdapterConfig:
    model_name_or_path: str = ""        # HF model name or local path (local backend)
    device: str = "auto"
    torch_dtype: str = "auto"           # "auto", "bfloat16", "float16", "float32"
    max_new_tokens: int = 2048
    temperature: float = 0.0            # 0.0 = greedy decoding
    top_p: float = 1.0
    do_sample: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False          # 4-bit quantized inference
    trust_remote_code: bool = True
    enabled: bool = True                # False → no-op adapter
    enable_thinking: bool = False       # Qwen3 <think> mode; must match training
    # API backend — when api_model is set, no local model is loaded and all
    # generation is forwarded to an OpenAI-compatible server (OpenAI/lmdeploy/vLLM).
    api_model: str = ""                 # e.g. "gpt-4o" or "qwen3-14b-adapter"
    api_key: str = ""                   # falls back to OPENAI_API_KEY; "EMPTY" for local servers
    api_base_url: str = ""              # e.g. "http://localhost:8000/v1"; empty = official OpenAI
```

Instantiate from a dict or OmegaConf config:

```python
cfg = MemoryAdapterConfig.from_mapping({
    "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    "load_in_4bit": True,
})
```

## Backends

The adapter supports two mutually exclusive backends:

- **Local HuggingFace** *(default)* — set `model_name_or_path`; the model is loaded
  onto the GPU and `generate()` runs inference locally.
- **API** — set `api_model` (and optionally `api_key` / `api_base_url`); no model is
  loaded locally and every call is forwarded to an OpenAI-compatible server. This is
  the recommended path for a fine-tuned adapter served via lmdeploy or vLLM, since it
  decouples adapter inference from the planner/critic process.

```yaml
# config.yaml — serve a fine-tuned adapter via lmdeploy and call it over the API
memory_adapter:
  enabled: true
  model_name_or_path: ""               # leave empty — model is not loaded locally
  api_model: "qwen3-14b-adapter"
  api_key: "EMPTY"
  api_base_url: "http://localhost:8000/v1"
```

See the **Deployment** section of the top-level `README.md` for the matching
`lmdeploy serve api_server` command.


## How the Adapter Transforms Memory

The adapter's system prompt (`SYSTEM_PROMPTS` in `prompts.py`) instructs it to draw on
three memory sources — **Spatial**, **Episodic**, and **Semantic** — and to:

1. **Parse the instruction** to identify the target object(s), destination, and final
   condition before planning.
2. **Build a `FORESIGHT_PLAN`** — an ordered, memory-grounded step sequence, visiting the
   most probable object locations first and handling multi-instance ("all/every") tasks
   sequentially.
3. **Derive `FEASIBILITY_CRITERIA`** — 1–4 key preconditions per interaction action
   (`pick`, `place`, `open`, `slice`, …) for the critic to verify.
4. **Derive a `FALLBACK_STRATEGY`** — concrete recovery actions for the most likely
   failures, using receptacle/object names taken from Spatial Memory (no vague
   references like "nearest receptacle").

The planner consumes `FORESIGHT_PLAN` + `FALLBACK_STRATEGY` (via `build_planner_context`),
while the critic consumes `FEASIBILITY_CRITERIA` (via `build_critic_context`).

