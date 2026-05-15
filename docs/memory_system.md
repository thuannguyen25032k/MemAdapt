# MemAdapt Memory System

This document describes the memory architecture introduced in MemAdapt, covering the four memory modules, the lifecycle, planner injection, evaluator integration, and configuration.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MemoryManager                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────┐ │
│  │SpatialMemory │  │TemporalMemory│  │EpisodicMemory│  │Sema-│ │
│  │(object locs) │  │(step history)│  │(past episodes│  │ntic │ │
│  │stale detect  │  │sliding window│  │cross-episode)│  │Mem  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──┬──┘ │
│         └─────────────────┴──────────────────┴─────────────┘   │
│                           retrieve() → MemoryContext            │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     MemoryPromptFormatter    │
              │  format_for_planner()        │
              │  format_for_critic()         │
              │  format_compact()            │
              └──────────────┬──────────────┘
                             │
             (adapter disabled)           (adapter enabled)
                    │                            │
                    ▼                            ▼
           ┌─────────────────┐      ┌──────────────────────┐
           │   VLM Planner   │      │    Memory Adapter    │
           │  (raw memory    │      │  (staleness reasoning│
           │   prefix)       │      │   + XML output)      │
           └─────────────────┘      └──────────┬───────────┘
                                               │
                                  ┌────────────┴──────────────┐
                                  ▼                            ▼
                           VLM Planner                   VLM Critic
                         (ADAPTED_CONTEXT             (FEASIBILITY_CRITERIA
                          + FORESIGHT_PLAN)            from adapter)
```

The memory system is **fully independent** of the simulator and model API. It uses pure Python + NumPy and stores data as JSON/JSONL files.

---

## Four Memory Types

### 1. SpatialMemory (`memory/spatial_memory.py`)

Tracks object locations, states, and spatial relations as a scene graph.

- Detects **stale locations** when an object is moved.
- Detects **conflicts** between relations.
- Key method: `update_from_observation(observation, info, step_id, ...)`
- Research focus: stale-memory reasoning under partial observability.

### 2. TemporalMemory (`memory/temporal_memory.py`)

Episode-scoped sliding-window of recent steps.

- Stores action, feedback, success/failure per step.
- Detects **repeated failures** on the same action.
- Compressed FIFO when `max_steps` is exceeded.
- Reset at episode start; not persisted across episodes (by design).

### 3. EpisodicMemory (`memory/episodic_memory.py`)

Cross-episode long-term store of summarised task attempts.

- Each episode becomes an `EpisodeRecord` with instruction, status, trajectory summary, key actions, failure reasons, success strategy.
- Deduplication via exact-match + similarity threshold.
- Retrieval scoring: task similarity, status, scene, objects, recency.

### 4. SemanticMemory (`memory/semantic_memory.py`)

Persistent domain facts, rules, and preconditions.

- Pre-seeded with 6 default facts (object visibility, manipulator reachability, etc.).
- `extract_facts_from_episode()` mines new facts from completed episodes.
- Deduplication via similarity threshold.
- Retrieval scoring: relevance, confidence, category, recency.

---

## Update / Retrieve / Finalize Lifecycle

```
Episode start
  └── planner.reset()
        └── memory_manager.reset_episode()        ← clears TemporalMemory only

Per step
  └── planner.update_info(info)
        └── memory_manager.update(...)             ← TemporalMemory + SpatialMemory

Per step (planner act)
  └── planner.act(obs, instruction)
        └── memory_manager.retrieve(query)         ← all four modules
        └── MemoryPromptFormatter.format_for_planner(ctx)
        └── memory_prompt prepended to planner prompt

Episode end
  └── memory_manager.finalize_episode(...)         ← EpisodicMemory + SemanticMemory
  └── memory_manager.save()                        ← if save_on_episode_end=True
```

---

## Planner Injection Point

In `VLMPlanner.act()`:

```python
memory_prompt = self._get_planner_memory_prompt(user_instruction, obs=obs)
if memory_prompt:
    prompt = memory_prompt + "\n\n" + prompt
```

The memory block is a **plain-text prefix** — no JSON examples, no code blocks. It is clearly marked:

```
[Retrieved Memory for Planning]
Retrieved memory is helpful but may be outdated. ...
Do not copy memory text into the final action output.

[Spatial Memory]
- Apple was last seen on the kitchen table at step 0, confidence 0.95.
...
```

> **Note — Memory Adapter integration:** The above describes the *direct* raw-memory
> injection path, active when `memory_adapter.enabled = false` (the default for
> evaluation without a trained adapter).  When the Memory Adapter is enabled
> (`memory_adapter.enabled = true`), retrieved memories are first passed through the
> `MemoryAdapter` before reaching the planner or critic.  The adapter transforms them
> into uncertainty-aware `ADAPTED_CONTEXT`, `FORESIGHT_PLAN`, and
> `FEASIBILITY_CRITERIA` blocks, which replace the raw memory prompt.  See
> [memory_adapter.md](memory_adapter.md) for details.

---

## Evaluator Lifecycle

Memory is wired in each of the four evaluators via `memory/integration.py`:

| Evaluator | Planner | Memory injection |
|---|---|---|
| EB-ALFRED | VLMPlanner | Full (per-step retrieval + finalize) |
| EB-Habitat | VLMPlanner | Full (per-step retrieval + finalize) |
| EB-Navigation | EBNavigationPlanner | finalize only (no `set_memory_manager`) |
| EB-Manipulation | ManipPlanner | finalize only (no `set_memory_manager`) |

Helper functions in `memory/integration.py`:

```python
create_memory_manager_from_config(cfg)       # create or return None
attach_memory_to_planner(planner, mm)        # safe no-op if planner unsupported
finalize_memory_episode(mm, planner, ...)    # episode-end hook
save_memory_if_configured(mm, cfg, ...)      # conditional save
compute_final_status(info)                   # "success" / "partial" / "failure"
```

---

## How to Enable Memory

Add to your experiment config (e.g. `eb-alf.yaml` or via command-line override):

```yaml
memory:
  enabled: true
  spatial_enabled: true
  temporal_enabled: true
  episodic_enabled: true
  semantic_enabled: true
  storage_dir: "./memory_store"
  top_k_per_memory: 5
  temporal_max_steps: 20
  use_embeddings: true
  allow_stale_warnings: true
  max_context_chars: 4000
  max_section_chars: 1200
  auto_save: false
  load_on_start: true
  save_on_episode_end: true
  save_on_end: true
```

The default in `config.yaml` is `memory: null`, which disables all memory — **existing behavior is fully preserved**.

To load saved memory across evaluation runs, set `load_on_start: true` with the same `storage_dir`.

---

## What Files Are Saved

Under `storage_dir/` (default `./memory_store/`):

| File | Contents |
|---|---|
| `spatial_memory.json` | Scene graph (nodes, relations, stale flags) |
| `temporal_memory.json` | Current episode steps + compressed summaries |
| `episodic_memory.json` | All past episode records |
| `semantic_memory.json` | All semantic facts |

All files are human-readable JSON, versioned with timestamps.

---

## Running the Demo

```bash
python examples/demo_memory_system.py
```

No simulator, model API, or external service required. The demo:
1. Creates a MemoryManager with a temporary storage directory.
2. Adds spatial memory (apple moves from table → fridge, triggering stale warning).
3. Adds temporal steps (failed find, successful open/pick).
4. Finalizes an episode (success).
5. Retrieves memory for a follow-up query.
6. Prints the planner and critic prompts, plus memory stats.
7. Saves all memory files.

---

## Known Limitations

1. **Navigation / Manipulation planners do not yet receive per-step memory injection** — `EBNavigationPlanner` and `ManipPlanner` do not inherit `VLMPlanner` and do not have `set_memory_manager()`. They benefit from `finalize_episode` (cross-episode learning) but not from per-step retrieval during planning.

2. **Current retrieval uses simple scoring / hash embeddings** — `HashEmbeddingProvider` uses a deterministic token-hash method. For production quality, swap in a real sentence-embedding model via a custom `EmbeddingProvider` subclass and set `use_embeddings: true`.

3. **Stale memory detection is heuristic** — Spatial stale marking uses location-change detection; semantic conflicts require manual or LLM-assisted verification.

4. **No memory pruning policy** — Long evaluation runs may accumulate many episodic/semantic entries. A future pruning or consolidation step is recommended for runs > 100 episodes.

---

## Memory Adapter (MemoryAdapter)

The **MemoryAdapter** is an optional LLM-backed module that transforms raw `MemoryContext` into structured guidance for the planner and critic.  It is **disabled by default** and loaded independently from the planner/critic models.

### How It Works

```
MemoryContext
     │
     ▼
MemoryAdapter.adapt(MemoryAdapterInput)
     │  (Hugging Face CausalLM generates structured output)
     ▼
MemoryAdapterOutput
   ├── planner_context  → injected into VLMPlanner prompt
   ├── critic_context   → injected into VLMCritic prompt
   ├── foresight_plan
   ├── feasibility_criteria
   ├── stale_memory_assessment
   └── confidence
```

### Enabling via `config.yaml`

```yaml
memory_adapter:
  enabled: true
  model_name_or_path: "Qwen/Qwen2.5-1.5B-Instruct"   # any HF causal LM
  device: auto
  torch_dtype: auto
  max_new_tokens: 512
  temperature: 0.0
  do_sample: false
  load_in_4bit: false          # set true to reduce VRAM
  trust_remote_code: true
  system_prompt_name: default  # or "planner_only" / "critic_only"
  max_input_chars: 6000
  max_memory_chars: 3500
```

### Hydra CLI Override

```bash
python -m embodiedbench.main \
  env=alfred \
  model_name=gpt-4o \
  memory.enabled=true \
  memory_adapter.enabled=true \
  memory_adapter.model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct \
  memory_adapter.load_in_4bit=true
```

### Lifecycle

The adapter is created **once per eval-set** (not per episode) and shared by the planner and critic:

```python
self.memory_adapter = create_memory_adapter_from_config(self.config)
attach_memory_adapter_to_planner(self.planner, self.memory_adapter)
attach_memory_adapter_to_critic(self.dual_critic, self.memory_adapter)
# ... run episodes ...
unload_memory_adapter(self.memory_adapter)   # frees GPU memory at run end
```

### Fallback Behaviour

If the adapter is absent, disabled, raises an exception, or returns empty/unsafe output, both the planner and critic transparently fall back to the raw `MemoryPromptFormatter` output.  **No behavior change unless the adapter is explicitly enabled.**

### Running the Demo

```bash
# Requires: pip install transformers torch
python examples/demo_memory_adapter.py

# Custom model:
python examples/demo_memory_adapter.py --model Qwen/Qwen2.5-1.5B-Instruct --mode planner
```

### Smoke Test (optional, downloads ~5 MB)

```bash
RUN_HF_ADAPTER_SMOKE=1 pytest tests/memory_adapter/test_memory_adapter_smoke.py -v
```

This verifies the full load→adapt→unload cycle with `sshleifer/tiny-gpt2`.  It checks **wiring only**, not output quality — the tiny model will produce low-quality/empty structured output.

```bash
# Custom model for smoke test:
ADAPTER_SMOKE_MODEL=gpt2 RUN_HF_ADAPTER_SMOKE=1 pytest tests/memory_adapter/test_memory_adapter_smoke.py -v
```

---

## Experiment & Ablation Modes

`setup_memory_experiment(cfg, planner, critic)` is the single entry-point used by every evaluator.  It reads `cfg.memory_experiment.mode` and wires up components accordingly.

### Modes

| Mode | MemoryManager | MemoryAdapter | Planner | Critic |
|---|---|---|---|---|
| `none` | ✗ | ✗ | — | — |
| `raw_planner` | ✓ | ✗ | ✓ | ✗ |
| `raw_planner_critic` | ✓ | ✗ | ✓ | ✓ |
| `adapted_planner` | ✓ | ✓ | ✓ | ✗ |
| `adapted_planner_critic` | ✓ | ✓ | ✓ | ✓ |
| *(absent)* | ✓ | ✓ | ✓ | ✓ |

*(absent)* = the `memory_experiment` key is not in the config → backward-compatible full attach.

### Config

```yaml
# embodiedbench/configs/config.yaml
memory_experiment:
  mode: none              # change to run a different ablation
  log_memory_outputs: true
  log_adapter_outputs: true
```

### CLI Override

```bash
python -m embodiedbench.main env=alfred model_name=gpt-4o \
  memory.enabled=true \
  memory_experiment.mode=adapted_planner_critic \
  memory_adapter.enabled=true \
  memory_adapter.model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct
```

### Test

```bash
pytest tests/memory_adapter/test_memory_experiment_modes.py -v
```

---

## Metrics & Logging (Step 21)

`MemoryExperimentMetrics` is a lightweight dataclass that accumulates per-episode counters for ablation analysis. It is injected into the planner and critic via `set_metrics()` and written into `episode_info['memory_metrics']` at the end of each episode.

### Counters

| Counter | Where incremented |
|---|---|
| `memory_retrieval_calls` | After `MemoryManager.retrieve()` in planner and critic |
| `planner_memory_injections` | When a non-empty memory prompt is returned to the planner |
| `critic_memory_injections` | When a non-empty memory prompt is returned to the critic |
| `adapter_planner_calls` / `adapter_critic_calls` | When `MemoryAdapter.adapt()` is entered |
| `adapter_calls` | Sum of both adapter paths |
| `adapter_fallbacks` | On code-fence / empty / exception in adapter output |
| `stale_warning_count` | `len(ctx.stale_memory_warnings)` after retrieve |
| `planning_hint_count` | `len(ctx.planning_hints)` after retrieve |
| `feasibility_constraint_count` | `len(ctx.feasibility_constraints)` after retrieve |
| `planner_memory_prompt_chars` | `len(prompt)` for each planner injection |
| `critic_memory_prompt_chars` | `len(prompt)` for each critic injection |
| `adapted_planner_prompt_chars` | When adapter provides non-empty planner context |
| `adapted_critic_prompt_chars` | When adapter provides non-empty critic context |
| `critic_rejections` | When `VLMCritic.evaluate()` returns `valid=False` |
| `replans`, `invalid_actions`, `env_steps`, `task_success`, `task_progress` | Copied from `episode_info` by `collect_episode_metrics()` |

### Usage

```python
from embodiedbench.memory.integration import (
    create_metrics_from_config,
    attach_metrics_to_planner,
    attach_metrics_to_critic,
    collect_episode_metrics,
)

# Once per eval-set:
self.metrics = create_metrics_from_config(self.config)
attach_metrics_to_planner(self.planner, self.metrics)
attach_metrics_to_critic(self.dual_critic, self.metrics)

# At episode start:
self.metrics.reset_episode()

# At episode end:
collect_episode_metrics(self.metrics, episode_info)
episode_info['memory_metrics'] = self.metrics.to_dict()
```

### Test

```bash
pytest tests/memory_adapter/test_memory_metrics.py -v
```

---

## Structured Logging & Training Data Collection (Step 22)

`MemoryExperimentLogger` writes per-episode artifacts capturing everything the adapter sees and produces — enabling offline analysis and future SFT/RL training.

### Output files

| File | Description |
|---|---|
| `<log_dir>/episodes/<episode_id>.json` | Full episode record (all prompts, adapter output, trajectory, metrics) |
| `<log_dir>/training_records.jsonl` | Compact SFT-ready row per episode (one JSON object per line) |

### Training row schema

```json
{
  "instruction":          "pick up the mug and put it in the drawer",
  "observation_or_state": "<raw MemoryContext text>",
  "retrieved_memory":     "<planner memory prompt>",
  "adapter_target": {
    "adapted_context":         "<adapted planner context>",
    "foresight_plan":          ["step 1: ...", "step 2: ..."],
    "feasibility_criteria":    ["object must be reachable"],
    "stale_memory_assessment": ["warning: stale location for mug"]
  },
  "outcome": {
    "success":  true,
    "progress": 1.0,
    "steps":    12,
    "replans":  2
  }
}
```

### Config

```yaml
memory_experiment:
  mode: adapted_planner_critic
  log_memory_outputs: true
  log_adapter_outputs: true
  log_dir: "./memory_logs"        # root directory for all logs
  save_training_records: true     # write training_records.jsonl
```

`log_dir` defaults to `"./memory_logs"` when not specified.

### Evaluator integration

`MemoryExperimentLogger` is created once per eval-set and used at the end of each episode:

```python
# Once per eval-set:
self.mem_logger = create_logger_from_config(self.config)

# At episode end:
if self.mem_logger is not None and self.mem_logger.enabled:
    ep_record = MemoryExperimentLogger.build_episode_log(
        episode_id=...,
        env_name="alfred",
        task_instruction=user_instruction,
        mode=...,
        planner=self.planner,
        critic=dual_critic,
        episode_info=episode_info,
        metrics=self.metrics,
        metadata={...},
    )
    self.mem_logger.log_episode(ep_record)
    self.mem_logger.append_training_record(ep_record)
```

`build_episode_log` degrades gracefully: every field falls back to `""` / `[]` / `None` when the corresponding planner/critic attribute is absent.

### MemoryEpisodeLog fields

| Field | Source |
|---|---|
| `planner_memory_prompt` | `planner.last_memory_prompt` |
| `adapted_planner_context` | `planner.last_adapted_memory_prompt` |
| `raw_memory_context` | `planner.last_memory_context.to_text()` |
| `foresight_plan` | `planner.last_adapted_memory_output.foresight_plan` |
| `feasibility_criteria` | `planner.last_adapted_memory_output.feasibility_criteria` |
| `stale_memory_assessment` | `planner.last_adapted_memory_output.stale_memory_assessment` |
| `critic_memory_prompt` | `critic.vlm.last_adapted_memory_prompt` |
| `critic_events` | `critic._episode_critic_records` |
| `planner_actions` | `planner.episode_act_feedback` (decoded) |

### Test

```bash
pytest tests/memory_adapter/test_memory_logging.py -v
```
