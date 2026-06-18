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
           │  (raw memory    │      │  (memory→guidance    │
           │   prefix)       │      │   + XML output)      │
           └─────────────────┘      └──────────┬───────────┘
                                               │
                                  ┌────────────┴──────────────┐
                                  ▼                            ▼
                           VLM Planner                   VLM Critic
                         (FORESIGHT_PLAN              (FEASIBILITY_CRITERIA
                          + FALLBACK_STRATEGY)         from adapter)
```

The memory system is **fully independent** of the simulator and model API. It uses pure Python + NumPy and stores data as JSON/JSONL files.

---

## Four Memory Types

### 1. SpatialMemory (`memory/spatial_memory.py`)

Tracks object locations, states, and spatial relations as a scene graph.

- Detects **stale locations** when an object is moved.
- Detects **conflicts** between relations.
- Key method: `update_from_observation(observation, info, step_id, ...)`
- Tracks object locations under partial observability.

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
> into `FORESIGHT_PLAN`, `FEASIBILITY_CRITERIA`, and `FALLBACK_STRATEGY`
> blocks, which replace the raw memory prompt.  See
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

## Memory Adapter Integration

The **MemoryAdapter** is an optional module that sits between `MemoryManager.retrieve()`
and the VLM planner/critic.  It is **disabled by default**.  For the full API, config
reference, and output format see [memory_adapter.md](memory_adapter.md).

### Lifecycle (per eval-set)

```python
self.memory_adapter = create_memory_adapter_from_config(self.config)
attach_memory_adapter_to_planner(self.planner, self.memory_adapter)
attach_memory_adapter_to_critic(self.dual_critic, self.memory_adapter)
# ... run episodes ...
unload_memory_adapter(self.memory_adapter)   # frees GPU memory at run end
```

If the adapter is absent, disabled, or returns empty output, both components fall back
transparently to the raw `MemoryPromptFormatter` output.

---

## Experiment & Ablation Modes

`setup_memory_experiment(cfg, planner, critic)` is the single entry-point used by every
evaluator.  It reads `cfg.memory_experiment.mode` and wires up components accordingly.

| Mode | MemoryManager | MemoryAdapter | Planner injection | Critic injection |
|---|---|---|---|---|
| `none` | ✗ | ✗ | — | — |
| `raw_planner` | ✓ | ✗ | ✓ | ✗ |
| `raw_planner_critic` | ✓ | ✗ | ✓ | ✓ |
| `adapted_planner` | ✓ | ✓ | ✓ | ✗ |
| `adapted_planner_critic` | ✓ | ✓ | ✓ | ✓ |
| *(key absent)* | ✓ | ✓ | ✓ | ✓ |

*(key absent)* = the `memory_experiment` key is not in the config → backward-compatible
full attach.

```yaml
# embodiedbench/configs/config.yaml
memory_experiment:
  mode: none              # change to run a different ablation
  log_memory_outputs: true
  log_adapter_outputs: true
```

---

## Metrics & Logging

`MemoryExperimentMetrics` accumulates per-episode counters and is written into
`episode_info['memory_metrics']` at episode end.

| Counter | Description |
|---|---|
| `memory_retrieval_calls` | Calls to `MemoryManager.retrieve()` |
| `planner_memory_injections` | Non-empty memory prompts returned to planner |
| `critic_memory_injections` | Non-empty memory prompts returned to critic |
| `adapter_calls` | Total `MemoryAdapter.adapt()` invocations |
| `adapter_fallbacks` | Fallbacks due to empty / exception adapter output |
| `planner_memory_prompt_chars` | Cumulative prompt chars injected to planner |
| `critic_memory_prompt_chars` | Cumulative prompt chars injected to critic |
| `critic_rejections` | Steps where `VLMCritic.evaluate()` returned `valid=False` |
| `replans`, `invalid_actions`, `env_steps`, `task_success`, `task_progress` | From `episode_info` |

`MemoryExperimentLogger` writes per-episode JSON artifacts and a compact
`training_records.jsonl` (SFT-ready rows) under `log_dir`.  Enable via:

```yaml
memory_experiment:
  mode: adapted_planner_critic
  log_memory_outputs: true
  log_adapter_outputs: true
  log_dir: "./memory_logs"
  save_training_records: true
```
