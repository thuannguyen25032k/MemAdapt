# Architecture

## Component Map

```
embodiedbench/
├── memory/                      # Four-type memory subsystem
│   ├── spatial_memory.py        # 3-D scene-graph of object locations
│   ├── temporal_memory.py       # Time-ordered episode event log
│   ├── episodic_memory.py       # Cross-episode task-attempt records
│   ├── semantic_memory.py       # Persistent domain facts and rules
│   ├── trajectory.py            # TrajectoryRecorder (per-episode capture)
│   ├── trajectory_schemas.py    # TrajectoryStep / TrajectoryEpisode
│   └── manager.py               # Unified retrieval interface (MemoryManager)
│
├── memory_adapter/              # Adapter runtime
│   ├── adapter.py               # MemoryAdapter class
│   ├── config.py                # MemoryAdapterConfig
│   ├── prompts.py               # Adapter input prompt builder
│   ├── parsing.py               # Structured XML output parser
│   ├── schemas.py               # MemoryAdapterInput / MemoryAdapterOutput
│   └── utils.py                 # Shared adapter utilities
│
├── memory_adapter_training/     # Stage 1: SFT pipeline
│   ├── trainer.py               # HuggingFace Trainer wrapper
│   ├── config.py                # SFT hyperparameters
│   ├── dataset.py               # Training data loading and collation
│   ├── modeling.py              # Model + LoRA initialisation
│   └── formatting.py            # Prompt/response formatting helpers
│
├── memory_adapter_rl/           # Stage 2: GRPO refinement (planned, not yet implemented)
│   ├── rewards.py               # Composite reward (format + 3-section quality)
│   ├── grpo.py                  # GRPO rollout & advantage normalisation
│   ├── trainer.py               # MemoryAdapterGRPOTrainer (TRL + CPU fallback)
│   ├── checkpoints.py           # Checkpoint save/load
│   ├── formatting.py            # XML prompt/validation helpers
│   ├── config.py                # RL hyperparameters
│   ├── schemas.py               # RewardSignal
│   ├── evaluation.py            # RL evaluation utilities
│   └── utils.py                 # Logging helpers
│
├── evaluation/                  # Benchmark evaluation
│   ├── runner.py                # run_experiment() — top-level evaluation entry point
│   ├── metrics.py               # Success rate, SPL, task progress, ...
│   ├── reporting.py             # JSON / CSV result writers
│   ├── aggregators.py           # Multi-episode result aggregation
│   ├── launcher.py              # Single / batch / suite runners
│   ├── sweeps.py                # Grid and seed sweeps
│   └── utils.py                 # Git hash, JSON helpers
```

## Information Flow

```
Episode (observation, task, step) 
    │
    ▼
Memory Manager ──(retrieve)──► Memory Bundle
                                    │
                                    ▼
                          ┌──────────────────┐
   Task instruction ──────►│  MemoryAdapter   │
   Retrieved memory ──────►│  (Qwen3-14B      │
                           │   + LoRA)        │
                           └────────┬─────────┘
                                    │ XML output
                       ┌────────────┴────────────┐
                       ▼                         ▼
             foresight_plan                feasibility_criteria
             + fallback_strategy                 │
                       │                         │
                       ▼                         ▼
                 ┌───────────┐           ┌──────────────┐
                 │VLM Planner│           │  VLM Critic  │
                 │ (frozen)  │           │   (frozen)   │
                 └─────┬─────┘           └──────┬───────┘
                       └──────────┬─────────────┘
                                  ▼
                               Action
```

## Reward Components (GRPO)

> **Planned, not yet implemented.** The GRPO refinement stage is a design specification
> only. See [grpo_training.md](grpo_training.md) for the full reward formula, component
> definitions, default weights, and the GRPO config YAML.
