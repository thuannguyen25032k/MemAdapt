# Architecture

## Component Map

```
embodiedbench/
├── memory/                      # Four-type memory subsystem
│   ├── spatial_memory.py        # 3-D scene-graph of object locations
│   ├── temporal_memory.py       # Time-ordered episode event log
│   ├── episodic_memory.py       # Cross-episode task-attempt records
│   ├── semantic_memory.py       # Persistent domain facts and rules
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
├── memory_dataset/              # Dataset construction pipeline
│   ├── recorder.py              # Episode recording
│   ├── hindsight.py             # Hindsight staleness labelling
│   ├── sample_builder.py        # SFT example formatting
│   ├── curation.py              # Dataset curation and filtering
│   ├── synthetic_supervision.py # Synthetic target generation
│   ├── schemas.py               # Dataset record schemas
│   └── statistics.py            # Dataset statistics
│
├── memory_adapter_training/     # Stage 1: SFT pipeline
│   ├── trainer.py               # HuggingFace Trainer wrapper
│   ├── config.py                # SFT hyperparameters
│   ├── dataset.py               # Training data loading and collation
│   ├── modeling.py              # Model + LoRA initialisation
│   └── formatting.py            # Prompt/response formatting helpers
│
├── memory_adapter_rl/           # Stage 2: GRPO refinement
│   ├── rewards.py               # 9-component reward function
│   ├── grpo.py                  # GRPO rollout & policy update
│   ├── preferences.py           # DPO / ORPO preference pairs
│   ├── datasets.py              # Preference dataset builder
│   ├── trainer.py               # RL trainer classes
│   ├── checkpoints.py           # Checkpoint save/load
│   ├── formatting.py            # XML prompt/validation helpers
│   ├── config.py                # RL hyperparameters
│   ├── schemas.py               # RewardSignal, PreferencePair
│   └── evaluation.py            # RL evaluation utilities
│
├── evaluation/                  # Benchmark evaluation
│   ├── runner.py                # run_experiment() — top-level evaluation entry point
│   ├── metrics.py               # Success rate, SPL, stale-misuse rate, ...
│   ├── reporting.py             # JSON / CSV result writers
│   └── aggregators.py           # Multi-episode result aggregation
│
├── analysis/                    # Qualitative analysis
│   ├── trajectory_analysis.py   # Trajectory failure analysis
│   ├── comparisons.py           # Run-to-run comparison
│   ├── visualization.py         # Plots and figures
│   ├── detectors.py             # Failure-mode detectors
│   ├── taxonomy.py              # Failure taxonomy
│   └── reporting.py             # Analysis report generation
│
└── experiments/                 # Ablation and multi-seed orchestration
    ├── schemas.py               # ExperimentSpec, ExperimentResult
    ├── registry.py              # ExperimentRegistry (CRUD)
    ├── ablations.py             # 11-condition ablation factory
    ├── aggregation.py           # Multi-seed aggregation + CI
    ├── reporting.py             # Markdown / CSV / LaTeX tables
    ├── visualization.py         # Publication figures
    ├── launcher.py              # Single / batch / suite runners
    ├── sweeps.py                # Grid and seed sweeps
    └── utils.py                 # Git hash, JSON helpers
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
   Observation ──────────►│  MemoryAdapter   │
   Task instruction ──────►│  (Qwen2.5-7B    │
                           │   + QLoRA)       │
                           └────────┬─────────┘
                                    │ XML output
                       ┌────────────┴────────────┐
                       ▼                         ▼
             adapted_context               adapted_context
             + foresight_plan              + feasibility_criteria
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

The composite reward is:

$$
R = w_{\text{success}} \cdot S
  + w_{\text{progress}} \cdot P
  - w_{\text{replan}} \cdot R_p
  - w_{\text{invalid}} \cdot I
  - w_{\text{stale}} \cdot M_s
  - w_{\text{halluc}} \cdot H
  + w_{\text{feasib}} \cdot F
  + w_{\text{foresight}} \cdot Q
  + w_{\text{xml}} \cdot X
$$

| Component | Symbol | Default Weight | Sign |
|---|---|---|---|
| Task success | $S$ | 1.0 | + |
| Task progress | $P$ | 0.5 | + |
| Replanning penalty | $R_p$ | 0.2 | − |
| Invalid action penalty | $I$ | 0.3 | − |
| Stale-memory misuse penalty | $M_s$ | 0.4 | − |
| Hallucination penalty | $H$ | 0.5 | − |
| Feasibility quality | $F$ | 0.3 | + |
| Foresight quality | $Q$ | 0.2 | + |
| XML structure validity | $X$ | 0.5 | + |
