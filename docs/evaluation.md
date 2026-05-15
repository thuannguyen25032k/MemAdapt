# Benchmark Evaluation

## Overview

`embodiedbench/evaluation/` provides the `run_experiment()` function (in `runner.py`)
that runs a trained MemAdapt checkpoint on any of the four EmbodiedBench environments
and records per-task success rates and additional metrics.  The runner patches the
evaluator config with the adapter checkpoint path and memory experiment mode, then
delegates to the existing benchmark evaluator — it does not re-implement simulator
logic.

## Supported Benchmarks

| ID | Environment | Tasks |
|---|---|---|
| `eb_alfred` | AI2-THOR (ALFRED) | pick-and-place, cleaning, heating, cooling |
| `eb_habitat` | Habitat 2.0 | fetch, rearrangement |
| `eb_manipulation` | Franka manipulation | peg-in-hole, stacking |
| `eb_nav` | Habitat navigation | object-goal navigation |

## Quick Start

```bash
python embodiedbench/main.py \
    --config embodiedbench/configs/eb-alf.yaml \
    --adapter_checkpoint outputs/memory_adapter_rl/grpo_qwen7b/checkpoint-final \
    --output_dir         outputs/eval/grpo_qwen7b_alfred
```

## Config Reference

```yaml
# embodiedbench/configs/eb-alf.yaml
benchmark: eb_alfred
split: test
num_episodes: 200
seed: 42

memory:
  use_spatial: true
  use_temporal: true
  use_episodic: true
  use_semantic: true
  max_entries: 50

adapter:
  enabled: true
  checkpoint: null    # override via CLI
  max_new_tokens: 512
  temperature: 0.1

planner:
  model: "gpt-4o"     # or local model path
  max_steps: 30

critic:
  model: "gpt-4o"
  enabled: true
```

## Output Structure

```
outputs/eval/grpo_qwen7b_alfred/
├── episode_results.jsonl    # per-episode pass/fail + metrics
├── summary.json             # aggregate metrics
└── visualisations/
    ├── success_by_task_type.png
    └── step_distribution.png
```

`summary.json` schema:

```json
{
  "benchmark":        "eb_alfred",
  "adapter":          "grpo_qwen7b",
  "num_episodes":     200,
  "success_rate":     0.623,
  "mean_steps":       14.2,
  "stale_misuse_rate": 0.041,
  "hallucination_rate": 0.028
}
```

## Evaluate RL Adapter Directly

```bash
python embodiedbench/scripts/evaluate_memory_adapter_rl.py \
    --checkpoint outputs/memory_adapter_rl/grpo_qwen7b/checkpoint-final \
    --benchmark  eb_alfred \
    --split      test \
    --output_dir outputs/rl_eval/grpo_qwen7b
```
