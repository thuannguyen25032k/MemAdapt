# Experiments & Ablation Studies

## Overview

`embodiedbench/experiments/` provides a complete orchestration layer for running
systematic ablations and multi-seed experiments with automatic aggregation, statistical
testing, and paper-ready table / figure generation.

## 11-Condition Ablation Suite

```bash
python embodiedbench/scripts/run_ablation_suite.py \
    --benchmark   eb_alfred \
    --seeds       1 2 3 4 5 \
    --grpo_checkpoint outputs/memory_adapter_rl/grpo_qwen7b/checkpoint-final \
    --sft_checkpoint  outputs/memory_adapter_training/qwen_qlora/checkpoint-final \
    --output_dir  outputs/ablations/eb_alfred
```

### Ablation Conditions

The 11-condition suite covers three orthogonal ablation axes:

**Axis 1 — Adapter type**

| Condition | Adapter | Inject Target | GRPO Rewards | Purpose |
|---|---|---|---|---|
| `baseline` | — | — | — | No-memory reference |
| `raw_memory` | raw (no adapter) | both | — | Raw-injection reference |
| `sft_adapter` | SFT only | both | — | Stage 1 only (no GRPO) |
| `grpo_adapter` (**full**) | GRPO | both | all | Full MemAdapt system |

**Axis 2 — Injection target** (using full GRPO adapter)

| Condition | Adapter | Inject Target | GRPO Rewards | Purpose |
|---|---|---|---|---|
| `planner_only` | GRPO | planner | all | Ablate critic guidance |
| `critic_only` | GRPO | critic | all | Ablate planner guidance |
| `planner_critic` | GRPO | both | all | Dual injection (same settings as `grpo_adapter`; serves as matched reference for this sub-axis) |

**Axis 3 — Reward component** (using full GRPO adapter, dual injection)

| Condition | Adapter | Inject Target | Excluded Reward | Purpose |
|---|---|---|---|---|
| `no_stale_penalty` | GRPO | both | stale penalty | Ablate staleness robustness |
| `no_xml_reward` | GRPO | both | XML validity | Ablate format reward |
| `no_feasibility` | GRPO | both | feasibility | Ablate feasibility guidance |
| `no_foresight` | GRPO | both | foresight quality | Ablate foresight planning |

### Output Layout

```
outputs/ablations/eb_alfred/
├── baseline/          seed_1/ seed_2/ ... aggregated/
├── raw_memory/        ...
├── grpo_adapter/      ...
├── ...
└── aggregated/
    ├── summary_table.md
    ├── summary_table.csv
    ├── summary_table.tex
    └── figures/
        ├── ablation_bar.png
        ├── reward_heatmap.png
        └── seed_variance.png
```

## Multi-Seed Experiment

```bash
python embodiedbench/scripts/run_multiseed_experiments.py \
    --config  embodiedbench/configs/experiments/grpo_adapter.yaml \
    --seeds   1 2 3 4 5 \
    --output_dir outputs/multiseed/grpo_adapter
```

## Generate Paper Tables

```bash
python embodiedbench/scripts/generate_paper_tables.py \
    --results_dir outputs/ablations/eb_alfred/aggregated \
    --output_dir  outputs/paper_tables
```

Generates:
- `table_main.tex` — main results table (LaTeX booktabs)
- `table_ablation.tex` — ablation table
- `table_main.csv` — CSV version for import into spreadsheets
- `table_ablation.md` — Markdown version for GitHub

## Python API

```python
from embodiedbench.experiments import (
    ExperimentRegistry,
    AblationSuite,
    run_ablation_suite,
    aggregate_seed_results,
    generate_markdown_table,
)

registry = ExperimentRegistry("outputs/ablations/eb_alfred")
suite = AblationSuite.default(benchmark="eb_alfred", seeds=[1, 2, 3])
results = run_ablation_suite(suite, registry=registry)
aggregated = aggregate_seed_results(results)
print(generate_markdown_table(aggregated))
```
