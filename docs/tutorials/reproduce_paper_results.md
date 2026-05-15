# Tutorial: Reproduce Paper Results

This tutorial reproduces the main results table from the paper using the provided
configs and scripts.

## Prerequisites

- Conda env `embench` installed (see [README](../../README.md))
- Trained GRPO checkpoint available at
  `outputs/memory_adapter_rl/grpo_qwen7b/checkpoint-final`
- Trained SFT checkpoint available at
  `outputs/memory_adapter_training/qwen_qlora/checkpoint-final`

If you do not have checkpoints, follow
[train_memory_adapter_tutorial.md](train_memory_adapter_tutorial.md) first.

## Step 1 — Run the Full Ablation Suite

```bash
python embodiedbench/scripts/run_ablation_suite.py \
    --benchmark   eb_alfred \
    --seeds       1 2 3 4 5 \
    --grpo_checkpoint outputs/memory_adapter_rl/grpo_qwen7b/checkpoint-final \
    --sft_checkpoint  outputs/memory_adapter_training/qwen_qlora/checkpoint-final \
    --output_dir  outputs/ablations/eb_alfred
```

This runs 11 conditions × 5 seeds = 55 evaluation runs (~55 h on A100).

To do a quick sanity-check with fewer seeds:
```bash
python embodiedbench/scripts/run_ablation_suite.py \
    --benchmark eb_alfred --seeds 1 2 3 \
    --grpo_checkpoint outputs/memory_adapter_rl/grpo_qwen7b/checkpoint-final \
    --sft_checkpoint  outputs/memory_adapter_training/qwen_qlora/checkpoint-final \
    --output_dir outputs/ablations/eb_alfred_3seed
```

## Step 2 — Multi-Seed Experiments (all 4 benchmarks)

```bash
for benchmark in eb_alfred eb_habitat eb_manipulation eb_nav; do
    python embodiedbench/scripts/run_multiseed_experiments.py \
        --config  embodiedbench/configs/experiments/grpo_adapter.yaml \
        --benchmark $benchmark \
        --seeds   1 2 3 4 5 \
        --output_dir outputs/multiseed/$benchmark
done
```

## Step 3 — Generate Paper Tables

```bash
python embodiedbench/scripts/generate_paper_tables.py \
    --results_dir outputs/ablations/eb_alfred/aggregated \
    --output_dir  outputs/paper_tables
```

Output files:
- `outputs/paper_tables/table_main.tex`
- `outputs/paper_tables/table_ablation.tex`
- `outputs/paper_tables/table_main.md`

## Step 4 — Verify Against Expected Numbers

See [docs/reproducibility.md](../reproducibility.md) for the expected success rates
per condition (ALFRED, 5 seeds).

```bash
cat outputs/ablations/eb_alfred/aggregated/summary_table.md
```

## Checking Metadata

Every run records its git hash and config hash:

```bash
cat outputs/ablations/eb_alfred/grpo_adapter/seed_1/metadata.json
```
