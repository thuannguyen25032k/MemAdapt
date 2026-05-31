# Tutorial: Reproduce Paper Results

This tutorial reproduces the main results from the paper by running the full
evaluation pipeline with the provided trained checkpoints.

## Prerequisites

- Conda env `embench` installed (see [README](../../README.md))
- Trained GRPO checkpoint available at
  `outputs/memory_adapter_rl/grpo_qwen7b/checkpoint-final`
- Trained SFT checkpoint available at
  `outputs/memory_adapter_training/qwen3_14b/checkpoint-final`

If you do not have checkpoints, follow
[train_memory_adapter_tutorial.md](train_memory_adapter_tutorial.md) first.

## Step 1 — Run Evaluation

Use `embodiedbench/main.py` to evaluate each condition.  Example for the full
GRPO adapter on ALFRED:

```bash
python embodiedbench/main.py \
    --benchmark eb_alfred \
    --mode adapted_memory \
    --grpo_checkpoint outputs/memory_adapter_rl/grpo_qwen7b/checkpoint-final \
    --output_dir outputs/evaluation/grpo_adapter
```

Repeat for other modes (`baseline`, `raw_memory`, `sft_adapter`) and benchmarks
(`eb_habitat`, `eb_navigation`).

## Step 2 — Evaluate RL Adapter Quality

Use the RL evaluation script to score adapter outputs from a checkpoint:

```bash
python embodiedbench/scripts/evaluate_memory_adapter_rl.py \
    --checkpoint outputs/memory_adapter_rl/grpo_qwen7b/checkpoint-final \
    --prompts    memory_adapter_dataset/alfred_memory_logs/sft_filtered/sft_targets_filtered.jsonl \
    --output_dir outputs/eval_rl \
    --generate
```

## Step 3 — Verify Against Expected Numbers

See [docs/reproducibility.md](../reproducibility.md) for the expected success rates
per condition.
```

## Checking Metadata

Every run records its git hash and config hash:

```bash
cat outputs/ablations/eb_alfred/grpo_adapter/seed_1/metadata.json
```
