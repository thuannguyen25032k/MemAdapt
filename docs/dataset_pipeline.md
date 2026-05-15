# Dataset Pipeline

## Overview

The memory-adapter training dataset is built from recorded benchmark episodes using
hindsight labelling — we know which memory entries turned out to be stale because we
can compare the agent's beliefs against the ground-truth state at each step.

## Pipeline Steps

```
1. Record episodes
       │
       ▼
2. Annotate staleness (hindsight)
       │
       ▼
3. Generate adapter target outputs (XML)
       │
       ▼
4. Format as SFT examples
       │
       ▼
5. Build HuggingFace Dataset
```

## Step 1 — Record Episodes

Run any benchmark with `--record_memory` to save memory snapshots:

```bash
python embodiedbench/main.py \
    --config embodiedbench/configs/eb-alf.yaml \
    --record_memory \
    --episodes_output_dir data/episodes/eb_alfred
```

Each episode is saved as `episodes/<episode_id>.json` containing:
- `steps[]` — list of (observation, action, memory_snapshot, success) tuples
- `task_instruction` — natural-language task string
- `final_success` — bool

## Step 2 — Build the Training Dataset

```bash
python embodiedbench/scripts/build_preference_dataset.py \
    --episodes_dir data/episodes/eb_alfred \
    --output_dir   data/memory_adapter_training/eb_alfred \
    --split_ratio  0.9
```

This script performs hindsight staleness annotation and formats the annotated episodes
into SFT training examples.  The script is named `build_preference_dataset.py` because
it also supports optional preference-pair generation for DPO/ORPO training; for
standard two-stage training (SFT → GRPO), the default output format is plain SFT.

This creates:
```
data/memory_adapter_training/eb_alfred/
├── train.jsonl   # ~90 % of episodes
├── val.jsonl     # ~10 % of episodes
└── metadata.json
```

Each `.jsonl` line is a JSON object.  In SFT mode the `target` field contains the
expert-generated adapter output; in preference mode, `chosen`/`rejected` fields are
populated for DPO/ORPO training:

```json
{
  "prompt": "<system>...<user>Task: ...\nMemory: ...\nObservation: ...</user>",
  "target": "<ADAPTED_CONTEXT>...</ADAPTED_CONTEXT>\n<FORESIGHT_PLAN>...</FORESIGHT_PLAN>\n...",
  "chosen": "<ADAPTED_CONTEXT>...</ADAPTED_CONTEXT>...",
  "rejected": "<ADAPTED_CONTEXT>[verbatim stale memory]</ADAPTED_CONTEXT>..."
}
```

## Data Statistics (expected)

| Split | ALFRED | Habitat | Manipulation | Navigation |
|---|---|---|---|---|
| Train | ~8 000 | ~4 000 | ~2 000 | ~2 000 |
| Val | ~900 | ~450 | ~220 | ~220 |
