# Dataset Pipeline

## Overview

The MemGuide dataset is built from recorded benchmark episodes (EB-ALFRED and
EB-Habitat).  A frontier LLM synthesizes structured guidance targets
(`FORESIGHT_PLAN`, `FEASIBILITY_CRITERIA`, `FALLBACK_STRATEGY`) for each episode;
behavioral consensus filtering then discards any target that degrades closed-loop
execution relative to baseline.  The surviving targets form the SFT training set.

## Pipeline Steps

```
1. Record benchmark runs
   (expert + novice planners;
    baseline vs. memory-adapter)
       │
       ▼
2. Frontier LLM synthesizes
   structured guidance targets
   (FORESIGHT_PLAN / FEASIBILITY_CRITERIA /
    FALLBACK_STRATEGY)
       │
       ▼
3. Behavioral consensus filtering
   (filter_sft_targets.py)
       │
       ▼
4. (optional) Paraphrase instructions
       │
       ▼
5. MemGuide — curated SFT targets
```

## Step 1 — Record Benchmark Runs

Filtering compares two planners — an **expert** (`InternVL3_5-38B`) and a **novice**
(`InternVL3_5-14B`) — each run **twice** per task: once without the adapter target
(`*_baseline`) and once with it (`*_memory_adapter`). Enable training-record logging via
the `memory_experiment` block:

```yaml
# embodiedbench/configs/config.yaml (or CLI override)
memory_experiment:
  mode: adapted_planner_critic
  save_training_records: true
  log_dir: "./alfred_memory_logs"
```

This produces, per domain folder (e.g. `alfred_memory_logs/`):

```
alfred_memory_logs/
├── training_records_38B.jsonl                 # adapter targets (expert outcome)
├── training_records_14B.jsonl                 # adapter targets (novice outcome)
├── InternVL3_5-38B_baseline/<cat>/results/episode_*_final_res.json
├── InternVL3_5-38B_memory_adapter/<cat>/results/episode_*_final_res.json
├── InternVL3_5-14B_baseline/<cat>/results/episode_*_final_res.json
└── InternVL3_5-14B_memory_adapter/<cat>/results/episode_*_final_res.json
```

Each `training_records_*.jsonl` line pairs a task instruction and its retrieved memory
with the frontier-LLM guidance target:

```json
{
  "instruction": "Pick up the mug and place it on the shelf.",
  "retrieved_memory": "[Spatial] mug last seen on table ...",
  "planner_prompt": "...",
  "adapter_target": {
    "foresight_plan": ["...", "..."],
    "feasibility_criteria": ["..."],
    "fallback_strategy": ["..."]
  },
  "outcome": {"success": true, "progress": 1.0, "steps": 12, "replans": 0}
}
```

## Step 2 — Behavioral Consensus Filtering

`filter_sft_targets.py` keeps only targets that **do not degrade** task progress for
**either** planner. A target is removed when, for the expert and/or the novice,

```
progress_with_adapter < progress_without_adapter − tolerance
```

```bash
python -m embodiedbench.memory_adapter_training.filter_sft_targets \
    --dataset-root memory_adapter_dataset \
    --output-dir   memory_adapter_dataset/sft_filtered \
    --tolerance    0.0
```

For each domain this writes:

```
<domain>/sft_filtered/
├── sft_targets_filtered.jsonl   # kept targets → SFT training set
└── removed_targets.jsonl        # rejected targets (for inspection)
```

The kept `sft_targets_filtered.jsonl` files are the direct input to
[sft_training.md](sft_training.md).

## Step 3 — (Optional) Paraphrase Instructions

`paraphrase_instructions.py` rewrites each task instruction with a frontier LLM to
increase linguistic diversity without altering task semantics:

```bash
python embodiedbench/scripts/paraphrase_instructions.py \
    --root    MemGuide \
    --model   gpt-5.5 \
    --api_key $OPENAI_API_KEY
```

## Data Statistics

The released **MemGuide** dataset contains the filtered SFT targets:

| Split   | Environment | Records |
|---------|-------------|---------|
| alfred  | EB-ALFRED   | 250     |
| habitat | EB-Habitat  | 240     |

MemGuide is available on the HuggingFace Hub:
[NMThuan032k/MemGuide](https://huggingface.co/datasets/NMThuan032k/MemGuide).
