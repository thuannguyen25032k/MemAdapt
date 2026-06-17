# MemAdapter: Structuring Retrieved Memories for VLM-Based Embodied Planning

<!-- <p align="center">
  <img src="docs/images/framework.png" width="720" alt="MemAdapt Framework"/>
</p>

<p align="center">
  <a href="#installation"><img src="https://img.shields.io/badge/Python-3.9%2B-blue" /></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-green" /></a>
  <a href="docs/reproducibility.md"><img src="https://img.shields.io/badge/Reproducible-Yes-brightgreen" /></a>
</p>

--- -->

## Overview

Memory-augmented Vision-Language Model (VLM) planning is a promising approach for
long-horizon embodied tasks under partial observability. Existing systems typically
retrieve task-relevant memories and inject them directly into the planner, but raw
memories are often verbose, heterogeneous, and unstructured, making them difficult for
the planner to use. We propose **MemAdapter**, a plug-and-play Memory Adapter that,
conditioned on the task instruction, converts retrieved memories into structured,
task-level guidance comprising a foresight plan for global task sequencing, feasibility
criteria for critic-based action verification, and a fallback strategy for failure
recovery. Producing such guidance requires specialized reasoning that a compact large
language model (LLM) cannot reliably perform without dedicated training. We therefore
train MemAdapter without manual annotation: we synthesize expert guidance targets with
a frontier LLM, retain only those that do not degrade closed-loop execution via
behavioral consensus filtering, and distill the filtered data into a compact 14B
adapter through supervised fine-tuning. We evaluate MemAdapter on 400 tasks from the
EB-ALFRED and EB-Habitat environments of EmbodiedBench. With a frozen
Qwen2.5-VL-72B-Instruct planner, our MemAdapter-enabled framework attains a 79.50% average
success rate, improving over the strongest memory-augmented framework under the same
planner (RoboMemory) by 12.75 points; it further surpasses the strongest standalone VLM
agent (Claude-3.5-Sonnet) by 8.75 points despite building on a far weaker standalone
planner, indicating that memory adaptation can offset raw differences in planner
capability. These results show that effective memory-augmented embodied planning
requires not only retrieving memory but also adapting it into explicit, verifiable,
and recovery-aware planning guidance.

---

## Key Contributions

- **MemAdapter** — We propose MemAdapter, a plug-and-play module between memory
  retrieval and the VLM planner that converts heterogeneous retrieved memories
  (spatial, temporal, episodic, and semantic) into structured planning guidance
  without modifying either component.
- **Structured guidance format** — We design a structured guidance format with three
  components, each targeting a distinct stage of closed-loop planning: a *foresight
  plan* (initial task hypothesis), *feasibility criteria* (per-action preconditions for
  the critic), and a *fallback strategy* (spatially grounded recovery actions).
- **Automated fine-tuning pipeline** — We develop an automated fine-tuning pipeline
  that needs no manual annotation, in which a frontier LLM synthesizes expert
  guidance targets, behavioral consensus filtering discards targets that degrade
  closed-loop execution, and supervised fine-tuning distills the rest into MemAdapter.
- **MemGuide dataset** — We release MemGuide, a memory-to-guidance dataset pairing
  task instructions and retrieved memories with their structured planning guidance, to
  support future research on memory-adapted embodied planning.
- **Evaluation on EmbodiedBench** — We evaluate our framework on EmbodiedBench, a
  standardized benchmark for vision-driven embodied agents, attaining a 79.50% average
  success rate. This exceeds the strongest same-planner memory-augmented framework
  (RoboMemory) by 12.75 points and the strongest standalone VLM agent
  (Claude-3.5-Sonnet) by 8.75 points, with especially pronounced gains on
  commonsense-reasoning tasks.

<!-- ---

## Architecture

MemAdapter sits between the memory retrieval system and the VLM planner/critic.
Conditioned on the task instruction and retrieved memories, it converts heterogeneous
memory into structured, task-level planning guidance. Both the planner and critic remain
frozen; MemAdapter is the sole trained component.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Embodied Task Loop                           │
│                                                                     │
│  Environment ──► Observation ──► Memory System                     │
│                                       │                            │
│                          (spatial / temporal /                     │
│                           episodic / semantic)                     │
│                                       │                            │
│                               Retrieved Memories                   │
│                          (verbose, heterogeneous,                  │
│                           and unstructured)                        │
│                                       │                            │
│         Task Instruction ─────────────┤                            │
│                                       ▼                            │
│                          ┌────────────────────────┐               │
│                          │      MemAdapter        │               │
│                          │  (Qwen3-14B + LoRA)    │               │
│                          │                        │               │
│                          │  converts memories to  │               │
│                          │  structured guidance:  │               │
│                          │  • foresight plan      │               │
│                          │  • feasibility criteria│               │
│                          │  • fallback strategy   │               │
│                          └────────────┬───────────┘               │
│                                       │                            │
│              ┌────────────────────────┴──────────────────┐        │
│              │                                           │         │
│   foresight plan + fallback strategy          feasibility criteria │
│              │                                           │         │
│              ▼                                           ▼         │
│     ┌─────────────────┐                       ┌──────────────────┐│
│     │   VLM Planner   │                       │   VLM Critic     ││
│     │   (frozen)      │                       │   (frozen)       ││
│     └────────┬────────┘                       └────────┬─────────┘│
│              └──────────────────┬─────────────────────┘           │
│                                 ▼                                  │
│                              Action                                │
└─────────────────────────────────────────────────────────────────────┘
``` -->

**Key design properties:**

- **Plug-and-play** — MemAdapter wraps any existing memory system and VLM backbone
  without modifying either component.
- **Task-conditioned guidance** — the adapter is conditioned on the task instruction,
  producing guidance that is relevant to the current goal rather than a generic memory
  summary.
- **Structured three-part output** — a single adapter pass produces a *foresight plan*
  and *fallback strategy* for the planner, and *feasibility criteria* for the critic,
  ensuring internal consistency across both decision-making roles.
- **Modality-agnostic** — compatible with all four memory modalities: spatial,
  temporal, episodic, and semantic.

---

## Training Pipeline

MemAdapter is the **only trained component** in the system. Training requires no manual
annotation and proceeds in two stages.

<!-- ```
Recorded Benchmark Episodes
(EB-ALFRED + EB-Habitat)
        │
        ▼
Frontier LLM synthesizes structured
guidance targets for each episode
(FORESIGHT_PLAN / FEASIBILITY_CRITERIA /
 FALLBACK_STRATEGY)
        │
        ▼
Behavioral Consensus Filtering
        │
        ▼
  MemGuide Dataset
  (task instruction + retrieved memories
   + filtered guidance targets)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Stage 1 — Supervised Fine-Tuning (SFT)                   │
│                                                           │
│  • Base model: Qwen3-14B + LoRA                           │
│  • Learns to produce structured XML guidance              │
│    (foresight plan, feasibility criteria,                 │
│     fallback strategy) from task + memories.              │
│  • Loss computed over assistant responses only;           │
│    same prompt format used at training and inference.     │
└──────────────────────────┬────────────────────────────────┘
                           │
                           ▼
                      SFT Adapter
                           │
                           ▼
┌───────────────────────────────────────────────────────────┐
│  Stage 2 — GRPO Refinement                                │
│                                                           │
│  • Optimises against closed-loop task-execution feedback. │
│  • Rewards: task success/progress, output format          │
│    validity, per-section quality (foresight, feasibility, │
│    fallback); penalises replanning and invalid actions.   │
│  • Planner and critic remain frozen throughout.           │
└──────────────────────────┬────────────────────────────────┘
                           │
                           ▼
                  MemAdapter (final)
                           │
                           ▼
              Benchmark Evaluation
              (EB-ALFRED / EB-Habitat)
``` -->

**Stage 1 — Supervised Fine-Tuning (SFT)** distills the filtered MemGuide targets into
the compact Qwen3-14B adapter via LoRA. The adapter learns to generate the three-part
structured guidance (foresight plan, feasibility criteria, fallback strategy) conditioned on a task instruction and retrieved memories, using the same prompt format at both training and inference time.

**Stage 2 — GRPO Refinement (Not implemented)** sharpens closed-loop performance. Rollouts are scored with a composite reward covering task success/progress, structural format validity, and per-section content quality, with penalties for excessive replanning and invalid actions.
This stage trains the adapter — not the planner — to produce guidance that makes the
frozen downstream components more reliable.

---

### Prerequisites

- Python 3.9+
- conda (recommended)
- CUDA 11.8+ (for training; evaluation can run on CPU)

### Step 1 — Clone

```bash
git clone https://github.com/thuannguyen25032k/MemAdapt.git
cd MemAdapt
```

### Step 2 — Create environment

```bash
# Primary environment (ALFRED + Habitat)
conda env create -f conda_envs/environment.yaml
conda activate embench
pip install -e .
```

### Step 3 — Install benchmark data

```bash
bash install.sh
```

### Step 4 — Verify installation

```bash
python -c "from embodiedbench.memory_adapter import MemoryAdapter; print('OK')"
pytest tests/ -q --tb=no
```

---

## Quickstart

### Minimal adapter usage

```python
from embodiedbench.memory_adapter import MemoryAdapter, MemoryAdapterInput
from embodiedbench.memory_adapter.config import MemoryAdapterConfig

cfg = MemoryAdapterConfig(model_name_or_path="Qwen/Qwen3-14B")
adapter = MemoryAdapter(cfg)

adapter_input = MemoryAdapterInput(
    task_instruction="Pick up the mug and place it on the shelf.",
    memory_context=memory_manager.retrieve("mug shelf"),
)
output = adapter.adapt(adapter_input)
print(output.foresight_plan)
print(output.feasibility_criteria)
print(output.fallback_strategy)
```

---

## Dataset Generation

See [docs/dataset_pipeline.md](docs/dataset_pipeline.md) for full details.

```bash
# Filter curated SFT targets from collected training records
python -m embodiedbench.memory_adapter_training.filter_sft_targets \
    --dataset-root memory_adapter_dataset \
    --output-dir   memory_adapter_dataset/sft_filtered
```

---

## SFT Training

See [docs/sft_training.md](docs/sft_training.md) for full details.

```bash
python -m embodiedbench.memory_adapter_training.train_sft \
    --config embodiedbench/configs/memory_adapter_training/qwen3_14b.yaml \
    --train_path memory_adapter_dataset/alfred_memory_logs/sft_filtered/sft_targets_filtered.jsonl \
                 memory_adapter_dataset/habitat_memory_logs/sft_filtered/sft_targets_filtered.jsonl \
    --output_dir outputs/memory_adapter_training/qwen3_14b
```

---

## Merging the Checkpoint

After SFT training the output directory contains a **LoRA adapter** (delta weights
only), not a full standalone model. Before running benchmark evaluation, merge the adapter back into the base model:

```bash
python embodiedbench/scripts/merge_adapter.py \
    --base_model  embodiedbench/memory_adapter/models/Qwen3-14B \
    --adapter_dir outputs/memory_adapter_training/qwen3_14b/checkpoint-final \
    --output_dir  outputs/merged/qwen3_14b_merged
```

The merged model is saved as a plain HuggingFace checkpoint — no PEFT dependency
needed at inference time.

## Benchmark Evaluation

```bash
python embodiedbench/main.py \
    env=eb-alf \
    adapter_checkpoint=outputs/merged/qwen3_14b_merged
```

---

## Ablation Studies

Set the `mode` field in `embodiedbench/configs/config.yaml` under `memory_experiment` to run different ablation conditions:

| `mode` value | Description |
|---|---|
| `baseline` | No memory, no adapter — pure planner + critic |
| `raw_memory` | Raw retrieved memory injected directly, no adaptation |
| `adapted_memory` | **Full MemAdapt system** — adapter injected into both planner and critic |
| `adapted_memory_planner_only` | Adapter injected into planner only |
| `adapted_memory_critic_only` | Adapter injected into critic only |
| `adapted_memory_planner_critic` | Explicit dual injection (equivalent to `adapted_memory`) |

---

## Reproducibility

See [docs/reproducibility.md](docs/reproducibility.md) for full details.

- Expected hardware: 1× A100 80 GB (training) / any CPU (evaluation stub)
- Expected runtime: SFT ~4 h, GRPO ~6 h on A100

---

## Project Structure

```
MemAdapt/
├── embodiedbench/
│   ├── memory/                  # Memory system (spatial, temporal, episodic, semantic) + trajectory recorder
│   ├── memory_adapter/          # MemAdapt runtime adapter
│   ├── memory_adapter_training/ # SFT training infrastructure
│   ├── memory_adapter_rl/       # GRPO RL refinement
│   ├── evaluation/              # Benchmark evaluation harness
│   ├── scripts/                 # CLI entry points
│   ├── configs/                 # YAML configs for all modules
│   ├── examples/                # Minimal runnable examples
│   ├── envs/                    # Benchmark environment wrappers
│   ├── evaluator/               # Original EmbodiedBench evaluators
│   └── main.py                  # Top-level benchmark runner
├── docs/                        # Full documentation
├── conda_envs/                  # Conda environment specs
├── Docker/                      # Docker build files
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Citation

If you use MemAdapt in your research, please cite:

```bibtex
@article{nguyen2026memadapt,
  title   = {MemAdapt: A Plug-and-Play Memory Adapter for Stale-Memory Reasoning
             in Embodied Agents},
  author  = {Nguyen, Minh Thuan and Le, Bao Long},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

The EmbodiedBench benchmark environments are subject to their own licenses;
see [Original_README.md](Original_README.md) for details.
