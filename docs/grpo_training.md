# GRPO Training

## Overview

Stage 2 fine-tunes the SFT checkpoint further using **Group Relative Policy Optimisation
(GRPO)**.  For each prompt, the model generates a group of $G$ candidate outputs; their
rewards are normalised within the group, and the policy is updated to favour
higher-reward completions.

GRPO requires no separate reference model (unlike DPO) and scales well to large group
sizes.

## Quick Start

```bash
python embodiedbench/scripts/train_memory_adapter_grpo.py \
    --config embodiedbench/configs/memory_adapter_rl/qwen_grpo.yaml \
    --sft_checkpoint outputs/memory_adapter_training/qwen_qlora/checkpoint-final \
    --output_dir     outputs/memory_adapter_rl/grpo_qwen7b
```

## Config Reference

```yaml
# embodiedbench/configs/memory_adapter_rl/qwen_grpo.yaml
run_name: memadapt_grpo_qwen7b
algorithm: grpo
seed: 42

model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
torch_dtype: bfloat16
load_in_4bit: false

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

# Data — GRPO rollout prompt files
train_data_path: "data/memory_adapter_rl/grpo_prompts_train.jsonl"
val_data_path:   "data/memory_adapter_rl/grpo_prompts_val.jsonl"
max_prompt_length: 1024
max_length: 2048

# Reward weights (must match RLRewardWeights in memory_adapter_rl/config.py)
reward_weights:
  w_success:   1.0   # task success bonus
  w_progress:  0.5   # task progress fraction
  w_replan:    0.2   # replanning penalty (per event)
  w_invalid:   0.3   # invalid action penalty (per occurrence)
  w_stale:     0.4   # stale-memory misuse penalty
  w_halluc:    0.5   # hallucinated object penalty
  w_feasib:    0.3   # feasibility quality bonus
  w_foresight: 0.2   # foresight quality bonus
  w_xml:       0.5   # XML structure validity bonus

# Training
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2.0e-5
warmup_ratio: 0.03
lr_scheduler_type: cosine
gradient_checkpointing: true
save_steps: 100

# GRPO-specific
grpo:
  num_generations: 8     # rollouts per prompt
  group_size: 8          # group size for relative reward normalisation
  kl_beta: 0.04          # KL divergence coefficient
  reward_normalization: true

# Logging
report_to: "wandb"
logging_steps: 10
```

## Reward Function

The composite reward for a single adapter output is:

$$
R = w_s \cdot S
  + w_p \cdot P
  - w_r \cdot R_p
  - w_i \cdot I
  - w_k \cdot M_s
  - w_h \cdot H
  + w_f \cdot F
  + w_q \cdot Q
  + w_x \cdot X
$$

where the components are:

| Symbol | Meaning | Type |
|---|---|---|
| $S$ | Task success | bonus |
| $P$ | Task progress fraction | bonus |
| $R_p$ | Replanning event count | penalty |
| $I$ | Invalid action count | penalty |
| $M_s$ | Stale-memory misuse count | penalty |
| $H$ | Hallucinated object count | penalty |
| $F$ | Feasibility quality score $\in [0,1]$ | bonus |
| $Q$ | Foresight quality score $\in [0,1]$ | bonus |
| $X$ | XML structure validity $\in \{0,1\}$ | bonus |

Bonus-type components are in $[0, 1]$; penalty-type components are raw counts
(unbounded), so the reward is not globally bounded — this is intentional to allow
strong penalties for repeated stale-memory misuse.

## Group Relative Normalisation

Within a group of $G$ outputs for the same prompt:

$$
\hat{R}_i = \frac{R_i - \mu_G}{\sigma_G + \varepsilon}
$$

The policy is updated via a KL-regularised objective:

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[
  \hat{R}_i \cdot \log \pi_\theta(y_i \mid x)
\right] + \beta \cdot D_{\text{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right]
$$

where $\beta = \texttt{kl\_beta} = 0.04$ controls the divergence penalty and
$\pi_{\text{ref}}$ is the SFT checkpoint frozen at the start of Stage 2.  Unlike a
PPO-style clipped surrogate, this formulation does not require a separate critic network.

## Debug Run

```bash
python embodiedbench/scripts/train_memory_adapter_grpo.py \
    --config embodiedbench/configs/memory_adapter_rl/debug_grpo_tiny.yaml \
    --output_dir /tmp/memadapt_grpo_debug
```

## Hardware Requirements

| Config | GPU | VRAM | Time |
|---|---|---|---|
| `qwen_grpo.yaml` | A100 80 GB | ~60 GB | ~6 h |
| `debug_grpo_tiny.yaml` | CPU | — | ~3 min |

## Checkpoint Layout

```
outputs/memory_adapter_rl/grpo_qwen7b/
├── checkpoint-step-200/
├── checkpoint-step-400/
├── checkpoint-final/       ← use this for evaluation
└── training_log.jsonl
```
