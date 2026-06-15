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
    --sft_checkpoint outputs/memory_adapter_training/qwen3_14b/checkpoint-final \
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
  w_success:     1.0   # task success bonus (env rollout)
  w_progress:    0.5   # task progress fraction (env rollout)
  w_format:      0.5   # structural validity of the 3 required sections
  w_foresight:   0.6   # FORESIGHT_PLAN quality bonus
  w_feasibility: 0.6   # FEASIBILITY_CRITERIA quality bonus
  w_fallback:    0.6   # FALLBACK_STRATEGY quality bonus
  w_replan:      0.1   # replanning penalty (per event)
  w_invalid:     0.1   # invalid action penalty (per occurrence)
  w_repetition:  0.3   # degeneracy / repetition penalty

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
  num_generations: 8     # rollouts per prompt (group size)
  kl_beta: 0.04          # KL divergence coefficient
  temperature: 0.9       # rollout sampling temperature
  top_p: 0.95            # nucleus sampling top-p
  max_new_tokens: 2048    # max tokens per completion

# Logging
report_to: "wandb"
logging_steps: 10
```

## Reward Function

The composite reward for a single adapter output is:

$$
R = w_s \cdot S
  + w_p \cdot P
  + w_x \cdot X
  + w_{fo} \cdot F_o
  + w_{fe} \cdot F_e
  + w_{fa} \cdot F_a
  - w_r \cdot R_p
  - w_i \cdot I
  - w_d \cdot D
$$

where the components are:

| Symbol | Meaning | Type |
|---|---|---|
| $S$ | Task success (env rollout) | bonus |
| $P$ | Task progress fraction (env rollout) | bonus |
| $X$ | Format validity: fraction of the 3 required sections present **and** non-empty $\in [0,1]$ | bonus |
| $F_o$ | `FORESIGHT_PLAN` quality $\in [0,1]$ | bonus |
| $F_e$ | `FEASIBILITY_CRITERIA` quality $\in [0,1]$ | bonus |
| $F_a$ | `FALLBACK_STRATEGY` quality $\in [0,1]$ | bonus |
| $R_p$ | Replanning event count (env rollout) | penalty |
| $I$ | Invalid action count (env rollout) | penalty |
| $D$ | Degeneracy / repetition penalty $\in [0,1]$ | penalty |

The three section-quality terms ($F_o, F_e, F_a$) are computed purely from the
response text, giving a dense learning signal even when no environment is in the
loop. The task-outcome terms ($S, P, R_p, I$) are zero when scoring offline
prompts and are populated from environment rollout columns during online GRPO.
Bonus and degeneracy components are in $[0, 1]$; the count penalties are raw
(unbounded), so the reward is not globally bounded — intentional, to strongly
discourage repeated invalid actions.

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
