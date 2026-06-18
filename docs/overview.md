# Overview

## Problem

Memory-augmented VLM planning is a promising approach for long-horizon embodied tasks
under partial observability.  Existing systems typically retrieve task-relevant memories
and inject them directly into the planner, but raw memories are often **verbose**,
**heterogeneous**, and **unstructured**, making them difficult for the planner to use
effectively.  Without adaptation, naïvely injecting retrieved memories can cause the
planner to generate infeasible sub-tasks and fail critic-raised checks.

## MemAdapter's Approach

MemAdapter inserts a trained, **plug-and-play** module between memory retrieval and the
VLM planner/critic pair.  Conditioned on the task instruction, it converts retrieved
memories into structured, task-level planning guidance without modifying the planner or
critic.

The adapter produces three structured output sections:
- `FORESIGHT_PLAN` — memory-grounded, ordered step sequence for global task sequencing.
- `FEASIBILITY_CRITERIA` — per-action preconditions for the critic to verify.
- `FALLBACK_STRATEGY` — spatially grounded recovery actions for failure cases.

MemAdapter is the sole trainable component in the system and requires **no manual
annotation**.  The released model and all reported results come from Stage 1; Stage 2 is
planned future work.

1. **Supervised Fine-Tuning (SFT)** — a frontier LLM synthesizes expert guidance targets
   from recorded benchmark episodes; behavioral consensus filtering discards targets that
   degrade closed-loop execution; the filtered data is distilled into a compact Qwen3-14B
   adapter via LoRA.
2. **GRPO Refinement** *(planned, not yet implemented)* — would optimise the adapter
   against closed-loop task-execution feedback (task success/progress, format validity,
   per-section quality) with the planner and critic frozen throughout. Code scaffolding
   exists under `memory_adapter_rl/` but has not been validated; see
   [grpo_training.md](grpo_training.md).

## Design Principles

- **Plug-and-play** — wraps any existing memory system and VLM backbone without
  modifying either component.
- **Task-conditioned** — the adapter is conditioned on the task instruction, so guidance
  is goal-relevant rather than a generic memory summary.
- **Structured three-part output** — a single adapter pass simultaneously produces
  guidance for the planner (foresight plan + fallback strategy) and the critic
  (feasibility criteria), ensuring internal consistency across both roles.
- **Modality-agnostic** — compatible with spatial, temporal, episodic, and semantic
  memory.
