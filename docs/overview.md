# Overview

## Problem

Embodied agents operating in dynamic environments accumulate memories about object
locations, past events, and task-relevant states.  These memories are crucial for
long-horizon task success — but they become **stale** as the environment evolves.

When the environment changes between steps, retrieved memories may be:
- **stale** — reflecting a past state that no longer holds,
- **contradictory** — conflicting with the current observation,
- **incomplete** — missing critical recent updates,
- **misleading** — causing the planner to commit to infeasible sub-tasks.

Naïvely injecting raw retrieved memories into a VLM planner causes the planner to:
- Hallucinate object locations that have changed.
- Plan sub-tasks that are no longer executable.
- Generate actions that fail feasibility checks raised by the VLM critic.

Discarding all memories wastes useful cross-episode history.  Injecting them verbatim
causes stale-memory interference.  Neither strategy is adequate for reliable long-horizon
reasoning under environment change.

## MemAdapt's Approach

MemAdapt inserts a **Memory Adapter** between the memory retrieval module and the
VLM planner/critic pair.  The adapter transforms retrieved memories into
**uncertainty-aware reasoning contexts** that both the planner and critic can reliably
consume, without modifying either component.

The Memory Adapter is the sole trainable component in the system, trained in two stages:

1. **Hindsight-supervised SFT** — teaches the adapter to perform memory reliability
   reasoning from hindsight-annotated trajectories: correctly identifying stale entries,
   hedging uncertain information, and grounding planner and critic guidance in current
   evidence.
2. **GRPO refinement** — optimises the adapter's memory reasoning against task-execution
   feedback, improving robustness to stale memories, reducing hallucination, and
   tightening feasibility reasoning.  The planner and critic remain frozen throughout.

The adapter produces three structured output sections:
- `FORESIGHT_PLAN` — memory-grounded, ordered step sequence for the planner.
- `FEASIBILITY_CRITERIA` — per-action preconditions for the critic to verify.
- `FALLBACK_STRATEGY` — concrete recovery actions for the most likely failures.

## Design Principles

- **Plug-and-play** — wraps any existing memory system and VLM backbone without modifying either.
- **Dual guidance** — a single adapter pass simultaneously grounds the planner (foresight)
  and the critic (feasibility), ensuring internal consistency across both roles.
- **Memory-modality agnostic** — compatible with spatial, temporal, episodic, and semantic memory.
- **Uncertainty-aware** — the adapter reasons about memory reliability before producing any output,
  preventing stale information from propagating unchecked into planning.
- **Reproducible** — fixed seeds, config hashes, and git-commit tracking throughout.
