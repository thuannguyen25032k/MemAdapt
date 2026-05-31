"""
memory_adapter_training

Supervised fine-tuning (SFT) pipeline for the Memory Adapter.

This package is intentionally isolated from the runtime inference code
(memory_adapter/, embodiedbench/planner/, embodiedbench/memory/).

Package layout
--------------
config.py            — MemoryAdapterTrainingConfig (YAML-backed dataclass)
dataset.py           — load curated SFT datasets into HF Dataset objects
formatting.py        — deterministic prompt/target formatting (no heavy deps)
collator.py          — chat-template tokenization, label masking, batching
modeling.py          — LoRA/QLoRA model construction (peft + bitsandbytes)
trainer.py           — HF Trainer wrapper with validation hooks
evaluation.py        — generation-based evaluation metrics
checkpoints.py       — save / load / merge / export LoRA adapters
utils.py             — seed, logging, miscellaneous helpers
train_sft.py         — CLI entry point tying the pipeline together
filter_sft_targets.py— filter curated targets by expert/novice degradation
"""
