#!/usr/bin/env python3
"""
upload_memguide_to_hf.py

Upload the MemGuide dataset (paraphrased SFT targets) to HuggingFace Hub.

The dataset will be created at:
  https://huggingface.co/datasets/<HF_USERNAME>/MemGuide

Two splits are uploaded:
  - alfred  → MemGuide/alfred_memory_logs/sft_filtered/sft_targets_filtered_paraphrased.jsonl
  - habitat → MemGuide/habitat_memory_logs/sft_filtered/sft_targets_filtered_paraphrased.jsonl

Usage
-----
# Login first (one-time), then run:
huggingface-cli login

python embodiedbench/scripts/upload_memguide_to_hf.py \
    --hf_username  NMThuan032k \
    --hf_token     hf_xxxxxxxxxxxxxxxxxxxx

# Dry-run (validate files, print what would be uploaded, write nothing):
python embodiedbench/scripts/upload_memguide_to_hf.py \
    --hf_username  NMThuan032k \
    --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset card (README.md that appears on the HF dataset page)
# ---------------------------------------------------------------------------
DATASET_CARD = """\
---
license: mit
task_categories:
  - robotics
  - text-generation
language:
  - en
tags:
  - embodied-ai
  - memory-adapter
  - planning
  - MemAdapter
  - EmbodiedBench
pretty_name: MemGuide
size_categories:
  - 100<n<1K
---

# MemGuide

**MemGuide** is the training dataset for [MemAdapter](https://github.com/thuannguyen25032k/MemAdapt),
a plug-and-play module that converts retrieved memories into structured planning guidance for
Vision-Language Model (VLM) embodied agents.

## Dataset Summary

Each record pairs a task instruction and retrieved memories (spatial, episodic, and semantic)
with structured planning guidance produced by a frontier LLM and filtered by behavioral consensus.
The dataset is used to fine-tune the MemAdapter (Qwen3-14B + LoRA).

## Splits

| Split   | Environment | Records |
|---------|-------------|---------|
| alfred  | EB-ALFRED   | 250     |
| habitat | EB-Habitat  | 240     |

## Schema

Each JSONL line contains:

| Field | Type | Description |
|---|---|---|
| `instruction` | `str` | Paraphrased natural-language task instruction |
| `retrieved_memory` | `str` | Spatial, episodic, and semantic memory context retrieved for the task |
| `planner_prompt` | `str` | Formatted planner prompt (foresight plan + fallback strategy) |
| `adapter_target` | `dict` | Structured guidance: `foresight_plan`, `feasibility_criteria`, `fallback_strategy` |
| `outcome` | `dict` | Closed-loop execution result: `success`, `progress`, `steps`, `replans` |
| `degradation_check` | `dict` | Behavioral consensus filter result: whether the guidance degraded either planner |

### `adapter_target` structure

```json
{
  "foresight_plan":       ["Step 1: ...", "Step 2: ..."],
  "feasibility_criteria": ["\"pick mug\": mug must be visible and reachable."],
  "fallback_strategy":    ["If cannot pick: navigate to table, retry pick."]
}
```

## Usage

```python
from datasets import load_dataset

ds = load_dataset("NMThuan032k/MemGuide")

# Access a single split
alfred_ds = ds["alfred"]
print(alfred_ds[0]["instruction"])
print(alfred_ds[0]["adapter_target"])
```

## Citation

```bibtex
@article{nguyen2026memadapter,
  title   = {MemAdapter: Structuring Retrieved Memories for VLM-Based Embodied Planning},
  author  = {Nguyen, Thuan},
  year    = {2026}
}
```

## License

[MIT](https://opensource.org/licenses/MIT)
"""

# ---------------------------------------------------------------------------
# Files to upload  {split_name: relative_path_from_repo_root}
# ---------------------------------------------------------------------------
SPLITS = {
    "alfred": "MemGuide/alfred_memory_logs/sft_filtered/sft_targets_filtered_paraphrased.jsonl",
    "habitat": "MemGuide/habitat_memory_logs/sft_filtered/sft_targets_filtered_paraphrased.jsonl",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_file(path: Path, split: str) -> int:
    """Return record count; raise if file is missing or contains bad JSON."""
    if not path.exists():
        raise FileNotFoundError(f"[{split}] File not found: {path}")
    records = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"[{split}] Bad JSON on line {lineno}: {e}") from e
            records += 1
    return records


def upload(hf_username: str, hf_token: str | None, root: Path, dry_run: bool) -> None:
    from huggingface_hub import HfApi, CommitOperationAdd

    repo_id = f"{hf_username}/MemGuide"
    api = HfApi()

    # --- Validate all files first ---
    print("Validating files …")
    for split, rel in SPLITS.items():
        path = root / rel
        count = validate_file(path, split)
        print(f"  ✓ [{split:>8}]  {count} records  →  {path.name}")

    if dry_run:
        print(f"\n[DRY-RUN] Would create/update dataset repo: {repo_id}")
        print(f"[DRY-RUN] Would upload {len(SPLITS)} split files + README.md")
        return

    # --- Create dataset repo (no-op if already exists) ---
    print(f"\nCreating dataset repo '{repo_id}' (skipped if already exists) …")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=hf_token,
    )

    # --- Build commit operations ---
    operations: list[CommitOperationAdd] = []

    # Dataset card
    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=DATASET_CARD.encode("utf-8"),
        )
    )

    # Data files — upload each split under data/<split>.jsonl
    for split, rel in SPLITS.items():
        path = root / rel
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"data/{split}.jsonl",
                path_or_fileobj=path,
            )
        )

    # --- Commit ---
    print(f"Uploading {len(operations)} files to {repo_id} …")
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="feat: add MemGuide paraphrased SFT dataset (alfred + habitat splits)",
        token=hf_token,
    )

    print(f"\n✓ Done!  View your dataset at:")
    print(f"  https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the MemGuide dataset to HuggingFace Hub."
    )
    parser.add_argument(
        "--hf_username",
        default="NMThuan032k",
        help="Your HuggingFace username (default: NMThuan032k).",
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace write token. Falls back to $HF_TOKEN env var. "
             "If omitted, uses the token stored by 'huggingface-cli login'.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repo root directory (default: current working directory).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate files and print what would be uploaded without writing anything.",
    )
    args = parser.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: 'huggingface_hub' not found.  Run:  pip install huggingface_hub")
        sys.exit(1)

    upload(
        hf_username=args.hf_username,
        hf_token=args.hf_token,
        root=Path(args.root),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
