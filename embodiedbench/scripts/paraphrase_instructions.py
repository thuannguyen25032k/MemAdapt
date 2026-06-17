#!/usr/bin/env python3
"""
paraphrase_instructions.py

Paraphrase the 'instruction' field of every record in the MemGuide JSONL
files without changing the meaning.  Both the top-level ``instruction`` and
the matching ``degradation_check.instruction`` are updated to the same
paraphrased string so the dataset stays internally consistent.

Outputs are written to new ``*_paraphrased.jsonl`` files; originals are
never overwritten.

Supports any OpenAI-compatible backend (lmdeploy / vLLM / OpenAI API).

Usage
-----
# With a local lmdeploy server
python embodiedbench/scripts/paraphrase_instructions.py \
    --api_base http://localhost:8000/v1 \
    --api_key  EMPTY \
    --model    qwen3-14b-adapter

# With OpenAI
python embodiedbench/scripts/paraphrase_instructions.py \
    --api_base https://api.openai.com/v1 \
    --api_key  $OPENAI_API_KEY \
    --model    gpt-4o-mini

# Dry-run: print first 5 paraphrases per file, write nothing
python embodiedbench/scripts/paraphrase_instructions.py \
    --api_base https://api.openai.com/v1 \
    --api_key  $OPENAI_API_KEY \
    --model    gpt-4o-mini \
    --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Files to process (relative to --root, default = repo root)
# ---------------------------------------------------------------------------
JSONL_FILES = [
    "MemGuide/alfred_memory_logs/sft_filtered/sft_targets_filtered.jsonl",
    "MemGuide/habitat_memory_logs/sft_filtered/sft_targets_filtered.jsonl",
]

SYSTEM_PROMPT = (
    "You are a paraphrasing assistant for robotics task instructions. "
    "Rewrite the given instruction in different words WITHOUT changing its meaning, "
    "the objects involved, the actions required, or the final goal. "
    "Use natural, fluent English. "
    "Output ONLY the paraphrased instruction — no explanation, no quotes, no extra text."
)


# ---------------------------------------------------------------------------
# Core paraphrase call
# ---------------------------------------------------------------------------

def paraphrase(instruction: str, client, model: str, retries: int = 3) -> str:
    """Call the LLM to paraphrase a single instruction.  Returns the original
    on permanent failure so the dataset is never left with a blank field."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": instruction},
                ],
                max_completion_tokens=128,
            )
            result = response.choices[0].message.content.strip()
            # Basic sanity check: non-empty and not an error message
            if result and len(result) > 5:
                return result
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [WARN] attempt {attempt + 1}/{retries} failed: {exc} — retrying in {wait}s")
            time.sleep(wait)

    print(f"  [WARN] All retries exhausted for: '{instruction}' — keeping original")
    return instruction


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def process_file(path: Path, client, model: str, dry_run: bool) -> None:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]
    total = len(records)

    tag = "[DRY-RUN] " if dry_run else ""
    print(f"\n{tag}Processing {path}  ({total} records)")

    updated: list[dict] = []
    for i, rec in enumerate(records):
        original = rec["instruction"]
        paraphrased = paraphrase(original, client, model)

        # Update both places the instruction appears
        rec["instruction"] = paraphrased
        if "degradation_check" in rec and "instruction" in rec["degradation_check"]:
            rec["degradation_check"]["instruction"] = paraphrased

        updated.append(rec)

        # Always show first 5; then every 50
        if i < 5 or (i + 1) % 50 == 0:
            prefix = f"  [{i + 1:>3}/{total}]"
            if i < 5:
                print(f"{prefix} ORIG : {original}")
                print(f"{'':>{len(prefix)}} NEW  : {paraphrased}")
            else:
                print(f"{prefix} ... progress tick")

        if dry_run and i >= 4:
            print(f"  (dry-run: stopping after 5 examples)")
            break

    if not dry_run:
        out_path = path.with_name(path.stem + "_paraphrased" + path.suffix)
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in updated) + "\n",
            encoding="utf-8",
        )
        print(f"  ✓ Saved {len(updated)} records → {out_path}")
    else:
        print(f"  [DRY-RUN] would write {len(updated)} records to {path.stem}_paraphrased{path.suffix}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paraphrase MemGuide task instructions via an OpenAI-compatible API."
    )
    parser.add_argument(
        "--api_base",
        default=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="Base URL of the OpenAI-compatible API server (default: OpenAI).",
    )
    parser.add_argument(
        "--api_key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key (use 'EMPTY' for local lmdeploy/vLLM servers).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name to use for paraphrasing (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print first 5 paraphrases per file without writing any output.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repo root directory (default: current working directory).",
    )
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: 'openai' package not found.  Run:  pip install openai")
        sys.exit(1)

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    root = Path(args.root)

    for rel in JSONL_FILES:
        path = root / rel
        if not path.exists():
            print(f"[SKIP] File not found: {path}")
            continue
        process_file(path, client, args.model, args.dry_run)

    print("\nAll done.")


if __name__ == "__main__":
    main()
