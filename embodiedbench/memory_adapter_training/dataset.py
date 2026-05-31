"""
memory_adapter_training/dataset.py

Load curated SFT records into HuggingFace ``Dataset`` objects.

Each record is normalised to a ``{"prompt", "response"}`` pair via
``formatting.format_sample`` so the rest of the pipeline (collator, trainer)
only ever sees that shape.

Input files may be:
  * JSONL (one JSON object per line), or
  * JSON (a list of objects).

Object shapes accepted by ``format_sample``:
  * Filtered SFT targets (``filter_sft_targets.py`` output) with a nested
    ``adapter_target`` dict.
  * Pre-formatted ``{"prompt", "response"}`` pairs.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple

from embodiedbench.memory_adapter_training.formatting import format_sample

logger = logging.getLogger("EB_logger")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"[Dataset] {path}:{lineno} decode error: {exc}")
    return records


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Expected list or dict in {path}")


def load_sft_records(path: "str | list[str]") -> List[Dict[str, str]]:
    """Load one or more files and return a merged list of ``{"prompt", "response"}`` pairs.

    *path* may be a single file path or a list of file paths; in the latter
    case all records are concatenated in order before being returned.

    Records whose response has no usable content are dropped.
    """
    paths = [path] if isinstance(path, str) else list(path)
    all_pairs: List[Dict[str, str]] = []
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Dataset file not found: {p}")
        raw = _load_jsonl(p) if p.endswith(".jsonl") else _load_json_list(p)
        for obj in raw:
            try:
                pair = format_sample(obj)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[Dataset] format_sample failed: {exc}")
                continue
            if pair["prompt"].strip() and pair["response"].strip():
                all_pairs.append(pair)
        logger.info(f"[Dataset] Loaded {len(all_pairs)} records (cumulative) from {p}")
    return all_pairs


def make_hf_dataset(records: List[Dict[str, str]]):  # noqa: ANN201
    """Convert a list of ``{"prompt", "response"}`` pairs into a HF Dataset."""
    from datasets import Dataset  # type: ignore

    return Dataset.from_list(records)


def split_train_val(
    records: List[Dict[str, str]],
    val_ratio: float,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Shuffle and split records into (train, val)."""
    if val_ratio <= 0.0 or len(records) < 2:
        return records, []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[n_val:], shuffled[:n_val]


def load_train_val_datasets(
    train_path: "str | list[str]",
    val_path: str = "",
    val_ratio: float = 0.0,
    seed: int = 42,
):  # noqa: ANN201
    """Load train (and optionally val) datasets as HF Dataset objects.

    If ``val_path`` is empty but ``val_ratio`` > 0, the train file is split
    into train/val internally.

    Returns
    -------
    (train_dataset, val_dataset) - val_dataset is None when no validation data.
    """
    train_records = load_sft_records(train_path)

    val_records: List[Dict[str, str]] = []
    if val_path and os.path.isfile(val_path):
        val_records = load_sft_records(val_path)
    elif val_path:
        logger.warning(f"[Dataset] val_path not found: {val_path}")
    elif val_ratio > 0.0:
        train_records, val_records = split_train_val(train_records, val_ratio, seed)
        logger.info(
            f"[Dataset] Auto-split: {len(train_records)} train / {len(val_records)} val "
            f"(val_ratio={val_ratio})"
        )

    train_ds = make_hf_dataset(train_records)
    val_ds = make_hf_dataset(val_records) if val_records else None
    return train_ds, val_ds
