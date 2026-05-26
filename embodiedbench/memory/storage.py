"""
memory/storage.py

JSON / JSONL persistence helpers.

Design goals:
- Create missing directories automatically.
- Write UTF-8 throughout.
- Never crash on a missing file — return a safe default instead.
- Pretty-print normal JSON files for human readability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str) -> None:
    """Create *path* (and any parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(path: str, data: Any) -> None:
    """Serialise *data* to *path* as pretty-printed UTF-8 JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str, default: Any = None) -> Any:
    """Load and return the JSON at *path*, or *default* on any error."""
    p = Path(path)
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def save_jsonl(path: str, rows: list) -> None:
    """Write *rows* to *path* as newline-delimited JSON (overwrites)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def append_jsonl(path: str, row: dict) -> None:
    """Append a single *row* to *path* as a JSONL line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list:
    """
    Load all rows from a JSONL file.
    Returns [] if the file is missing.  Malformed lines are silently skipped.
    """
    p = Path(path)
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows
