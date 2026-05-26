"""
memory/utils.py

Shared utilities for the MemAdapt memory system.

Centralises helpers that were previously scattered across individual modules:

  - text similarity / set-overlap / list deduplication  (used by semantic/episodic memory)
  - safe config-key access helpers                      (used by integration.py)
  - safe attribute / string helpers                     (used by logging.py)
"""

from __future__ import annotations

from typing import Any, Optional

from embodiedbench.memory.embeddings import hybrid_score, lexical_overlap_score


# ---------------------------------------------------------------------------
# Text / scoring helpers
# ---------------------------------------------------------------------------

def similarity(
    text_a: str,
    text_b: str,
    emb_a: Optional[list] = None,
    emb_b: Optional[list] = None,
    emb_weight: float = 0.6,
    lex_weight: float = 0.4,
) -> float:
    """
    Compute text similarity using hybrid scoring when embeddings are available,
    falling back to lexical Jaccard overlap otherwise.
    """
    if emb_a and emb_b:
        return hybrid_score(text_a, text_b, emb_a, emb_b, emb_weight, lex_weight)
    return lexical_overlap_score(text_a, text_b)


def set_overlap(set_a: set, set_b: set) -> float:
    """Jaccard overlap between two sets; returns 0.0 if either is empty."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def list_union(existing: list, incoming: list) -> list:
    """
    Return the union of two lists, preserving order, case-insensitive dedup.
    Items already in *existing* (by lowercased value) are not re-added.
    """
    seen = {x.lower() for x in existing}
    result = list(existing)
    for item in incoming:
        if item.lower() not in seen:
            result.append(item)
            seen.add(item.lower())
    return result


# ---------------------------------------------------------------------------
# Safe config-key access helpers
# ---------------------------------------------------------------------------

_MISSING = object()  # sentinel for "key not present"


def _get_cfg_key(cfg: Any, key: str, default: Any = None) -> Any:
    """
    Safely extract *key* from *cfg* regardless of whether *cfg* is a dict,
    an OmegaConf DictConfig, or a plain object with attributes.

    Returns *default* if the key/attribute does not exist or on any error.
    """
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    # Attribute-based (OmegaConf, dataclass, namespace, …)
    try:
        val = getattr(cfg, key, _MISSING)
        if val is not _MISSING:
            return val
    except Exception:
        pass
    # Fallback: try .get() for OmegaConf structs / dict-like proxies
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def _cfg_enabled(section: Any) -> bool:
    """Return True only when *section* exists and has ``enabled=True``."""
    if section is None:
        return False
    return bool(_get_cfg_key(section, "enabled", False))


def _cfg_flag(section: Any, key: str, default: bool = True) -> bool:
    """
    Return a boolean flag from a config section.
    Falls back to *default* when the section is None or the key is absent.
    """
    if section is None:
        return default
    return bool(_get_cfg_key(section, key, default))


# ---------------------------------------------------------------------------
# Safe attribute / string helpers
# ---------------------------------------------------------------------------

def safe_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely read ``obj.attr``; return *default* on AttributeError or any error."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def safe_str(obj: Any, attr: str) -> str:
    """
    Safely read a string attribute from *obj*.
    Returns ``""`` when the attribute is absent, None, or not a string.
    """
    val = safe_attr(obj, attr)
    return val if isinstance(val, str) else ""
