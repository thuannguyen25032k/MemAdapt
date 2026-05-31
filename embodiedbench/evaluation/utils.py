"""
evaluation/utils.py

JSON and config helpers for the evaluation harness.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def is_json_serializable(obj: Any) -> bool:
    """Return True if *obj* can be serialised to JSON without error."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Config patching helpers
# ---------------------------------------------------------------------------

def patch_config_for_mode(
    base_config: Dict[str, Any],
    mode: str,
    adapter_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a copy of *base_config* with the memory_experiment block updated
    for the given *mode* (does not mutate *base_config*).
    """
    from embodiedbench.evaluation.runner import _MODE_TO_EXPERIMENT_MODE  # local import to avoid cycles

    cfg = dict(base_config)
    cfg["memory_experiment"] = {
        "mode": _MODE_TO_EXPERIMENT_MODE.get(mode, "none"),
        "log_memory_outputs": True,
        "log_adapter_outputs": True,
        "save_training_records": False,
        "adapter_checkpoint": adapter_checkpoint or "",
    }
    return cfg
