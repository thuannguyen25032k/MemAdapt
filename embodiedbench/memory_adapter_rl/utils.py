"""
memory_adapter_rl/utils.py

Shared utilities for the GRPO refinement pipeline.
"""

from __future__ import annotations

import logging


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logging for the RL scripts."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
