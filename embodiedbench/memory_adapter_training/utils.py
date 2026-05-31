"""
memory_adapter_training/utils.py

Miscellaneous training utilities.
"""

from __future__ import annotations

import logging
import random
from typing import Tuple

logger = logging.getLogger("EB_logger")


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logging(level: str = "INFO") -> None:
    """Configure root and EB_logger loggers at the requested level."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=numeric,
    )
    logging.getLogger("EB_logger").setLevel(numeric)


def count_parameters(model) -> Tuple[int, int]:  # noqa: ANN001
    """
    Return (total_params, trainable_params) for *model*.

    Works with both vanilla ``nn.Module`` and PEFT-wrapped models.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
