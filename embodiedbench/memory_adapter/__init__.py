"""
embodiedbench/memory_adapter/__init__.py

Memory Adapter — an independent LLM/VLM-based module that transforms
retrieved MemoryContext into structured planner/critic guidance.

Usage
-----
from embodiedbench.memory_adapter import (
    MemoryAdapter,
    MemoryAdapterConfig,
    MemoryAdapterInput,
    MemoryAdapterOutput,
    build_adapter_prompt,
    parse_adapter_output,
)
"""

from embodiedbench.memory_adapter.schemas import MemoryAdapterInput, MemoryAdapterOutput
from embodiedbench.memory_adapter.config import MemoryAdapterConfig
from embodiedbench.memory_adapter.prompts import build_adapter_prompt
from embodiedbench.memory_adapter.parsing import parse_adapter_output
from embodiedbench.memory_adapter.adapter import MemoryAdapter

__all__ = [
    "MemoryAdapter",
    "MemoryAdapterConfig",
    "MemoryAdapterInput",
    "MemoryAdapterOutput",
    "build_adapter_prompt",
    "parse_adapter_output",
]
