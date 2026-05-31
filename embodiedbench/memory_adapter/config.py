"""
embodiedbench/memory_adapter/config.py

Configuration dataclass for the Memory Adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class MemoryAdapterConfig:
    """
    Configuration for MemoryAdapter.

    Two mutually exclusive backends:
    - Local HF: set model_name_or_path; model is loaded onto the GPU.
    - API:      set api_model (+ optionally api_key / api_base_url); no local model is loaded.
                Any OpenAI-compatible server is supported (OpenAI, lmdeploy, vLLM, …).
    """

    model_name_or_path: str = ""
    device: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    enabled: bool = True
    enable_thinking: bool = False  # Qwen3: suppresses <think>…</think> when False; ignored for API backend
    # API backend — when set, overrides local model loading
    api_model: str = ""        # e.g. "gpt-4o" or "qwen3-14b-adapter" (lmdeploy/vLLM)
    api_key: str = ""          # falls back to OPENAI_API_KEY env var; use "EMPTY" for local servers
    api_base_url: str = ""     # e.g. "http://localhost:8000/v1"; empty = official OpenAI

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name_or_path":  self.model_name_or_path,
            "device":              self.device,
            "torch_dtype":         self.torch_dtype,
            "max_new_tokens":      self.max_new_tokens,
            "temperature":         self.temperature,
            "top_p":               self.top_p,
            "do_sample":           self.do_sample,
            "load_in_8bit":        self.load_in_8bit,
            "load_in_4bit":        self.load_in_4bit,
            "trust_remote_code":   self.trust_remote_code,
            "enabled":             self.enabled,
            "enable_thinking":     self.enable_thinking,
            "api_model":           self.api_model,
            "api_key":             self.api_key,
            "api_base_url":        self.api_base_url,
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> "MemoryAdapterConfig":
        """
        Build from any dict-like object (plain dict, OmegaConf DictConfig, etc.).
        Unknown keys are silently ignored.
        """
        if mapping is None:
            return cls()

        # Normalise to a plain dict
        if hasattr(mapping, "items"):
            raw: Dict[str, Any] = {k: v for k, v in mapping.items()}
        elif hasattr(mapping, "__dict__"):
            raw = dict(vars(mapping))
        else:
            raw = {}

        _FIELDS = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in raw.items() if k in _FIELDS}
        return cls(**filtered)
