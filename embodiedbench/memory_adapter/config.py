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
    All configuration required to instantiate and run a MemoryAdapter.

    Fields
    ------
    model_name_or_path   : HuggingFace model name or local path.
    device               : "auto", "cpu", "cuda", "cuda:0", etc.
    torch_dtype          : "auto", "float16", "bfloat16", "float32".
    max_new_tokens       : maximum tokens to generate per adapter call.
    temperature          : sampling temperature (ignored when do_sample=False).
    top_p                : nucleus-sampling probability mass.
    do_sample            : enable stochastic sampling.
    load_in_8bit         : enable bitsandbytes 8-bit quantization.
    load_in_4bit         : enable bitsandbytes 4-bit quantization.
    trust_remote_code    : passed to from_pretrained.
    enabled              : when False the adapter is a no-op (returns empty output).
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
    # Qwen3-style thinking mode — set to False to suppress <think>...</think> output
    # via apply_chat_template(enable_thinking=False). Ignored for OpenAI backend.
    enable_thinking: bool = False
    # OpenAI backend (optional — set to use GPT instead of a local HF model)
    openai_model: str = ""        # e.g. "gpt-4o"
    openai_api_key: str = ""      # if empty, falls back to OPENAI_API_KEY env var

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
            "openai_model":        self.openai_model,
            "openai_api_key":      self.openai_api_key,
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
