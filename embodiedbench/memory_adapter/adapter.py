"""
embodiedbench/memory_adapter/adapter.py

MemoryAdapter — a HuggingFace (or OpenAI) LLM-backed module that transforms
retrieved MemoryContext into structured planner / critic guidance.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from typing import Any, Optional, Union

from embodiedbench.memory_adapter.config import MemoryAdapterConfig
from embodiedbench.memory_adapter.schemas import MemoryAdapterInput, MemoryAdapterOutput
from embodiedbench.memory_adapter.prompts import build_adapter_prompt, messages_to_text
from embodiedbench.memory_adapter.parsing import parse_adapter_output

logger = logging.getLogger("EB_logger")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def _resolve_dtype(dtype_str: str):
    """Resolve a string dtype name to a torch dtype or 'auto'."""
    if not _TORCH_AVAILABLE:
        return "auto"
    _MAP = {
        "float16":  torch.float16,
        "fp16":     torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16":     torch.bfloat16,
        "float32":  torch.float32,
        "fp32":     torch.float32,
        "auto":     "auto",
    }
    return _MAP.get(dtype_str.lower(), "auto")


# ---------------------------------------------------------------------------
# Formatted output builders
# ---------------------------------------------------------------------------

def build_planner_context(output: MemoryAdapterOutput) -> str:
    """Format MemoryAdapterOutput into a planner-injection string.

    Returns an empty string when the output carries no substantive content
    so callers can skip injection entirely.
    """
    if output.is_empty():
        return ""

    lines = []

    if output.foresight_plan:
        lines.append(
            "**Foresight Plan**: This plan may be helpful for you to complete the task."
        )
        for step in output.foresight_plan:
            lines.append(f"- {step}")
        lines.append("")

    # if output.fallback_strategy:
    #     lines.append(
    #         "**Fallback Strategy**: Follow these fallback strategies when an action fails:"
    #     )
    #     for rule in output.fallback_strategy:
    #         lines.append(f"- {rule}")
    
    lines.append(
        "\nAlways verify these against the live observation. If the image clearly contradicts these, trust the image."
    )
    return "\n".join(lines).strip()


def build_critic_context(output: MemoryAdapterOutput) -> str:
    """Format MemoryAdapterOutput into a critic-injection string."""
    if not output.feasibility_criteria:
        return ""

    lines = []

    lines.append("Feasibility criteria:")
    for c in output.feasibility_criteria:
        lines.append(f"- {c}")
    lines.append("")

    lines.append(
        "Reject or request replanning if the proposed action violates these criteria."
    )
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# MemoryAdapter
# ---------------------------------------------------------------------------

class MemoryAdapter:
    """
    Memory Adapter backed by a HuggingFace causal-LM (or OpenAI) that
    transforms retrieved ``MemoryContext`` into structured planner/critic
    guidance.  The adapter owns its own model and is fully decoupled from
    ``VLMPlanner`` and ``VLMCritic``.
    """

    def __init__(
        self,
        config: Union[MemoryAdapterConfig, Any] = None,
        model_name_or_path: Optional[str] = None,
    ) -> None:
        # Normalise config
        if config is None:
            self.config = MemoryAdapterConfig()
        elif isinstance(config, MemoryAdapterConfig):
            self.config = config
        else:
            self.config = MemoryAdapterConfig.from_mapping(config)

        if model_name_or_path:
            self.config.model_name_or_path = model_name_or_path

        self.tokenizer = None
        self.model = None
        self.device: str = "cpu"
        self._use_device_map: bool = False  # set True when device_map="auto" is active
        # Cache the last adapt() result so the critic can reuse it without a second inference.
        self.last_output: Optional[MemoryAdapterOutput] = None
        # Cached API client (created lazily on first generate() call).
        self._api_client = None
        # Set externally (e.g. env.log_path) to enable per-call debug logs.
        self.log_path: Optional[str] = None
        self._episode_index: int = 0   # incremented at the start of each episode
        self._log_call_index: int = 0  # resets to 0 each episode

        if self.config.enabled and self.config.model_name_or_path and not self.config.api_model:
            self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load tokenizer and model from HuggingFace."""
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for MemoryAdapter. "
                "Install with: pip install transformers"
            )

        path = self.config.model_name_or_path
        logger.info(f"[MemoryAdapter] Loading tokenizer from '{path}'")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Resolve device and whether accelerate's device_map="auto" should be used.
        # _use_device_map=True means inputs must NOT be manually moved in generate();
        # accelerate handles placement internally.
        if self.config.device == "auto":
            self.device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
            self._use_device_map = True
        else:
            self.device = self.config.device

        dtype = _resolve_dtype(self.config.torch_dtype)

        load_kwargs: dict = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype":       dtype,
        }

        # Quantization
        if self.config.load_in_4bit or self.config.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig as _BnBConfig
                load_kwargs["quantization_config"] = _BnBConfig(
                    load_in_4bit=self.config.load_in_4bit,
                    load_in_8bit=self.config.load_in_8bit,
                )
                self._use_device_map = True  # bitsandbytes requires device_map
            except ImportError:
                logger.warning(
                    "[MemoryAdapter] bitsandbytes not available; "
                    "quantization disabled."
                )

        # device_map="auto" shards across all available GPUs (required for large /
        # quantized models). For an explicit device (e.g. "cuda:0") pass it directly.
        load_kwargs["device_map"] = "auto" if self._use_device_map else self.device

        logger.info(f"[MemoryAdapter] Loading model from '{path}' "
                    f"(device={self.device}, dtype={self.config.torch_dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        logger.info("[MemoryAdapter] Model loaded.")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, messages: "list[dict]") -> str:
        """
        Run the causal-LM on the chat ``messages`` (system + user) and return
        the generated text.

        ``messages`` is a list of ``{"role", "content"}`` dicts as produced by
        ``build_adapter_prompt``. Override this method in subclasses / mocks for
        unit testing.
        """
        # --- API backend (any OpenAI-compatible server: OpenAI, lmdeploy, vLLM, …) ---
        if self.config.api_model:
            try:
                from openai import OpenAI as _OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
            if self._api_client is None:
                api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
                client_kwargs: dict = {"api_key": api_key}
                if self.config.api_base_url:
                    client_kwargs["base_url"] = self.config.api_base_url
                self._api_client = _OpenAI(**client_kwargs)

            _is_reasoning = self.config.api_model.startswith(("o1", "o3", "o4"))
            _create_kwargs: dict = {
                "model": self.config.api_model,
                "messages": messages,
                "max_completion_tokens": self.config.max_new_tokens,
            }
            if not _is_reasoning:
                _create_kwargs["temperature"] = self.config.temperature
            resp = self._api_client.chat.completions.create(**_create_kwargs)
            return resp.choices[0].message.content or ""

        # --- Local HuggingFace backend ---
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "MemoryAdapter model is not loaded. "
                "Ensure config.enabled=True and model_name_or_path is set."
            )

        # Use apply_chat_template when available (required for Qwen3 and most
        # instruction-tuned models) so the system/user turns are properly
        # formatted and enable_thinking is respected.
        if self.tokenizer.chat_template is not None:
            try:
                chat_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.config.enable_thinking,
                )
            except TypeError:
                # Older tokenizers may not support enable_thinking; fall back.
                chat_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self.tokenizer(chat_text, return_tensors="pt", truncation=False)
        else:
            # No chat template: flatten the messages into a single string.
            chat_text = messages_to_text(messages)
            inputs = self.tokenizer(chat_text, return_tensors="pt", truncation=False)

        if _TORCH_AVAILABLE:
            if self._use_device_map:
                # device_map="auto" shards layers across GPUs but does NOT move
                # inputs automatically. Move inputs to the device that holds the
                # first layer (embed_tokens) so there is no device mismatch.
                first_device = next(self.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
            elif self.device.startswith("cuda"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs: dict = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample":      self.config.do_sample,
        }
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"]       = self.config.top_p
        else:
            # Explicitly unset sampling params that the model's generation_config.json
            # may carry as defaults. Without this, transformers warns that temperature/
            # top_p/top_k are invalid flags for greedy decoding.
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"]       = None
            gen_kwargs["top_k"]       = None

        with (torch.no_grad() if _TORCH_AVAILABLE else nullcontext()):
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Strip Qwen3-style thinking blocks if present (e.g. when enable_thinking=True
        # or the model emits them regardless).
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].lstrip("\n")

        return text

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, adapter_input: MemoryAdapterInput) -> "list[dict]":
        return build_adapter_prompt(adapter_input, self.config)

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _save_log(
        self,
        adapter_input: MemoryAdapterInput,
        messages: "list[dict]",
        raw: str,
        output: MemoryAdapterOutput,
    ) -> None:
        """
        Write one JSON record to ``{log_path}/adapter_logs/call_{N:04d}.json``.
        Silently skipped when ``self.log_path`` is None.
        """
        if not self.log_path:
            return
        try:
            log_dir = os.path.join(
                self.log_path, "adapter_logs",
                f"episode_{self._episode_index:04d}",
            )
            os.makedirs(log_dir, exist_ok=True)
            filename = f"call_{self._log_call_index:04d}.json"
            record = {
                "call_index": self._log_call_index,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "input": adapter_input.to_dict(),
                "messages": messages,
                "raw_output": raw,
                "parsed_output": output.to_dict(),
            }
            with open(os.path.join(log_dir, filename), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            self._log_call_index += 1
        except Exception as e:
            logger.debug(f"[MemoryAdapter] _save_log failed: {e}")

    # ------------------------------------------------------------------
    # Main adapt() entry point
    # ------------------------------------------------------------------

    def adapt(self, adapter_input: MemoryAdapterInput) -> MemoryAdapterOutput:
        """
        Transform retrieved memory into structured guidance.

        Returns an empty MemoryAdapterOutput if the adapter is disabled.
        Never raises — errors are captured into parse_error.
        """
        if not self.config.enabled:
            return MemoryAdapterOutput(
                parse_error="MemoryAdapter is disabled (config.enabled=False)."
            )

        try:
            messages = self.build_prompt(adapter_input)
            raw      = self.generate(messages)
            output   = parse_adapter_output(raw)
        except Exception as e:
            logger.warning(f"[MemoryAdapter] adapt() failed: {e}")
            return MemoryAdapterOutput(raw_output="", parse_error=str(e))

        # Save debug log before caching
        self._save_log(adapter_input, messages, raw, output)
        self.last_output = output
        
        return output

    def reset_last_output(self) -> None:
        """Clear the cached adapt() result. Call at the start of each episode."""
        self.last_output = None
        self._episode_index += 1
        self._log_call_index = 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Delete model/tokenizer and free CUDA memory if available."""
        self.model     = None
        self.tokenizer = None
        self._api_client = None
        if _TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("[MemoryAdapter] Model unloaded.")
