"""
embodiedbench/memory_adapter/adapter.py

MemoryAdapter — an independent Hugging Face LLM-backed module that transforms
retrieved MemoryContext into structured planner / critic guidance.

Design
------
- The adapter owns its own tokenizer + model (loaded from HuggingFace).
- generate() is a distinct method so tests can subclass and override it
  without touching HuggingFace at all.
- planner / critic context strings are built on-demand by callers via
  build_planner_context() / build_critic_context().
- The adapter is self-contained: no coupling to VLMPlanner or VLMCritic.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional, Union

from embodiedbench.memory_adapter.config import MemoryAdapterConfig
from embodiedbench.memory_adapter.schemas import MemoryAdapterInput, MemoryAdapterOutput
from embodiedbench.memory_adapter.prompts import build_adapter_prompt
from embodiedbench.memory_adapter.parsing import parse_adapter_output

logger = logging.getLogger("EB_logger")

# ---------------------------------------------------------------------------
# Optional HuggingFace / torch imports
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
            "**Foresight Plan**: Use this foresight plan as an initial hypothesis plan for the task."
        )
        for step in output.foresight_plan:
            lines.append(f"- {step}")
        lines.append("")

    if output.fallback_strategy:
        lines.append(
            "**Fallback Strategy**: Follow these fallback strategies when an action fails:"
        )
        for rule in output.fallback_strategy:
            lines.append(f"- {rule}")
    
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
        # Cache the last adapt() result so the critic can reuse it without a second inference.
        self.last_output: Optional[MemoryAdapterOutput] = None
        # Cached OpenAI client (created lazily on first generate() call).
        self._openai_client = None
        # Set externally (e.g. env.log_path) to enable per-call debug logs.
        self.log_path: Optional[str] = None
        self._episode_index: int = 0   # incremented at the start of each episode
        self._log_call_index: int = 0  # resets to 0 each episode

        if self.config.enabled and self.config.model_name_or_path and not self.config.openai_model:
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

        # Resolve device
        if _TORCH_AVAILABLE and self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device if self.config.device != "auto" else "cpu"

        dtype = _resolve_dtype(self.config.torch_dtype)

        load_kwargs: dict = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype":       dtype,
        }

        # Quantization
        if self.config.load_in_4bit or self.config.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig as _BnBConfig
                bnb_cfg = _BnBConfig(
                    load_in_4bit=self.config.load_in_4bit,
                    load_in_8bit=self.config.load_in_8bit,
                )
                load_kwargs["quantization_config"] = bnb_cfg
            except ImportError:
                logger.warning(
                    "[MemoryAdapter] bitsandbytes not available; "
                    "quantization disabled."
                )

        if self.device not in ("auto", "cpu") and not (
            self.config.load_in_4bit or self.config.load_in_8bit
        ):
            load_kwargs["device_map"] = self.device
        elif self.config.device == "auto":
            load_kwargs["device_map"] = "auto"

        logger.info(f"[MemoryAdapter] Loading model from '{path}' "
                    f"(device={self.device}, dtype={self.config.torch_dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        logger.info("[MemoryAdapter] Model loaded.")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Run the causal-LM on the prompt and return the generated text.

        Override this method in subclasses / mocks for unit testing.
        """
        # --- OpenAI backend ---
        if self.config.openai_model:
            try:
                from openai import OpenAI as _OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI backend. "
                    "Install with: pip install openai"
                )
            # Cache client on first use to avoid recreating it every call
            if self._openai_client is None:
                api_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
                self._openai_client = _OpenAI(api_key=api_key)

            resp = self._openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            return resp.choices[0].message.content or ""

        # --- Local HuggingFace backend ---
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "MemoryAdapter model is not loaded. "
                "Ensure config.enabled=True and model_name_or_path is set."
            )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,  # Let the model handle long inputs with its context window (and risk truncation if it exceeds it).
        )


        if _TORCH_AVAILABLE and self.device.startswith("cuda"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs: dict = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample":      self.config.do_sample,
        }
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"]       = self.config.top_p

        with (torch.no_grad() if _TORCH_AVAILABLE else _nullctx()):
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, adapter_input: MemoryAdapterInput) -> str:
        return build_adapter_prompt(adapter_input, self.config)

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _save_log(
        self,
        adapter_input: MemoryAdapterInput,
        prompt: str,
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
                "prompt": prompt,
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
            prompt = self.build_prompt(adapter_input)
            raw    = self.generate(prompt)
            output = parse_adapter_output(raw)
        except Exception as e:
            logger.warning(f"[MemoryAdapter] adapt() failed: {e}")
            return MemoryAdapterOutput(raw_output="", parse_error=str(e))

        # Save debug log before caching
        self._save_log(adapter_input, prompt, raw, output)
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
        if _TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("[MemoryAdapter] Model unloaded.")


# ---------------------------------------------------------------------------
# Tiny context-manager stub for when torch is unavailable
# ---------------------------------------------------------------------------

class _nullctx:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
