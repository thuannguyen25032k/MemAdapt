"""
memory_adapter_training/collator.py

Data collator for Memory Adapter SFT training.

Each sample is a ``{"prompt": ..., "response": ...}`` pair. The collator:

1. Renders the prompt with the tokenizer's chat template
   (``apply_chat_template(..., add_generation_prompt=True)``) so the training
   input matches inference exactly (Qwen3 ``<|im_start|>`` formatting).
2. Appends the tokenized response plus an EOS token.
3. Masks the prompt tokens in ``labels`` with ``-100`` so loss is computed
   only over the assistant response.
4. Pads the batch to a uniform length.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

from embodiedbench.memory_adapter_training.formatting import to_chat_messages

logger = logging.getLogger("EB_logger")


class MemoryAdapterDataCollator:
    """Tokenise + label-mask a batch of ``{"prompt", "response"}`` samples.

    Parameters
    ----------
    tokenizer       : HF tokenizer (ideally with a chat template).
    max_seq_length  : sequences are truncated to this length.
    enable_thinking : passed to ``apply_chat_template`` for Qwen3-style models;
                      should match the inference setting (default False).
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 2048,
        enable_thinking: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.enable_thinking = enable_thinking

    # ------------------------------------------------------------------
    def _encode_prompt(self, prompt: str) -> List[int]:
        """Render + tokenize the system+user prompt up to the assistant turn."""
        messages = to_chat_messages(prompt)
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                # Tokenizers without the enable_thinking kwarg.
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
        # Fallback for tokenizers without a chat template (e.g. gpt2 in tests):
        # flatten system + user into a single string.
        text = "\n\n".join(m["content"] for m in messages)
        return self.tokenizer(text, add_special_tokens=True)["input_ids"]

    def _encode_one(self, prompt: str, response: str):
        prompt_ids = self._encode_prompt(prompt)
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            response_ids = response_ids + [eos_id]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + list(response_ids)

        # Keep the response fully supervised: when the combined length exceeds
        # max_seq_length, truncate the *prompt* from the left (dropping the
        # oldest context) rather than the response from the right. The response
        # carries the training signal, so it must never be cut.
        if len(input_ids) > self.max_seq_length:
            keep_response = len(response_ids)
            budget_for_prompt = self.max_seq_length - keep_response
            if budget_for_prompt <= 0:
                # Pathological case: the response alone exceeds the budget.
                # Keep the tail of the response so at least some signal remains.
                logger.warning(
                    "Response (%d tokens) exceeds max_seq_length (%d); the "
                    "response itself was truncated. Increase "
                    "dataset.max_seq_length.",
                    keep_response,
                    self.max_seq_length,
                )
                response_ids = response_ids[: self.max_seq_length]
                input_ids = list(response_ids)
                labels = list(response_ids)
            else:
                # Drop the oldest prompt tokens, keep the most recent context.
                logger.warning(
                    "Prompt+response (%d tokens) exceeds max_seq_length (%d); "
                    "truncating the prompt from the left to keep the response "
                    "supervised. Increase dataset.max_seq_length to avoid losing "
                    "context.",
                    len(input_ids),
                    self.max_seq_length,
                )
                prompt_ids = prompt_ids[-budget_for_prompt:]
                input_ids = prompt_ids + response_ids
                labels = [-100] * len(prompt_ids) + list(response_ids)
        return input_ids, labels

    # ------------------------------------------------------------------
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded = [self._encode_one(f["prompt"], f["response"]) for f in features]
        all_input_ids = [e[0] for e in encoded]
        all_labels = [e[1] for e in encoded]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        max_len = max(len(ids) for ids in all_input_ids)

        input_ids, attention, labels = [], [], []
        for ids, lbls in zip(all_input_ids, all_labels):
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention.append([1] * len(ids) + [0] * pad)
            labels.append(lbls + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
