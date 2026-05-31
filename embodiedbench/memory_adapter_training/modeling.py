"""
memory_adapter_training/modeling.py

Utilities to load a base causal-LM, optionally quantise it (QLoRA), and wrap
it with a PEFT LoRA adapter ready for SFT.

All heavy imports (torch, transformers, peft) are deferred to function bodies
so that the rest of the package can be imported without a GPU or large models.
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger("EB_logger")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_base_model(cfg):  # noqa: ANN001
    """
    Load a base causal-LM from *cfg* (MemoryAdapterTrainingConfig).

    Supports
    --------
    * full precision (default)
    * bf16 / fp16
    * 4-bit QLoRA via bitsandbytes (cfg.model.load_in_4bit)
    * 8-bit via bitsandbytes (cfg.model.load_in_8bit)

    Returns
    -------
    model : PreTrainedModel
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_cfg = cfg.model
    model_name = model_cfg.model_name_or_path

    # ---- dtype ----
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(model_cfg.torch_dtype, torch.float32)

    # ---- quantisation config ----
    bnb_config = None
    if model_cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch_dtype = None  # BnB handles dtype
    elif model_cfg.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = None

    # ---- kwargs ----
    kwargs = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=model_cfg.trust_remote_code,
        device_map="auto",
    )
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["dtype"] = torch_dtype  # `torch_dtype` renamed to `dtype` in transformers ≥ 4.50

    if model_cfg.attn_implementation:
        kwargs["attn_implementation"] = model_cfg.attn_implementation
    elif model_cfg.use_flash_attention:
        # use_flash_attention=True is an alias for attn_implementation="flash_attention_2"
        kwargs["attn_implementation"] = "flash_attention_2"

    logger.info(f"[Modeling] Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    model.config.use_cache = False  # needed for gradient checkpointing
    return model


def load_tokenizer(cfg):  # noqa: ANN001
    """Load the tokenizer, ensuring a pad token exists."""
    from transformers import AutoTokenizer

    tok_path = cfg.resolve_tokenizer_path()
    logger.info(f"[Modeling] Loading tokenizer: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def sync_model_tokenizer_config(model, tokenizer) -> None:
    """
    Align model.config and model.generation_config with the tokenizer's
    special-token IDs to suppress the "tokenizer has new PAD/BOS/EOS tokens"
    warning emitted by ``Trainer.__init__``.
    """
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id


def build_lora_config(cfg, model=None):  # noqa: ANN001
    """
    Construct a ``peft.LoraConfig`` from *cfg.lora*.

    When ``cfg.lora.target_modules`` is None, PEFT's ``"all-linear"`` is used,
    which auto-targets every linear layer of the model (architecture-agnostic).
    """
    from peft import LoraConfig, TaskType  # type: ignore

    lora_cfg = cfg.lora
    target_modules = lora_cfg.target_modules or "all-linear"

    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def apply_lora(model, lora_config):  # noqa: ANN001
    """Wrap *model* with a PEFT LoRA adapter."""
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore

    # Prepare for kbit if quantised
    if getattr(model, "is_quantized", False) or getattr(
        getattr(model, "config", None), "quantization_config", None
    ):
        logger.info("[Modeling] Preparing quantised model for k-bit training")
        model = prepare_model_for_kbit_training(model)
    else:
        # Required when gradient_checkpointing=True on a non-quantised model:
        # ensures the input embeddings keep a grad_fn so the backward pass can
        # reach the LoRA parameters through the checkpointed activations.
        model.enable_input_require_grads()

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def get_trainable_model(cfg) -> Tuple:  # noqa: ANN001
    """
    Full pipeline: load base model + tokenizer, optionally apply LoRA.

    Returns
    -------
    (model, tokenizer)
    """
    if cfg.model.use_unsloth:
        return load_unsloth_model(cfg)

    model = load_base_model(cfg)
    tokenizer = load_tokenizer(cfg)

    # Align model config with tokenizer special tokens before LoRA wrapping so
    # HF Trainer does not emit "tokenizer has new PAD/BOS/EOS tokens" warning.
    sync_model_tokenizer_config(model, tokenizer)

    if cfg.lora.enabled:
        lora_config = build_lora_config(cfg, model)
        model = apply_lora(model, lora_config)

    return model, tokenizer


def load_unsloth_model(cfg) -> Tuple:  # noqa: ANN001
    """
    Load + LoRA-wrap a model via Unsloth's ``FastLanguageModel``.

    Unsloth must be imported before transformers/peft to apply its patches, so
    the import lives here. The returned model/tokenizer are HF-compatible and
    plug directly into the existing Trainer / collator / checkpoint code.
    """
    import torch
    from unsloth import FastLanguageModel  # type: ignore

    model_cfg = cfg.model
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": None,
    }
    dtype = dtype_map.get(model_cfg.torch_dtype, None)

    logger.info(f"[Modeling] Loading Unsloth model: {model_cfg.model_name_or_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg.model_name_or_path,
        max_seq_length=cfg.dataset.max_seq_length,
        dtype=dtype,
        load_in_4bit=model_cfg.load_in_4bit,
        load_in_8bit=model_cfg.load_in_8bit,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if cfg.lora.enabled:
        lora_cfg = cfg.lora
        target_modules = lora_cfg.target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        logger.info("[Modeling] Applying Unsloth LoRA adapter")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            bias=lora_cfg.bias,
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth" if cfg.training.gradient_checkpointing else False,
            random_state=cfg.training.seed,
        )

    sync_model_tokenizer_config(model, tokenizer)
    return model, tokenizer
