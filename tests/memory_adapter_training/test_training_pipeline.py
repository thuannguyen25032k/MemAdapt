"""
tests/memory_adapter_training/test_training_pipeline.py

Unit tests for the Memory Adapter LoRA/QLoRA SFT pipeline.

All tests run without a GPU and without downloading large models. The collator
tests use the small gpt2 tokenizer (which has no chat template, exercising the
fallback path); the Qwen3 chat-template path is covered at integration time.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filtered_record(**kwargs) -> Dict[str, Any]:
    """A record in the filter_sft_targets.py output shape."""
    base = {
        "instruction": "Rinse off a ladle and move it to the table.",
        "retrieved_memory": "[Spatial Memory] Ladle in DiningTable.",
        "adapter_target": {
            "foresight_plan": ["Step 1: Find a DiningTable.", "Step 2: Pick up the Ladle."],
            "feasibility_criteria": ["\"pick\": robot must be near the Ladle."],
            "fallback_strategy": ["If \"cannot pick\": navigate to DiningTable, retry pick."],
        },
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config_creation(self):
        from embodiedbench.memory_adapter_training.config import MemoryAdapterTrainingConfig
        cfg = MemoryAdapterTrainingConfig()
        assert cfg.lora.r == 16
        assert cfg.lora.enabled is True
        assert cfg.model.enable_thinking is False

    def test_yaml_round_trip(self, tmp_path):
        from embodiedbench.memory_adapter_training.config import MemoryAdapterTrainingConfig
        cfg = MemoryAdapterTrainingConfig()
        cfg.training.seed = 999
        cfg.lora.r = 8
        cfg.dataset.val_ratio = 0.2
        yaml_path = str(tmp_path / "cfg.yaml")
        cfg.save_yaml(yaml_path)
        loaded = MemoryAdapterTrainingConfig.from_yaml(yaml_path)
        assert loaded.training.seed == 999
        assert loaded.lora.r == 8
        assert loaded.dataset.val_ratio == 0.2

    def test_is_qlora(self):
        from embodiedbench.memory_adapter_training.config import MemoryAdapterTrainingConfig
        cfg = MemoryAdapterTrainingConfig()
        cfg.model.load_in_4bit = True
        assert cfg.is_qlora() is True
        cfg.model.load_in_4bit = False
        assert cfg.is_qlora() is False

    def test_qwen3_yaml_loads(self):
        import os
        from embodiedbench.memory_adapter_training.config import MemoryAdapterTrainingConfig
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "embodiedbench", "configs", "memory_adapter_training", "qwen3_14b.yaml",
        )
        cfg = MemoryAdapterTrainingConfig.from_yaml(path)
        assert "Qwen3-14B" in cfg.model.model_name_or_path
        assert cfg.lora.enabled is True


# ---------------------------------------------------------------------------
# 2. Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_build_target_text_uses_xml_tags(self):
        from embodiedbench.memory_adapter_training.formatting import build_target_text
        target = build_target_text(
            foresight_plan=["step 1"],
            feasibility_criteria=["ok"],
            fallback_strategy=["recover"],
        )
        assert "<FORESIGHT_PLAN>" in target and "</FORESIGHT_PLAN>" in target
        assert "<FEASIBILITY_CRITERIA>" in target
        assert "<FALLBACK_STRATEGY>" in target
        # Must NOT use the colon-section format
        assert "FORESIGHT_PLAN:" not in target

    def test_format_sample_from_filtered_record(self):
        from embodiedbench.memory_adapter_training.formatting import format_sample
        pair = format_sample(_filtered_record())
        assert set(pair.keys()) == {"prompt", "response"}
        assert "Rinse off a ladle" in pair["prompt"]
        assert "<FORESIGHT_PLAN>" in pair["response"]
        assert "Pick up the Ladle" in pair["response"]

    def test_format_sample_passthrough_pair(self):
        from embodiedbench.memory_adapter_training.formatting import format_sample
        pair = format_sample({"prompt": "p", "response": "r"})
        assert pair == {"prompt": "p", "response": "r"}

    def test_parse_target_text(self):
        from embodiedbench.memory_adapter_training.formatting import (
            build_target_text,
            parse_target_text,
        )
        target = build_target_text(
            foresight_plan=["my plan"],
            feasibility_criteria=["constraints"],
            fallback_strategy=["fallback action"],
        )
        parsed = parse_target_text(target)
        assert any("my plan" in i for i in parsed["foresight_plan"])
        assert any("constraints" in i for i in parsed["feasibility_criteria"])
        assert any("fallback action" in i for i in parsed["fallback_strategy"])


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------

class TestDataset:
    def test_load_filtered_jsonl(self, tmp_path):
        from embodiedbench.memory_adapter_training.dataset import load_sft_records
        f = tmp_path / "train.jsonl"
        f.write_text(
            json.dumps(_filtered_record()) + "\n"
            + json.dumps(_filtered_record(instruction="Put a mug on the table.")) + "\n"
        )
        records = load_sft_records(str(f))
        assert len(records) == 2
        assert all(set(r) == {"prompt", "response"} for r in records)

    def test_load_prompt_response_pairs(self, tmp_path):
        from embodiedbench.memory_adapter_training.dataset import load_sft_records
        f = tmp_path / "train.jsonl"
        f.write_text(json.dumps({"prompt": "hello", "response": "world"}) + "\n")
        records = load_sft_records(str(f))
        assert records == [{"prompt": "hello", "response": "world"}]

    def test_split_train_val(self):
        from embodiedbench.memory_adapter_training.dataset import split_train_val
        recs = [{"prompt": str(i), "response": str(i)} for i in range(10)]
        train, val = split_train_val(recs, val_ratio=0.2, seed=0)
        assert len(val) == 2 and len(train) == 8

    def test_make_hf_dataset(self):
        from embodiedbench.memory_adapter_training.dataset import make_hf_dataset
        ds = make_hf_dataset([{"prompt": "a", "response": "b"}])
        assert len(ds) == 1
        assert ds[0]["prompt"] == "a" and ds[0]["response"] == "b"

    def test_missing_file_raises(self):
        from embodiedbench.memory_adapter_training.dataset import load_sft_records
        with pytest.raises(FileNotFoundError):
            load_sft_records("/nonexistent/path/train.jsonl")


# ---------------------------------------------------------------------------
# 4. Collator  (gpt2 tokenizer -> chat-template fallback path)
# ---------------------------------------------------------------------------

class TestCollator:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        return tok

    def test_basic_batch(self, tokenizer):
        from embodiedbench.memory_adapter_training.collator import MemoryAdapterDataCollator
        collator = MemoryAdapterDataCollator(tokenizer, max_seq_length=512)
        batch = collator([
            {"prompt": "p1", "response": "r1"},
            {"prompt": "p2", "response": "r2"},
        ])
        assert batch["input_ids"].shape[0] == 2
        assert "labels" in batch and "attention_mask" in batch

    def test_labels_mask_prompt(self, tokenizer):
        from embodiedbench.memory_adapter_training.collator import MemoryAdapterDataCollator
        # The shared system prompt is ~1.3k tokens, so use a realistic seq length
        # (the production config uses 4096) to keep the response within budget.
        collator = MemoryAdapterDataCollator(tokenizer, max_seq_length=2048)
        batch = collator([{"prompt": "a longer prompt here", "response": "the response"}])
        labels = batch["labels"][0].tolist()
        assert -100 in labels                       # prompt masked
        assert any(x != -100 for x in labels)       # response supervised

    def test_truncation(self, tokenizer):
        from embodiedbench.memory_adapter_training.collator import MemoryAdapterDataCollator
        collator = MemoryAdapterDataCollator(tokenizer, max_seq_length=32)
        batch = collator([{"prompt": "word " * 300, "response": "word " * 300}])
        assert batch["input_ids"].shape[1] <= 32

    def test_long_prompt_keeps_response_supervised(self, tokenizer):
        from embodiedbench.memory_adapter_training.collator import MemoryAdapterDataCollator
        # A very long prompt with a short response: the prompt must be truncated
        # (from the left) so the response stays within budget and supervised.
        collator = MemoryAdapterDataCollator(tokenizer, max_seq_length=64)
        batch = collator([{"prompt": "word " * 300, "response": "the answer"}])
        labels = batch["labels"][0].tolist()
        assert batch["input_ids"].shape[1] <= 64
        assert any(x != -100 for x in labels)       # response still supervised


# ---------------------------------------------------------------------------
# 5. Modeling
# ---------------------------------------------------------------------------

class TestModeling:
    def test_build_lora_config_defaults_to_all_linear(self):
        from embodiedbench.memory_adapter_training.modeling import build_lora_config
        from embodiedbench.memory_adapter_training.config import MemoryAdapterTrainingConfig
        cfg = MemoryAdapterTrainingConfig()
        cfg.lora.target_modules = None
        lora_config = build_lora_config(cfg)
        assert lora_config.target_modules == "all-linear"

    def test_build_lora_config_honors_explicit_modules(self):
        from embodiedbench.memory_adapter_training.modeling import build_lora_config
        from embodiedbench.memory_adapter_training.config import MemoryAdapterTrainingConfig
        cfg = MemoryAdapterTrainingConfig()
        cfg.lora.target_modules = ["q_proj", "v_proj"]
        lora_config = build_lora_config(cfg)
        assert set(lora_config.target_modules) == {"q_proj", "v_proj"}


# ---------------------------------------------------------------------------
# 6. Utils
# ---------------------------------------------------------------------------

class TestUtils:
    def test_set_seed_runs(self):
        from embodiedbench.memory_adapter_training.utils import set_seed
        set_seed(42)

    def test_setup_logging_runs(self):
        from embodiedbench.memory_adapter_training.utils import setup_logging
        setup_logging("WARNING")

    def test_count_parameters(self):
        from embodiedbench.memory_adapter_training.utils import count_parameters
        import torch.nn as nn
        total, trainable = count_parameters(nn.Linear(4, 4))
        assert total > 0 and trainable == total


# ---------------------------------------------------------------------------
# 7. Checkpoints  (mocked peft model)
# ---------------------------------------------------------------------------

class TestCheckpoints:
    def test_save_lora_adapter(self, tmp_path):
        from embodiedbench.memory_adapter_training.checkpoints import save_lora_adapter
        mock_model = MagicMock()
        save_lora_adapter(mock_model, str(tmp_path))
        mock_model.save_pretrained.assert_called_once_with(str(tmp_path))

    def test_merge_and_unload(self):
        from embodiedbench.memory_adapter_training.checkpoints import merge_and_unload
        merged = MagicMock()
        peft_model = MagicMock()
        peft_model.merge_and_unload.return_value = merged
        assert merge_and_unload(peft_model) is merged

    def test_export_merged_model(self, tmp_path):
        from embodiedbench.memory_adapter_training.checkpoints import export_merged_model
        merged = MagicMock()
        peft_model = MagicMock()
        peft_model.merge_and_unload.return_value = merged
        export_merged_model(peft_model, str(tmp_path))
        merged.save_pretrained.assert_called_once_with(str(tmp_path))
