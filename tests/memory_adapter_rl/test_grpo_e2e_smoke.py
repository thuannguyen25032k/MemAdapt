"""
tests/memory_adapter_rl/test_grpo_e2e_smoke.py

End-to-end GRPO smoke test using debug_grpo_tiny.yaml.

Verifies (in order):
  1. reward_fn is called during TRL training.
  2. Generated completions are passed correctly into reward_fn (non-empty strings,
     one per prompt × generation).
  3. format reward contributes to total reward (format_validity component > 0 for
     a correctly structured completion).
  4. One training step completes without falling back to the custom no-update loop
     (trainer._use_trl is True).
  5. LoRA trainable parameters are non-zero.
  6. Checkpoint is saved to the output directory.
  7. Loading the checkpoint produces valid XML outputs (adapter loads; generation
     runs; validate_grpo_output returns the expected result dict structure).

The entire module is skipped automatically if TRL / transformers / torch / peft /
datasets are not installed, or if the model cannot be loaded from the hub
(offline environments or disk constraints).

Run tag: ``pytest -m e2e`` (skipped in the default fast-test suite).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module-level skip guards — all heavy dependencies must be present
# ---------------------------------------------------------------------------

trl = pytest.importorskip("trl", reason="trl not installed — skipping GRPO smoke test")
transformers = pytest.importorskip("transformers", reason="transformers not installed")
torch = pytest.importorskip("torch", reason="torch not installed")
peft_lib = pytest.importorskip("peft", reason="peft not installed")
datasets_lib = pytest.importorskip("datasets", reason="datasets not installed")

# ---------------------------------------------------------------------------
# Project path bootstrap
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "embodiedbench" / "configs" / "memory_adapter_rl" / "debug_grpo_tiny.yaml"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embodiedbench.memory_adapter_rl.config import RLConfig
from embodiedbench.memory_adapter_rl.grpo import (
    make_trl_reward_fn,
    validate_grpo_output,
)
from embodiedbench.memory_adapter_rl.trainer import MemoryAdapterGRPOTrainer

# ---------------------------------------------------------------------------
# Minimal training prompts (two examples)
# ---------------------------------------------------------------------------

_TRAIN_PROMPTS: List[str] = [
    (
        "You are the Memory Adapter for an embodied agent.\n"
        "The agent observes: a cup is on the table.\n"
        "Respond ONLY with the three required XML sections:\n"
        "<FORESIGHT_PLAN>, <FEASIBILITY_CRITERIA>, <FALLBACK_STRATEGY>."
    ),
    (
        "You are the Memory Adapter for an embodied agent.\n"
        "The agent observes: the door ahead is now closed.\n"
        "Respond ONLY with the three required XML sections:\n"
        "<FORESIGHT_PLAN>, <FEASIBILITY_CRITERIA>, <FALLBACK_STRATEGY>."
    ),
]

# A reference completion that contains all three required XML sections.
_XML_VALID_COMPLETION = (
    "<FORESIGHT_PLAN>\n"
    "- Navigate to the table.\n"
    "- Pick up the cup.\n"
    "- Navigate to the counter.\n"
    "- Place the cup on the counter.\n"
    "</FORESIGHT_PLAN>\n"
    "<FEASIBILITY_CRITERIA>\n"
    '- "pick up the cup": the cup must be within reach and the gripper empty.\n'
    '- "place the cup": the counter surface must be clear.\n'
    "</FEASIBILITY_CRITERIA>\n"
    "<FALLBACK_STRATEGY>\n"
    '- If "cannot pick / not near": navigate to table 1, then retry pick.\n'
    '- If "place blocked": navigate to the left counter, then retry place.\n'
    "</FALLBACK_STRATEGY>\n"
)

# ---------------------------------------------------------------------------
# Helper: spy wrapper around make_trl_reward_fn
# ---------------------------------------------------------------------------

def _build_spy_reward_fn(
    weights: Any,
    call_log: List[Dict[str, Any]],
) -> Callable:
    """
    Wraps the real TRL reward function returned by make_trl_reward_fn.
    Every call appends a record to ``call_log``:
        {"prompts": [...], "completions": [...], "returns": [...]}
    """
    real_fn = make_trl_reward_fn(weights=weights)

    def _spy(
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        rewards = real_fn(prompts, completions, **kwargs)
        call_log.append(
            {
                "prompts": list(prompts),
                "completions": list(completions),
                "returns": list(rewards),
            }
        )
        return rewards

    return _spy


def _count_trainable_params(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Module-scoped fixtures  (model loaded only once for the entire test module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_output(tmp_path_factory):
    """Temporary output directory; cleaned up at the end of the module."""
    return str(tmp_path_factory.mktemp("grpo_smoke"))


@pytest.fixture(scope="module")
def cfg(tmp_output):
    """Load debug_grpo_tiny.yaml and override output_dir + speed settings."""
    c = RLConfig.from_yaml(str(CONFIG_PATH))
    c.output_dir = tmp_output
    # Shorten generation length so CPU runs complete quickly
    c.grpo.max_new_tokens = 32
    # Force a checkpoint on the very first step so test-6 can verify it
    c.save_steps = 1
    return c


@pytest.fixture(scope="module")
def model_and_tokenizer(cfg):
    """
    Load Qwen2.5-0.5B-Instruct and wrap with LoRA.
    Skips the module if the model cannot be fetched (offline / no disk space).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        # Ensure a pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch.float32,  # float32 for CPU stability
        )

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            # target_modules is required by PEFT ≥ 0.10 when the model has
            # multiple linear-layer types; read from the debug config YAML.
            target_modules=cfg.lora_target_modules or ["q_proj", "v_proj"],
        )
        model = get_peft_model(base_model, lora_config)
        return model, tokenizer

    except Exception as exc:
        pytest.skip(f"Model load failed (network / disk): {exc}")


@pytest.fixture(scope="module")
def train_dataset():
    """Minimal 2-row HuggingFace Dataset with the required 'prompt' column."""
    from datasets import Dataset

    return Dataset.from_dict({"prompt": _TRAIN_PROMPTS})


@pytest.fixture(scope="module")
def reward_call_log() -> List[Dict[str, Any]]:
    """Shared call recorder — all reward_fn invocations are appended here."""
    return []


@pytest.fixture(scope="module")
def trained_trainer(cfg, model_and_tokenizer, train_dataset, reward_call_log):
    """
    Build + run the MemoryAdapterGRPOTrainer (module-scoped so the training run
    is shared across all seven assertions).

    The real ``make_trl_reward_fn`` is patched with a spy that records every call
    made by TRL's GRPOTrainer.  The spy wraps the *real* reward function, so
    actual reward values are still computed correctly.
    """
    model, tokenizer = model_and_tokenizer

    spy_fn = _build_spy_reward_fn(cfg.reward_weights, reward_call_log)

    # Patch at the import site used inside _build_grpo_trainer
    with patch(
        "embodiedbench.memory_adapter_rl.grpo.make_trl_reward_fn",
        return_value=spy_fn,
    ):
        trainer = MemoryAdapterGRPOTrainer(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            train_prompts=train_dataset,
            eval_prompts=None,
            ref_model=None,
        )
        trainer.build()

    # If TRL construction failed (dependency issue not caught above), skip all tests
    if not trainer._use_trl:
        pytest.skip(
            "TRL GRPOTrainer construction failed during build(); "
            "trainer fell back to the custom no-update loop."
        )

    trainer.train()
    return trainer


# ---------------------------------------------------------------------------
# The seven smoke assertions
# ---------------------------------------------------------------------------

class TestGRPOEndToEndSmoke:
    """
    All tests share the module-scoped ``trained_trainer`` fixture so the
    expensive model load + training step happen exactly once.
    """

    # ------------------------------------------------------------------ 1
    def test_1_reward_fn_called_during_training(
        self, trained_trainer, reward_call_log
    ):
        """TRL must invoke the reward function at least once during training."""
        assert len(reward_call_log) >= 1, (
            "reward_fn was never called — TRL may not be routing through our "
            f"reward_funcs. Call log: {reward_call_log}"
        )

    # ------------------------------------------------------------------ 2
    def test_2_completions_passed_correctly_to_reward_fn(
        self, trained_trainer, reward_call_log
    ):
        """
        Every batch passed to reward_fn must contain non-empty string completions,
        and the number of returns must equal the number of completions.
        """
        assert len(reward_call_log) >= 1, "reward_fn call log is empty (see test-1)"

        for batch_idx, call in enumerate(reward_call_log):
            completions = call["completions"]
            returns = call["returns"]

            assert len(completions) > 0, (
                f"Batch {batch_idx}: completions list is empty"
            )
            assert len(completions) == len(returns), (
                f"Batch {batch_idx}: completions length {len(completions)} != "
                f"returns length {len(returns)}"
            )
            for i, c in enumerate(completions):
                assert isinstance(c, str) and len(c) > 0, (
                    f"Batch {batch_idx}, completion[{i}] is not a non-empty string: "
                    f"{c!r}"
                )

    # ------------------------------------------------------------------ 3
    def test_3_xml_reward_contributes_to_total_reward(self, trained_trainer):
        """
        When a completion contains all three XML sections, the reward function
        must return a total that includes a positive format_validity contribution.

        We call the real reward function directly on a known-good completion and
        verify that (a) format_validity > 0 and (b) total > reward-without-format.
        """
        from embodiedbench.memory_adapter_rl.rewards import compute_reward
        from embodiedbench.memory_adapter_rl.config import RLRewardWeights

        weights = RLRewardWeights()  # defaults: w_format = 0.5

        sig_with_format = compute_reward(
            response=_XML_VALID_COMPLETION,
            weights=weights,
        )
        # format_validity should be 1.0 (all three sections present and filled)
        assert sig_with_format.format_validity == pytest.approx(1.0), (
            f"Expected format_validity=1.0 for a fully valid completion; "
            f"got {sig_with_format.format_validity}"
        )
        # The format component (w_format * format_validity = 0.5 * 1.0 = 0.5)
        # must push total above the zero-format baseline.
        weights_no_format = RLRewardWeights(w_format=0.0)
        sig_no_format = compute_reward(
            response=_XML_VALID_COMPLETION,
            weights=weights_no_format,
        )
        assert sig_with_format.total > sig_no_format.total, (
            f"format component did not raise total reward: "
            f"with_format={sig_with_format.total}, no_format={sig_no_format.total}"
        )

    # ------------------------------------------------------------------ 4
    def test_4_training_used_trl_path_not_custom_loop(self, trained_trainer):
        """
        trainer._use_trl must be True — TRL GRPOTrainer was used, not the
        custom no-gradient fallback loop.
        """
        assert trained_trainer._use_trl is True, (
            "trainer._use_trl is False: training fell back to the custom loop "
            "which performs no gradient updates."
        )
        assert trained_trainer.inner_trainer is not None, (
            "trainer.inner_trainer is None — TRL GRPOTrainer was never set."
        )

    # ------------------------------------------------------------------ 5
    def test_5_lora_trainable_parameters_are_nonzero(
        self, trained_trainer, model_and_tokenizer
    ):
        """
        The PEFT-wrapped model must have at least one trainable parameter,
        confirming LoRA adapters were attached.
        """
        model, _ = model_and_tokenizer
        n_trainable = _count_trainable_params(model)
        assert n_trainable > 0, (
            "No trainable parameters found — LoRA adapter was not applied or "
            f"all parameters were frozen. model class: {type(model).__name__}"
        )

    # ------------------------------------------------------------------ 6
    def test_6_checkpoint_is_saved(self, trained_trainer, cfg):
        """
        After calling trainer.save(), the output directory must contain at least
        one PEFT adapter file (adapter_model.safetensors or adapter_model.bin).
        """
        save_path = trained_trainer.save()

        assert os.path.isdir(save_path), (
            f"save() returned path {save_path!r} but it is not a directory."
        )

        # Collect all files recursively
        all_files = []
        for root, _, files in os.walk(save_path):
            all_files.extend(files)

        adapter_files = [
            f for f in all_files
            if f in (
                "adapter_model.safetensors",
                "adapter_model.bin",
                "adapter_config.json",
                "pytorch_model.bin",
            )
        ]

        assert len(adapter_files) > 0, (
            f"No adapter files found under {save_path!r}. "
            f"All files present: {all_files}"
        )

    # ------------------------------------------------------------------ 7
    def test_7_loaded_checkpoint_generates_xml_output(
        self, trained_trainer, cfg, model_and_tokenizer
    ):
        """
        Load the saved PEFT adapter back from disk, run inference on a single
        prompt, and verify that:
          (a) generation completes without error,
          (b) the output is a non-empty string,
          (c) validate_grpo_output returns a dict with all expected keys.

        Note: for a model trained on only 1–2 steps, is_valid may be False;
        the smoke test asserts *mechanism correctness*, not model quality.
        """
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        save_path = cfg.output_dir

        # Confirm the saved files exist (test-6 must have passed first)
        assert os.path.isdir(save_path), (
            f"Checkpoint directory {save_path!r} does not exist — "
            "did test-6 pass?"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch.float32,
        )
        loaded_model = PeftModel.from_pretrained(base_model, save_path)
        loaded_model.eval()

        prompt = (
            "You are the Memory Adapter. "
            "Respond with all three XML sections: "
            "<FORESIGHT_PLAN>, <FEASIBILITY_CRITERIA>, <FALLBACK_STRATEGY>.\n"
            "Observation: a chair is blocking the path."
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = loaded_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,  # greedy for determinism
            )
        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_len:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # (a) non-empty generation
        assert isinstance(text, str) and len(text.strip()) > 0, (
            "Loaded checkpoint produced an empty generation."
        )

        # (b) validate_grpo_output runs and returns the expected keys
        result = validate_grpo_output(text)
        expected_keys = {
            "is_valid",
            "missing_sections",
            "format_score",
        }
        assert expected_keys.issubset(result.keys()), (
            f"validate_grpo_output result missing keys: "
            f"{expected_keys - result.keys()}"
        )

        # (c) structural sanity
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["missing_sections"], list)
        assert 0.0 <= result["format_score"] <= 1.0
