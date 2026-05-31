"""
tests/memory_adapter_rl/test_grpo_pipeline.py

Unit tests for the GRPO refinement pipeline (CPU-only, no heavy deps required).

Covers: config loading, the composite reward aligned to the three production
sections (FORESIGHT_PLAN / FEASIBILITY_CRITERIA / FALLBACK_STRATEGY), advantage
normalisation, rollout generation, the TRL reward function, and the trainer's
custom (no-gradient) loop + save path.
"""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock

import pytest

from embodiedbench.memory_adapter_rl.config import GRPOConfig, RLConfig, RLRewardWeights
from embodiedbench.memory_adapter_rl.grpo import (
    RolloutGroup,
    generate_candidate_group,
    make_trl_reward_fn,
    normalize_rewards,
    validate_grpo_output,
)
from embodiedbench.memory_adapter_rl.rewards import (
    REQUIRED_SECTIONS,
    compute_reward,
    score_fallback_quality,
    score_feasibility_quality,
    score_foresight_quality,
    score_format_validity,
    score_repetition_penalty,
)
from embodiedbench.memory_adapter_rl.schemas import RewardSignal
from embodiedbench.memory_adapter_rl.trainer import MemoryAdapterGRPOTrainer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GOOD_RESPONSE = (
    "<FORESIGHT_PLAN>\n"
    "- Navigate to the kitchen table.\n"
    "- Pick up the mug.\n"
    "- Navigate to the counter.\n"
    "- Place the mug on the counter.\n"
    "</FORESIGHT_PLAN>\n\n"
    "<FEASIBILITY_CRITERIA>\n"
    '- "pick up the mug": the mug must be within reach and the gripper empty.\n'
    '- "place the mug": the counter surface must be clear.\n'
    "</FEASIBILITY_CRITERIA>\n\n"
    "<FALLBACK_STRATEGY>\n"
    '- If "cannot pick / not near": navigate to table 1, then retry pick.\n'
    '- If "place blocked": navigate to the left counter, then retry place.\n'
    "</FALLBACK_STRATEGY>\n"
)

_BAD_RESPONSE = "I will pick up the mug and place it on the counter."


@pytest.fixture()
def grpo_cfg() -> RLConfig:
    cfg = RLConfig(run_name="test_grpo", algorithm="grpo", output_dir=tempfile.mkdtemp())
    cfg.grpo = GRPOConfig(num_generations=2, max_new_tokens=64, rollout_batch_size=1)
    return cfg


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_grpo_config_defaults():
    cfg = GRPOConfig()
    assert cfg.num_generations == 8
    assert cfg.kl_beta == pytest.approx(0.04)
    assert cfg.advantage_epsilon == pytest.approx(1e-8)


def test_rlconfig_with_grpo_block():
    cfg = RLConfig._from_dict({
        "algorithm": "grpo",
        "grpo": {"num_generations": 4, "kl_beta": 0.02},
        "reward_weights": {"w_foresight": 0.9},
    })
    assert isinstance(cfg.grpo, GRPOConfig)
    assert cfg.grpo.num_generations == 4
    assert cfg.reward_weights.w_foresight == pytest.approx(0.9)


def test_reward_weights_target_sections():
    w = RLRewardWeights()
    # The three guidance sections must carry the dominant quality weights.
    assert w.w_foresight > 0 and w.w_feasibility > 0 and w.w_fallback > 0
    assert w.w_foresight >= w.w_format


# ---------------------------------------------------------------------------
# Reward — schema alignment
# ---------------------------------------------------------------------------

def test_required_sections_match_production():
    assert REQUIRED_SECTIONS == [
        "FORESIGHT_PLAN", "FEASIBILITY_CRITERIA", "FALLBACK_STRATEGY",
    ]


def test_format_validity_full_vs_empty():
    assert score_format_validity(_GOOD_RESPONSE) == pytest.approx(1.0)
    assert score_format_validity("") == pytest.approx(0.0)
    assert score_format_validity(_BAD_RESPONSE) == pytest.approx(0.0)


def test_empty_sections_score_zero_format():
    # Tags present but empty must not earn format reward (no gaming).
    empty = (
        "<FORESIGHT_PLAN>\nN/A\n</FORESIGHT_PLAN>\n"
        "<FEASIBILITY_CRITERIA>\n\n</FEASIBILITY_CRITERIA>\n"
        "<FALLBACK_STRATEGY>\nNone\n</FALLBACK_STRATEGY>\n"
    )
    assert score_format_validity(empty) == pytest.approx(0.0)


def test_per_section_quality_positive_on_good_response():
    assert score_foresight_quality(_GOOD_RESPONSE) > 0.5
    assert score_feasibility_quality(_GOOD_RESPONSE) > 0.5
    assert score_fallback_quality(_GOOD_RESPONSE) > 0.5


def test_per_section_quality_zero_on_bad_response():
    assert score_foresight_quality(_BAD_RESPONSE) == 0.0
    assert score_feasibility_quality(_BAD_RESPONSE) == 0.0
    assert score_fallback_quality(_BAD_RESPONSE) == 0.0


def test_repetition_penalty_detects_degeneracy():
    repetitive = "\n".join(["- the the the the the"] * 10)
    assert score_repetition_penalty(repetitive) > 0.3
    assert score_repetition_penalty(_GOOD_RESPONSE) < 0.5


def test_good_response_outscores_bad():
    good = compute_reward(_GOOD_RESPONSE)
    bad = compute_reward(_BAD_RESPONSE)
    assert isinstance(good, RewardSignal)
    assert good.total > bad.total
    assert good.format_validity == pytest.approx(1.0)


def test_task_success_increases_reward():
    base = compute_reward(_GOOD_RESPONSE, task_success=False).total
    success = compute_reward(_GOOD_RESPONSE, task_success=True, task_progress=1.0).total
    assert success > base


def test_reward_deterministic():
    a = compute_reward(_GOOD_RESPONSE, task_success=True, task_progress=0.8)
    b = compute_reward(_GOOD_RESPONSE, task_success=True, task_progress=0.8)
    assert a.total == pytest.approx(b.total)


# ---------------------------------------------------------------------------
# Advantage normalisation
# ---------------------------------------------------------------------------

def test_normalize_rewards_zero_mean_unit_sample_var():
    adv = normalize_rewards([1.0, 2.0, 3.0, 4.0, 5.0])
    n = len(adv)
    mean = sum(adv) / n
    assert abs(mean) < 1e-6
    sample_var = sum((a - mean) ** 2 for a in adv) / (n - 1)
    assert abs(sample_var - 1.0) < 0.01


def test_normalize_rewards_single_and_empty():
    assert normalize_rewards([3.5]) == [0.0]
    assert normalize_rewards([]) == []


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def test_validate_grpo_output_good_and_bad():
    good = validate_grpo_output(_GOOD_RESPONSE)
    assert good["is_valid"] is True
    assert good["missing_sections"] == []
    assert good["format_score"] == pytest.approx(1.0)

    bad = validate_grpo_output(_BAD_RESPONSE)
    assert bad["is_valid"] is False
    assert set(bad["missing_sections"]) == set(REQUIRED_SECTIONS)


# ---------------------------------------------------------------------------
# TRL reward function
# ---------------------------------------------------------------------------

def test_trl_reward_fn_signature_and_values():
    fn = make_trl_reward_fn()
    rewards = fn(
        prompts=["p1", "p2"],
        completions=[_GOOD_RESPONSE, _BAD_RESPONSE],
    )
    assert len(rewards) == 2
    assert rewards[0] > rewards[1]


def test_trl_reward_fn_reads_episode_columns():
    fn = make_trl_reward_fn()
    no_success = fn(["p"], [_GOOD_RESPONSE])[0]
    with_success = fn(["p"], [_GOOD_RESPONSE], task_success=[True], task_progress=[1.0])[0]
    assert with_success > no_success


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------

def test_generate_candidate_group_mock():
    group = generate_candidate_group(
        prompt="test prompt", model=None, tokenizer=None, num_generations=3,
    )
    assert isinstance(group, RolloutGroup)
    assert len(group.responses) == 3
    assert len(group.rewards) == 3
    assert len(group.advantages) == 3
    assert len(group.xml_valid) == 3
    assert all(group.xml_valid)  # mock responses are fully structured


def test_rollout_group_helpers():
    group = RolloutGroup(prompt="p", responses=["a", "b", "c"], rewards=[0.1, 0.9, 0.3],
                         xml_valid=[True, False, True])
    assert group.best_response == "b"
    assert group.mean_reward == pytest.approx(0.4333, abs=1e-3)
    assert group.xml_validity_rate == pytest.approx(2 / 3)


def test_rollout_group_json_serializable():
    group = generate_candidate_group("serialize me", None, None, num_generations=2)
    restored = json.loads(group.to_json())
    assert restored["prompt"] == "serialize me"
    assert len(restored["responses"]) == 2


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def test_grpo_trainer_init(grpo_cfg):
    trainer = MemoryAdapterGRPOTrainer(
        cfg=grpo_cfg, model=MagicMock(), tokenizer=MagicMock(),
        train_prompts=["p1", "p2"],
    )
    assert trainer.cfg.algorithm == "grpo"
    assert trainer._trainer is None


def test_grpo_trainer_custom_loop(grpo_cfg):
    trainer = MemoryAdapterGRPOTrainer(
        cfg=grpo_cfg, model=None, tokenizer=None, train_prompts=["a", "b"],
    )
    trainer._use_trl = False
    trainer._trainer = None
    metrics = trainer._custom_train_loop()
    assert metrics["train/num_groups"] == 2
    assert metrics["train/xml_validity_rate"] == pytest.approx(1.0)
    assert "train/mean_reward" in metrics


def test_grpo_trainer_save(grpo_cfg):
    model = MagicMock()
    model.save_pretrained = MagicMock()
    trainer = MemoryAdapterGRPOTrainer(
        cfg=grpo_cfg, model=model, tokenizer=MagicMock(), train_prompts=[],
    )
    trainer._use_trl = False
    path = trainer.save()
    assert path == grpo_cfg.output_dir
    model.save_pretrained.assert_called_once_with(grpo_cfg.output_dir)


def test_checkpoint_manager_save_list(grpo_cfg):
    from embodiedbench.memory_adapter_rl.checkpoints import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        grpo_cfg.output_dir = tmpdir
        mgr = CheckpointManager(output_dir=tmpdir, cfg=grpo_cfg)
        model = MagicMock()
        model.save_pretrained = MagicMock()
        mgr.save(model=model, tokenizer=MagicMock(), step=10, tag="grpo")
        assert len(mgr.list_checkpoints()) >= 1


# ---------------------------------------------------------------------------
# Prompt parity with production schema
# ---------------------------------------------------------------------------

def test_rl_prompt_matches_sft_user_content():
    from embodiedbench.memory_adapter_rl.formatting import build_rl_prompt
    from embodiedbench.memory_adapter.prompts import build_adapter_user_content

    instruction = "Put a clean mug on the counter."
    memory = "[Spatial Memory] mug on table 1."
    assert build_rl_prompt(instruction, memory) == build_adapter_user_content(instruction, memory)
