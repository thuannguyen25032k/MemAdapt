"""
tests/evaluation/test_evaluation_harness.py

Unit tests for Step-29 evaluation harness.

All tests are mock-only — no simulator, no GPU, no real model download.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_episode_dict(
    success: bool = True,
    num_steps: int = 10,
    num_invalid: int = 1,
    planner_steps: int = 2,
    runtime: float = 5.0,
    stale_detected: bool = False,
    stale_recovered: bool = False,
    adapter_calls: int = 0,
) -> Dict[str, Any]:
    return {
        "task_success": int(success),
        "task_progress": 1.0 if success else 0.3,
        "num_steps": num_steps,
        "num_invalid_actions": num_invalid,
        "planner_steps": planner_steps,
        "episode_elapsed_seconds": runtime,
        "memory_metrics": {
            "planner_calls": 3,
            "critic_calls": 1,
            "adapter_calls": adapter_calls,
            "adapter_fallbacks": 0,
            "stale_memory_detected": int(stale_detected),
            "stale_memory_recovered": int(stale_recovered),
        },
    }


def _make_episode_result(**kwargs):
    from embodiedbench.evaluation.schemas import EpisodeResult
    return EpisodeResult(**kwargs)


def _make_experiment_result(
    benchmark: str = "eb_alfred",
    mode: str = "adapted_memory",
    n_episodes: int = 5,
    success_rate: float = 0.8,
):
    from embodiedbench.evaluation.schemas import ExperimentConfig, ExperimentResult
    from embodiedbench.evaluation.metrics import episode_result_from_evaluator_dict, compute_aggregate_metrics

    cfg = ExperimentConfig(
        benchmark=benchmark, mode=mode,
        num_episodes=n_episodes, seed=42,
    )
    n_success = int(n_episodes * success_rate)
    episodes = [
        episode_result_from_evaluator_dict(
            _make_episode_dict(success=(i < n_success)),
            benchmark=benchmark, mode=mode,
            episode_id=f"ep_{i:04d}",
        )
        for i in range(n_episodes)
    ]
    agg = compute_aggregate_metrics(episodes, label=mode, benchmark=benchmark, mode=mode)
    return ExperimentResult(config=cfg, episodes=episodes, summary=agg.to_dict())


# ---------------------------------------------------------------------------
# 1. Experiment schema serialisation
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_experiment_config_round_trip(self):
        from embodiedbench.evaluation.schemas import ExperimentConfig
        cfg = ExperimentConfig(
            benchmark="eb_alfred", mode="adapted_memory",
            num_episodes=20, seed=99,
        )
        d = cfg.to_dict()
        loaded = ExperimentConfig.from_dict(d)
        assert loaded.benchmark == "eb_alfred"
        assert loaded.seed == 99

    def test_experiment_config_json_round_trip(self):
        from embodiedbench.evaluation.schemas import ExperimentConfig
        cfg = ExperimentConfig(mode="baseline", num_episodes=5)
        s = cfg.to_json()
        loaded = ExperimentConfig.from_json(s)
        assert loaded.mode == "baseline"
        assert loaded.num_episodes == 5

    def test_episode_result_round_trip(self):
        from embodiedbench.evaluation.schemas import EpisodeResult
        ep = EpisodeResult(
            episode_id="ep_001", benchmark="eb_alfred", mode="raw_memory",
            task_success=True, num_steps=12, num_replans=2,
        )
        d = ep.to_dict()
        loaded = EpisodeResult.from_dict(d)
        assert loaded.task_success is True
        assert loaded.num_replans == 2

    def test_aggregate_metrics_round_trip(self):
        from embodiedbench.evaluation.schemas import AggregateMetrics
        agg = AggregateMetrics(
            success_rate=0.75, avg_replans=1.5,
            stale_memory_recovery_rate=0.6,
        )
        d = agg.to_dict()
        loaded = AggregateMetrics.from_dict(d)
        assert loaded.success_rate == 0.75
        assert loaded.stale_memory_recovery_rate == 0.6

    def test_experiment_result_to_dict_contains_episodes(self):
        result = _make_experiment_result(n_episodes=3)
        d = result.to_dict()
        assert len(d["episodes"]) == 3
        assert "summary" in d

    def test_all_outputs_json_serializable(self):
        result = _make_experiment_result()
        d = result.to_dict()
        # Should not raise
        s = json.dumps(d)
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# 2. Runner works with mock episode_fn
# ---------------------------------------------------------------------------

class TestRunner:
    def _mock_episodes(self, config):
        return [_make_episode_dict(success=(i % 2 == 0)) for i in range(4)]

    def test_run_experiment_basic(self, tmp_path):
        from embodiedbench.evaluation.schemas import ExperimentConfig
        from embodiedbench.evaluation.runner import run_experiment

        cfg = ExperimentConfig(
            benchmark="eb_alfred", mode="baseline",
            num_episodes=4, output_dir=str(tmp_path),
            save_episode_jsons=True,
        )
        result = run_experiment(cfg, episode_fn=self._mock_episodes)
        assert len(result.episodes) == 4
        assert "success_rate" in result.summary

    def test_run_experiment_creates_json(self, tmp_path):
        from embodiedbench.evaluation.schemas import ExperimentConfig
        from embodiedbench.evaluation.runner import run_experiment

        cfg = ExperimentConfig(
            experiment_id="test_run",
            benchmark="eb_alfred", mode="raw_memory",
            output_dir=str(tmp_path), save_episode_jsons=True,
        )
        run_experiment(cfg, episode_fn=self._mock_episodes)
        out_file = os.path.join(str(tmp_path), "test_run_result.json")
        assert os.path.isfile(out_file)
        with open(out_file) as fh:
            d = json.load(fh)
        assert d["config"]["mode"] == "raw_memory"

    def test_run_experiment_invalid_benchmark_raises(self, tmp_path):
        from embodiedbench.evaluation.schemas import ExperimentConfig
        from embodiedbench.evaluation.runner import run_experiment
        cfg = ExperimentConfig(benchmark="unknown_bench", output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown benchmark"):
            run_experiment(cfg, episode_fn=self._mock_episodes)

    def test_run_experiment_invalid_mode_raises(self, tmp_path):
        from embodiedbench.evaluation.schemas import ExperimentConfig
        from embodiedbench.evaluation.runner import run_experiment
        cfg = ExperimentConfig(
            benchmark="eb_alfred", mode="rl_ppo",
            output_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match="Unknown mode"):
            run_experiment(cfg, episode_fn=self._mock_episodes)

    def test_adapter_modes_handled(self, tmp_path):
        from embodiedbench.evaluation.schemas import ExperimentConfig
        from embodiedbench.evaluation.runner import run_experiment

        for mode in [
            "adapted_memory",
            "adapted_memory_planner_only",
            "adapted_memory_critic_only",
            "adapted_memory_planner_critic",
        ]:
            cfg = ExperimentConfig(
                benchmark="eb_alfred", mode=mode,
                output_dir=str(tmp_path), save_episode_jsons=False,
            )
            # episode_fn bypass — no real adapter loading needed
            result = run_experiment(cfg, episode_fn=self._mock_episodes)
            assert result.config.mode == mode


# ---------------------------------------------------------------------------
# 3. Aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_aggregate_results(self):
        from embodiedbench.evaluation.aggregators import aggregate_results
        r1 = _make_experiment_result(n_episodes=5, success_rate=0.8)
        r2 = _make_experiment_result(n_episodes=5, success_rate=0.6)
        agg = aggregate_results([r1, r2], label="merged")
        assert agg.num_episodes == 10
        # success rate should be between 0.6 and 0.8
        assert 0.6 <= agg.success_rate <= 0.8

    def test_compare_modes(self):
        from embodiedbench.evaluation.aggregators import compare_modes
        mode_results = {
            "baseline": [_make_experiment_result(mode="baseline", success_rate=0.5)],
            "adapted_memory": [_make_experiment_result(mode="adapted_memory", success_rate=0.8)],
        }
        comparison = compare_modes(mode_results)
        assert "baseline" in comparison
        assert "adapted_memory" in comparison
        assert comparison["adapted_memory"].success_rate > comparison["baseline"].success_rate

    def test_cross_seed_summary(self):
        from embodiedbench.evaluation.aggregators import cross_seed_summary
        results = [
            _make_experiment_result(n_episodes=5, success_rate=0.7),
            _make_experiment_result(n_episodes=5, success_rate=0.9),
        ]
        summary = cross_seed_summary(results)
        assert "success_rate_mean" in summary
        assert "success_rate_std" in summary
        assert 0.6 <= summary["success_rate_mean"] <= 0.9

    def test_deterministic_aggregation(self):
        """Same inputs always produce the same output."""
        from embodiedbench.evaluation.aggregators import aggregate_results
        results = [_make_experiment_result(n_episodes=8, success_rate=0.75)]
        a1 = aggregate_results(results)
        a2 = aggregate_results(results)
        assert a1.success_rate == a2.success_rate
        assert a1.stale_memory_recovery_rate == a2.stale_memory_recovery_rate

    def test_aggregate_from_directory(self, tmp_path):
        from embodiedbench.evaluation.aggregators import aggregate_from_directory
        from embodiedbench.evaluation.reporting import save_experiment_result_json
        r = _make_experiment_result(benchmark="eb_alfred", mode="baseline")
        save_experiment_result_json(r, str(tmp_path / "baseline_result.json"))
        metrics = aggregate_from_directory(str(tmp_path))
        assert len(metrics) == 1
        assert metrics[0].mode == "baseline"


# ---------------------------------------------------------------------------
# 4. Stale-memory recovery metric  (Step 29D)
# ---------------------------------------------------------------------------

class TestStaleRecoveryMetric:
    def test_stale_recovery_rate_no_stale(self):
        from embodiedbench.evaluation.metrics import compute_stale_memory_recovery_rate
        from embodiedbench.evaluation.schemas import EpisodeResult
        eps = [
            EpisodeResult(task_success=True, stale_memory_detected=False),
            EpisodeResult(task_success=False, stale_memory_detected=False),
        ]
        assert compute_stale_memory_recovery_rate(eps) == 0.0

    def test_stale_recovery_rate_all_recovered(self):
        from embodiedbench.evaluation.metrics import compute_stale_memory_recovery_rate
        from embodiedbench.evaluation.schemas import EpisodeResult
        eps = [
            EpisodeResult(task_success=True, stale_memory_detected=True),
            EpisodeResult(task_success=True, stale_memory_detected=True),
        ]
        assert compute_stale_memory_recovery_rate(eps) == 1.0

    def test_stale_recovery_rate_partial(self):
        from embodiedbench.evaluation.metrics import compute_stale_memory_recovery_rate
        from embodiedbench.evaluation.schemas import EpisodeResult
        eps = [
            EpisodeResult(task_success=True,  stale_memory_detected=True),
            EpisodeResult(task_success=False, stale_memory_detected=True,
                          task_progress=0.2),  # not recovered
            EpisodeResult(task_success=False, stale_memory_detected=True,
                          stale_memory_recovered=True),  # explicit recovery flag
        ]
        rate = compute_stale_memory_recovery_rate(eps)
        assert abs(rate - 2/3) < 1e-9

    def test_stale_recovery_via_progress(self):
        from embodiedbench.evaluation.metrics import compute_stale_memory_recovery_rate
        from embodiedbench.evaluation.schemas import EpisodeResult
        eps = [
            EpisodeResult(task_success=False, stale_memory_detected=True,
                          task_progress=0.75),  # >0.5 → recovered
        ]
        assert compute_stale_memory_recovery_rate(eps) == 1.0

    def test_stale_recovery_rate_in_aggregate_metrics(self):
        from embodiedbench.evaluation.metrics import compute_aggregate_metrics
        from embodiedbench.evaluation.schemas import EpisodeResult
        eps = [
            EpisodeResult(task_success=True,  stale_memory_detected=True),
            EpisodeResult(task_success=False, stale_memory_detected=True),
        ]
        agg = compute_aggregate_metrics(eps)
        assert agg.stale_memory_recovery_rate == 0.5


# ---------------------------------------------------------------------------
# 5. Markdown reporting
# ---------------------------------------------------------------------------

class TestMarkdownReporting:
    def test_build_markdown_table(self):
        from embodiedbench.evaluation.reporting import build_markdown_table
        from embodiedbench.evaluation.schemas import AggregateMetrics
        metrics = [
            AggregateMetrics(mode="baseline", benchmark="eb_alfred",
                             success_rate=0.5, avg_replans=3.0),
            AggregateMetrics(mode="adapted_memory", benchmark="eb_alfred",
                             success_rate=0.8, avg_replans=1.5),
        ]
        table = build_markdown_table(metrics)
        assert "| Mode |" in table
        assert "baseline" in table
        assert "adapted_memory" in table
        assert "0.800" in table

    def test_save_markdown_report(self, tmp_path):
        from embodiedbench.evaluation.reporting import save_markdown_report
        from embodiedbench.evaluation.schemas import AggregateMetrics
        metrics = [AggregateMetrics(mode="baseline", success_rate=0.5)]
        path = str(tmp_path / "report.md")
        save_markdown_report(metrics, path, title="Test Report")
        assert os.path.isfile(path)
        content = open(path).read()
        assert "# Test Report" in content
        assert "baseline" in content

    def test_markdown_report_with_extra_sections(self, tmp_path):
        from embodiedbench.evaluation.reporting import save_markdown_report
        from embodiedbench.evaluation.schemas import AggregateMetrics
        metrics = [AggregateMetrics(mode="adapted_memory", success_rate=0.8)]
        path = str(tmp_path / "report.md")
        save_markdown_report(
            metrics, path,
            extra_sections={"Notes": "This is a custom section."},
        )
        content = open(path).read()
        assert "## Notes" in content
        assert "custom section" in content


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

class TestVisualization:
    def _make_metrics(self):
        from embodiedbench.evaluation.schemas import AggregateMetrics
        return [
            AggregateMetrics(mode="baseline", benchmark="eb_alfred",
                             success_rate=0.5, avg_replans=3.0,
                             stale_memory_recovery_rate=0.0,
                             avg_invalid_actions=2.0),
            AggregateMetrics(mode="adapted_memory", benchmark="eb_alfred",
                             success_rate=0.8, avg_replans=1.5,
                             stale_memory_recovery_rate=0.7,
                             avg_invalid_actions=0.8),
        ]

    def test_save_all_plots(self, tmp_path):
        from embodiedbench.evaluation.visualization import save_all_plots
        metrics = self._make_metrics()
        paths = save_all_plots(metrics, str(tmp_path))
        assert len(paths) > 0
        for path in paths.values():
            assert os.path.isfile(path)

    def test_individual_plot_returns_fig(self):
        from embodiedbench.evaluation.visualization import plot_success_rate_by_mode
        metrics = self._make_metrics()
        fig, ax = plot_success_rate_by_mode(metrics)
        assert fig is not None

    def test_stale_recovery_plot(self):
        from embodiedbench.evaluation.visualization import plot_stale_recovery_rate
        metrics = self._make_metrics()
        fig, ax = plot_stale_recovery_rate(metrics)
        assert fig is not None


# ---------------------------------------------------------------------------
# 7. CSV export
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_export_csv(self, tmp_path):
        from embodiedbench.evaluation.reporting import export_csv
        from embodiedbench.evaluation.schemas import AggregateMetrics
        metrics = [
            AggregateMetrics(mode="baseline", success_rate=0.5),
            AggregateMetrics(mode="adapted_memory", success_rate=0.8),
        ]
        path = str(tmp_path / "results.csv")
        export_csv(metrics, path)
        assert os.path.isfile(path)
        rows = open(path).readlines()
        assert len(rows) == 3  # header + 2 data rows
        assert "Mode" in rows[0]
        assert "baseline" in rows[1]

    def test_generate_full_report(self, tmp_path):
        from embodiedbench.evaluation.reporting import generate_full_report
        from embodiedbench.evaluation.schemas import AggregateMetrics
        metrics = [AggregateMetrics(mode="raw_memory", success_rate=0.6)]
        paths = generate_full_report(metrics, str(tmp_path))
        assert "json" in paths
        assert "csv" in paths
        assert "markdown" in paths
        for p in paths.values():
            assert os.path.isfile(p)


# ---------------------------------------------------------------------------
# 8. Experiment modes handled correctly
# ---------------------------------------------------------------------------

class TestExperimentModes:
    def test_mode_to_experiment_mode_mapping(self):
        from embodiedbench.evaluation.runner import _MODE_TO_EXPERIMENT_MODE
        assert _MODE_TO_EXPERIMENT_MODE["baseline"] == "none"
        assert _MODE_TO_EXPERIMENT_MODE["raw_memory"] == "raw_memory"
        assert _MODE_TO_EXPERIMENT_MODE["adapted_memory"] == "adapted_memory"

    def test_build_evaluator_config_sets_mode(self):
        from embodiedbench.evaluation.runner import _build_evaluator_config
        from embodiedbench.evaluation.schemas import ExperimentConfig
        cfg = ExperimentConfig(mode="adapted_memory_planner_only")
        ev_cfg = _build_evaluator_config(cfg)
        assert ev_cfg["memory_experiment"]["mode"] == "adapted_memory_planner_only"

    def test_patch_config_for_mode(self):
        from embodiedbench.evaluation.utils import patch_config_for_mode
        base = {"model_name": "gpt4"}
        patched = patch_config_for_mode(base, "raw_memory")
        assert patched["memory_experiment"]["mode"] == "raw_memory"
        # Original not mutated
        assert "memory_experiment" not in base

    def test_all_valid_modes_accepted(self, tmp_path):
        from embodiedbench.evaluation.schemas import ExperimentConfig, VALID_MODES
        from embodiedbench.evaluation.runner import run_experiment

        def mock_ep(cfg):
            return [_make_episode_dict()]

        for mode in VALID_MODES:
            cfg = ExperimentConfig(
                benchmark="eb_alfred", mode=mode,
                output_dir=str(tmp_path), save_episode_jsons=False,
            )
            result = run_experiment(cfg, episode_fn=mock_ep)
            assert result.config.mode == mode


# ---------------------------------------------------------------------------
# 9. Adapter checkpoint loading path
# ---------------------------------------------------------------------------

class TestAdapterCheckpoint:
    def test_missing_checkpoint_raises(self, tmp_path):
        from embodiedbench.evaluation.runner import _maybe_load_adapter
        from embodiedbench.evaluation.schemas import ExperimentConfig
        cfg = ExperimentConfig(
            mode="adapted_memory",
            adapter_checkpoint=str(tmp_path / "nonexistent_ckpt"),
        )
        with pytest.raises(FileNotFoundError):
            _maybe_load_adapter(cfg)

    def test_baseline_mode_skips_adapter_load(self, tmp_path):
        from embodiedbench.evaluation.runner import _maybe_load_adapter
        from embodiedbench.evaluation.schemas import ExperimentConfig
        cfg = ExperimentConfig(mode="baseline", adapter_checkpoint=None)
        result = _maybe_load_adapter(cfg)
        assert result is None

    def test_raw_memory_skips_adapter_load(self, tmp_path):
        from embodiedbench.evaluation.runner import _maybe_load_adapter
        from embodiedbench.evaluation.schemas import ExperimentConfig
        cfg = ExperimentConfig(mode="raw_memory", adapter_checkpoint=None)
        result = _maybe_load_adapter(cfg)
        assert result is None

    def test_adapted_mode_no_checkpoint_warns(self, tmp_path, caplog):
        from embodiedbench.evaluation.runner import _maybe_load_adapter
        from embodiedbench.evaluation.schemas import ExperimentConfig
        import logging
        cfg = ExperimentConfig(mode="adapted_memory", adapter_checkpoint=None)
        with caplog.at_level(logging.WARNING, logger="EB_logger"):
            result = _maybe_load_adapter(cfg)
        assert result is None
        assert "adapter_checkpoint" in caplog.text.lower() or "none" in caplog.text.lower()


# ---------------------------------------------------------------------------
# 10. Deterministic aggregation (property-based smoke test)
# ---------------------------------------------------------------------------

class TestDeterministicAggregation:
    def test_same_episodes_same_metrics(self):
        from embodiedbench.evaluation.metrics import compute_aggregate_metrics
        from embodiedbench.evaluation.schemas import EpisodeResult

        eps = [
            EpisodeResult(task_success=True,  num_steps=10, num_replans=1,
                          stale_memory_detected=True,  stale_memory_recovered=True),
            EpisodeResult(task_success=False, num_steps=15, num_replans=3,
                          stale_memory_detected=True,  stale_memory_recovered=False),
            EpisodeResult(task_success=True,  num_steps=8,  num_replans=0,
                          stale_memory_detected=False),
        ]
        for _ in range(5):
            agg = compute_aggregate_metrics(eps)
            assert abs(agg.success_rate - 2/3) < 1e-9
            assert abs(agg.avg_replans - (1+3+0)/3) < 1e-9
            assert abs(agg.stale_memory_recovery_rate - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# 11. All outputs JSON serializable
# ---------------------------------------------------------------------------

class TestJsonSerializable:
    def test_experiment_result_serializable(self):
        from embodiedbench.evaluation.utils import is_json_serializable
        result = _make_experiment_result()
        assert is_json_serializable(result.to_dict())

    def test_aggregate_metrics_serializable(self):
        from embodiedbench.evaluation.utils import is_json_serializable
        from embodiedbench.evaluation.aggregators import aggregate_results
        result = _make_experiment_result()
        agg = aggregate_results([result])
        assert is_json_serializable(agg.to_dict())

    def test_episode_result_serializable(self):
        from embodiedbench.evaluation.utils import is_json_serializable
        from embodiedbench.evaluation.schemas import EpisodeResult
        ep = EpisodeResult(
            task_success=True, num_steps=5, stale_memory_detected=True,
            extra={"custom_key": 42},
        )
        assert is_json_serializable(ep.to_dict())
