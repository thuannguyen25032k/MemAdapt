"""
evaluation/aggregators.py

Aggregation utilities: merge episodes across runs, compare modes, summarise
across seeds, and load results from a directory.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from .schemas import AggregateMetrics, EpisodeResult, ExperimentResult
from .metrics import compute_aggregate_metrics

logger = logging.getLogger("EB_logger")


# ---------------------------------------------------------------------------
# Core aggregator
# ---------------------------------------------------------------------------

def aggregate_results(
    results: List[ExperimentResult],
    label: str = "",
) -> AggregateMetrics:
    """
    Merge all episodes from multiple ExperimentResult objects (sharing the
    same benchmark + mode) and compute aggregate metrics.
    """
    all_episodes: List[EpisodeResult] = []
    for r in results:
        all_episodes.extend(r.episodes)

    benchmark = results[0].config.benchmark if results else ""
    mode = results[0].config.mode if results else ""

    return compute_aggregate_metrics(
        all_episodes,
        label=label or f"{mode}/{benchmark}",
        benchmark=benchmark,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Mode comparison
# ---------------------------------------------------------------------------

def compare_modes(
    mode_results: Dict[str, List[ExperimentResult]],
) -> Dict[str, AggregateMetrics]:
    """
    Map ``{mode_name → [ExperimentResult, …]}`` to
    ``{mode_name → AggregateMetrics}`` (single or multi-seed per mode).
    """
    comparison: Dict[str, AggregateMetrics] = {}
    for mode, results in mode_results.items():
        if not results:
            continue
        comparison[mode] = aggregate_results(results, label=mode)
    return comparison


# ---------------------------------------------------------------------------
# Cross-seed summary
# ---------------------------------------------------------------------------

def cross_seed_summary(
    results: List[ExperimentResult],
) -> Dict[str, float]:
    """
    Compute mean ± std for key float metrics across multiple seeds of the
    same mode/benchmark. Returns keys like ``success_rate_mean`` / ``_std``.
    """
    import statistics

    per_seed_metrics: Dict[str, List[float]] = {}
    for r in results:
        agg = compute_aggregate_metrics(r.episodes)
        for k, v in agg.to_dict().items():
            if isinstance(v, float):
                per_seed_metrics.setdefault(k, []).append(v)

    summary: Dict[str, float] = {}
    for key, values in per_seed_metrics.items():
        summary[f"{key}_mean"] = statistics.mean(values)
        summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


# ---------------------------------------------------------------------------
# Load from directory
# ---------------------------------------------------------------------------

def aggregate_from_directory(
    directory: str,
    mode_filter: Optional[str] = None,
    benchmark_filter: Optional[str] = None,
) -> List[AggregateMetrics]:
    """
    Load all ``*_result.json`` files from *directory*, optionally filter by
    mode / benchmark, and return one AggregateMetrics per (benchmark, mode).
    """
    if not os.path.isdir(directory):
        logger.warning(f"[Aggregators] Directory not found: {directory}")
        return []

    # Group by (benchmark, mode)
    groups: Dict[tuple, List[ExperimentResult]] = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith("_result.json"):
            continue
        path = os.path.join(directory, fname)
        try:
            with open(path, encoding="utf-8") as fh:
                r = ExperimentResult.from_dict(json.load(fh))
        except Exception as exc:
            logger.warning(f"[Aggregators] Skipping {path}: {exc}")
            continue

        if mode_filter and r.config.mode != mode_filter:
            continue
        if benchmark_filter and r.config.benchmark != benchmark_filter:
            continue

        key = (r.config.benchmark, r.config.mode)
        groups.setdefault(key, []).append(r)

    aggregates = []
    for (benchmark, mode), results in groups.items():
        agg = aggregate_results(
            results,
            label=f"{mode}/{benchmark} (n={len(results)} runs)",
        )
        aggregates.append(agg)
        logger.info(
            f"[Aggregators] {benchmark}/{mode}: "
            f"n_ep={agg.num_episodes}, success={agg.success_rate:.3f}, "
            f"stale_recovery={agg.stale_memory_recovery_rate:.3f}"
        )
    return aggregates
