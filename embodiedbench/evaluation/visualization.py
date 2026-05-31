"""
evaluation/visualization.py

Matplotlib-based plots for Step-29 evaluation harness.

All functions are self-contained and return (fig, axes) so callers can
further customise or directly save. A ``save_all_plots`` convenience
function writes every standard chart to a directory.

Plots
-----
plot_success_rate_by_mode       — bar chart
plot_replans_by_mode            — bar chart
plot_invalid_actions_by_mode    — bar chart
plot_stale_recovery_rate        — bar chart (the novelty metric)
plot_trajectory_length          — bar chart
plot_adapter_usage              — stacked bar (usage vs fallback)
plot_metric_comparison_grid     — 2×3 grid of all key metrics
save_all_plots                  — write PNG files to a directory
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

from .schemas import AggregateMetrics

logger = logging.getLogger("EB_logger")

# Colour palette (colourblind-friendly)
_PALETTE = [
    "#4878CF",  # blue        — baseline
    "#6ACC65",  # green       — raw_memory
    "#D65F5F",  # red         — adapted_memory
    "#B47CC7",  # purple      — planner_only
    "#C4AD66",  # gold        — critic_only
    "#77BEDB",  # light blue  — planner_critic
]

_DEFAULT_FIG_SIZE = (7, 4)


def _mode_labels(metrics: List[AggregateMetrics]) -> List[str]:
    return [m.label or m.mode for m in metrics]


def _bar_chart(
    metrics: List[AggregateMetrics],
    attr: str,
    ylabel: str,
    title: str,
    figsize: Tuple[int, int] = _DEFAULT_FIG_SIZE,
    higher_better: bool = True,
):
    """Generic bar chart helper. Returns (fig, ax)."""
    import matplotlib.pyplot as plt

    labels = _mode_labels(metrics)
    values = [getattr(m, attr, 0.0) for m in metrics]
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(metrics))]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(max(values) * 1.15, 0.1))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    direction = "↑ higher better" if higher_better else "↓ lower better"
    ax.set_xlabel(direction, fontsize=8, color="grey")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_success_rate_by_mode(
    metrics: List[AggregateMetrics],
    figsize: Tuple[int, int] = _DEFAULT_FIG_SIZE,
):
    """Bar chart: success rate per mode."""
    return _bar_chart(
        metrics, "success_rate", "Success Rate", "Task Success Rate by Mode",
        figsize=figsize, higher_better=True,
    )


def plot_replans_by_mode(
    metrics: List[AggregateMetrics],
    figsize: Tuple[int, int] = _DEFAULT_FIG_SIZE,
):
    """Bar chart: average replans per mode."""
    return _bar_chart(
        metrics, "avg_replans", "Avg Replans", "Average Replans by Mode",
        figsize=figsize, higher_better=False,
    )


def plot_invalid_actions_by_mode(
    metrics: List[AggregateMetrics],
    figsize: Tuple[int, int] = _DEFAULT_FIG_SIZE,
):
    """Bar chart: average invalid actions per mode."""
    return _bar_chart(
        metrics, "avg_invalid_actions", "Avg Invalid Actions",
        "Average Invalid Actions by Mode",
        figsize=figsize, higher_better=False,
    )


def plot_stale_recovery_rate(
    metrics: List[AggregateMetrics],
    figsize: Tuple[int, int] = _DEFAULT_FIG_SIZE,
):
    """
    Bar chart: stale-memory recovery rate per mode.

    This is the core MemAdapt novelty metric: measures how often the agent
    successfully recovers after detecting stale / conflicting memory.
    """
    return _bar_chart(
        metrics, "stale_memory_recovery_rate",
        "Stale Memory Recovery Rate",
        "Stale-Memory Recovery Rate by Mode (MemAdapt Novelty Metric)",
        figsize=figsize, higher_better=True,
    )


def plot_trajectory_length(
    metrics: List[AggregateMetrics],
    figsize: Tuple[int, int] = _DEFAULT_FIG_SIZE,
):
    """Bar chart: average trajectory length per mode."""
    return _bar_chart(
        metrics, "avg_trajectory_length", "Avg Trajectory Length",
        "Average Trajectory Length by Mode",
        figsize=figsize, higher_better=False,
    )


def plot_adapter_usage(
    metrics: List[AggregateMetrics],
    figsize: Tuple[int, int] = _DEFAULT_FIG_SIZE,
):
    """
    Grouped bar chart: adapter usage rate vs adapter fallback rate per mode.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = _mode_labels(metrics)
    usage = [m.adapter_usage_rate for m in metrics]
    fallback = [m.adapter_fallback_rate for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, usage, width, label="Adapter Usage", color=_PALETTE[2])
    ax.bar(x + width / 2, fallback, width, label="Fallback Rate", color=_PALETTE[0])

    ax.set_ylabel("Rate")
    ax.set_title("Adapter Usage and Fallback Rate by Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    return fig, ax


def plot_metric_comparison_grid(
    metrics: List[AggregateMetrics],
    figsize: Tuple[int, int] = (14, 8),
):
    """
    2×3 grid showing all six key metrics side-by-side.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    configs = [
        ("success_rate",               "Success Rate ↑",          True),
        ("avg_replans",                "Avg Replans ↓",           False),
        ("avg_invalid_actions",        "Avg Invalid Actions ↓",   False),
        ("avg_steps",                  "Avg Steps",               False),
        ("stale_memory_recovery_rate", "Stale Recovery Rate ↑",   True),
        ("adapter_usage_rate",         "Adapter Usage Rate",      True),
    ]
    labels = _mode_labels(metrics)
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(metrics))]

    for ax, (attr, ylabel, _) in zip(axes, configs):
        values = [getattr(m, attr, 0.0) for m in metrics]
        bars = ax.bar(labels, values, color=colours, edgecolor="white")
        ax.set_title(ylabel, fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=7,
            )

    fig.suptitle("MemAdapt Evaluation — Key Metrics Comparison", fontsize=12)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Batch save
# ---------------------------------------------------------------------------

def save_all_plots(
    metrics: List[AggregateMetrics],
    output_dir: str,
    dpi: int = 150,
) -> Dict[str, str]:
    """
    Save all standard plots to *output_dir* as PNG files.

    Returns
    -------
    dict mapping plot name → file path.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend

    os.makedirs(output_dir, exist_ok=True)
    saved: Dict[str, str] = {}

    plot_fns = [
        ("success_rate",    plot_success_rate_by_mode),
        ("replans",         plot_replans_by_mode),
        ("invalid_actions", plot_invalid_actions_by_mode),
        ("stale_recovery",  plot_stale_recovery_rate),
        ("trajectory",      plot_trajectory_length),
        ("adapter_usage",   plot_adapter_usage),
        ("grid",            plot_metric_comparison_grid),
    ]

    for name, fn in plot_fns:
        try:
            fig, _ = fn(metrics)
            path = os.path.join(output_dir, f"{name}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            fig.clf()
            import matplotlib.pyplot as plt
            plt.close(fig)
            saved[name] = path
            logger.debug(f"[Visualization] Saved {name} → {path}")
        except Exception as exc:
            logger.warning(f"[Visualization] Plot '{name}' failed: {exc}")

    logger.info(f"[Visualization] {len(saved)} plots saved to {output_dir}")
    return saved
