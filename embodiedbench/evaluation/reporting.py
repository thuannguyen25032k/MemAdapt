"""
evaluation/reporting.py

Report generation: JSON dumps, CSV/TSV export, and paper-ready Markdown
tables for ExperimentResult / AggregateMetrics.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .schemas import AggregateMetrics, ExperimentResult

logger = logging.getLogger("EB_logger")

# Columns shown in tables (in display order)
_TABLE_COLUMNS = [
    ("mode",                        "Mode"),
    ("benchmark",                   "Benchmark"),
    ("num_episodes",                "N"),
    ("success_rate",                "Success ↑"),
    ("avg_replans",                 "Replans ↓"),
    ("avg_invalid_actions",         "Invalid Acts ↓"),
    ("avg_steps",                   "Steps"),
    ("stale_memory_recovery_rate",  "Stale Recovery ↑"),
    ("adapter_usage_rate",          "Adapter Usage"),
    ("adapter_fallback_rate",       "Fallback Rate"),
]

# Float precision in reports
_FLOAT_FMT = "{:.3f}"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return _FLOAT_FMT.format(value)
    return str(value)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_experiment_result_json(result: ExperimentResult, path: str) -> None:
    """Write a full ExperimentResult to *path* as pretty JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(result.to_json())
    logger.info(f"[Reporting] ExperimentResult saved → {path}")


def save_aggregate_metrics_json(
    metrics: List[AggregateMetrics], path: str
) -> None:
    """Write a list of AggregateMetrics to *path* as pretty JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([m.to_dict() for m in metrics], fh, indent=2)
    logger.info(f"[Reporting] AggregateMetrics saved → {path}")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def export_csv(
    metrics: List[AggregateMetrics],
    path: str,
    delimiter: str = ",",
) -> None:
    """
    Export a list of AggregateMetrics as CSV (columns from ``_TABLE_COLUMNS``).
    Use ``delimiter="\\t"`` for TSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    col_keys = [c[0] for c in _TABLE_COLUMNS]
    col_headers = [c[1] for c in _TABLE_COLUMNS]

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(col_headers)
        for m in metrics:
            d = m.to_dict()
            writer.writerow([_fmt(d.get(k, "")) for k in col_keys])

    logger.info(f"[Reporting] CSV saved → {path}")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def build_markdown_table(metrics: List[AggregateMetrics]) -> str:
    """
    Build a Markdown table string from a list of AggregateMetrics.

    Returns
    -------
    str — ready to embed in a README or paper supplement.
    """
    col_keys = [c[0] for c in _TABLE_COLUMNS]
    col_headers = [c[1] for c in _TABLE_COLUMNS]

    # Header row
    header = "| " + " | ".join(col_headers) + " |"
    separator = "| " + " | ".join(["---"] * len(col_headers)) + " |"

    rows = [header, separator]
    for m in metrics:
        d = m.to_dict()
        row = "| " + " | ".join(_fmt(d.get(k, "")) for k in col_keys) + " |"
        rows.append(row)

    return "\n".join(rows)


def save_markdown_report(
    metrics: List[AggregateMetrics],
    path: str,
    title: str = "MemAdapt Evaluation Results",
    extra_sections: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write a Markdown report: an aggregate table plus optional extra sections
    given as ``{section_title → section_body}``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append(build_markdown_table(metrics))
    lines.append("")

    if extra_sections:
        for sec_title, sec_body in extra_sections.items():
            lines.append(f"## {sec_title}")
            lines.append("")
            lines.append(sec_body)
            lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info(f"[Reporting] Markdown report saved → {path}")


# ---------------------------------------------------------------------------
# Full report bundle
# ---------------------------------------------------------------------------

def generate_full_report(
    metrics: List[AggregateMetrics],
    output_dir: str,
    title: str = "MemAdapt Evaluation Results",
) -> Dict[str, str]:
    """
    Generate JSON + CSV + Markdown reports in *output_dir*.

    Returns
    -------
    dict mapping format name → file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    json_path = os.path.join(output_dir, "aggregate_metrics.json")
    save_aggregate_metrics_json(metrics, json_path)
    paths["json"] = json_path

    csv_path = os.path.join(output_dir, "aggregate_metrics.csv")
    export_csv(metrics, csv_path)
    paths["csv"] = csv_path

    md_path = os.path.join(output_dir, "report.md")
    save_markdown_report(metrics, md_path, title=title)
    paths["markdown"] = md_path

    logger.info(f"[Reporting] Full report bundle written to {output_dir}")
    return paths
