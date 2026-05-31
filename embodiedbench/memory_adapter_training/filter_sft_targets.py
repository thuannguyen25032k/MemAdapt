"""
filter_sft_targets.py

Filter "good" memory-adapter targets for supervised fine-tuning (SFT).

Procedure
---------
For every task instruction the adapter target (the *plan* produced by the
memory adapter) is evaluated with two planner agents:

    * expert  : InternVL3_5-38B
    * novice  : InternVL3_5-14B

Each planner runs the task **twice**:

    * without the adapter target  -> ``*_baseline``          run
    * with    the adapter target  -> ``*_memory_adapter``    run

A plan is considered **low quality** and is *removed* from the training set
when it *degrades* the task progress of **either** planner, i.e. when

    progress_with_adapter  <  progress_without_adapter - tolerance

for the expert and/or the novice agent.  All remaining samples are kept as
SFT targets.

Data layout (per domain folder, e.g. ``alfred_memory_logs``)
-----------------------------------------------------------
    training_records_38B.jsonl              # adapter targets (one per task)
    training_records_14B.jsonl              # same adapter targets, novice outcome
    InternVL3_5-38B_baseline/<cat>/results/episode_*_final_res.json
    InternVL3_5-38B_memory_adapter/<cat>/results/episode_*_final_res.json
    InternVL3_5-14B_baseline/<cat>/results/episode_*_final_res.json
    InternVL3_5-14B_memory_adapter/<cat>/results/episode_*_final_res.json

Each ``episode_*_final_res.json`` file contains an ``instruction`` and a
``task_progress`` field which are used to match runs to adapter targets
(instructions are unique within a domain).

Usage
-----
    python -m embodiedbench.memory_adapter_training.filter_sft_targets \
        --dataset-root memory_adapter_dataset \
        --output-dir memory_adapter_dataset/sft_filtered

    # single domain only
    python -m embodiedbench.memory_adapter_training.filter_sft_targets \
        --domains alfred_memory_logs --tolerance 0.0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration of the four evaluation runs
# ---------------------------------------------------------------------------

EXPERT_MODEL = "InternVL3_5-38B"
NOVICE_MODEL = "InternVL3_5-14B"

# Run-folder names relative to a domain directory.
EXPERT_BASELINE = f"{EXPERT_MODEL}_baseline"
EXPERT_ADAPTER = f"{EXPERT_MODEL}_memory_adapter"
NOVICE_BASELINE = f"{NOVICE_MODEL}_baseline"
NOVICE_ADAPTER = f"{NOVICE_MODEL}_memory_adapter"

# Canonical adapter-target source (the targets are identical across the two
# training-record files; only the recorded outcome differs).
RECORDS_FILE = "training_records_38B.jsonl"


# ---------------------------------------------------------------------------
# Result loading helpers
# ---------------------------------------------------------------------------

def load_progress_by_instruction(run_dir: str) -> Dict[str, float]:
    """
    Scan ``<run_dir>/<category>/results/episode_*_final_res.json`` and return a
    mapping ``instruction -> task_progress``.

    If the same instruction appears more than once (it should not within a
    domain) the *minimum* progress is kept -- the conservative choice when
    deciding whether a plan is harmful.
    """
    progress: Dict[str, float] = {}
    pattern = os.path.join(run_dir, "*", "results", "episode_*_final_res.json")
    for path in glob.glob(pattern):
        try:
            with open(path, "r") as fh:
                res = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        instruction = res.get("instruction")
        if instruction is None:
            continue
        prog = res.get("task_progress")
        if prog is None:
            continue
        prog = float(prog)
        if instruction in progress:
            progress[instruction] = min(progress[instruction], prog)
        else:
            progress[instruction] = prog
    return progress


def load_records(records_path: str) -> List[dict]:
    """Load a JSONL file of adapter-target training records."""
    records: List[dict] = []
    with open(records_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

@dataclass
class SampleVerdict:
    instruction: str
    expert_without: Optional[float]
    expert_with: Optional[float]
    novice_without: Optional[float]
    novice_with: Optional[float]
    expert_degraded: bool = False
    novice_degraded: bool = False
    evaluable: bool = True          # all four runs were found
    keep: bool = False
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "expert_progress_without_adapter": self.expert_without,
            "expert_progress_with_adapter": self.expert_with,
            "novice_progress_without_adapter": self.novice_without,
            "novice_progress_with_adapter": self.novice_with,
            "expert_degraded": self.expert_degraded,
            "novice_degraded": self.novice_degraded,
            "evaluable": self.evaluable,
            "keep": self.keep,
            "reasons": self.reasons,
        }


def evaluate_sample(
    instruction: str,
    expert_baseline: Dict[str, float],
    expert_adapter: Dict[str, float],
    novice_baseline: Dict[str, float],
    novice_adapter: Dict[str, float],
    *,
    tolerance: float,
    keep_unverified: bool,
) -> SampleVerdict:
    """Decide whether the adapter target for *instruction* should be kept."""
    e_without = expert_baseline.get(instruction)
    e_with = expert_adapter.get(instruction)
    n_without = novice_baseline.get(instruction)
    n_with = novice_adapter.get(instruction)

    verdict = SampleVerdict(
        instruction=instruction,
        expert_without=e_without,
        expert_with=e_with,
        novice_without=n_without,
        novice_with=n_with,
    )

    missing = [
        name
        for name, val in (
            ("expert_baseline", e_without),
            ("expert_memory_adapter", e_with),
            ("novice_baseline", n_without),
            ("novice_memory_adapter", n_with),
        )
        if val is None
    ]

    if missing:
        verdict.evaluable = False
        verdict.reasons.append("missing runs: " + ", ".join(missing))
        verdict.keep = keep_unverified
        if not keep_unverified:
            verdict.reasons.append("dropped (cannot verify degradation)")
        return verdict

    # Degradation = with-adapter progress strictly below without-adapter
    # progress (beyond the tolerance band).
    if e_with < e_without - tolerance:
        verdict.expert_degraded = True
        verdict.reasons.append(
            f"expert progress dropped {e_without:.3f} -> {e_with:.3f}"
        )
    if n_with < n_without - tolerance:
        verdict.novice_degraded = True
        verdict.reasons.append(
            f"novice progress dropped {n_without:.3f} -> {n_with:.3f}"
        )

    verdict.keep = not (verdict.expert_degraded or verdict.novice_degraded)
    if verdict.keep:
        verdict.reasons.append("kept (no degradation for either planner)")
    return verdict


# ---------------------------------------------------------------------------
# Domain-level driver
# ---------------------------------------------------------------------------

def filter_domain(
    domain_dir: str,
    output_dir: str,
    *,
    tolerance: float,
    keep_unverified: bool,
) -> dict:
    """Filter adapter targets for a single domain folder."""
    records_path = os.path.join(domain_dir, RECORDS_FILE)
    if not os.path.isfile(records_path):
        raise FileNotFoundError(f"records file not found: {records_path}")

    records = load_records(records_path)

    expert_baseline = load_progress_by_instruction(
        os.path.join(domain_dir, EXPERT_BASELINE)
    )
    expert_adapter = load_progress_by_instruction(
        os.path.join(domain_dir, EXPERT_ADAPTER)
    )
    novice_baseline = load_progress_by_instruction(
        os.path.join(domain_dir, NOVICE_BASELINE)
    )
    novice_adapter = load_progress_by_instruction(
        os.path.join(domain_dir, NOVICE_ADAPTER)
    )

    os.makedirs(output_dir, exist_ok=True)
    kept_path = os.path.join(output_dir, "sft_targets_filtered.jsonl")
    removed_path = os.path.join(output_dir, "removed_targets.jsonl")
    report_path = os.path.join(output_dir, "filter_report.json")

    n_kept = n_removed = n_unverified = 0
    verdict_records: List[dict] = []

    with open(kept_path, "w") as kept_fh, open(removed_path, "w") as removed_fh:
        for record in records:
            instruction = record.get("instruction", "")
            verdict = evaluate_sample(
                instruction,
                expert_baseline,
                expert_adapter,
                novice_baseline,
                novice_adapter,
                tolerance=tolerance,
                keep_unverified=keep_unverified,
            )

            # Attach diagnostics to the record so downstream consumers can
            # inspect why a sample was kept/removed.
            enriched = dict(record)
            enriched["degradation_check"] = verdict.as_dict()
            verdict_records.append(verdict.as_dict())

            if not verdict.evaluable:
                n_unverified += 1

            if verdict.keep:
                kept_fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                n_kept += 1
            else:
                removed_fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                n_removed += 1

    report = {
        "domain_dir": domain_dir,
        "tolerance": tolerance,
        "keep_unverified": keep_unverified,
        "total_records": len(records),
        "kept": n_kept,
        "removed": n_removed,
        "unverified": n_unverified,
        "expert_degraded": sum(
            1 for v in verdict_records if v["expert_degraded"]
        ),
        "novice_degraded": sum(
            1 for v in verdict_records if v["novice_degraded"]
        ),
        "outputs": {
            "kept": kept_path,
            "removed": removed_path,
            "report": report_path,
        },
    }

    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_domains(dataset_root: str) -> List[str]:
    """Return domain sub-directories that contain a records file."""
    domains: List[str] = []
    for name in sorted(os.listdir(dataset_root)):
        path = os.path.join(dataset_root, name)
        if os.path.isdir(path) and os.path.isfile(
            os.path.join(path, RECORDS_FILE)
        ):
            domains.append(name)
    return domains


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter good adapter targets for SFT by removing plans "
        "that degrade expert (38B) or novice (14B) planner progress."
    )
    parser.add_argument(
        "--dataset-root",
        default="memory_adapter_dataset",
        help="Root folder containing the per-domain log directories.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Domain sub-folders to process (default: auto-discover all that "
        "contain a training-records file).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write filtered outputs. Defaults to "
        "<domain_dir>/sft_filtered for each domain.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Allowed progress drop before a plan counts as degrading "
        "(default 0.0 = any drop removes the sample).",
    )
    parser.add_argument(
        "--keep-unverified",
        action="store_true",
        help="Keep samples that miss one or more of the four runs "
        "(default: drop them).",
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise SystemExit(f"dataset root not found: {dataset_root}")

    domains = args.domains or _discover_domains(dataset_root)
    if not domains:
        raise SystemExit(f"no domains with {RECORDS_FILE} found in {dataset_root}")

    grand_total = grand_kept = grand_removed = 0
    for domain in domains:
        domain_dir = os.path.join(dataset_root, domain)
        out_dir = (
            os.path.join(args.output_dir, domain)
            if args.output_dir
            else os.path.join(domain_dir, "sft_filtered")
        )
        report = filter_domain(
            domain_dir,
            out_dir,
            tolerance=args.tolerance,
            keep_unverified=args.keep_unverified,
        )
        grand_total += report["total_records"]
        grand_kept += report["kept"]
        grand_removed += report["removed"]
        print(
            f"[{domain}] total={report['total_records']} "
            f"kept={report['kept']} removed={report['removed']} "
            f"(expert_degraded={report['expert_degraded']}, "
            f"novice_degraded={report['novice_degraded']}, "
            f"unverified={report['unverified']}) -> {report['outputs']['kept']}"
        )

    print(
        f"\nTOTAL: {grand_total} records -> kept {grand_kept}, "
        f"removed {grand_removed}"
    )


if __name__ == "__main__":
    main()
