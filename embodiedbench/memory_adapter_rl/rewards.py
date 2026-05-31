"""
memory_adapter_rl/rewards.py

Interpretable, modular reward functions for GRPO refinement of the Memory
Adapter.

The Memory Adapter is trained to emit three XML sections that the downstream
planner and critic consume (see ``memory_adapter.prompts``):

    <FORESIGHT_PLAN>       ordered, grounded steps to complete the task
    <FEASIBILITY_CRITERIA> preconditions the critic must check before actions
    <FALLBACK_STRATEGY>    recovery rules for likely invalid actions

Composite reward
----------------
    R = w_success     * task_success
      + w_progress    * task_progress
      + w_format      * format_validity
      + w_foresight   * foresight_quality
      + w_feasibility * feasibility_quality
      + w_fallback    * fallback_quality
      - w_replan      * replan_count
      - w_invalid     * invalid_action_count
      - w_repetition  * repetition_penalty

``task_success`` / ``task_progress`` / ``replan_count`` / ``invalid_action_count``
are environment-rollout signals (0 when scoring offline prompts). The format and
per-section quality terms are computed purely from the response text, giving a
dense learning signal even without an environment in the loop.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .config import RLRewardWeights
from .schemas import RewardSignal

# ---------------------------------------------------------------------------
# Required output sections (must match memory_adapter.prompts.ALL_SECTIONS)
# ---------------------------------------------------------------------------

SECTION_FORESIGHT_PLAN = "FORESIGHT_PLAN"
SECTION_FEASIBILITY_CRITERIA = "FEASIBILITY_CRITERIA"
SECTION_FALLBACK_STRATEGY = "FALLBACK_STRATEGY"

REQUIRED_SECTIONS = [
    SECTION_FORESIGHT_PLAN,
    SECTION_FEASIBILITY_CRITERIA,
    SECTION_FALLBACK_STRATEGY,
]

_TAG_RE = re.compile(r"<([A-Z_]+)>(.*?)</\1>", re.S)

# Action verbs the planner emits; a grounded foresight plan uses these.
_NAV_VERBS = ("navigate", "find", "go to", "move to")
_INTERACT_VERBS = (
    "pick up", "pick", "place", "put", "open", "close",
    "turn on", "turn off", "slice", "toggle", "drop",
)
_EMPTY_TOKENS = {"", "n/a", "none", "none detected", "-"}


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def extract_sections(response: str) -> dict:
    """Return ``{section_name: inner_text}`` for every well-formed XML tag."""
    found = {}
    for m in _TAG_RE.finditer(response or ""):
        found[m.group(1)] = m.group(2).strip()
    return found


def _bullets(section_text: str) -> List[str]:
    """Split a section body into cleaned, non-empty bullet lines."""
    out: List[str] = []
    for line in (section_text or "").splitlines():
        line = line.strip().lstrip("-").strip()
        if line and line.lower() not in _EMPTY_TOKENS:
            out.append(line)
    return out


def _is_nonempty(section_text: str) -> bool:
    return len(_bullets(section_text)) > 0


def _contains_any(text: str, needles) -> bool:
    t = text.lower()
    return any(n in t for n in needles)


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------

def score_format_validity(response: str) -> float:
    """
    Fraction of the three required sections that are present AND non-empty [0, 1].

    A section that is present but empty (or "N/A"/"None") scores nothing, so the
    model cannot game the format reward with blank tags.
    """
    sections = extract_sections(response)
    present_and_filled = sum(
        1 for s in REQUIRED_SECTIONS if _is_nonempty(sections.get(s, ""))
    )
    return present_and_filled / len(REQUIRED_SECTIONS)


# ---------------------------------------------------------------------------
# Per-section quality heuristics
# ---------------------------------------------------------------------------

def score_foresight_quality(response: str) -> float:
    """
    Quality of FORESIGHT_PLAN [0, 1].

    Rewards an ordered, grounded, action-oriented plan:
      * +0.4 for >= 3 steps (a real multi-step plan), +0.2 for >= 2 steps.
      * +0.3 when a navigation step precedes the first interaction step
        (the agent must reach an object before manipulating it).
      * +0.3 for action-verb coverage (navigate + interact verbs present).
    """
    body = extract_sections(response).get(SECTION_FORESIGHT_PLAN, "")
    steps = _bullets(body)
    if not steps:
        return 0.0

    score = 0.0
    if len(steps) >= 3:
        score += 0.4
    elif len(steps) >= 2:
        score += 0.2

    nav_idx = next((i for i, s in enumerate(steps) if _contains_any(s, _NAV_VERBS)), None)
    int_idx = next((i for i, s in enumerate(steps) if _contains_any(s, _INTERACT_VERBS)), None)
    if nav_idx is not None and (int_idx is None or nav_idx <= int_idx):
        score += 0.3

    joined = " ".join(steps)
    if _contains_any(joined, _NAV_VERBS) and _contains_any(joined, _INTERACT_VERBS):
        score += 0.3

    return min(score, 1.0)


def score_feasibility_quality(response: str) -> float:
    """
    Quality of FEASIBILITY_CRITERIA [0, 1].

    The prompt asks for entries shaped ``"<sub-task>": <condition>`` covering
    interaction preconditions:
      * +0.4 when at least one bullet uses the ``"...": condition`` form.
      * +0.3 for referencing interaction actions (pick/place/open/... ).
      * +0.3 for >= 2 distinct criteria.
    """
    body = extract_sections(response).get(SECTION_FEASIBILITY_CRITERIA, "")
    criteria = _bullets(body)
    if not criteria:
        return 0.0

    score = 0.0
    if any(":" in c for c in criteria):
        score += 0.4
    if _contains_any(" ".join(criteria), _INTERACT_VERBS):
        score += 0.3
    if len(set(criteria)) >= 2:
        score += 0.3

    return min(score, 1.0)


def score_fallback_quality(response: str) -> float:
    """
    Quality of FALLBACK_STRATEGY [0, 1].

    The prompt asks for entries shaped ``If "<invalid condition>": <recovery>``:
      * +0.4 when at least one bullet starts with a conditional ("if ...").
      * +0.3 for a recovery action (retry / navigate to an alternative).
      * +0.3 for >= 2 distinct fallback rules.
    """
    body = extract_sections(response).get(SECTION_FALLBACK_STRATEGY, "")
    rules = _bullets(body)
    if not rules:
        return 0.0

    score = 0.0
    if any(r.lower().startswith("if") or '":' in r for r in rules):
        score += 0.4
    if _contains_any(" ".join(rules), ("retry", "instead", "then", "navigate", "try")):
        score += 0.3
    if len(set(rules)) >= 2:
        score += 0.3

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Degeneracy penalty
# ---------------------------------------------------------------------------

def score_repetition_penalty(response: str) -> float:
    """
    Degeneracy penalty in [0, 1] (higher = more repetitive / worse).

    GRPO can collapse onto repetitive text. We penalise (a) duplicate
    non-empty lines and (b) low token diversity, so the reward favours diverse,
    informative responses.
    """
    text = response or ""
    if not text.strip():
        return 1.0

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line_penalty = 0.0
    if lines:
        unique_ratio = len(set(lines)) / len(lines)
        line_penalty = 1.0 - unique_ratio  # 0 when all lines unique

    tokens = text.split()
    token_penalty = 0.0
    if len(tokens) >= 20:
        token_diversity = len(set(tokens)) / len(tokens)
        # Map diversity in [0, 0.5] -> penalty [1, 0]; >0.5 diversity -> 0.
        token_penalty = max(0.0, 1.0 - token_diversity / 0.5)

    return round(min(1.0, max(line_penalty, token_penalty)), 4)


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

def compute_reward(
    response: str,
    task_success: bool = False,
    task_progress: float = 0.0,
    replan_count: int = 0,
    invalid_action_count: int = 0,
    weights: Optional[RLRewardWeights] = None,
) -> RewardSignal:
    """
    Compute the full interpretable reward for one Memory Adapter response.

    Parameters
    ----------
    response             : raw text output of the adapter.
    task_success         : did the episode succeed? (environment rollout)
    task_progress        : fraction of subtasks completed [0, 1].
    replan_count         : number of replanning events (penalised).
    invalid_action_count : number of invalid actions (penalised).
    weights              : RLRewardWeights (defaults used when None).
    """
    weights = weights or RLRewardWeights()

    format_validity = score_format_validity(response)
    foresight_quality = score_foresight_quality(response)
    feasibility_quality = score_feasibility_quality(response)
    fallback_quality = score_fallback_quality(response)
    repetition_penalty = score_repetition_penalty(response)

    total = (
        weights.w_success     * float(task_success)
        + weights.w_progress    * float(task_progress)
        + weights.w_format      * format_validity
        + weights.w_foresight   * foresight_quality
        + weights.w_feasibility * feasibility_quality
        + weights.w_fallback    * fallback_quality
        - weights.w_replan      * replan_count
        - weights.w_invalid     * invalid_action_count
        - weights.w_repetition  * repetition_penalty
    )

    return RewardSignal(
        task_success=float(task_success),
        task_progress=float(task_progress),
        replan_count=replan_count,
        invalid_action_count=invalid_action_count,
        format_validity=format_validity,
        foresight_quality=foresight_quality,
        feasibility_quality=feasibility_quality,
        fallback_quality=fallback_quality,
        repetition_penalty=repetition_penalty,
        total=round(total, 6),
    )
