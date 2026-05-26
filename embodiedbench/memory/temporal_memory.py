"""
memory/temporal_memory.py

TemporalMemory — episode-scoped, step-level interaction history.

Stores a sliding window of ``TemporalStep`` objects.  When the window
overflows it compresses older steps into compact text summaries (default)
or drops them FIFO.  Retrieval scores steps by lexical overlap, recency,
and failure/rejection bonuses; embedding similarity is used when an
``EmbeddingProvider`` is supplied.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from embodiedbench.memory.base import (
    BaseMemory,
    MemoryItem,
    MemoryQuery,
    RetrievedMemory,
    now_ts,
    truncate_text,
)
from embodiedbench.memory.embeddings import (
    EmbeddingProvider,
    hybrid_score,
    lexical_overlap_score,
)
from embodiedbench.memory.storage import load_json, save_json


# ---------------------------------------------------------------------------
# TemporalStep
# ---------------------------------------------------------------------------

def _safe_action_repr(action: Any) -> str:
    """Convert an action of any type to a JSON-safe string."""
    if action is None:
        return ""
    if isinstance(action, (int, float, str, bool)):
        return str(action)
    if isinstance(action, list):
        return "[" + ", ".join(_safe_action_repr(a) for a in action) + "]"
    return repr(action)


def _safe_info(info: dict) -> dict:
    """
    Return a shallow copy of *info* with non-serializable values replaced by
    their string repr.  Skips ``scene_objects`` / ``inventory_objects``
    (potentially large lists) — stores only their counts instead.
    """
    if not info:
        return {}
    safe: dict = {}
    for k, v in info.items():
        if k in ("scene_objects", "inventory_objects"):
            safe[k + "_count"] = len(v) if isinstance(v, list) else repr(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        else:
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = repr(v)
    return safe


@dataclass
class TemporalStep:
    """
    A single step in the episode interaction history.

    ``action`` may be int, list[int], str, or a sentinel (-1/-2/-3).
    ``info`` may contain non-serializable values sanitized by ``_safe_info()``.
    """

    step_id: int = 0
    task_instruction: str = ""
    action: Any = field(default=None, repr=False)
    action_text: str = ""
    env_feedback: str = ""
    success: Optional[bool] = None
    planner_output: Optional[str] = None
    critic_output: Optional[str] = None
    critic_rejected: bool = False
    info: dict = field(default_factory=dict)
    created_at: float = field(default_factory=now_ts)

    def to_dict(self) -> dict:
        return {
            "step_id":              self.step_id,
            "task_instruction":     self.task_instruction,
            "action":               _safe_action_repr(self.action),
            "action_text":          self.action_text,
            "env_feedback":         self.env_feedback,
            "success":              self.success,
            "planner_output":       self.planner_output,
            "critic_output":        self.critic_output,
            "critic_rejected":      self.critic_rejected,
            "info":                 _safe_info(self.info),
            "created_at":           self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalStep":
        return cls(
            step_id=int(d.get("step_id", 0)),
            task_instruction=d.get("task_instruction", ""),
            action=d.get("action"),
            action_text=d.get("action_text", ""),
            env_feedback=d.get("env_feedback", ""),
            success=d.get("success"),
            planner_output=d.get("planner_output"),
            critic_output=d.get("critic_output"),
            critic_rejected=bool(d.get("critic_rejected", False)),
            info=dict(d.get("info") or {}),
            created_at=float(d.get("created_at", now_ts())),
        )

    def to_memory_item(self) -> MemoryItem:
        """Convert this step into a ``MemoryItem`` for retrieval."""
        parts = []
        if self.task_instruction:
            parts.append(f"Task: {self.task_instruction}")
        if self.action_text:
            parts.append(f"Action: {self.action_text}")
        elif self.action is not None:
            parts.append(f"Action: {_safe_action_repr(self.action)}")
        if self.env_feedback:
            parts.append(f"Feedback: {self.env_feedback}")
        if self.critic_rejected and self.critic_output:
            parts.append(f"Critic: {self.critic_output}")

        # Higher importance for failures and critic rejections.
        _importance_map = {
            "critic_rejected": 0.85,
            "failed":          0.75,
            "success":         0.55,
        }
        if self.critic_rejected:
            importance = _importance_map["critic_rejected"]
        elif self.success is False:
            importance = _importance_map["failed"]
        elif self.success is True:
            importance = _importance_map["success"]
        else:
            importance = 0.5

        return MemoryItem(
            memory_type="temporal",
            content=" | ".join(parts),
            metadata={
                "step_id":         self.step_id,
                "action_text":     self.action_text,
                "success":         self.success,
                "critic_rejected": self.critic_rejected,
                "env_feedback":    self.env_feedback,
            },
            importance=importance,
            confidence=1.0,
            source="temporal_memory",
        )

    def short_summary(self, max_chars: int = 300) -> str:
        """One-line summary suitable for prompt injection."""
        _status = {True: "✓", False: "✗"}
        success_str = (
            "[critic-rejected]" if self.critic_rejected
            else _status.get(self.success, "")
        )
        action_str = self.action_text or _safe_action_repr(self.action) or "?"
        line = f"Step {self.step_id}: action={action_str}; success={success_str}; feedback={self.env_feedback or ''}"
        return truncate_text(line, max_chars)


# ---------------------------------------------------------------------------
# TemporalMemory
# ---------------------------------------------------------------------------

class TemporalMemory(BaseMemory):
    """
    Episode-scoped sliding-window memory of step-level interactions.

    After every ``env.step()`` call ``append_step()``.  At most ``max_steps``
    full steps are kept.  On overflow: compress to summaries (default) or
    drop FIFO.
    """

    def __init__(
        self,
        max_steps: int = 20,
        max_summaries: int = 10,
        embedding_provider: Optional[EmbeddingProvider] = None,
        storage_path: Optional[str] = None,
        compress_on_overflow: bool = True,
    ):
        self.max_steps = max_steps
        self.max_summaries = max_summaries
        self.embedding_provider = embedding_provider
        self.storage_path = storage_path
        self.compress_on_overflow = compress_on_overflow

        self.steps: list[TemporalStep] = []
        self.summaries: list[str] = []
        # Track total steps added (including compressed ones) for step_id inference
        self._total_steps_added: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def append_step(
        self,
        task_instruction: str = "",
        action: Any = None,
        action_text: str = "",
        env_feedback: str = "",
        success: Optional[bool] = None,
        planner_output: Optional[str] = None,
        critic_output: Optional[str] = None,
        critic_rejected: bool = False,
        info: Optional[dict] = None,
        step_id: Optional[int] = None,
    ) -> TemporalStep:
        """
        Create a ``TemporalStep`` and add it to the buffer.  Returns it.
        ``step_id`` defaults to the total number of steps seen so far.
        """
        if step_id is None:
            step_id = self._total_steps_added

        # Derive success from info if not explicitly provided
        if success is None and info:
            raw = info.get("last_action_success")
            if raw is not None:
                success = bool(raw)

        # Derive action_text from info if not provided
        if not action_text and info:
            action_text = str(info.get("action_description", "")) or action_text

        # Derive env_feedback from info if not provided
        if not env_feedback and info:
            env_feedback = str(info.get("env_feedback", ""))

        step = TemporalStep(
            step_id=step_id,
            task_instruction=task_instruction,
            action=action,
            action_text=action_text,
            env_feedback=env_feedback,
            success=success,
            planner_output=planner_output,
            critic_output=critic_output,
            critic_rejected=critic_rejected,
            info=_safe_info(info or {}),
        )
        self.steps.append(step)
        self._total_steps_added += 1
        self.compress_if_needed()
        return step

    def update(self, *args, **kwargs) -> None:
        """
        ``BaseMemory.update()`` implementation — delegates to ``append_step()``.

        Extra positional args are ignored to allow flexible call sites.
        """
        _KEYS = {
            "task_instruction", "action", "action_text",
            "env_feedback", "success",
            "planner_output", "critic_output", "critic_rejected",
            "info", "step_id",
        }
        self.append_step(**{k: v for k, v in kwargs.items() if k in _KEYS})

    # ------------------------------------------------------------------
    # Overflow / compression
    # ------------------------------------------------------------------

    def compress_if_needed(self) -> None:
        """
        If ``self.steps`` exceeds ``max_steps``, compress or drop the oldest
        overflow steps.  Merges summary blocks when ``max_summaries`` is exceeded.
        """
        overflow = len(self.steps) - self.max_steps
        if overflow <= 0:
            return

        old_steps, self.steps = self.steps[:overflow], self.steps[overflow:]

        if self.compress_on_overflow:
            self.summaries.append(self._compress_steps(old_steps))
            while len(self.summaries) > self.max_summaries:
                self.summaries[0:2] = [self.summaries[0] + " | " + self.summaries[1]]
        # else: FIFO — old_steps are simply discarded

    def _compress_steps(self, steps: list) -> str:
        """Produce a compact summary string from a list of TemporalStep objects."""
        lines = []
        for s in steps:
            success_str = "OK" if s.success else ("FAIL" if s.success is False else "?")
            action_str = s.action_text or _safe_action_repr(s.action) or "?"
            fb = truncate_text(s.env_feedback, 80)
            lines.append(f"step{s.step_id}:{action_str}[{success_str}]{(' fb:' + fb) if fb else ''}")
        return "; ".join(lines)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: MemoryQuery, top_k: int = 5) -> list:
        """
        Score and return the top_k most relevant ``TemporalStep`` objects as
        ``RetrievedMemory``.  Steps are re-sorted chronologically for prompts.

        Scoring weights: lexical/hybrid 0.45·0.70, embedding 0.25·0.00,
        recency 0.15, failure/reject 0.10, success 0.05.
        """
        if not self.steps:
            return []

        query_text = query.text_for_retrieval() or query.task_instruction
        n = len(self.steps)

        use_emb = self.embedding_provider is not None
        # Weight redistribution when no embeddings
        w_lex = 0.70 if not use_emb else 0.45
        w_emb = 0.00 if not use_emb else 0.25
        w_rec = 0.15
        w_fail = 0.10
        w_succ = 0.05

        query_emb: Optional[list] = None
        if use_emb and query_text:
            try:
                query_emb = self.embedding_provider.embed_text(query_text)
            except Exception:
                query_emb = None

        results: list[tuple[float, str, TemporalStep]] = []

        for rank, step in enumerate(self.steps):
            item_text = step.to_memory_item().content
            step_emb: Optional[list] = None
            if use_emb and query_emb is not None:
                try:
                    step_emb = self.embedding_provider.embed_text(item_text)
                except Exception:
                    step_emb = None

            # 1. Lexical / hybrid similarity
            if use_emb and query_emb is not None and step_emb is not None:
                sim = hybrid_score(
                    query_text, item_text,
                    query_emb, step_emb,
                    embedding_weight=w_emb / (w_lex + w_emb),
                    lexical_weight=w_lex / (w_lex + w_emb),
                )
            else:
                sim = lexical_overlap_score(query_text, item_text)

            # 2. Recency bonus: most recent step → 1.0, oldest → 0.0
            recency = rank / (n - 1) if n > 1 else 1.0  # rank 0 = oldest

            # 3. Failure / rejection bonus
            fail_bonus = 0.0
            if step.critic_rejected:
                fail_bonus = 1.0
            elif step.success is False:
                fail_bonus = 0.8

            # 4. Success relevance bonus (small; only if query overlaps)
            succ_bonus = 0.0
            if step.success is True:
                lex = lexical_overlap_score(query_text, item_text)
                succ_bonus = lex  # proportional to relevance

            score = (
                w_lex * sim
                + w_rec * recency
                + w_fail * fail_bonus
                + w_succ * succ_bonus
            )
            score = min(1.0, max(0.0, score))

            # Build reason string
            if step.critic_rejected:
                reason = "critic rejection"
            elif step.success is False:
                reason = "recent failed action"
            elif step.success is True and succ_bonus > 0.1:
                reason = "recent relevant success"
            else:
                reason = "recent relevant step"

            results.append((score, reason, step))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:top_k]

        # Re-sort chronologically so the planner sees steps in time order.
        # Scoring is only used for *selection* (which top_k steps to include),
        # not for the presentation order.
        top.sort(key=lambda x: x[2].step_id)

        return [
            RetrievedMemory(
                item=step.to_memory_item(),
                score=score,
                reason=reason,
            )
            for score, reason, step in top
        ]

    # ------------------------------------------------------------------
    # Summarisation helpers
    # ------------------------------------------------------------------

    def summarize_recent_history(self, max_steps: int = 10) -> str:
        """
        Return a concise bullet list of the most recent steps.
        Includes older compressed summaries as a preamble when available.
        """
        lines: list[str] = []

        if self.summaries:
            lines.append("[Earlier history (compressed)]")
            for s in self.summaries[-3:]:   # show at most 3 summary blocks
                lines.append(f"  {truncate_text(s, 200)}")
            lines.append("")

        recent = self.steps[-max_steps:]
        if recent:
            lines.append("[Recent steps]")
            for step in recent:
                lines.append(f"- {step.short_summary(200)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_recent_failures(self, limit: int = 5) -> list:
        """Return the most recent steps where success=False."""
        failures = [s for s in self.steps if s.success is False]
        return failures[-limit:]

    def get_recent_rejections(self, limit: int = 5) -> list:
        """Return the most recent steps where critic_rejected=True."""
        rejections = [s for s in self.steps if s.critic_rejected]
        return rejections[-limit:]

    def get_recent_successes(self, limit: int = 5) -> list:
        """Return the most recent steps where success=True."""
        successes = [s for s in self.steps if s.success is True]
        return successes[-limit:]

    def detect_repeated_failures(self) -> list:
        """
        Detect action texts that have failed more than once.
        Returns a list of warning strings for each repeated-failure action.
        """
        fail_counts: dict[str, int] = {}
        for step in self.steps:
            if step.success is False:
                key = _normalize_action_key(step.action_text or _safe_action_repr(step.action))
                fail_counts[key] = fail_counts.get(key, 0) + 1

        warnings: list[str] = []
        for action_key, count in fail_counts.items():
            if count >= 2:
                warnings.append(
                    f"Action '{action_key}' has failed {count} times. "
                    "Consider an alternative approach."
                )
        return warnings

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def to_prompt_context(self, memories: list) -> str:
        """
        Format temporal memory into a prompt-ready string.

        1. **Do-not-repeat block** — actions that have failed ≥2 times.
        2. **Chronological interaction history** — retrieved steps sorted by
           step_id, with inline ``⚠ same action still failing`` loop tags.
        """
        if not memories:
            return ""

        sections: list[str] = []

        # ── 1. Do-not-repeat warnings ──────────────────────────────────
        rep_warnings = self.detect_repeated_failures()
        if rep_warnings:
            warn_lines = ["Do not repeat these failed actions:"]
            for w in rep_warnings:
                warn_lines.append(f"- {w}")
            sections.append("\n".join(warn_lines))

        # ── 2. Chronological interaction history ───────────────────────
        # memories are already chronological from retrieve() — just format them.
        hist_lines = ["Recent relevant interactions:"]
        prev_action: Optional[str] = None
        prev_failed: bool = False

        for rm in memories:
            meta        = rm.item.metadata
            step_num    = meta.get("step_id", "?")
            action_str  = meta.get("action_text", "?")
            success_val = meta.get("success")
            feedback    = meta.get("env_feedback", "")
            reason_tag  = f" [{rm.reason}]" if rm.reason else ""

            # Detect action loops
            norm_action = _normalize_action_key(action_str)
            loop_tag = ""
            if (
                success_val is False
                and prev_failed
                and norm_action == prev_action
            ):
                loop_tag = " ⚠ same action still failing"

            # step_num matches the 0-based step index used in the action history prompt
            hist_lines.append(
                f"- Step {step_num}{reason_tag}: "
                f"action={action_str}; "
                f"feedback={feedback}"
                f"{loop_tag}"
            )

            prev_action = norm_action
            prev_failed = success_val is False

        sections.append("\n".join(hist_lines))

        body = "\n\n".join(sections)
        return body

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.storage_path
        if not target:
            return
        data = {
            "max_steps":     self.max_steps,
            "max_summaries": self.max_summaries,
            "steps":         [s.to_dict() for s in self.steps],
            "summaries":     list(self.summaries),
            "_total_steps_added": self._total_steps_added,
        }
        save_json(target, data)

    def load(self, path: Optional[str] = None) -> None:
        target = path or self.storage_path
        if not target:
            return
        data = load_json(target, default=None)
        if data is None:
            return  # missing file — keep empty state
        self.max_steps     = int(data.get("max_steps",     self.max_steps))
        self.max_summaries = int(data.get("max_summaries", self.max_summaries))
        self.steps         = [TemporalStep.from_dict(d) for d in (data.get("steps") or [])]
        self.summaries     = list(data.get("summaries") or [])
        self._total_steps_added = int(data.get("_total_steps_added", len(self.steps)))

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Clear all steps and summaries for a new episode."""
        self.steps = []
        self.summaries = []
        self._total_steps_added = 0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TemporalMemory(steps={len(self.steps)}, "
            f"summaries={len(self.summaries)}, "
            f"max_steps={self.max_steps})"
        )


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _normalize_action_key(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for action deduplication."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())
