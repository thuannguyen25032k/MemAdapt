"""
memory/episodic_memory.py

EpisodicMemory — persistent store of task-specific execution trajectories.

Lifecycle
---------
1. After each episode ``MemoryManager.finalize_episode()`` calls
   ``add_episode_from_trajectory()``, building an ``EpisodeRecord`` and
   running the VLM-adjudicated add/update/remove/noop pipeline.
2. ``retrieve(query, top_k)`` returns the top-k scored ``RetrievedMemory`` objects.

No-VLM behaviour: always ``add`` (trajectories are never silently discarded).
Memory Adapter: ``EpisodeRecord.to_memory_item()`` exposes full trajectory text.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from embodiedbench.memory.base import (
    BaseMemory,
    MemoryItem,
    MemoryQuery,
    RetrievedMemory,
    UpdateDecision,
    normalize_text,
    now_ts,
)
from embodiedbench.memory.embeddings import EmbeddingProvider
from embodiedbench.memory.storage import load_json, save_json
from embodiedbench.memory.utils import (
    similarity as _similarity,
)


# ---------------------------------------------------------------------------
# EpisodeRecord
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """
    A single task-specific execution trajectory.

    Attributes
    ----------
    steps :
        Ordered list of ``{"step_id": int, "action": str, "feedback": str}``
        dicts, extracted from ``TemporalMemory`` at episode end.
    final_status :
        One of ``"success"``, ``"failure"``, ``"partial"``, ``"unknown"``.
    """    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_instruction: str = ""
    env_name: str = ""
    task_type: str = ""              # e.g. eval_set name ("base", "spatial", ...); "" if unknown
    scene_name: str = ""             # AI2-THOR sceneName (e.g. "FloorPlan3"); "" for Habitat
    final_status: str = "unknown"    # "success" | "failure" | "partial" | "unknown"
    steps: list = field(default_factory=list)            # list[dict]
    embedding: Optional[list] = None
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)

    def touch(self) -> None:
        """Refresh ``updated_at`` to the current timestamp."""
        self.updated_at = now_ts()

    def trajectory_text(self) -> str:
        """
        Human-readable step-by-step trajectory string, including failed steps.

        Example::

            Step 0: navigate to the left counter in the kitchen.
            Step 1: pick up the cup. Env feedback: The action executed successfully.
        """
        if not self.steps:
            return "[No steps recorded]"
        return "\n".join(_format_step(i, s) for i, s in enumerate(self.steps))

    def text_for_retrieval(self) -> str:
        """Return the human instruction only — retrieval matches on task intent."""
        return self.task_instruction or ""

    def to_memory_item(self) -> MemoryItem:
        """Expose as a ``MemoryItem`` for the retrieval pipeline and Memory Adapter."""
        content_parts = [
            f"Human instruction: {self.task_instruction}",
            f"Outcome: {self.final_status}",
        ]
        content_parts.append(self.trajectory_text())  # full sequence

        return MemoryItem(
            memory_type="episodic",
            content="\n".join(content_parts),
            metadata={
                "episode_id":       self.id,
                "env_name":         self.env_name,
                "task_type":        self.task_type,
                "scene_name":       self.scene_name,
                "status":           self.final_status,
            },
            embedding=self.embedding,
            source="episodic_memory",
        )

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "task_instruction": self.task_instruction,
            "env_name":         self.env_name,
            "task_type":        self.task_type,
            "scene_name":       self.scene_name,
            "final_status":     self.final_status,
            "steps":            list(self.steps),
            "embedding":        self.embedding,
            "created_at":       self.created_at,
            "updated_at":       self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeRecord":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            task_instruction=d.get("task_instruction", ""),
            env_name=d.get("env_name", ""),
            task_type=d.get("task_type", "") or d.get("scene_id", ""),
            scene_name=d.get("scene_name", ""),
            final_status=d.get("final_status", "unknown"),
            steps=list(d.get("steps") or []),
            embedding=d.get("embedding"),
            created_at=float(d.get("created_at", now_ts())),
            updated_at=float(d.get("updated_at", now_ts())),
        )


# ---------------------------------------------------------------------------
# EpisodicVectorStore
# ---------------------------------------------------------------------------

class EpisodicVectorStore:
    """
    Thin vector-database layer over a shared ``list[EpisodeRecord]``.

    Holds a *reference* to ``EpisodicMemory.episodes`` so mutations are
    immediately reflected in subsequent queries.
    """

    def __init__(
        self,
        episodes: list,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self._episodes = episodes  # shared mutable reference
        self._embedding_provider = embedding_provider

    def embed(self, text: str) -> Optional[list]:
        """Return an embedding vector for *text*, or ``None`` on failure."""
        if self._embedding_provider is None or not text:
            return None
        try:
            return self._embedding_provider.embed_text(text)
        except Exception:
            return None

    def top_k_similar(
        self,
        text: str,
        k: int,
        exclude_ids: Optional[set] = None,
    ) -> list:
        """
        Return up to *k* ``(score, EpisodeRecord)`` tuples sorted by
        descending hybrid (cosine + lexical) similarity to *text*.
        """
        if not text or not self._episodes:
            return []
        exclude_ids = exclude_ids or set()
        query_norm = normalize_text(text)
        query_emb = self.embed(text)

        scored = [
            (_similarity(query_norm, normalize_text(ep.text_for_retrieval()),
                         query_emb, ep.embedding), ep)
            for ep in self._episodes
            if ep.id not in exclude_ids and ep.text_for_retrieval()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# EpisodicUpdater
# ---------------------------------------------------------------------------

class EpisodicUpdater:
    """
    Decides what to do with a new ``EpisodeRecord``.

    Retrieves top-S similar episodes, calls the VLM to choose
    ``add`` / ``update`` / ``remove`` / ``noop``, then returns an
    ``UpdateDecision``.  Falls back to ``add`` without a ``vlm_call``.
    """

    def __init__(
        self,
        vector_store: EpisodicVectorStore,
        vlm_call: Optional[Callable[[str], str]] = None,
        top_s: int = 3,
    ) -> None:
        self._store = vector_store
        self.vlm_call = vlm_call
        self.top_s = top_s

    def process(self, record: EpisodeRecord) -> UpdateDecision:
        """Return an ``UpdateDecision`` for *record*."""
        if not record.task_instruction:
            return UpdateDecision("noop")
        if self.vlm_call is None:
            return UpdateDecision("add")
        similar = self._store.top_k_similar(
            record.text_for_retrieval(), self.top_s, exclude_ids={record.id}
        )
        try:
            return self._vlm_process(record, similar)
        except Exception:
            return UpdateDecision("add")

    def _vlm_process(self, record: EpisodeRecord, similar: list) -> UpdateDecision:
        if not similar:
            return UpdateDecision("add")

        new_traj = record.trajectory_text()
        items_text = "\n".join(
            f"{i+1}. [score={score:.2f}] Task: \"{ep.task_instruction}\" "
            f"({ep.final_status}) | {ep.trajectory_text()}"
            for i, (score, ep) in enumerate(similar)
        )
        prompt = (
            "You are managing an episodic memory database for an embodied robot.\n\n"
            f"New episode:\n"
            f"Task: \"{record.task_instruction}\"\n"
            f"Outcome: {record.final_status}\n"
            f"{new_traj}\n\n"
            f"Most similar existing episodes (top-{len(similar)}):\n"
            f"{items_text}\n\n"
            "Decide what to do with the new episode. Choose ONE action:\n"
            "- \"add\"    : genuinely new trajectory information worth recording\n"
            "- \"update\" : better or more complete version of an existing episode\n"
            "- \"remove\" : an existing episode is now obsolete\n"
            "- \"noop\"   : fully redundant with an existing episode\n\n"
            "Respond with JSON only (no markdown):\n"
            "{\"action\": \"add\"} or\n"
            "{\"action\": \"update\", \"target_index\": 1} or\n"
            "{\"action\": \"remove\", \"target_index\": 2} or\n"
            "{\"action\": \"noop\"}"
        )
        return self._parse_decision(self.vlm_call(prompt), similar)

    @staticmethod
    def _parse_decision(raw: str, similar: list) -> UpdateDecision:
        """Parse the VLM JSON response into an ``UpdateDecision``."""
        if not raw:
            return UpdateDecision("add")
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return UpdateDecision("add")
        try:
            parsed = json.loads(match.group())
            action = str(parsed.get("action", "add")).lower()
            if action not in ("add", "update", "remove", "noop"):
                action = "add"
            if action in ("noop", "add"):
                return UpdateDecision(action)
            idx = max(0, min(int(parsed.get("target_index", 1)) - 1, len(similar) - 1))
            return UpdateDecision(action, target_id=similar[idx][1].id)
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            return UpdateDecision("add")


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory(BaseMemory):
    """
    Persistent, cross-episode store of task execution trajectories.

    New episodes are adjudicated by ``EpisodicUpdater`` (VLM-driven when
    ``vlm_call`` is supplied) before being stored in ``EpisodicVectorStore``.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        storage_path: Optional[str] = None,
        max_episodes: Optional[int] = None,
        vlm_call: Optional[Callable[[str], str]] = None,
        top_s: int = 3,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.storage_path = storage_path
        self.max_episodes = max_episodes
        self.episodes: list[EpisodeRecord] = []

        self._vector_store = EpisodicVectorStore(self.episodes, embedding_provider)
        self._updater = EpisodicUpdater(
            vector_store=self._vector_store,
            vlm_call=vlm_call,
            top_s=top_s,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_episode_from_trajectory(
        self,
        task_instruction: str,
        final_status: str = "unknown",
        steps: Optional[list] = None,
        env_name: str = "",
        task_type: str = "",
        scene_name: str = "",
        episode_id: Optional[str] = None,
    ) -> Optional[EpisodeRecord]:
        """
        Build an ``EpisodeRecord`` from raw trajectory data, run the VLM
        update pipeline, and store the result.  Returns the stored record,
        or ``None`` if the updater decided ``noop``.

        Only successful trajectories are stored; failed episodes are skipped.
        """
        task_instruction = (task_instruction or "").strip()
        if not task_instruction:
            return None

        # Only successful trajectories are worth storing.
        if final_status != "success":
            return None

        cleaned_steps: list = list(steps or [])

        record = EpisodeRecord(
            id=episode_id or str(uuid.uuid4()),
            task_instruction=task_instruction,
            env_name=env_name,
            task_type=task_type,
            scene_name=scene_name,
            final_status=final_status,
            steps=cleaned_steps,
        )

        if self.embedding_provider is not None:
            try:
                record.embedding = self.embedding_provider.embed_text(
                    record.text_for_retrieval()
                )
            except Exception:
                record.embedding = None

        return self._apply_decision(record, self._updater.process(record))

    def _apply_decision(
        self,
        record: EpisodeRecord,
        decision: UpdateDecision,
    ) -> Optional[EpisodeRecord]:
        if decision.action == "noop":
            return None

        if decision.action == "remove" and decision.target_id:
            self._remove_by_id(decision.target_id)
            self.episodes.append(record)
            self._enforce_max_episodes()
            return record

        if decision.action == "update" and decision.target_id:
            existing = self._find_by_id(decision.target_id)
            if existing is not None:
                existing.task_instruction = record.task_instruction
                existing.final_status     = record.final_status
                existing.steps            = record.steps
                existing.embedding        = record.embedding
                existing.touch()
                return existing

        # "add", or "update" whose target was not found (fallback)
        self.episodes.append(record)
        self._enforce_max_episodes()
        return record

    def update(self, *args, **kwargs) -> None:
        """No-op — use ``add_episode_from_trajectory()`` instead."""

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: MemoryQuery, top_k: int = 5) -> list:
        """
        Score every stored episode against *query* and return the top_k as
        ``RetrievedMemory`` objects.

        Scoring weights: base_sim×0.9, scene_match×0.1.
        """
        if not self.episodes:
            return []

        query_text = query.task_instruction.strip()
        if not query_text:
            return []
        query_text_norm = normalize_text(query_text)

        query_emb: Optional[list] = None
        if self.embedding_provider is not None:
            try:
                query_emb = self.embedding_provider.embed_text(query_text)
            except Exception:
                pass

        results = []
        for ep in self.episodes:
            # 1. Base semantic similarity (normalized text)
            base_sim = _similarity(query_text_norm, normalize_text(ep.text_for_retrieval()), query_emb, ep.embedding)

            # 2. Scene / env match bonus
            scene_bonus = 1.0 if (query.scene_name and ep.scene_name == query.scene_name) else 0.0

            score = min(1.0, max(0.0,
                0.9 * base_sim
                + 0.1 * scene_bonus
            ))
            results.append((score, ep))

        results.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedMemory(item=ep.to_memory_item(), score=score, reason=_episode_reason(ep))
            for score, ep in results[0:top_k]  # skip the top-1 match (too close to be useful as a "similar episode")
        ]

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def successful_trajectory_guide(self, memories: list) -> str:
        """
        Return a step-by-step guide from the best successful episode in
        *memories*, or ``""`` when no successful episode is present.
        """
        success_eps = [
            rm for rm in memories
            if rm.item.metadata.get("status") == "success"
        ]
        if not success_eps:
            return ""

        # Use the highest-scored successful episode
        best = success_eps[0]
        ep = self._find_by_id(best.item.metadata.get("episode_id", ""))
        if ep is None:
            # Fall back to content stored in the MemoryItem
            return f"Successful trajectory guide (follow these steps):\n{best.item.content}"

        lines = [
            "Successful trajectory guide (follow these steps):",
            f"Human instruction: {ep.task_instruction}",
        ]
        lines.extend(_format_step(i, s) for i, s in enumerate(ep.steps))
        return "\n".join(lines)

    def to_prompt_context(self, memories: list) -> str:
        """
        Render retrieved successful episodes as a planner-ready context string.

        Each episode shows the task instruction followed by every step with
        its action and environment feedback.

        Example output::

            Similar successful episodes:
            Human instruction: Extract a cup from the left counter and move it to the sofa.
            Step 0: navigate to the left counter in the kitchen.
            Step 1: pick up the cup. Env feedback: The action executed successfully.
        """
        if not memories:
            return ""

        lines: list[str] = []
        lines.append("Similar successful episodes:")
        for rm in memories:
            ep = self._find_by_id(rm.item.metadata.get("episode_id", ""))
            if ep is None:
                lines.append(rm.item.content)
            else:
                traj   = ep.trajectory_text()
                lines.append(f"Human instruction: {ep.task_instruction}\n{traj}")

        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.storage_path
        if not target:
            return
        save_json(target, {
            "max_episodes": self.max_episodes,
            "episodes": [e.to_dict() for e in self.episodes],
        })

    def load(self, path: Optional[str] = None) -> None:
        target = path or self.storage_path
        if not target:
            return
        data = load_json(target, default=None)
        if data is None:
            return
        self.max_episodes = data.get("max_episodes", self.max_episodes)
        loaded = [EpisodeRecord.from_dict(d) for d in (data.get("episodes") or [])]
        # Mutate in-place to keep EpisodicVectorStore's shared reference valid.
        self.episodes.clear()
        self.episodes.extend(loaded)

    # ------------------------------------------------------------------
    # BaseMemory lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Episodic memory is persistent across episodes — this is a no-op."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_by_id(self, episode_id: str) -> Optional[EpisodeRecord]:
        return next((ep for ep in self.episodes if ep.id == episode_id), None)

    def _remove_by_id(self, episode_id: str) -> bool:
        for i, ep in enumerate(self.episodes):
            if ep.id == episode_id:
                self.episodes.pop(i)
                return True
        return False

    def _enforce_max_episodes(self) -> None:
        if self.max_episodes is None or len(self.episodes) <= self.max_episodes:
            return
        self.episodes.sort(key=lambda e: e.created_at, reverse=True)
        del self.episodes[self.max_episodes:]

    def __len__(self) -> int:
        return len(self.episodes)

    def __repr__(self) -> str:  # pragma: no cover
        return f"EpisodicMemory(episodes={len(self.episodes)})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_step(i: int, s: dict) -> str:

    act = (s.get("action") or "?").strip()
    action_part = act if act.endswith(('.', '!', '?')) else f"{act}."
    fb = (s.get("feedback") or "").strip()
    fb_part = f" Env feedback: {fb}" if fb else ""
    return f"Step {i}: {action_part}{fb_part}"


def _episode_reason(ep: EpisodeRecord) -> str:
    """Human-readable retrieval reason for a ``RetrievedMemory`` object."""
    _labels = {
        "success": "similar successful episode",
        "failure": "similar failed episode",
        "partial": "similar partial episode",
    }
    return _labels.get(ep.final_status, "similar episode")
