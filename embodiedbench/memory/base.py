"""
memory/base.py

Core data models and abstract interface for the MemAdapt memory system.
Designed to be environment-agnostic across EB-ALFRED, EB-Habitat,
EB-Navigation, and EB-Manipulation.
"""

from __future__ import annotations

import abc
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_ts() -> float:
    """Return current Unix timestamp."""
    return time.time()


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars* characters, appending '…' if cut."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "…"


def normalize_text(text: str) -> str:
    """Lower-case, collapse whitespace."""
    return " ".join(text.lower().split())


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Serialize *obj* to a JSON string.
    Non-serializable values are replaced with their repr() string.
    """
    def _default(o):
        return repr(o)
    return json.dumps(obj, default=_default, **kwargs)


# ---------------------------------------------------------------------------
# UpdateDecision
# ---------------------------------------------------------------------------

@dataclass
class UpdateDecision:
    """
    Decision returned by a memory updater (SemanticMemory or EpisodicMemory)
    for a single candidate item.

    action : str
        One of ``"add"``, ``"update"``, ``"remove"``, ``"noop"``.
    target_id : Optional[str]
        ID of the existing item to update or remove (None for add/noop).
    new_content : Optional[str]
        Replacement content when ``action == "update"``; None otherwise.
    """
    action: str                          # "add" | "update" | "remove" | "noop"
    target_id: Optional[str] = None
    new_content: Optional[str] = None


# ---------------------------------------------------------------------------
# MemoryItem
# ---------------------------------------------------------------------------

@dataclass
class MemoryItem:
    """
    A single memory entry.

    memory_type should be one of: "spatial", "temporal", "episodic", "semantic".
    The field is a plain string so callers are not constrained to that list.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = "episodic"
    content: str = ""
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list] = None        # list[float] when populated
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    importance: float = 0.5
    confidence: float = 1.0
    source: str = ""

    # ------------------------------------------------------------------
    def touch(self) -> None:
        """Update the *updated_at* timestamp to now."""
        self.updated_at = now_ts()

    def short_text(self, max_chars: int = 300) -> str:
        """Return a truncated version of *content* for display."""
        return truncate_text(self.content, max_chars)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "memory_type":  self.memory_type,
            "content":      self.content,
            "metadata":     self.metadata,
            "embedding":    self.embedding,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
            "importance":   self.importance,
            "confidence":   self.confidence,
            "source":       self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryItem":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            memory_type=d.get("memory_type", "episodic"),
            content=d.get("content", ""),
            metadata=d.get("metadata", {}),
            embedding=d.get("embedding"),
            created_at=d.get("created_at", now_ts()),
            updated_at=d.get("updated_at", now_ts()),
            importance=float(d.get("importance", 0.5)),
            confidence=float(d.get("confidence", 1.0)),
            source=d.get("source", ""),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MemoryItem(id={self.id!r}, type={self.memory_type!r}, "
            f"importance={self.importance:.2f}, content={self.short_text(60)!r})"
        )


# ---------------------------------------------------------------------------
# MemoryQuery
# ---------------------------------------------------------------------------

@dataclass
class MemoryQuery:
    """
    A query used to retrieve memories.

    Flexible enough for all four EB environments.
    """

    task_instruction: str = ""
    recent_actions: list = field(default_factory=list)        # list[str]
    env_name: Optional[str] = None
    scene_id: Optional[str] = None

    # ------------------------------------------------------------------
    def text_for_retrieval(self) -> str:
        """
        Combine all textual fields into a single retrieval string.
        This is the string that gets embedded / matched against stored memories.
        """
        parts: list[str] = []

        if self.task_instruction:
            parts.append(f"Task: {self.task_instruction}")
        if self.recent_actions:
            parts.append(f"Recent actions: {', '.join(self.recent_actions)}")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "task_instruction":     self.task_instruction,
            "recent_actions":       list(self.recent_actions),
            "env_name":             self.env_name,
            "scene_id":             self.scene_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryQuery":
        return cls(
            task_instruction=d.get("task_instruction", ""),
            recent_actions=list(d.get("recent_actions") or []),
            env_name=d.get("env_name"),
            scene_id=d.get("scene_id"),
        )


# ---------------------------------------------------------------------------
# RetrievedMemory
# ---------------------------------------------------------------------------

@dataclass
class RetrievedMemory:
    """A MemoryItem paired with its retrieval score and an optional reason."""

    item: MemoryItem = field(default_factory=MemoryItem)
    score: float = 0.0
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "item":   self.item.to_dict(),
            "score":  self.score,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievedMemory":
        return cls(
            item=MemoryItem.from_dict(d.get("item", {})),
            score=float(d.get("score", 0.0)),
            reason=d.get("reason"),
        )


# ---------------------------------------------------------------------------
# MemoryContext
# ---------------------------------------------------------------------------

@dataclass
class MemoryContext:
    """
    Aggregated memory context returned to the planner or critic.

    Sections map to the four memory types.
    """

    spatial_context: str = ""
    temporal_context: str = ""
    episodic_context: str = ""
    semantic_context: str = ""
    retrieved_items: list = field(default_factory=list)            # list[RetrievedMemory]

    # ------------------------------------------------------------------
    def is_empty(self) -> bool:
        """Return True when no meaningful memory content exists."""
        return not any([
            self.spatial_context,
            self.temporal_context,
            self.episodic_context,
            self.semantic_context,
            self.retrieved_items,
        ])

    # ------------------------------------------------------------------
    def compact(self, max_chars: int = 2000) -> str:
        """Return a compact combined string of all non-empty sections, truncated."""
        sections = []
        if self.spatial_context:
            sections.append(f"[Spatial Memory]\n{self.spatial_context}")
        if self.temporal_context:
            sections.append(f"[Temporal Memory]\n{self.temporal_context}")
        if self.episodic_context:
            sections.append(f"[Episodic Memory]\n{self.episodic_context}")
        if self.semantic_context:
            sections.append(f"[Semantic Memory]\n{self.semantic_context}")
        ctx = "\n\n".join(sections)
        return truncate_text(ctx, max_chars)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "spatial_context":  self.spatial_context,
            "temporal_context": self.temporal_context,
            "episodic_context": self.episodic_context,
            "semantic_context": self.semantic_context,
            "retrieved_items":  [r.to_dict() for r in self.retrieved_items],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryContext":
        return cls(
            spatial_context=d.get("spatial_context", ""),
            temporal_context=d.get("temporal_context", ""),
            episodic_context=d.get("episodic_context", ""),
            semantic_context=d.get("semantic_context", ""),
            retrieved_items=[
                RetrievedMemory.from_dict(r)
                for r in (d.get("retrieved_items") or [])
            ],
        )


# ---------------------------------------------------------------------------
# BaseMemory abstract class
# ---------------------------------------------------------------------------

class BaseMemory(abc.ABC):
    """
    Abstract base class for all MemAdapt memory implementations.

    Subclasses must implement update, retrieve, save, and load.
    The remaining methods have default implementations.
    """

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Store or update a memory entry from new experience."""

    @abc.abstractmethod
    def retrieve(self, query: MemoryQuery, top_k: int = 5) -> list:
        """
        Retrieve the *top_k* most relevant MemoryItems for *query*.

        Returns:
            list[RetrievedMemory]
        """

    @abc.abstractmethod
    def save(self, path: Optional[str] = None) -> None:
        """Persist memory state. Uses ``self.storage_path`` when *path* is None."""

    @abc.abstractmethod
    def load(self, path: Optional[str] = None) -> None:
        """Load persisted state. Uses ``self.storage_path`` when *path* is None."""

    # ------------------------------------------------------------------
    def to_prompt_context(self, memories: list) -> str:
        """
        Convert a list of RetrievedMemory objects into a prompt-ready string.

        Default implementation: numbered list of items sorted by score descending.
        Subclasses may override for richer formatting.
        """
        if not memories:
            return ""

        sorted_mems = sorted(memories, key=lambda m: m.score, reverse=True)
        lines: list[str] = ["[Retrieved Memories]"]
        for i, rm in enumerate(sorted_mems, 1):
            reason_suffix = f" ({rm.reason})" if rm.reason else ""
            line = f"{i}. [{rm.item.memory_type}]{reason_suffix} {rm.item.content}"
            lines.append(line)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """
        Clear any episode-scoped state (e.g., working memory).
        Override in subclasses that maintain per-episode buffers.
        Default: no-op.
        """
