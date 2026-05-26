"""
memory/semantic_memory.py

SemanticMemory: persistent cross-episode knowledge store.

Lifecycle
---------
1. Seeded with hardcoded commonsense defaults on construction.
2. After each episode: ``MemoryCreator`` calls the VLM for candidate facts;
   ``MemoryUpdater`` decides add/update/remove/noop per candidate.
3. ``retrieve(query, top_k)`` returns top-k relevant facts.

Both ``MemoryCreator`` and ``MemoryUpdater`` require a ``vlm_call`` to
function; without one, no new facts are generated.

Memory Adapter: ``SemanticFact.to_memory_item()`` produces standard
``MemoryItem`` objects the adapter can consume directly.
"""

from __future__ import annotations

import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from embodiedbench.memory.base import (
    BaseMemory,
    MemoryItem,
    MemoryQuery,
    RetrievedMemory,
    UpdateDecision,
    normalize_text,
)
from embodiedbench.memory.embeddings import (
    EmbeddingProvider,
)
from embodiedbench.memory.storage import load_json, save_json
from embodiedbench.memory.utils import (
    similarity as _similarity,
    set_overlap as _overlap,
    list_union as _merge_lists,
)


# Valid semantic categories
_VALID_CATEGORIES = frozenset({
    "precondition", "affordance", "safety", "failure_avoidance",
    "search_strategy", "task_rule", "environment_rule", "general",
})


# ---------------------------------------------------------------------------
# SemanticFact
# ---------------------------------------------------------------------------

@dataclass
class SemanticFact:
    """A single reusable semantic fact for embodied planning."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    category: str = "general"
    related_objects: list = field(default_factory=list)  # list[str]
    related_actions: list = field(default_factory=list)  # list[str]
    embedding: Optional[list] = None                     # list[float]

    def touch(self) -> None:
        """No-op retained for API compatibility."""
        pass

    def short_summary(self, max_chars: int = 300) -> str:
        text = f"[{self.category}] {self.content}"
        return text[:max_chars] if len(text) > max_chars else text

    def to_dict(self) -> dict:
        return {
            "id":                self.id,
            "content":           self.content,
            "category":          self.category,
            "related_objects":   list(self.related_objects),
            "related_actions":   list(self.related_actions),
            "embedding":         self.embedding,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticFact":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            content=d.get("content", ""),
            category=d.get("category", "general"),
            related_objects=list(d.get("related_objects") or []),
            related_actions=list(d.get("related_actions") or []),
            embedding=d.get("embedding"),
        )

    def to_memory_item(self) -> MemoryItem:
        parts = [self.content]
        if self.related_objects:
            parts.append(f"objects: {', '.join(self.related_objects)}")
        if self.related_actions:
            parts.append(f"actions: {', '.join(self.related_actions)}")
        return MemoryItem(
            memory_type="semantic",
            content=" | ".join(parts),
            metadata={
                "fact_id":         self.id,
                "category":        self.category,
                "related_objects": list(self.related_objects),
                "related_actions": list(self.related_actions),
            },
        )

# ---------------------------------------------------------------------------
# SemanticVectorStore
# ---------------------------------------------------------------------------

class SemanticVectorStore:
    """
    Thin vector-database layer over a shared ``list[SemanticFact]``.

    Holds a reference to ``SemanticMemory.facts`` so mutations are immediately
    reflected in subsequent queries.
    """

    def __init__(
        self,
        facts: list,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self._facts = facts  # shared mutable reference
        self._embedding_provider = embedding_provider

    def embed(self, text: str) -> Optional[list]:
        """Return embedding for *text*, or ``None`` if provider is unavailable."""
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
        Return up to *k* ``(score, SemanticFact)`` tuples sorted by
        descending similarity to *text*.
        """
        if not text or not self._facts:
            return []
        exclude_ids = exclude_ids or set()
        query_norm = normalize_text(text)
        query_emb = self.embed(text)

        scored = [
            (_similarity(query_norm, normalize_text(f.content), query_emb, f.embedding), f)
            for f in self._facts
            if f.id not in exclude_ids and f.content
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# MemoryCreator
# ---------------------------------------------------------------------------

class MemoryCreator:
    """
    Generates candidate fact dicts from an episode summary via VLM.

    Returns ``[]`` when no ``vlm_call`` is provided or the response cannot
    be parsed.
    """

    _CANDIDATE_SCHEMA = (
        '{"facts": [{'
        '"content": "...", '
        '"category": "precondition|affordance|safety|failure_avoidance|search_strategy|recovery|task_rule|general", '
        '"related_objects": [...], '
        '"related_actions": [...]'
        "}]}"
    )

    def __init__(
        self,
        vlm_call: Optional[Callable[[str], str]] = None,
        max_candidates: int = 4,
    ) -> None:
        self.vlm_call = vlm_call
        self.max_candidates = max_candidates

    def create_from_episode(
        self,
        episode_summary: str,
        task_instruction: str = "",
        success: Optional[bool] = None,
    ) -> list:
        """
        Generate candidate fact dicts for this episode via VLM.

        Returns an empty list when no ``vlm_call`` is configured or when the
        VLM response cannot be parsed.  Each dict has keys: ``content``,
        ``category``, ``related_objects``, ``related_actions``, ``source_episode_id``.
        """
        if self.vlm_call is None:
            return []
        try:
            candidates = self._vlm_create(
                episode_summary, task_instruction, success,
            )
            if candidates:
                return candidates[: self.max_candidates]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # VLM internals
    # ------------------------------------------------------------------

    def _vlm_create(
        self,
        episode_summary: str,
        task_instruction: str,
        success: Optional[bool],
    ) -> list:
        if success is True:
            outcome = "success"
        elif success is False:
            outcome = "failure"
        else:
            outcome = "unknown"
        prompt = (
            "You are the Long-Term Semantic Memory Creator for an embodied robot. "
            "Convert this completed episode into stable, task-relevant facts for future episodes.\n\n"
            f"Task: {task_instruction}\n"
            f"Outcome: {outcome}\n"
            f"Episode summary: {episode_summary}\n\n"
            "Semantic memory should store time-independent knowledge, not episodic details.\n\n"
            "RULES:\n"
            "- Extract only reusable facts about object affordances, action preconditions, search priors, "
            "failure causes, recovery patterns, or safety constraints.\n"
            "- Do NOT store exact locations, receptacle IDs, room-specific placements, navigation order, "
            "or single-episode object positions.\n"
            "- Generalize from specific objects to object categories when appropriate.\n"
            "- For failed episodes, store what pattern failed and the better alternative.\n"
            "- For successful episodes, store only patterns that transfer to future tasks.\n"
            "- Do not repeat action-rule basics unless the episode reveals a non-obvious constraint.\n"
            "- Each fact must be concise and useful for at least 3 future tasks.\n"
            "- Return an empty list if the episode contains no reusable semantic fact.\n\n"
            "Categories: precondition|affordance|failure_avoidance|search_strategy|recovery|task_rule|safety|general\n\n"
            "JSON only:\n"
            f"{self._CANDIDATE_SCHEMA}"
        )
        raw = self.vlm_call(prompt)
        return self._parse_vlm_candidates(raw)

    @staticmethod
    def _parse_vlm_candidates(raw: str) -> list:
        """Parse JSON from VLM response; return [] on any error."""
        if not raw:
            return []
        # Extract first {...} block
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group())
            facts = parsed.get("facts", [])
            result = []
            for f in facts:
                if not isinstance(f, dict) or not f.get("content"):
                    continue
                category = f.get("category", "general")
                if category not in _VALID_CATEGORIES:
                    category = "general"
                result.append({
                    "content": str(f["content"]).strip(),
                    "category": category,
                    "related_objects": list(f.get("related_objects") or []),
                    "related_actions": list(f.get("related_actions") or []),
                })
            return result
        except (json.JSONDecodeError, TypeError, KeyError):
            return []


# ---------------------------------------------------------------------------
# MemoryUpdater
# ---------------------------------------------------------------------------

class MemoryUpdater:
    """
    Decides what to do with each candidate fact produced by ``MemoryCreator``.

    Compares each candidate against the top-S similar existing facts and
    returns an ``UpdateDecision`` (add / update / remove / noop) via VLM.
    Falls back to ``noop`` when no ``vlm_call`` is provided.
    """

    def __init__(
        self,
        vector_store: SemanticVectorStore,
        vlm_call: Optional[Callable[[str], str]] = None,
        top_s: int = 3,
        dedup_threshold: float = 0.85,
    ) -> None:
        self._store = vector_store
        self.vlm_call = vlm_call
        self.top_s = top_s
        self.dedup_threshold = dedup_threshold

    def process(self, candidate: dict) -> UpdateDecision:
        """Return an ``UpdateDecision`` for *candidate* (must have a ``"content"`` key)."""
        content = candidate.get("content", "").strip()
        if not content:
            return UpdateDecision("noop")

        if self.vlm_call is None:
            return UpdateDecision("noop")

        similar = self._store.top_k_similar(content, self.top_s)

        try:
            return self._vlm_process(content, similar)
        except Exception:
            return UpdateDecision("add")

    def _vlm_process(
        self,
        content: str,
        similar: list,
    ) -> UpdateDecision:
        if not similar:
            return UpdateDecision("add")

        items_text = "\n".join(
            f"{i+1}. [score={score:.2f}] \"{fact.content}\""
            for i, (score, fact) in enumerate(similar)
        )
        prompt = (
            "You are managing a semantic memory database for an embodied robot.\n\n"
            f"New candidate memory item:\n\"{content}\"\n\n"
            f"Most similar existing memory items (top-{len(similar)}):\n{items_text}\n\n"
            "Decide what to do with the candidate. Choose ONE action:\n"
            "- \"add\"    : the candidate contains new, useful information\n"
            "- \"update\" : the candidate improves or corrects an existing item\n"
            "- \"remove\" : an existing item is now outdated because of the candidate\n"
            "- \"noop\"   : the candidate is fully redundant with existing items\n\n"
            "Respond with JSON only (no markdown):\n"
            "{\"action\": \"add\"} or\n"
            "{\"action\": \"update\", \"target_index\": 1, \"new_content\": \"...\"} or\n"
            "{\"action\": \"remove\", \"target_index\": 2} or\n"
            "{\"action\": \"noop\"}"
        )
        raw = self.vlm_call(prompt)
        return self._parse_vlm_decision(raw, similar)

    @staticmethod
    def _parse_vlm_decision(raw: str, similar: list) -> UpdateDecision:
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
            target_id = similar[idx][1].id
            if action == "remove":
                return UpdateDecision("remove", target_id=target_id)
            return UpdateDecision("update", target_id=target_id, new_content=parsed.get("new_content") or None)
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            return UpdateDecision("add")


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class SemanticMemory(BaseMemory):
    """
    Persistent, cross-episode knowledge store backed by a vector database.

    Facts are stored in a ``SemanticVectorStore`` that supports cosine /
    lexical similarity retrieval.  New facts are proposed by ``MemoryCreator``
    and adjudicated by ``MemoryUpdater`` (VLM-enhanced when ``vlm_call`` is
    supplied).  Updates across independent candidates are parallelised via
    ``ThreadPoolExecutor``.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        storage_path: Optional[str] = None,
        dedup_threshold: float = 0.85,
        max_facts: Optional[int] = None,
        vlm_call: Optional[Callable[[str], str]] = None,
        top_s: int = 3,
        max_update_workers: int = 4,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.storage_path = storage_path
        self.dedup_threshold = dedup_threshold
        self.max_facts = max_facts
        self.facts: list[SemanticFact] = []

        self._vector_store = SemanticVectorStore(self.facts, embedding_provider)
        self._creator = MemoryCreator(vlm_call=vlm_call)
        self._updater = MemoryUpdater(
            vector_store=self._vector_store,
            vlm_call=vlm_call,
            top_s=top_s,
            dedup_threshold=dedup_threshold,
        )
        self._max_update_workers = max_update_workers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[list]:
        """Return an embedding for *text*, or None if unavailable."""
        if self.embedding_provider is None:
            return None
        try:
            return self.embedding_provider.embed_text(text)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Fact management
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        related_objects: Optional[list] = None,
        related_actions: Optional[list] = None,
        fact_id: Optional[str] = None,
    ) -> SemanticFact:
        """
        Add a fact, merging into an existing near-duplicate if one exists.

        Returns the added or updated SemanticFact.
        """
        content = content.strip()
        if not content:
            raise ValueError("Fact content must not be empty.")

        related_objects = list(related_objects or [])
        related_actions = list(related_actions or [])

        # Deduplication: merge into existing near-duplicate if found
        existing = self.find_similar_fact(content)
        if existing is not None:
            existing.related_objects = _merge_lists(existing.related_objects, related_objects)
            existing.related_actions = _merge_lists(existing.related_actions, related_actions)
            if existing.embedding is None:
                existing.embedding = self._embed(existing.content)
            existing.touch()
            return existing

        fact = SemanticFact(
            id=fact_id or str(uuid.uuid4()),
            content=content,
            category=category,
            related_objects=related_objects,
            related_actions=related_actions,
            embedding=self._embed(content),
        )
        self.facts.append(fact)
        self._enforce_max_facts()
        return fact

    def update_fact(self, fact_id: str, **updates) -> Optional[SemanticFact]:
        """
        Update fields of an existing fact.
        If *content* changes, the embedding is recomputed.
        Returns the updated fact, or None if not found.
        """
        fact = self._find_by_id(fact_id)
        if fact is None:
            return None

        content_changed = False
        for key, val in updates.items():
            if key == "content":
                fact.content = str(val).strip()
                content_changed = True
            elif key == "category":
                fact.category = str(val)
            elif key == "related_objects":
                fact.related_objects = list(val)
            elif key == "related_actions":
                fact.related_actions = list(val)

        if content_changed:
            fact.embedding = self._embed(fact.content)

        fact.touch()
        return fact

    def remove_fact(self, fact_id: str) -> bool:
        """Remove a fact by id. Returns True if found and removed."""
        for i, f in enumerate(self.facts):
            if f.id == fact_id:
                self.facts.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_facts_by_category(self, category: str) -> list:
        """Return all facts whose category matches exactly."""
        return [f for f in self.facts if f.category == category]

    def get_facts_for_object(self, object_name: str) -> list:
        """Return facts that mention *object_name* in related_objects or content."""
        name_lower = object_name.lower()
        result = []
        for f in self.facts:
            obj_match = any(name_lower in o.lower() for o in f.related_objects)
            content_match = name_lower in f.content.lower()
            if obj_match or content_match:
                result.append(f)
        return result

    def get_facts_for_action(self, action_name: str) -> list:
        """Return facts that mention *action_name* in related_actions or content."""
        name_lower = action_name.lower()
        result = []
        for f in self.facts:
            act_match = any(name_lower in a.lower() for a in f.related_actions)
            content_match = name_lower in f.content.lower()
            if act_match or content_match:
                result.append(f)
        return result

    def find_similar_fact(self, content: str) -> Optional[SemanticFact]:
        """
        Return the first existing fact that is similar enough to *content*
        to be considered a duplicate, or None.

        Uses embedding cosine similarity when available, falls back to lexical
        overlap.  Threshold: self.dedup_threshold.
        """
        norm_content = normalize_text(content)
        query_emb = self._embed(content)

        for fact in self.facts:
            score = _similarity(
                norm_content, normalize_text(fact.content),
                query_emb, fact.embedding,
            )
            if score >= self.dedup_threshold:
                return fact
        return None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: MemoryQuery, top_k: int = 5) -> list:
        """
        Score every fact against the query and return top_k ``RetrievedMemory``.

        Weighted sum: base_sim×0.90, act_overlap×0.10.
        """
        if not self.facts:
            return []

        query_text = query.text_for_retrieval() or query.task_instruction
        query_text_norm = normalize_text(query_text) if query_text else ""
        query_actions = set(
            a.lower()
            for a in (query.recent_actions or [])
        )

        query_emb: Optional[list] = None
        if query_text:
            query_emb = self._embed(query_text)

        results: list[tuple[float, str, SemanticFact]] = []

        for fact in self.facts:
            fact_text = fact.content
            fact_emb = fact.embedding

            # 1. Base similarity (normalized text on both sides)
            base_sim = _similarity(query_text_norm, normalize_text(fact_text), query_emb, fact_emb)

            # 2. Action overlap bonus  [0, 1]
            fact_actions = set(a.lower() for a in fact.related_actions)
            act_bonus = _overlap(query_actions, fact_actions)
            if not act_bonus and query_actions:
                # word-boundary check to avoid substring false-positives
                fact_words = set(normalize_text(fact_text).split())
                matches = sum(1 for a in query_actions if a in fact_words)
                act_bonus = min(1.0, matches / len(query_actions))

            score = min(1.0, max(0.0,
                0.90 * base_sim
                + 0.10 * act_bonus
            ))

            reason = _reason_for_category(fact.category)
            results.append((score, reason, fact))

        results.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedMemory(item=fact.to_memory_item(), score=score, reason=reason)
            for score, reason, fact in results[:top_k]
        ]

    # ------------------------------------------------------------------
    # Episode-derived fact extraction (new pipeline)
    # ------------------------------------------------------------------

    def update_from_episode(
        self,
        episode_summary: str,
        task_instruction: str = "",
        success: Optional[bool] = None,
        episode_id: Optional[str] = None,
    ) -> list:
        """
        Full Creator → VectorStore → Updater pipeline for one episode.

        Generates candidate facts, adjudicates each via the updater (add /
        update / remove / noop) concurrently, then applies decisions serially.
        Returns the list of ``SemanticFact`` objects that were added or updated.
        """
        candidates = self._creator.create_from_episode(
            episode_summary=episode_summary,
            task_instruction=task_instruction,
            success=success,
        )
        if not candidates:
            return []

        # Collect decisions in parallel (safe: each worker only reads; writes
        # are serialized in the main thread below)
        decisions: list[tuple[dict, UpdateDecision]] = []
        with ThreadPoolExecutor(max_workers=min(self._max_update_workers, len(candidates))) as pool:
            future_map = {
                pool.submit(self._updater.process, cand): cand
                for cand in candidates
            }
            for future in as_completed(future_map):
                cand = future_map[future]
                try:
                    decision = future.result()
                except Exception:
                    decision = UpdateDecision("add")
                decisions.append((cand, decision))

        # Apply decisions serially to avoid race conditions on self.facts
        affected: list[SemanticFact] = []
        for cand, decision in decisions:
            fact = self._apply_decision(cand, decision)
            if fact is not None:
                affected.append(fact)

        return affected

    def _apply_decision(
        self,
        candidate: dict,
        decision: UpdateDecision,
    ) -> Optional["SemanticFact"]:
        """Apply a single ``UpdateDecision`` and return the affected fact (if any)."""
        if decision.action == "noop":
            return None

        if decision.action == "remove" and decision.target_id:
            self.remove_fact(decision.target_id)
            return None

        if decision.action == "update" and decision.target_id:
            updates: dict = {}
            if decision.new_content:
                updates["content"] = decision.new_content
            existing = self._find_by_id(decision.target_id)
            if existing is None:
                # fall through to add
                pass
            else:
                existing.related_objects = _merge_lists(
                    existing.related_objects,
                    list(candidate.get("related_objects") or []),
                )
                existing.related_actions = _merge_lists(
                    existing.related_actions,
                    list(candidate.get("related_actions") or []),
                )
                if updates:
                    self.update_fact(decision.target_id, **updates)
                else:
                    existing.touch()
                return existing

        # action == "add"  (or update with missing target — treat as add)
        return self.add_fact(
            content=candidate["content"],
            category=candidate.get("category", "general"),
            related_objects=list(candidate.get("related_objects") or []),
            related_actions=list(candidate.get("related_actions") or []),
        )

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def to_prompt_context(self, memories: list) -> str:
        if not memories:
            return ""

        lines = ["Relevant rules and commonsense knowledge:"]
        for rm in memories:
            meta = rm.item.metadata
            category = meta.get("category", "general")
            content = rm.item.content.split(" | ")[0]   # drop appended object/action lists
            lines.append(f"- [{category}] {content}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.storage_path
        if not target:
            return
        data = {
            "dedup_threshold": self.dedup_threshold,
            "max_facts":       self.max_facts,
            "facts":           [f.to_dict() for f in self.facts],
        }
        save_json(target, data)

    def load(self, path: Optional[str] = None) -> None:
        target = path or self.storage_path
        if not target:
            return
        data = load_json(target, default=None)
        if data is None:
            return
        self.dedup_threshold = float(data.get("dedup_threshold", self.dedup_threshold))
        self.max_facts = data.get("max_facts", self.max_facts)
        loaded = [SemanticFact.from_dict(d) for d in (data.get("facts") or [])]
        # Replace list contents so vector store shared reference stays valid
        self.facts.clear()
        self.facts.extend(loaded)

    # ------------------------------------------------------------------
    # BaseMemory: update() and reset_episode()
    # ------------------------------------------------------------------

    def update(self, *args, **kwargs) -> None:
        """Delegates to ``add_fact()`` when called with a ``content`` kwarg; otherwise no-op."""
        if "content" in kwargs:
            _KEYS = {"content", "category", "related_objects", "related_actions", "fact_id"}
            self.add_fact(**{k: v for k, v in kwargs.items() if k in _KEYS})

    def reset_episode(self) -> None:
        """Semantic memory is persistent — episode resets are intentional no-ops."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_by_id(self, fact_id: str) -> Optional[SemanticFact]:
        return next((f for f in self.facts if f.id == fact_id), None)

    def _enforce_max_facts(self) -> None:
        """If max_facts is set and exceeded, drop the oldest facts (FIFO)."""
        if self.max_facts is None or len(self.facts) <= self.max_facts:
            return
        del self.facts[self.max_facts:]

    def __len__(self) -> int:
        return len(self.facts)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SemanticMemory(facts={len(self.facts)}, threshold={self.dedup_threshold})"


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _reason_for_category(category: str) -> str:
    _MAP = {
        "precondition":      "relevant precondition",
        "affordance":        "relevant object affordance",
        "safety":            "safety constraint",
        "failure_avoidance": "failure avoidance rule",
        "search_strategy":   "search strategy hint",
        "task_rule":         "task rule",
        "environment_rule":  "environment rule",
    }
    return _MAP.get(category, "semantic similarity")
