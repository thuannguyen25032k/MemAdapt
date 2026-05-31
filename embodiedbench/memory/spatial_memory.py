"""SpatialMemory: 3-D scene-graph tracking objects, receptacles and their spatial
relationships across episode steps.

Lifecycle: build_from_metadata → update (each step) → retrieve → to_prompt_context.
"""

from __future__ import annotations

import uuid
import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from embodiedbench.memory.base import (
    BaseMemory,
    MemoryItem,
    MemoryQuery,
    RetrievedMemory,
    truncate_text,
    normalize_text,
)
from embodiedbench.memory.embeddings import EmbeddingProvider
from embodiedbench.memory.storage import load_json, save_json
from embodiedbench.memory.utils import similarity as _sim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_LOCATION_RELATIONS = {"on", "in", "contains"}
_STALE_THRESHOLD = 0.25
_NEAR_THRESHOLD = 1.5  # metres (XZ distance)

_RECEPTACLE_KEYWORDS = {
    "table", "counter", "refrigerator", "fridge", "cabinet", "sink",
    "shelf", "shelving", "stand", "bin", "trash", "drawer", "sofa",
    "couch", "chair", "bed", "desk", "tv", "television", "toilet",
    "bathtub", "microwave", "oven", "stove", "bench", "rack",
    "coffee", "nightstand", "dresser", "armchair",
}

_STOP_WORDS = {"the", "a", "an", "of", "to", "in", "on", "at", "is", "and", "or", "it", "its"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_scalar(v: Any) -> Any:
    """Return a JSON-safe scalar/dict/list from v."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe_scalar(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_scalar(vv) for k, vv in v.items()}
    return str(v)


def _norm(s: str) -> str:
    return normalize_text(s)


def _extract_base_name(raw: str) -> str:
    """Strip AI2-THOR instance suffixes from a raw object id.

    e.g. ``Mug_ff353859(Clone)_copy_44`` -> ``Mug``.
    Clean Habitat names pass through unchanged.
    """
    s = re.sub(r'\(Clone\).*$', '', raw)
    s = re.sub(r'_[0-9a-f]{6,}.*$', '', s)
    s = re.sub(r'_copy_\d+.*$', '', s)
    return s.strip('_') or raw


def _format_position(pos: Any) -> str:
    """Return '(x, y, z)' string from a position dict, or '' if unavailable."""
    if not isinstance(pos, dict) or "cx_norm" in pos or "x" not in pos:
        return ""
    try:
        return f"({float(pos['x']):.2f}, {float(pos.get('y', 0.0)):.2f}, {float(pos.get('z', 0.0)):.2f})"
    except (TypeError, ValueError):
        return ""


def _extract_words(text: str) -> set:
    """Return stop-word-filtered word set from text (for lexical scoring)."""
    if not text:
        return set()
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if w not in _STOP_WORDS and len(w) > 2}


# ---------------------------------------------------------------------------
# SpatialNode
# ---------------------------------------------------------------------------

@dataclass
class SpatialNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: str = "object"  # object | receptacle | room | area | agent | unknown
    position: Optional[Any] = None
    scene: str = ""
    state: dict = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "position": _safe_scalar(self.position),
            "scene": self.scene,
            "state": _safe_scalar(self.state),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpatialNode":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            name=d.get("name", ""),
            node_type=d.get("node_type", "object"),
            position=d.get("position"),
            scene=d.get("scene", ""),
            state=dict(d.get("state") or {}),
            confidence=float(d.get("confidence", 1.0)),
        )


# ---------------------------------------------------------------------------
# SpatialRelation
# ---------------------------------------------------------------------------

@dataclass
class SpatialRelation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""
    relation: str = ""   # in | on | contains | near | left_of | right_of
    object_id: str = ""
    confidence: float = 1.0
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "relation": self.relation,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpatialRelation":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            subject_id=d.get("subject_id", ""),
            relation=d.get("relation", ""),
            object_id=d.get("object_id", ""),
            confidence=float(d.get("confidence", 1.0)),
            evidence=d.get("evidence", ""),
        )


# ---------------------------------------------------------------------------
# SpatialMemory
# ---------------------------------------------------------------------------

class SpatialMemory(BaseMemory):
    """3-D scene graph for spatial memory.

    Nodes are objects/receptacles; edges are spatial relations (in, on, near, ...).
    Supports AI2-THOR (objectId / objectType / parentReceptacles) and Habitat.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        storage_path: Optional[str] = None,
        stale_confidence_decay: float = 0.5,
    ):
        self.embedding_provider = embedding_provider
        self.storage_path = storage_path
        self.stale_confidence_decay = stale_confidence_decay
        self.nodes: dict[str, SpatialNode] = {}
        self.relations: dict[str, SpatialRelation] = {}
        self.name_index: defaultdict = defaultdict(set)  # normalised name -> set[node_id]

    # --- Index helpers ---

    def rebuild_index(self) -> None:
        """Rebuild name_index from scratch (called after load)."""
        self.name_index = defaultdict(set)
        for node in self.nodes.values():
            self.name_index[_norm(node.name)].add(node.id)

    def _index_add(self, node: SpatialNode) -> None:
        self.name_index[_norm(node.name)].add(node.id)

    # --- Lookup ---

    def find_nodes_by_name(self, name: str) -> list:
        ids = self.name_index.get(_norm(name), set())
        return [self.nodes[i] for i in ids if i in self.nodes]

    def find_node(self, name: str, node_type: Optional[str] = None) -> Optional[SpatialNode]:
        candidates = self.find_nodes_by_name(name)
        if not candidates:
            return None
        if node_type:
            typed = [n for n in candidates if n.node_type == node_type]
            if typed:
                return typed[0]
        return max(candidates, key=lambda n: n.confidence)

    # --- Graph mutation ---

    def add_or_update_object(
        self,
        name: str,
        node_type: str = "object",
        position: Optional[Any] = None,
        scene: str = "",
        state: Optional[dict] = None,
        confidence: float = 1.0,
        node_id: Optional[str] = None,
    ) -> SpatialNode:
        """Upsert a node by name (or explicit node_id). Returns the node."""
        state = dict(state or {})
        existing: Optional[SpatialNode] = (
            self.nodes.get(node_id) if node_id else self.find_node(name, node_type)
        )

        if existing is not None:
            location_changed = bool(
                (scene and existing.scene and _norm(scene) != _norm(existing.scene))
                or (position is not None and existing.position is not None
                    and position != existing.position)
            )
            if location_changed:
                self._stale_location_relations(existing.id)
            existing.name = name
            if not (existing.node_type == "receptacle" and node_type == "object"):
                existing.node_type = node_type
            if position is not None:
                existing.position = _safe_scalar(position)
            if scene:
                existing.scene = scene
            existing.state.update(state)
            existing.confidence = confidence
            return existing

        node = SpatialNode(
            id=node_id or str(uuid.uuid4()),
            name=name, node_type=node_type,
            position=_safe_scalar(position),
            scene=scene, state=state, confidence=confidence,
        )
        self.nodes[node.id] = node
        self._index_add(node)
        return node

    def _stale_location_relations(self, subject_id: str) -> None:
        """Decay confidence of location relations when an object moves."""
        for rel in self.relations.values():
            if (rel.subject_id == subject_id and rel.relation in _LOCATION_RELATIONS) or \
               (rel.object_id == subject_id and rel.relation == "contains"):
                rel.confidence *= self.stale_confidence_decay

    def add_relation(
        self,
        subject_id: str,
        relation: str,
        object_id: str,
        confidence: float = 1.0,
        evidence: str = "",
        relation_id: Optional[str] = None,
    ) -> SpatialRelation:
        """Add or update a relation triple. Skips self-referential relations."""
        if subject_id == object_id:
            logger.debug("Skipping self-referential relation '%s' for %s.", relation, subject_id)
            return SpatialRelation(subject_id=subject_id, relation=relation,
                                   object_id=object_id, confidence=0.0)
        for rel in self.relations.values():
            if rel.subject_id == subject_id and rel.relation == relation and rel.object_id == object_id:
                rel.confidence = confidence
                rel.evidence = evidence or rel.evidence
                return rel
        new_rel = SpatialRelation(
            id=relation_id or str(uuid.uuid4()),
            subject_id=subject_id, relation=relation, object_id=object_id,
            confidence=confidence, evidence=evidence,
        )
        self.relations[new_rel.id] = new_rel
        return new_rel

    # --- Graph construction ---

    def _parse_object_dicts(self, objs: list, scene_name: str = "") -> None:
        """Parse simulator object dicts into graph nodes and 'in' relations.

        Pass 1: group by base type, assign stable display names
        (Cabinet, Cabinet_2, ...) matching the planner's action space.
        Pass 2: create nodes and resolve parentReceptacles via the display-name map.
        """
        entries: list[tuple[str, str, dict]] = []
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            raw_id = obj.get("objectId") or obj.get("name") or obj.get("id") or ""
            base_name = obj.get("objectType") or obj.get("type") or _extract_base_name(raw_id)
            if base_name and isinstance(base_name, str):
                entries.append((raw_id, base_name, obj))

        type_groups: dict[str, list[tuple[str, dict]]] = {}
        for raw_id, base_name, obj in entries:
            type_groups.setdefault(base_name, []).append((raw_id, obj))
        for instances in type_groups.values():
            instances.sort(key=lambda x: x[0])

        raw_id_to_display: dict[str, str] = {}
        ordered: list[tuple[str, str, dict]] = []
        for base_name, instances in type_groups.items():
            for idx, (raw_id, obj) in enumerate(instances):
                display_name = base_name if idx == 0 else f"{base_name}_{idx + 1}"
                raw_id_to_display[raw_id] = display_name
                ordered.append((raw_id, display_name, obj))

        for _raw_id, display_name, obj in ordered:
            receptacles = obj.get("parentReceptacles") or []
            if isinstance(receptacles, str):
                receptacles = [receptacles]

            state = {sf: obj[sf] for sf in
                     ("isOpen", "isPickedUp", "isSliced", "isToggled", "isBroken", "isDirty", "isCooked")
                     if sf in obj}

            inferred_type = "object" if receptacles else (
                "receptacle"
                if any(kw in display_name.lower() for kw in _RECEPTACLE_KEYWORDS)
                else "object"
            )

            node = self.add_or_update_object(
                name=display_name, node_type=inferred_type,
                position=obj.get("position"), scene=scene_name,
                state=state, confidence=1.0,
            )

            for rec_raw in receptacles:
                if not rec_raw or not isinstance(rec_raw, str):
                    continue
                rec_display = raw_id_to_display.get(rec_raw) or rec_raw.split("|")[0]
                rec_node = self.add_or_update_object(
                    name=rec_display, node_type="receptacle", confidence=0.9,
                )
                self.add_relation(node.id, "in", rec_node.id,
                                  confidence=1.0, evidence="parentReceptacles")

    def build_from_metadata(self, metadata: dict) -> None:
        """Build/refresh the scene graph from simulator metadata.

        Supports AI2-THOR (objects list of dicts with objectType/parentReceptacles/
        sceneName/position/state flags) and Habitat (objects list of dicts with
        objectType only + separate receptacles list of strings).
        """
        if not isinstance(metadata, dict):
            return
        scene_name = metadata.get("sceneName", "") or ""
        objects = metadata.get("objects", [])
        if objects and isinstance(objects[0], dict):
            self._parse_object_dicts(objects, scene_name=scene_name)
        for rec_name in metadata.get("receptacles", []):
            if rec_name:
                self.add_or_update_object(name=rec_name, node_type="receptacle", scene=scene_name)
        self._infer_spatial_relations()

    def update(self, metadata: Optional[dict] = None, **_kwargs) -> None:
        """Refresh the graph from one step's simulator metadata."""
        if metadata:
            self.build_from_metadata(metadata)

    def _infer_spatial_relations(self) -> None:
        """Infer near / left_of / right_of relations from 3-D positions of receptacles."""
        candidates = [
            n for n in self.nodes.values()
            if isinstance(n.position, dict) and "x" in n.position
            and n.node_type in ("receptacle", "room", "area")
        ]
        for i, n1 in enumerate(candidates):
            for j, n2 in enumerate(candidates):
                if i >= j:
                    continue
                try:
                    dx = float(n1.position["x"]) - float(n2.position["x"])
                    dz = float(n1.position.get("z", 0)) - float(n2.position.get("z", 0))
                except (TypeError, ValueError):
                    continue
                dist = (dx ** 2 + dz ** 2) ** 0.5
                if dist < 0.01:
                    continue
                if dist < _NEAR_THRESHOLD:
                    self.add_relation(n1.id, "near", n2.id, confidence=0.8, evidence="position_inference")
                if abs(dx) > abs(dz) and abs(dx) > 0.3:
                    if dx > 0:
                        self.add_relation(n1.id, "right_of", n2.id, confidence=0.75, evidence="position_inference")
                        self.add_relation(n2.id, "left_of",  n1.id, confidence=0.75, evidence="position_inference")
                    else:
                        self.add_relation(n1.id, "left_of",  n2.id, confidence=0.75, evidence="position_inference")
                        self.add_relation(n2.id, "right_of", n1.id, confidence=0.75, evidence="position_inference")

    # --- Retrieval ---

    def retrieve(self, query: MemoryQuery, top_k: int = 5) -> list:
        """Return top-K receptacles, top-K objects, and top-K scored relations.

        Scoring: semantic (embedding) when available, lexical (Jaccard) otherwise.
        Relations are candidates if at least one endpoint is in the top-K sets,
        then ranked by similarity of their text to the query.
        """
        if not self.nodes:
            return []

        q_text = query.task_instruction or ""
        rec_nodes = [n for n in self.nodes.values() if n.node_type in ("receptacle", "room", "area")]
        obj_nodes = [n for n in self.nodes.values() if n.node_type == "object"]

        if self.embedding_provider is not None:
            all_vecs = self.embedding_provider.embed_batch(
                [q_text] + [n.name for n in rec_nodes] + [n.name for n in obj_nodes]
            )
            q_vec = all_vecs[0]
            rec_vecs = all_vecs[1: 1 + len(rec_nodes)]
            obj_vecs = all_vecs[1 + len(rec_nodes):]
            rec_scored = sorted(
                [(_sim(q_text, n.name, q_vec, v), n) for n, v in zip(rec_nodes, rec_vecs)],
                key=lambda x: x[0], reverse=True,
            )
            obj_scored = sorted(
                [(_sim(q_text, n.name, q_vec, v), n) for n, v in zip(obj_nodes, obj_vecs)],
                key=lambda x: x[0], reverse=True,
            )
        else:
            q_vec = None
            q_lex = " ".join(_extract_words(q_text))
            rec_scored = sorted(
                [(_sim(q_lex, n.name), n) for n in rec_nodes],
                key=lambda x: x[0], reverse=True,
            )
            obj_scored = sorted(
                [(_sim(q_lex, n.name), n) for n in obj_nodes],
                key=lambda x: x[0], reverse=True,
            )

        # --- Build relation sentences ---
        # Object-centric: one sentence per object showing its primary location,
        #   e.g. "Ladle in CounterTop"
        # Receptacle-centric: one synthesised sentence per receptacle listing all
        #   contents, e.g. "CounterTop contains Ladle, Spoon, SaltShaker"
        # Both pools are scored against the query and the top-K are returned.

        obj_rel_sentences: list[tuple[str, str]] = []   # (sentence, reason)
        seen_obj_rels: set = set()
        for obj_node in obj_nodes:
            for rel in self.relations.values():
                if (rel.subject_id == obj_node.id
                        and rel.relation in ("in", "on")
                        and rel.confidence >= _STALE_THRESHOLD):
                    rec_node = self.nodes.get(rel.object_id)
                    if rec_node:
                        key = (obj_node.id, rel.relation, rec_node.id)
                        if key not in seen_obj_rels:
                            seen_obj_rels.add(key)
                            obj_rel_sentences.append((
                                f"{obj_node.name} {rel.relation} {rec_node.name}",
                                "object_relation",
                            ))
                        break  # one primary location per object

        rec_contents: dict = defaultdict(list)
        for rel in self.relations.values():
            if rel.relation in ("in", "on") and rel.confidence >= _STALE_THRESHOLD:
                o = self.nodes.get(rel.subject_id)
                r = self.nodes.get(rel.object_id)
                if o and r and o.node_type == "object":
                    rec_contents[r.id].append(o.name)

        rec_rel_sentences: list[tuple[str, str]] = []
        for rec_id, obj_name_list in rec_contents.items():
            rec_node = self.nodes.get(rec_id)
            if rec_node and obj_name_list:
                names_str = ", ".join(sorted(set(obj_name_list)))
                rec_rel_sentences.append((
                    f"{rec_node.name} contains {names_str}",
                    "receptacle_relation",
                ))

        all_rel_pairs = obj_rel_sentences + rec_rel_sentences
        # Score all relation sentences against the query
        rel_items: list = []
        if all_rel_pairs:
            sentences = [s for s, _ in all_rel_pairs]
            reasons   = [r for _, r in all_rel_pairs]

            if self.embedding_provider is not None:
                rel_vecs = self.embedding_provider.embed_batch(sentences)
                rel_scored = sorted(
                    [(_sim(q_text, s, q_vec, v), s, reason)
                     for s, reason, v in zip(sentences, reasons, rel_vecs)],
                    key=lambda x: x[0], reverse=True,
                )
            else:
                rel_scored = sorted(
                    [(_sim(q_lex, s), s, reason) for s, reason in zip(sentences, reasons)],
                    key=lambda x: x[0], reverse=True,
                )

            for score, sentence, reason in rel_scored[:top_k]:
                rel_items.append(RetrievedMemory(
                    item=MemoryItem(
                        memory_type="spatial",
                        content=truncate_text(sentence, 500),
                        metadata={"relation_type": reason},
                        importance=0.65, confidence=1.0, source="spatial_memory",
                    ),
                    score=score, reason=reason,
                ))

        # Assemble deduplicated output
        seen_content: set = set()
        output: list = []

        def _add(score: float, reason: str, item: MemoryItem) -> None:
            key = item.content[:80]
            if key not in seen_content:
                seen_content.add(key)
                output.append(RetrievedMemory(item=item, score=score, reason=reason))

        for score, node in rec_scored[:top_k]:
            _add(score, "receptacle", self.node_to_memory_item(node))
        for score, node in obj_scored[:top_k]:
            _add(score, "object", self.node_to_memory_item(node))
        for rm in sorted(rel_items, key=lambda x: x.score, reverse=True):
            _add(rm.score, rm.reason, rm.item)

        return output

    # --- MemoryItem builders ---

    def node_to_memory_item(self, node: SpatialNode) -> MemoryItem:
        """Convert a SpatialNode to a MemoryItem."""
        loc_str = f" in {node.scene}" if node.scene else ""
        pos_str = _format_position(node.position)
        loc_str = loc_str or (f" at {pos_str}" if pos_str else "")
        content = f"{node.name} last seen{loc_str}, confidence {node.confidence:.2f}."
        if node.state:
            state_pairs = ", ".join(f"{k}={v}" for k, v in list(node.state.items())[:4])
            content += f" State: [{state_pairs}]."
        return MemoryItem(
            memory_type="spatial",
            content=truncate_text(content, 300),
            metadata={"node_id": node.id, "node_type": node.node_type, "scene": node.scene},
            importance=0.7, confidence=node.confidence, source="spatial_memory",
        )

    def relation_to_memory_item(self, rel: SpatialRelation) -> MemoryItem:
        """Convert a SpatialRelation to a MemoryItem."""
        subj = self.nodes.get(rel.subject_id)
        obj = self.nodes.get(rel.object_id)
        content = (
            f"{subj.name if subj else rel.subject_id} {rel.relation} "
            f"{obj.name if obj else rel.object_id}, confidence {rel.confidence:.2f}."
        )
        return MemoryItem(
            memory_type="spatial",
            content=truncate_text(content, 500),
            metadata={
                "relation_id": rel.id, "relation": rel.relation,
                "subject_id": rel.subject_id, "object_id": rel.object_id,
            },
            importance=0.65, confidence=rel.confidence, source="spatial_memory",
        )

    # --- Prompt rendering ---

    def to_prompt_context(self, memories: list) -> str:
        """Render retrieved memories as a structured spatial context string.

        Output format:
          The task-related objects in the environment: obj1, obj2, ...
          The task-related receptacles in the environment: rec1, rec2, ...
          Their relations:
            obj1 in rec1
            ...
        """
        if not memories:
            return ""

        obj_names = [
            self.nodes[rm.item.metadata["node_id"]].name
            for rm in memories
            if rm.reason == "object" and rm.item.metadata.get("node_id") in self.nodes
        ]
        rec_names = [
            self.nodes[rm.item.metadata["node_id"]].name
            for rm in memories
            if rm.reason == "receptacle" and rm.item.metadata.get("node_id") in self.nodes
        ]

        lines: list[str] = [
            f"The task-related objects in the environment: {', '.join(obj_names) or 'None'}",
            f"The task-related receptacles in the environment: {', '.join(rec_names) or 'None'}",
        ]

        rel_lines: list[str] = []
        seen_sentences: set[str] = set()
        for rm in memories:
            if rm.reason not in ("object_relation", "receptacle_relation"):
                continue
            sentence = rm.item.content
            if sentence not in seen_sentences:
                seen_sentences.add(sentence)
                rel_lines.append(sentence)

        if rel_lines:
            lines.append("Their relations:")
            lines.extend(f"  {r}" for r in rel_lines)

        return "\n".join(lines)

    # --- Episode lifecycle ---

    def reset_episode(self) -> None:
        """Clear all nodes, relations, and the name index for a new episode."""
        self.nodes = {}
        self.relations = {}
        self.name_index = defaultdict(set)

    # --- Persistence ---

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.storage_path
        if not target:
            return
        save_json(target, {
            "stale_confidence_decay": self.stale_confidence_decay,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
        })

    def load(self, path: Optional[str] = None) -> None:
        target = path or self.storage_path
        if not target:
            return
        data = load_json(target, default=None)
        if data is None:
            return
        self.stale_confidence_decay = float(data.get("stale_confidence_decay", self.stale_confidence_decay))
        self.nodes = {d["id"]: SpatialNode.from_dict(d) for d in (data.get("nodes") or [])}
        self.relations = {d["id"]: SpatialRelation.from_dict(d) for d in (data.get("relations") or [])}
        self.rebuild_index()
