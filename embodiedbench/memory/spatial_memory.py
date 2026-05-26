"""
memory/spatial_memory.py

SpatialMemory: a 3-D scene-graph that tracks objects, receptacles, and their
spatial relationships across episode steps.

Graph lifecycle:
1. ``build_from_metadata`` — initialise from simulator metadata at episode start.
2. ``update`` — metadata-driven graph refresh each step.
3. ``retrieve`` — scored ``RetrievedMemory`` items for retrieval.
4. ``to_prompt_context`` — concise grouped scene snippet for the planner.
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
# Helpers
# ---------------------------------------------------------------------------

def _safe_scalar(v: Any) -> Any:
    """Return a JSON-safe scalar/dict/list representation of v."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe_scalar(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_scalar(vv) for k, vv in v.items()}
    return str(v)


def _norm(s: str) -> str:
    return normalize_text(s)


def _format_position(pos: Any) -> str:
    """Return a human-readable 3-D position string, or '' for image-space coords.

    * ``{"x": 1.2, "y": 0.0, "z": -3.4}``  -> ``"(1.20, 0.00, -3.40)"``
    * ``{"cx_norm": ..., "cy_norm": ...}``   -> ``""``  (image-space - omitted)
    """
    if not isinstance(pos, dict):
        return ""
    if "cx_norm" in pos or "cy_norm" in pos:
        return ""
    if "x" in pos:
        try:
            x = float(pos["x"])
            y = float(pos.get("y", 0.0))
            z = float(pos.get("z", 0.0))
            return f"({x:.2f}, {y:.2f}, {z:.2f})"
        except (TypeError, ValueError):
            return ""
    return ""


# ---------------------------------------------------------------------------
# SpatialNode
# ---------------------------------------------------------------------------

@dataclass
class SpatialNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: str = "object"  # object | room | area | receptacle | agent | unknown
    position: Optional[Any] = None
    room: str = ""
    state: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "position": _safe_scalar(self.position),
            "room": self.room,
            "state": _safe_scalar(self.state),
            "attributes": _safe_scalar(self.attributes),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpatialNode":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            name=d.get("name", ""),
            node_type=d.get("node_type", "object"),
            position=d.get("position"),
            room=d.get("room", ""),
            state=dict(d.get("state") or {}),
            attributes=dict(d.get("attributes") or {}),
            confidence=float(d.get("confidence", 1.0)),
        )


# ---------------------------------------------------------------------------
# SpatialRelation
# ---------------------------------------------------------------------------

@dataclass
class SpatialRelation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""
    relation: str = ""   # contains | on | in | near | left_of | right_of
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

# Relation types that indicate an object's location — used for stale detection.
_LOCATION_RELATIONS = {"on", "in", "contains"}

# Relations below this confidence are considered stale and excluded from output.
_STALE_THRESHOLD = 0.25

# XZ distance threshold (metres) below which two receptacles are considered "near".
_NEAR_THRESHOLD = 1.5

# Keywords that identify a node as a receptacle when no parentReceptacles data
# is available (e.g. Habitat).  Defined at module level to avoid re-creation
# on every call to _parse_object_dicts.
_RECEPTACLE_KEYWORDS = {
    "table", "counter", "refrigerator", "fridge", "cabinet", "sink",
    "shelf", "shelving", "stand", "bin", "trash", "drawer", "sofa",
    "couch", "chair", "bed", "desk", "tv", "television", "toilet",
    "bathtub", "microwave", "oven", "stove", "bench", "rack",
    "coffee", "nightstand", "dresser", "armchair",
}


class SpatialMemory(BaseMemory):
    """3-D scene graph that tracks objects, receptacles, and their spatial
    relationships.

    The graph is built once from simulator metadata at episode start
    (``build_from_metadata``), then incrementally refreshed each step via
    ``update``.  ``node_to_memory_item`` and ``relation_to_memory_item``
    produce standard ``MemoryItem`` objects that the Memory Adapter can
    consume to generate foresight plans and feasibility criteria.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,  # reserved for future embedding-based retrieval
        storage_path: Optional[str] = None,
        stale_confidence_decay: float = 0.5,
    ):
        self.storage_path = storage_path
        self.stale_confidence_decay = stale_confidence_decay

        self.nodes: dict[str, SpatialNode] = {}
        self.relations: dict[str, SpatialRelation] = {}
        self.name_index: defaultdict = defaultdict(set)  # normalised name -> set[node_id]

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def rebuild_index(self) -> None:
        self.name_index = defaultdict(set)
        for node in self.nodes.values():
            self.name_index[_norm(node.name)].add(node.id)

    def _index_add(self, node: SpatialNode) -> None:
        self.name_index[_norm(node.name)].add(node.id)

    # ------------------------------------------------------------------
    # Find helpers
    # ------------------------------------------------------------------

    def find_nodes_by_name(self, name: str) -> list:
        key = _norm(name)
        ids = self.name_index.get(key, set())
        return [self.nodes[i] for i in ids if i in self.nodes]

    def find_node(self, name: str, node_type: Optional[str] = None) -> Optional[SpatialNode]:
        candidates = self.find_nodes_by_name(name)
        if not candidates:
            return None
        if node_type:
            typed = [n for n in candidates if n.node_type == node_type]
            if typed:
                return typed[0]
        candidates.sort(key=lambda n: n.confidence, reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # add_or_update_object
    # ------------------------------------------------------------------

    def add_or_update_object(
        self,
        name: str,
        node_type: str = "object",
        position: Optional[Any] = None,
        room: str = "",
        state: Optional[dict] = None,
        attributes: Optional[dict] = None,
        confidence: float = 1.0,
        node_id: Optional[str] = None,
    ) -> SpatialNode:
        state = dict(state or {})
        attributes = dict(attributes or {})

        existing: Optional[SpatialNode] = None
        if node_id and node_id in self.nodes:
            existing = self.nodes[node_id]
        else:
            existing = self.find_node(name, node_type)

        if existing is not None:
            location_changed = bool(
                (room and existing.room and _norm(room) != _norm(existing.room))
                or (position is not None and existing.position is not None and position != existing.position)
            )
            if location_changed:
                self._stale_location_relations(existing.id)

            existing.name = name
            # Never downgrade receptacle -> object
            if not (existing.node_type == "receptacle" and node_type == "object"):
                existing.node_type = node_type
            if position is not None:
                existing.position = _safe_scalar(position)
            if room:
                existing.room = room
            existing.state.update(state)
            existing.attributes.update(attributes)
            existing.confidence = confidence
            return existing

        node = SpatialNode(
            id=node_id or str(uuid.uuid4()),
            name=name,
            node_type=node_type,
            position=_safe_scalar(position),
            room=room,
            state=state,
            attributes=attributes,
            confidence=confidence,
        )
        self.nodes[node.id] = node
        self._index_add(node)
        return node

    def _stale_location_relations(self, subject_id: str) -> None:
        for rel in self.relations.values():
            is_loc_subj = rel.subject_id == subject_id and rel.relation in _LOCATION_RELATIONS
            is_contains_obj = rel.object_id == subject_id and rel.relation == "contains"
            if is_loc_subj or is_contains_obj:
                rel.confidence *= self.stale_confidence_decay

    # ------------------------------------------------------------------
    # add_relation
    # ------------------------------------------------------------------

    def add_relation(
        self,
        subject_id: str,
        relation: str,
        object_id: str,
        confidence: float = 1.0,
        evidence: str = "",
        relation_id: Optional[str] = None,
    ) -> SpatialRelation:
        # Guard against self-referential relations
        if subject_id == object_id:
            logger.debug(
                "add_relation: skipping self-referential relation '%s' for node %s.",
                relation, subject_id,
            )
            # Return a dummy relation rather than creating one
            dummy = SpatialRelation(
                id=str(uuid.uuid4()),
                subject_id=subject_id,
                relation=relation,
                object_id=object_id,
                confidence=0.0,
            )
            return dummy

        # Update existing identical triple
        for rel in self.relations.values():
            if (
                rel.subject_id == subject_id
                and rel.relation == relation
                and rel.object_id == object_id
            ):
                rel.confidence = confidence
                rel.evidence = evidence or rel.evidence
                return rel

        new_rel = SpatialRelation(
            id=relation_id or str(uuid.uuid4()),
            subject_id=subject_id,
            relation=relation,
            object_id=object_id,
            confidence=confidence,
            evidence=evidence,
        )
        self.relations[new_rel.id] = new_rel
        return new_rel

    # ------------------------------------------------------------------
    # _parse_object_dicts  (internal - called by build_from_metadata)
    # ------------------------------------------------------------------

    def _parse_object_dicts(self, objs: list) -> None:
        for obj in objs:
            if not isinstance(obj, dict):
                continue

            name = (
                obj.get("name")
                or obj.get("objectType")
                or obj.get("type")
                or obj.get("id")
                or ""
            )
            if not name or not isinstance(name, str):
                continue
            position = obj.get("position")
            receptacles = obj.get("parentReceptacles") or []
            if isinstance(receptacles, str):
                receptacles = [receptacles]

            state = {}
            for sf in ("isOpen", "isPickedUp", "isSliced", "isToggled", "isBroken", "isDirty", "isCooked"):
                if sf in obj:
                    state[sf] = obj[sf]

            # Determine node type:
            # * If the object has parentReceptacles it is definitely an object (not furniture).
            # * If it lacks parentReceptacles AND its name suggests it is a container/surface,
            #   classify it as a receptacle.  Otherwise default to "object" to avoid polluting
            #   the scene graph with every small pickable item typed as "receptacle" (which
            #   breaks environments like Habitat that never populate parentReceptacles).
            if receptacles:
                inferred_type = "object"
            else:
                name_lower = name.lower()
                inferred_type = (
                    "receptacle"
                    if any(kw in name_lower for kw in _RECEPTACLE_KEYWORDS)
                    else "object"
                )
            node = self.add_or_update_object(
                name=name,
                node_type=inferred_type,
                position=position,
                state=state,
                confidence=1.0,
            )

            for rec_name in receptacles:
                if not rec_name or not isinstance(rec_name, str):
                    continue
                # Strip AI2-THOR pipe-suffixes e.g. "Fridge|1|2|3"
                clean_rec = rec_name.split("|")[0]
                rec_node = self.add_or_update_object(
                    name=clean_rec,
                    node_type="receptacle",
                    confidence=0.9,
                )
                self.add_relation(
                    subject_id=node.id,
                    relation="in",
                    object_id=rec_node.id,
                    confidence=1.0,
                    evidence="parentReceptacles",
                )

    # ------------------------------------------------------------------
    # build_from_metadata  (episode initialisation)
    # ------------------------------------------------------------------

    def build_from_metadata(self, metadata: dict) -> None:
        """Build the initial 3-D scene graph from simulator metadata.

        Supports:
        * **AI2-THOR** - ``metadata["objects"]`` is a list of dicts with
          ``objectType``, ``position``, ``parentReceptacles``, and state flags.
        * **Habitat** - ``metadata["objects"]`` is a list of minimal dicts with
          only ``objectType``; ``metadata["receptacles"]`` is a list of str.
        """
        if not isinstance(metadata, dict):
            return

        objects = metadata.get("objects", [])
        if objects:
            if isinstance(objects[0], dict):
                self._parse_object_dicts(objects)
            elif isinstance(objects[0], str):
                for name in objects:
                    if name:
                        self.add_or_update_object(name=name)

        # Habitat-style explicit receptacles list
        for rec_name in metadata.get("receptacles", []):
            if rec_name:
                self.add_or_update_object(
                    name=rec_name, node_type="receptacle"
                )

        # Infer geometric relations from 3-D positions (left_of/right_of/near)
        self._infer_spatial_relations()

    # ------------------------------------------------------------------
    # update  (metadata-driven graph updates)
    # ------------------------------------------------------------------

    def update(self, metadata: Optional[dict] = None, **_kwargs) -> None:
        """Update the scene graph from simulator metadata for one agent step.

        Delegates to ``build_from_metadata`` so that the scene graph stays
        in sync with the ground-truth simulator state every step.
        Extra keyword arguments (e.g. ``step_id``) are accepted and ignored
        for call-site compatibility.
        """
        if not metadata:
            return
        self.build_from_metadata(metadata)

    # ------------------------------------------------------------------
    # _infer_spatial_relations  (geometry -> left_of / right_of / near)
    # ------------------------------------------------------------------

    def _infer_spatial_relations(self) -> None:
        """Infer directional relations for receptacle-level nodes with 3-D positions.

        Adds ``left_of`` / ``right_of`` (X-axis) and ``near`` (XZ Euclidean
        distance < 1.5 m).  Only receptacle/room/area nodes participate.
        """
        candidates = [
            n for n in self.nodes.values()
            if n.position and isinstance(n.position, dict)
            and "x" in n.position
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
                    self.add_relation(
                        n1.id, "near", n2.id,
                        confidence=0.8, evidence="position_inference",
                    )

                if abs(dx) > abs(dz) and abs(dx) > 0.3:
                    if dx > 0:
                        self.add_relation(n1.id, "right_of", n2.id, confidence=0.75,
                                          evidence="position_inference")
                        self.add_relation(n2.id, "left_of", n1.id, confidence=0.75,
                                          evidence="position_inference")
                    else:
                        self.add_relation(n1.id, "left_of", n2.id, confidence=0.75,
                                          evidence="position_inference")
                        self.add_relation(n2.id, "right_of", n1.id, confidence=0.75,
                                          evidence="position_inference")

    # ------------------------------------------------------------------
    # retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: MemoryQuery, top_k: int = 5) -> list:
        """Return top-K receptacles, top-K objects, and their relations.

        Strategy:
        1. Score every receptacle node by name–instruction similarity; pick top-K.
        2. Score every object node by name–instruction similarity; pick top-K.
        3. Collect all non-stale relations where at least one endpoint appears in
           the selected sets and append them as additional ``RetrievedMemory`` items.
        4. If no relations are found, the result is the two ranked lists — the
           caller (Memory Adapter) receives the pure name-based context.
        """
        if not self.nodes:
            return []

        q_text = " ".join(_extract_words(query.task_instruction))

        # 1. Rank receptacles by name similarity
        rec_scored = sorted(
            [
                (_sim(q_text, node.name), node)
                for node in self.nodes.values()
                if node.node_type in ("receptacle", "room", "area")
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        top_rec_ids = {node.id for _, node in rec_scored[:top_k]}

        # 2. Rank objects by name similarity
        obj_scored = sorted(
            [
                (_sim(q_text, node.name), node)
                for node in self.nodes.values()
                if node.node_type == "object"
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        top_obj_ids = {node.id for _, node in obj_scored[:top_k]}

        # 3. Relations where at least one endpoint is in the selected sets
        selected_ids = top_rec_ids | top_obj_ids
        seen_rel: set = set()
        rel_items: list = []
        for rel in self.relations.values():
            if rel.confidence < _STALE_THRESHOLD:
                continue
            if rel.subject_id not in selected_ids and rel.object_id not in selected_ids:
                continue
            key = (rel.subject_id, rel.relation, rel.object_id)
            if key in seen_rel:
                continue
            seen_rel.add(key)
            rel_items.append(
                RetrievedMemory(
                    item=self.relation_to_memory_item(rel),
                    score=rel.confidence,
                    reason="spatial relation",
                )
            )

        # 4. Assemble (deduped)
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
        rel_items.sort(key=lambda x: x.score, reverse=True)
        for rm in rel_items:
            _add(rm.score, rm.reason, rm.item)

        return output

    # ------------------------------------------------------------------
    # MemoryItem builders  (used by retrieve + Memory Adapter)
    # ------------------------------------------------------------------

    def node_to_memory_item(self, node: SpatialNode) -> MemoryItem:
        """Convert a ``SpatialNode`` to a standard ``MemoryItem``.

        The Memory Adapter can consume these items to generate foresight plans
        and feasibility criteria from the spatial context.
        """
        loc_str = f" in {node.room}" if node.room else ""
        pos_str = _format_position(node.position)
        loc_str = loc_str or (f" at {pos_str}" if pos_str else "")
        content = f"{node.name} last seen{loc_str}, confidence {node.confidence:.2f}."
        if node.state:
            state_pairs = ", ".join(f"{k}={v}" for k, v in list(node.state.items())[:4])
            content += f" State: [{state_pairs}]."
        return MemoryItem(
            memory_type="spatial",
            content=truncate_text(content, 300),
            metadata={
                "node_id": node.id,
                "node_type": node.node_type,
                "room": node.room,
            },
            importance=0.7,
            confidence=node.confidence,
            source="spatial_memory",
        )

    def relation_to_memory_item(self, rel: SpatialRelation) -> MemoryItem:
        """Convert a ``SpatialRelation`` to a standard ``MemoryItem``."""
        subj = self.nodes.get(rel.subject_id)
        obj = self.nodes.get(rel.object_id)
        subj_name = subj.name if subj else rel.subject_id
        obj_name = obj.name if obj else rel.object_id
        content = f"{subj_name} {rel.relation} {obj_name}, confidence {rel.confidence:.2f}."
        return MemoryItem(
            memory_type="spatial",
            content=truncate_text(content, 300),
            metadata={
                "relation_id": rel.id,
                "relation": rel.relation,
                "subject_id": rel.subject_id,
                "object_id": rel.object_id,
            },
            importance=0.65,
            confidence=rel.confidence,
            source="spatial_memory",
        )

    # ------------------------------------------------------------------
    # to_prompt_context  (renders only the retrieved items, grouped)
    # ------------------------------------------------------------------

    def to_prompt_context(self, memories: list) -> str:
        """Format retrieved spatial memories in the preferred structured format:

        The task-related objects in the environment: obj1, obj2, ...
        The task-related receptacles in the environment: rec1, rec2, ...
        Their relations:              (only when relations exist)
          <relation lines>
        """
        if not memories:
            return ""

        # --- Node lists from retrieve() output ---
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

        lines: list[str] = []
        lines.append(
            f"The task-related objects in the environment: "
            f"{', '.join(obj_names) if obj_names else 'None'}"
        )
        lines.append(
            f"The task-related receptacles in the environment: "
            f"{', '.join(rec_names) if rec_names else 'None'}"
        )

        # --- Relations section (only when we have them) ---
        # Collect all node IDs referenced in retrieved items for relation filtering
        relevant_ids: set[str] = set()
        for rm in memories:
            meta = rm.item.metadata
            for key in ("node_id", "subject_id", "object_id"):
                nid = meta.get(key)
                if nid:
                    relevant_ids.add(nid)

        # Expand to parent receptacles
        for rel in self.relations.values():
            if rel.relation in ("in", "on") and rel.subject_id in relevant_ids:
                relevant_ids.add(rel.object_id)
            if rel.relation == "contains" and rel.object_id in relevant_ids:
                relevant_ids.add(rel.subject_id)

        rel_lines: list[str] = []

        # Containment relations
        receptacle_contents: dict[str, list[str]] = {}
        for rel in self.relations.values():
            if rel.confidence < _STALE_THRESHOLD:
                continue
            subj = self.nodes.get(rel.subject_id)
            obj = self.nodes.get(rel.object_id)
            if subj is None or obj is None:
                continue
            if rel.relation == "contains" and rel.subject_id in relevant_ids:
                receptacle_contents.setdefault(subj.name, [])
                if obj.name not in receptacle_contents[subj.name]:
                    receptacle_contents[subj.name].append(obj.name)
            elif rel.relation in ("in", "on") and rel.object_id in relevant_ids:
                receptacle_contents.setdefault(obj.name, [])
                if subj.name not in receptacle_contents[obj.name]:
                    receptacle_contents[obj.name].append(subj.name)
        for rec_name, objects in receptacle_contents.items():
            if objects:
                rel_lines.append(f"{rec_name} contains {', '.join(objects)}")

        # Room-based fallback lines
        covered_node_ids: set[str] = set()
        for rel in self.relations.values():
            if rel.relation in ("in", "on", "contains"):
                covered_node_ids.add(rel.subject_id)
                covered_node_ids.add(rel.object_id)
        for nid in relevant_ids:
            if nid in covered_node_ids:
                continue
            node = self.nodes.get(nid)
            if node and node.room:
                rel_lines.append(f"{node.name} last seen in {node.room}")

        if rel_lines:
            lines.append("Their relations:")
            lines.extend(f"  {r}" for r in rel_lines)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # reset_episode
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Clear the entire scene graph for the new episode.

        All nodes, relations, and the name index are wiped so that the next
        ``build_from_metadata`` call starts from a clean slate.
        """
        self.nodes = {}
        self.relations = {}
        self.name_index = defaultdict(set)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.storage_path
        if not target:
            return
        data = {
            "stale_confidence_decay": self.stale_confidence_decay,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
        }
        save_json(target, data)

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_STOP_WORDS = {"the", "a", "an", "of", "to", "in", "on", "at", "is", "and", "or", "it", "its"}


def _extract_words(text: str) -> set:
    if not text:
        return set()
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}
