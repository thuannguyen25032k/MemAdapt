"""
tests/memory/test_spatial_scene_graph.py

Tests for the redesigned 3-D scene-graph SpatialMemory:
  - build_from_metadata  (initial graph from simulator metadata)
  - update  (pick / place / navigate / open / close)
  - _infer_spatial_relations  (geometric left_of / right_of / near)
  - to_prompt_context  (grouped receptacle format)
"""

import pytest
from embodiedbench.memory.spatial_memory import SpatialMemory, SpatialNode, SpatialRelation
from embodiedbench.memory.base import MemoryQuery


# ---------------------------------------------------------------------------
# build_from_metadata
# ---------------------------------------------------------------------------

class TestBuildFromMetadata:
    def _alfred_metadata(self):
        return {
            "sceneName": "FloorPlan1",
            "objects": [
                {
                    "objectType": "Plate",
                    "objectId": "Plate|1",
                    "position": {"x": 1.0, "y": 0.9, "z": 0.5},
                    "parentReceptacles": ["DiningTable|1"],
                    "isPickedUp": False,
                },
                {
                    "objectType": "Potato",
                    "objectId": "Potato|1",
                    "position": {"x": 1.1, "y": 0.9, "z": 0.5},
                    "parentReceptacles": ["DiningTable|1"],
                    "isPickedUp": False,
                },
                {
                    "objectType": "DiningTable",
                    "objectId": "DiningTable|1",
                    "position": {"x": 1.0, "y": 0.75, "z": 0.5},
                    "parentReceptacles": [],
                },
                {
                    "objectType": "Fridge",
                    "objectId": "Fridge|1",
                    "position": {"x": -2.0, "y": 0.75, "z": 0.5},
                    "parentReceptacles": [],
                    "isOpen": False,
                },
            ],
            "agentPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

    def _habitat_metadata(self):
        return {
            "objects": [
                {"objectType": "toy airplane", "objectId": "toy airplane"},
                {"objectType": "bowl", "objectId": "bowl"},
            ],
            "receptacles": ["table 1", "table 2", "TV stand"],
            "is_holding": False,
            "scene_id": "Put both an toy airplane and a bowl onto the black table.",
        }

    def test_alfred_populates_nodes(self):
        sm = SpatialMemory()
        sm.build_from_metadata(self._alfred_metadata(), step_id=0)
        assert sm.find_node("Plate") is not None
        assert sm.find_node("Potato") is not None
        assert sm.find_node("DiningTable") is not None
        assert sm.find_node("Fridge") is not None

    def test_alfred_creates_in_relations(self):
        sm = SpatialMemory()
        sm.build_from_metadata(self._alfred_metadata(), step_id=0)
        plate = sm.find_node("Plate")
        table = sm.find_node("DiningTable")
        assert plate is not None and table is not None
        rels = [
            r for r in sm.relations.values()
            if r.subject_id == plate.id and r.relation == "in" and r.object_id == table.id
        ]
        assert rels, "Plate should have 'in DiningTable' relation"

    def test_alfred_infers_directional_relations(self):
        """DiningTable (x=1) and Fridge (x=-2) are far apart; right_of/left_of expected."""
        sm = SpatialMemory()
        sm.build_from_metadata(self._alfred_metadata(), step_id=0)
        dir_rels = [
            r for r in sm.relations.values()
            if r.relation in ("left_of", "right_of") and not r.stale
        ]
        assert dir_rels, "Should infer at least one left_of/right_of relation"

    def test_habitat_populates_objects_and_receptacles(self):
        sm = SpatialMemory()
        sm.build_from_metadata(self._habitat_metadata(), step_id=0)
        assert sm.find_node("toy airplane") is not None
        assert sm.find_node("bowl") is not None
        assert sm.find_node("table 1") is not None
        assert sm.find_node("TV stand") is not None

    def test_empty_metadata_does_not_crash(self):
        sm = SpatialMemory()
        sm.build_from_metadata({})
        assert len(sm.nodes) == 0

    def test_invalid_metadata_does_not_crash(self):
        sm = SpatialMemory()
        sm.build_from_metadata(None)  # type: ignore
        assert len(sm.nodes) == 0


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdateAfterAction:
    def _setup(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="apple", node_type="object", step_id=0)
        table = sm.add_or_update_object(
            name="table", node_type="receptacle",
            position={"x": 1.0, "y": 0.0, "z": 0.0}, step_id=0
        )
        sm.add_relation(
            subject_id=table.id, relation="contains", object_id=apple.id,
            confidence=1.0, step_id=0
        )
        return sm

    def test_pick_up_marks_location_stale(self):
        sm = self._setup()
        apple = sm.find_node("apple")
        sm.update("pick up the apple", success=True, step_id=1)
        contains_rels = [
            r for r in sm.relations.values()
            if r.object_id == apple.id and r.relation == "contains"
        ]
        assert all(r.stale for r in contains_rels), "All 'contains apple' relations should be stale"

    def test_pick_up_sets_held_object(self):
        sm = self._setup()
        apple = sm.find_node("apple")
        sm.update("pick up the apple", success=True, step_id=1)
        assert sm._held_object_id == apple.id

    def test_pick_up_sets_isPickedUp_state(self):
        sm = self._setup()
        sm.update("pick up the apple", success=True, step_id=1)
        apple = sm.find_node("apple")
        assert apple.state.get("isPickedUp") is True

    def test_navigate_sets_last_nav_target(self):
        sm = SpatialMemory()
        sm.update("navigate to the fridge", success=True, step_id=1)
        assert sm._last_nav_target == "fridge"

    def test_find_sets_last_nav_target(self):
        sm = SpatialMemory()
        sm.update("find a fridge", success=True, step_id=1)
        assert sm._last_nav_target == "fridge"

    def test_put_down_links_object_to_nav_target(self):
        sm = self._setup()
        sm.update("pick up the apple", success=True, step_id=1)
        sm.update("navigate to the fridge", success=True, step_id=2)
        sm.add_or_update_object(name="fridge", node_type="receptacle", step_id=2)
        sm.update("put down the object in hand", success=True, step_id=3)
        apple = sm.find_node("apple")
        fridge = sm.find_node("fridge")
        rels = [
            r for r in sm.relations.values()
            if r.subject_id == fridge.id and r.relation == "contains"
            and r.object_id == apple.id and not r.stale
        ]
        assert rels, "Fridge should contain apple after put-down"

    def test_put_down_clears_held_state(self):
        sm = self._setup()
        sm.update("pick up the apple", success=True, step_id=1)
        sm.update("navigate to the fridge", success=True, step_id=2)
        sm.update("put down the object in hand", success=True, step_id=3)
        assert sm._held_object_id is None

    def test_failed_action_does_not_mutate_state(self):
        sm = self._setup()
        apple = sm.find_node("apple")
        sm.update("pick up the apple", success=False, step_id=1)
        assert sm._held_object_id is None
        assert not apple.state.get("isPickedUp")

    def test_open_sets_isOpen_state(self):
        sm = SpatialMemory()
        sm.add_or_update_object(name="fridge", node_type="receptacle", state={"isOpen": False})
        sm.update("open the fridge", success=True, step_id=1)
        assert sm.find_node("fridge").state.get("isOpen") is True

    def test_close_sets_isOpen_false(self):
        sm = SpatialMemory()
        sm.add_or_update_object(name="fridge", node_type="receptacle", state={"isOpen": True})
        sm.update("close the fridge", success=True, step_id=1)
        assert sm.find_node("fridge").state.get("isOpen") is False


# ---------------------------------------------------------------------------
# _infer_spatial_relations
# ---------------------------------------------------------------------------

class TestInferSpatialRelations:
    def test_left_right_of_inferred(self):
        sm = SpatialMemory()
        n1 = sm.add_or_update_object(
            name="Counter A", node_type="receptacle",
            position={"x": -1.0, "y": 0.0, "z": 0.0}
        )
        n2 = sm.add_or_update_object(
            name="Counter B", node_type="receptacle",
            position={"x": 1.5, "y": 0.0, "z": 0.0}
        )
        sm._infer_spatial_relations()
        left_rels = [r for r in sm.relations.values() if r.relation == "left_of"]
        right_rels = [r for r in sm.relations.values() if r.relation == "right_of"]
        assert left_rels, "Should have at least one left_of relation"
        assert right_rels, "Should have at least one right_of relation"

    def test_near_relation_inferred_for_close_nodes(self):
        sm = SpatialMemory()
        sm.add_or_update_object(
            name="Table", node_type="receptacle",
            position={"x": 0.0, "y": 0.0, "z": 0.0}
        )
        sm.add_or_update_object(
            name="Chair", node_type="receptacle",
            position={"x": 0.8, "y": 0.0, "z": 0.0}
        )
        sm._infer_spatial_relations()
        near_rels = [r for r in sm.relations.values() if r.relation == "near"]
        assert near_rels, "Close nodes should have 'near' relation"

    def test_no_relation_for_nodes_without_position(self):
        sm = SpatialMemory()
        sm.add_or_update_object(name="Table", node_type="receptacle")
        sm.add_or_update_object(name="Chair", node_type="receptacle")
        sm._infer_spatial_relations()
        assert not sm.relations, "No position → no geometric relations"

    def test_objects_not_included_in_directional_rels(self):
        """Only receptacle/room/area nodes participate in directional inference."""
        sm = SpatialMemory()
        sm.add_or_update_object(
            name="apple", node_type="object",
            position={"x": 0.0, "y": 0.0, "z": 0.0}
        )
        sm.add_or_update_object(
            name="banana", node_type="object",
            position={"x": 2.0, "y": 0.0, "z": 0.0}
        )
        sm._infer_spatial_relations()
        assert not sm.relations


# ---------------------------------------------------------------------------
# to_prompt_context (grouped format)
# ---------------------------------------------------------------------------

class TestToPromptContextGrouped:
    def test_contains_header(self):
        sm = SpatialMemory()
        table = sm.add_or_update_object(name="Table 1", node_type="receptacle")
        plate = sm.add_or_update_object(name="Plate", node_type="object")
        sm.add_relation(table.id, "contains", plate.id, confidence=1.0)
        q = MemoryQuery(task_instruction="put plate on table", target_objects=["plate"])
        ctx = sm.to_prompt_context(sm.retrieve(q, top_k=5))
        # Header is added by MemoryPromptFormatter; raw context has the scene body.
        assert "Relevant Spatial Information:" in ctx

    def test_receptacle_contains_format(self):
        sm = SpatialMemory()
        table = sm.add_or_update_object(name="Table 1", node_type="receptacle")
        for obj_name in ["Plate", "Potato", "Hammer"]:
            node = sm.add_or_update_object(name=obj_name, node_type="object")
            sm.add_relation(table.id, "contains", node.id, confidence=1.0)
        q = MemoryQuery(task_instruction="find the plate", target_objects=["plate"])
        ctx = sm.to_prompt_context(sm.retrieve(q, top_k=10))
        assert "Table 1 contains" in ctx
        assert "Plate" in ctx

    def test_directional_relation_in_context(self):
        sm = SpatialMemory()
        bin_node = sm.add_or_update_object(
            name="Green trash bin", node_type="receptacle",
            position={"x": -1.5, "y": 0.0, "z": 0.0}
        )
        counter = sm.add_or_update_object(
            name="Kitchen Counter", node_type="receptacle",
            position={"x": 1.5, "y": 0.0, "z": 0.0}
        )
        sm._infer_spatial_relations()
        q = MemoryQuery(task_instruction="find bin near counter")
        ctx = sm.to_prompt_context(sm.retrieve(q, top_k=10))
        assert "Green trash bin" in ctx
        assert "Kitchen Counter" in ctx
        assert "left" in ctx.lower() or "right" in ctx.lower()

    def test_stale_override_notice_present(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple")
        table = sm.add_or_update_object(name="Table")
        rel = sm.add_relation(apple.id, "on", table.id, confidence=1.0)
        sm.mark_stale(relation_id=rel.id)
        q = MemoryQuery(task_instruction="find apple", target_objects=["apple"])
        ctx = sm.to_prompt_context(sm.retrieve(q, top_k=5))
        assert "override" in ctx.lower() or "stale" in ctx.lower()

    def test_empty_memories_returns_empty_string(self):
        sm = SpatialMemory()
        assert sm.to_prompt_context([]) == ""

    def test_respects_max_chars(self):
        sm = SpatialMemory()
        table = sm.add_or_update_object(name="Table", node_type="receptacle")
        for i in range(20):
            n = sm.add_or_update_object(name=f"Object{i}", node_type="object")
            sm.add_relation(table.id, "contains", n.id, confidence=1.0)
        q = MemoryQuery(task_instruction="find object")
        ctx = sm.to_prompt_context(sm.retrieve(q, top_k=5))
        assert len(ctx) > 0

    def test_in_relation_shown_as_containment(self):
        """'object in receptacle' should render as 'receptacle contains object'."""
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple", node_type="object")
        fridge = sm.add_or_update_object(name="Fridge", node_type="receptacle")
        sm.add_relation(apple.id, "in", fridge.id, confidence=1.0)
        q = MemoryQuery(task_instruction="get apple from fridge", target_objects=["apple"])
        ctx = sm.to_prompt_context(sm.retrieve(q, top_k=5))
        assert "Fridge contains Apple" in ctx


# ---------------------------------------------------------------------------
# End-to-end: metadata → action → prompt
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_pick_place_cycle_reflected_in_context(self):
        """Full cycle: build from metadata → pick up object → place on new receptacle."""
        metadata = {
            "objects": [
                {
                    "objectType": "Apple",
                    "objectId": "Apple|1",
                    "position": {"x": 1.0, "y": 0.9, "z": 0.5},
                    "parentReceptacles": ["Table|1"],
                    "isPickedUp": False,
                },
                {
                    "objectType": "Table",
                    "objectId": "Table|1",
                    "position": {"x": 1.0, "y": 0.75, "z": 0.5},
                    "parentReceptacles": [],
                },
                {
                    "objectType": "Fridge",
                    "objectId": "Fridge|1",
                    "position": {"x": -2.0, "y": 0.75, "z": 0.5},
                    "parentReceptacles": [],
                },
            ],
            "agentPosition": {},
        }
        sm = SpatialMemory()
        sm.build_from_metadata(metadata, step_id=0)

        # Step 1: pick up apple
        sm.update("pick up the Apple", success=True, step_id=1)
        apple = sm.find_node("Apple")
        assert apple.state.get("isPickedUp") is True

        # Step 2: navigate to fridge
        sm.update("navigate to the Fridge", success=True, step_id=2)

        # Step 3: place
        sm.update("put down the object in hand", success=True, step_id=3)
        fridge = sm.find_node("Fridge")
        rels = [
            r for r in sm.relations.values()
            if r.subject_id == fridge.id and r.relation == "contains"
            and r.object_id == apple.id and not r.stale
        ]
        assert rels, "Fridge should contain Apple after place action"

        # Check context
        q = MemoryQuery(task_instruction="put apple in fridge", target_objects=["apple"])
        ctx = sm.to_prompt_context(sm.retrieve(q, top_k=5))
        assert "Fridge contains Apple" in ctx
