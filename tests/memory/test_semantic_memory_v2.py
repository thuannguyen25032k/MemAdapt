"""
tests/memory/test_semantic_memory_v2.py

Tests for the revised SemanticMemory:
  - SemanticVectorStore (top_k_similar, embed)
  - MemoryCreator       (VLM path only)
  - UpdateDecision      (dataclass)
  - MemoryUpdater       (VLM path only)
  - SemanticMemory      (update_from_episode, load persistence, to_prompt_context header)

Run with:
    conda run -n embench python -m pytest tests/memory/test_semantic_memory_v2.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from embodiedbench.memory.base import MemoryQuery
from embodiedbench.memory.embeddings import HashEmbeddingProvider
from embodiedbench.memory.semantic_memory import (
    MemoryCreator,
    MemoryUpdater,
    SemanticFact,
    SemanticMemory,
    SemanticVectorStore,
    UpdateDecision,
    _reason_for_category,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_sm(provider=None, **kwargs) -> SemanticMemory:
    return SemanticMemory(embedding_provider=provider, **kwargs)


def _hash_provider(dim: int = 32) -> HashEmbeddingProvider:
    return HashEmbeddingProvider(dim=dim)


# ===========================================================================
# UpdateDecision
# ===========================================================================

class TestUpdateDecision:
    def test_defaults(self):
        d = UpdateDecision("add")
        assert d.action == "add"
        assert d.target_id is None
        assert d.new_content is None

    def test_update_decision(self):
        d = UpdateDecision("update", target_id="abc", new_content="new text")
        assert d.action == "update"
        assert d.target_id == "abc"
        assert d.new_content == "new text"

    def test_remove_decision(self):
        d = UpdateDecision("remove", target_id="xyz")
        assert d.action == "remove"
        assert d.target_id == "xyz"

    def test_noop_decision(self):
        d = UpdateDecision("noop")
        assert d.action == "noop"


# ===========================================================================
# SemanticVectorStore
# ===========================================================================

class TestSemanticVectorStore:
    def _store_with_facts(self, provider=None):
        facts = [
            SemanticFact(content="Open the fridge before placing items inside."),
            SemanticFact(content="Tables are common surfaces for objects."),
            SemanticFact(content="If an apple is not found, search nearby receptacles."),
        ]
        store = SemanticVectorStore(facts, provider)
        return store, facts

    def test_top_k_similar_returns_sorted(self):
        store, _ = self._store_with_facts()
        results = store.top_k_similar("fridge placement", k=3)
        assert len(results) <= 3
        scores = [s for s, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self):
        store, _ = self._store_with_facts()
        results = store.top_k_similar("open fridge", k=1)
        assert len(results) == 1

    def test_top_k_empty_store(self):
        store = SemanticVectorStore([])
        results = store.top_k_similar("anything", k=5)
        assert results == []

    def test_top_k_empty_query(self):
        store, _ = self._store_with_facts()
        results = store.top_k_similar("", k=3)
        assert results == []

    def test_top_k_excludes_ids(self):
        store, facts = self._store_with_facts()
        exclude = {facts[0].id, facts[1].id}
        results = store.top_k_similar("fridge", k=5, exclude_ids=exclude)
        result_ids = {f.id for _, f in results}
        assert result_ids.isdisjoint(exclude)

    def test_top_k_with_embedding_provider(self):
        provider = _hash_provider()
        store, _ = self._store_with_facts(provider=provider)
        results = store.top_k_similar("open the fridge", k=2)
        assert len(results) == 2
        for score, fact in results:
            assert 0.0 <= score <= 1.0

    def test_embed_without_provider_returns_none(self):
        store = SemanticVectorStore([])
        assert store.embed("hello") is None

    def test_embed_with_provider_returns_vector(self):
        provider = _hash_provider(dim=16)
        store = SemanticVectorStore([], provider)
        vec = store.embed("hello world")
        assert vec is not None
        assert len(vec) == 16

    def test_shared_reference_reflects_mutations(self):
        """Vector store reflects facts added after construction."""
        facts: list = []
        store = SemanticVectorStore(facts)
        facts.append(SemanticFact(content="New fact added later."))
        results = store.top_k_similar("new fact", k=1)
        assert len(results) == 1


# ===========================================================================
# MemoryCreator (no VLM → empty list)
# ===========================================================================

class TestMemoryCreatorNoVlm:
    def test_no_vlm_returns_empty(self):
        creator = MemoryCreator()
        candidates = creator.create_from_episode(
            episode_summary="Opened fridge, placed apple inside.",
            task_instruction="Put apple in fridge.",
            success=True,
        )
        assert candidates == []

    def test_no_vlm_on_failure_returns_empty(self):
        creator = MemoryCreator()
        candidates = creator.create_from_episode(
            episode_summary="Failed to pick up bowl.",
            success=False,
        )
        assert candidates == []

    def test_empty_summary_returns_empty(self):
        creator = MemoryCreator()
        assert creator.create_from_episode(episode_summary="") == []


# ===========================================================================
# MemoryCreator (VLM path)
# ===========================================================================

class TestMemoryCreatorVlm:
    def _vlm_response(self) -> str:
        return json.dumps({
            "facts": [
                {
                    "content": "Always open container before placing objects inside.",
                    "category": "precondition",
                    "related_objects": ["container"],
                    "related_actions": ["open", "place"],
                },
                {
                    "content": "Apples are often found on counters or in fridges.",
                    "category": "affordance",
                    "related_objects": ["apple", "counter", "fridge"],
                    "related_actions": [],
                },
            ]
        })

    def test_vlm_path_used_when_available(self):
        vlm = MagicMock(return_value=self._vlm_response())
        creator = MemoryCreator(vlm_call=vlm)
        candidates = creator.create_from_episode(
            episode_summary="Placed apple in fridge.", success=True,
        )
        assert vlm.called
        assert any(c["category"] == "precondition" for c in candidates)

    def test_vlm_fallback_on_bad_json(self):
        vlm = MagicMock(return_value="not valid json {{{")
        creator = MemoryCreator(vlm_call=vlm)
        # VLM parse failure → empty list (no rule-based fallback)
        candidates = creator.create_from_episode(
            episode_summary="Placed apple in fridge.", success=True,
            task_instruction="Put apple in fridge.",
        )
        assert candidates == []

    def test_vlm_fallback_on_exception(self):
        vlm = MagicMock(side_effect=RuntimeError("network error"))
        creator = MemoryCreator(vlm_call=vlm)
        # VLM exception → empty list (no rule-based fallback)
        candidates = creator.create_from_episode(
            episode_summary="Done.", success=True, task_instruction="Task.",
        )
        assert candidates == []

    def test_vlm_invalid_category_becomes_general(self):
        bad_response = json.dumps({
            "facts": [{"content": "Something.", "category": "TOTALLY_INVALID",
                        "related_objects": [], "related_actions": []}]
        })
        vlm = MagicMock(return_value=bad_response)
        creator = MemoryCreator(vlm_call=vlm)
        candidates = creator.create_from_episode(
            episode_summary="Done.", success=True,
        )
        for c in candidates:
            assert c["category"] in (
                "precondition", "affordance", "safety", "failure_avoidance",
                "search_strategy", "task_rule", "environment_rule", "general",
            )

    def test_vlm_max_candidates_respected(self):
        big_response = json.dumps({
            "facts": [
                {"content": f"Fact {i}.", "category": "general",
                 "related_objects": [], "related_actions": []}
                for i in range(10)
            ]
        })
        vlm = MagicMock(return_value=big_response)
        creator = MemoryCreator(vlm_call=vlm, max_candidates=3)
        candidates = creator.create_from_episode(
            episode_summary="Many things happened.", success=True,
        )
        assert len(candidates) <= 3


# ===========================================================================
# MemoryUpdater (no VLM → noop)
# ===========================================================================

class TestMemoryUpdaterNoVlm:
    def _updater(self, facts=None):
        facts = facts or []
        store = SemanticVectorStore(facts)
        return MemoryUpdater(store)  # no vlm_call

    def test_noop_when_store_empty(self):
        updater = self._updater()
        decision = updater.process({"content": "Open fridge before placing items."})
        assert decision.action == "noop"

    def test_noop_on_existing_content(self):
        fact = SemanticFact(content="To place objects inside the fridge the fridge must be open.")
        updater = self._updater(facts=[fact])
        decision = updater.process({"content": "Open fridge."})
        assert decision.action == "noop"

    def test_noop_on_empty_content(self):
        updater = self._updater()
        decision = updater.process({"content": ""})
        assert decision.action == "noop"

    def test_process_returns_update_decision(self):
        updater = self._updater()
        decision = updater.process({"content": "Something new."})
        assert isinstance(decision, UpdateDecision)


# ===========================================================================
# MemoryUpdater (VLM path)
# ===========================================================================

class TestMemoryUpdaterVlm:
    def _updater_with_vlm(self, vlm_response: str, facts=None):
        facts = facts or [SemanticFact(content="Fridge must be open to place items.")]
        store = SemanticVectorStore(facts)
        vlm = MagicMock(return_value=vlm_response)
        return MemoryUpdater(store, vlm_call=vlm, top_s=3), vlm

    def test_vlm_add_decision(self):
        updater, vlm = self._updater_with_vlm('{"action": "add"}')
        decision = updater.process({"content": "Navigate before manipulating objects."})
        assert vlm.called
        assert decision.action == "add"

    def test_vlm_noop_decision(self):
        updater, _ = self._updater_with_vlm('{"action": "noop"}')
        decision = updater.process({"content": "Open fridge to place items."})
        assert decision.action == "noop"

    def test_vlm_update_decision(self):
        updater, _ = self._updater_with_vlm(
            '{"action": "update", "target_index": 1, "new_content": "Updated rule."}'
        )
        decision = updater.process({"content": "Fridge must be open."})
        assert decision.action == "update"
        assert decision.target_id is not None

    def test_vlm_remove_decision(self):
        updater, _ = self._updater_with_vlm('{"action": "remove", "target_index": 1}')
        decision = updater.process({"content": "Some obsolete fact."})
        assert decision.action == "remove"
        assert decision.target_id is not None

    def test_vlm_fallback_on_bad_response(self):
        updater, _ = self._updater_with_vlm("not json at all <<<")
        decision = updater.process({"content": "New fact."})
        # Bad JSON → defaults to "add"
        assert decision.action == "add"

    def test_vlm_fallback_on_exception(self):
        facts = [SemanticFact(content="Existing fact.")]
        store = SemanticVectorStore(facts)
        vlm = MagicMock(side_effect=RuntimeError("timeout"))
        updater = MemoryUpdater(store, vlm_call=vlm)
        # VLM exception → defaults to "add"
        decision = updater.process({"content": "New fact."})
        assert decision.action == "add"


# ===========================================================================
# SemanticMemory.update_from_episode
# ===========================================================================

class TestUpdateFromEpisode:
    def _vlm_task_rule(self) -> MagicMock:
        return MagicMock(return_value=json.dumps({
            "facts": [{"content": "Always open container before placing.",
                        "category": "task_rule", "related_objects": [],
                        "related_actions": ["open", "place"]}]
        }))

    def _vlm_failure(self) -> MagicMock:
        return MagicMock(return_value=json.dumps({
            "facts": [{"content": "Avoid repeating failed pick-up attempts.",
                        "category": "failure_avoidance", "related_objects": [],
                        "related_actions": []}]
        }))

    def test_success_adds_task_rule(self):
        sm = SemanticMemory(vlm_call=self._vlm_task_rule())
        # Updater also needs VLM — mock to always "add"
        sm._updater.vlm_call = MagicMock(return_value='{"action": "add"}')
        added = sm.update_from_episode(
            episode_summary="Opened fridge and placed apple.",
            task_instruction="Put apple in fridge.",
            success=True,
        )
        assert len(added) >= 1
        assert any(f.category == "task_rule" for f in added)

    def test_failure_adds_failure_avoidance(self):
        sm = SemanticMemory(vlm_call=self._vlm_failure())
        sm._updater.vlm_call = MagicMock(return_value='{"action": "add"}')
        added = sm.update_from_episode(
            episode_summary="Failed to pick up bowl.",
            success=False,
        )
        assert any(f.category == "failure_avoidance" for f in added)

    def test_no_vlm_returns_empty(self):
        sm = _make_sm()
        added = sm.update_from_episode(episode_summary="Something happened.")
        assert added == []

    def test_no_candidates_returns_empty(self):
        sm = _make_sm()
        added = sm.update_from_episode(episode_summary="")
        assert added == []

    def test_parallelization_does_not_crash(self):
        vlm_creator = MagicMock(return_value=json.dumps({
            "facts": [
                {"content": "Fact one about searching.", "category": "search_strategy",
                 "related_objects": ["cup"], "related_actions": []},
                {"content": "Fact two about navigation.", "category": "task_rule",
                 "related_objects": [], "related_actions": []},
            ]
        }))
        sm = SemanticMemory(vlm_call=vlm_creator,
                            max_update_workers=4)
        sm._updater.vlm_call = MagicMock(return_value='{"action": "add"}')
        added = sm.update_from_episode(
            episode_summary="Object not visible.",
            success=False,
        )
        assert isinstance(added, list)

    def test_dedup_prevents_repeated_add(self):
        vlm_creator = MagicMock(return_value=json.dumps({
            "facts": [{"content": "Open fridge before placing.", "category": "task_rule",
                        "related_objects": [], "related_actions": []}]
        }))
        sm = SemanticMemory(vlm_call=vlm_creator)
        sm._updater.vlm_call = MagicMock(return_value='{"action": "add"}')
        sm.update_from_episode(
            episode_summary="Opened fridge and placed apple.",
            task_instruction="Put apple in fridge.",
            success=True,
        )
        count_after_first = len(sm)
        # Same episode again
        sm.update_from_episode(
            episode_summary="Opened fridge and placed apple.",
            task_instruction="Put apple in fridge.",
            success=True,
        )
        assert len(sm) <= count_after_first + 1

    def test_vlm_path_invoked_when_supplied(self):
        vlm_response = json.dumps({
            "facts": [{
                "content": "Always check container state before placing.",
                "category": "precondition",
                "related_objects": [],
                "related_actions": ["open", "place"],
            }]
        })
        vlm = MagicMock(return_value=vlm_response)
        sm = SemanticMemory(vlm_call=vlm)
        sm._updater.vlm_call = MagicMock(return_value='{"action": "add"}')
        sm.update_from_episode(
            episode_summary="Placed object in container.",
            success=True,
            task_instruction="Put object in container.",
        )
        assert vlm.called

    def test_updater_remove_action_removes_fact(self):
        sm = _make_sm()
        # Add a fact, then use VLM updater to remove it
        fact = sm.add_fact(content="Old outdated rule about tables.")
        assert len(sm) == 1
        # VLM always returns remove targeting index 1
        vlm = MagicMock(return_value='{"action": "remove", "target_index": 1}')
        sm._updater.vlm_call = vlm
        sm._creator.vlm_call = MagicMock(return_value=json.dumps({
            "facts": [{"content": "New better rule.", "category": "general",
                        "related_objects": [], "related_actions": []}]
        }))
        sm.update_from_episode(episode_summary="New information.", success=True,
                                task_instruction="Task.")
        # The remove decision should have removed the old fact
        assert sm._find_by_id(fact.id) is None


# ===========================================================================
# SemanticMemory persistence (load shares vector store reference)
# ===========================================================================

class TestSemanticMemoryPersistence:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sem.json")
            sm = _make_sm()
            sm.add_fact(content="open container before placing", category="precondition")
            sm.storage_path = path
            sm.save()

            sm2 = SemanticMemory(storage_path=path)
            sm2.load()
            assert len(sm2) == len(sm)

    def test_load_preserves_vector_store_reference(self):
        """After load(), vector store must still see the new facts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sem.json")
            sm = _make_sm()
            sm.add_fact(content="Apples are often on counters.", category="affordance")
            sm.storage_path = path
            sm.save()

            sm2 = SemanticMemory(storage_path=path)
            sm2.load()
            results = sm2._vector_store.top_k_similar("apple counter", k=1)
            assert len(results) == 1

    def test_load_missing_file_is_noop(self):
        sm = SemanticMemory(storage_path="/nonexistent/path.json")
        sm.load()  # should not raise
        assert len(sm) == 0


# ===========================================================================
# SemanticMemory.to_prompt_context
# ===========================================================================

class TestSemanticMemoryPromptContext:
    def _sm_with_facts(self):
        sm = _make_sm()
        sm.add_fact(content="open fridge before placing items inside", category="precondition",
                    related_actions=["open", "place"], importance=0.9)
        sm.add_fact(content="avoid repeating failed actions", category="failure_avoidance",
                    related_actions=["repeat"], importance=0.85)
        return sm

    def test_header_present(self):
        sm = self._sm_with_facts()
        query = MemoryQuery(task_instruction="open the fridge and place apple inside")
        results = sm.retrieve(query, top_k=3)
        ctx = sm.to_prompt_context(results)
        # Header is added by MemoryPromptFormatter; raw context has the rules body.
        assert "Relevant rules and commonsense knowledge:" in ctx

    def test_relevant_rules_line_present(self):
        sm = self._sm_with_facts()
        query = MemoryQuery(task_instruction="place the apple")
        results = sm.retrieve(query, top_k=3)
        ctx = sm.to_prompt_context(results)
        assert "Relevant rules" in ctx

    def test_empty_memories_returns_empty(self):
        sm = _make_sm()
        assert sm.to_prompt_context([]) == ""

    def test_category_labels_present(self):
        sm = self._sm_with_facts()
        query = MemoryQuery(
            task_instruction="open fridge, place apple",
            recent_actions=["open fridge"],
        )
        results = sm.retrieve(query, top_k=5)
        ctx = sm.to_prompt_context(results)
        assert "[" in ctx and "]" in ctx  # at least one [category] label


# ===========================================================================
# SemanticMemory.retrieve via vector store
# ===========================================================================

class TestSemanticMemoryRetrieveViaStore:
    def test_retrieve_uses_vector_store(self):
        """Retrieve should delegate base similarity through the vector store."""
        provider = _hash_provider(dim=64)
        sm = SemanticMemory(embedding_provider=provider)
        sm.add_fact(content="open the fridge before placing items inside", category="precondition")
        query = MemoryQuery(
            task_instruction="open the fridge and place apple inside",
            recent_actions=["navigate to fridge"],
        )
        results = sm.retrieve(query, top_k=5)
        assert len(results) > 0
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_retrieve_top_k_respected(self):
        sm = _make_sm()
        sm.add_fact(content="fact one", category="general")
        sm.add_fact(content="fact two", category="general")
        sm.add_fact(content="fact three", category="general")
        results = sm.retrieve(MemoryQuery(task_instruction="task"), top_k=2)
        assert len(results) <= 2

    def test_retrieve_empty_returns_empty(self):
        sm = _make_sm()
        results = sm.retrieve(MemoryQuery(task_instruction="task"))
        assert results == []


# ===========================================================================
# _reason_for_category utility
# ===========================================================================

class TestReasonForCategory:
    @pytest.mark.parametrize("cat,expected_substr", [
        ("precondition", "precondition"),
        ("affordance", "affordance"),
        ("safety", "safety"),
        ("failure_avoidance", "failure"),
        ("search_strategy", "search"),
        ("task_rule", "task"),
        ("unknown_cat", "semantic"),
    ])
    def test_known_and_unknown(self, cat, expected_substr):
        reason = _reason_for_category(cat)
        assert expected_substr in reason
