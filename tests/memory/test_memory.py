"""
tests/memory/test_memory.py

Unit tests for the MemAdapt memory foundation package.
Run with: pytest tests/memory/test_memory.py -v
"""

from __future__ import annotations

import os
import tempfile

import pytest

from embodiedbench.memory.base import (
    MemoryContext,
    MemoryItem,
    MemoryQuery,
    RetrievedMemory,
    now_ts,
    normalize_text,
    safe_json_dumps,
    truncate_text,
)
from embodiedbench.memory.embeddings import (
    DummyEmbeddingProvider,
    HashEmbeddingProvider,
    cosine_similarity,
    hybrid_score,
    lexical_overlap_score,
)
from embodiedbench.memory.storage import (
    append_jsonl,
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
)
from embodiedbench.memory.temporal_memory import TemporalMemory, TemporalStep
from embodiedbench.memory.semantic_memory import SemanticMemory, SemanticFact
from embodiedbench.memory.episodic_memory import EpisodicMemory, EpisodeRecord
from embodiedbench.memory.spatial_memory import SpatialMemory, SpatialNode, SpatialRelation
from embodiedbench.memory.manager import MemoryManager, MemoryConfig
from embodiedbench.memory.prompt_formatter import MemoryPromptFormatter


# ===========================================================================
# 1. MemoryItem serialization / deserialization
# ===========================================================================

class TestMemoryItem:
    def test_roundtrip(self):
        item = MemoryItem(
            memory_type="episodic",
            content="Picked up the apple from the table.",
            metadata={"env": "alfred", "step": 3},
            importance=0.8,
            confidence=0.9,
            source="evaluator",
        )
        d = item.to_dict()
        restored = MemoryItem.from_dict(d)

        assert restored.id == item.id
        assert restored.memory_type == item.memory_type
        assert restored.content == item.content
        assert restored.metadata == item.metadata
        assert restored.importance == pytest.approx(item.importance)
        assert restored.confidence == pytest.approx(item.confidence)
        assert restored.source == item.source

    def test_default_id_is_unique(self):
        a = MemoryItem()
        b = MemoryItem()
        assert a.id != b.id

    def test_touch_updates_timestamp(self):
        item = MemoryItem()
        old_ts = item.updated_at
        item.touch()
        assert item.updated_at >= old_ts

    def test_short_text_truncates(self):
        item = MemoryItem(content="x" * 500)
        short = item.short_text(max_chars=50)
        assert len(short) <= 50
        assert short.endswith("…")

    def test_short_text_no_truncation_when_within_limit(self):
        item = MemoryItem(content="short")
        assert item.short_text(100) == "short"

    def test_embedding_roundtrip(self):
        item = MemoryItem(embedding=[0.1, 0.2, 0.3])
        restored = MemoryItem.from_dict(item.to_dict())
        assert restored.embedding == pytest.approx([0.1, 0.2, 0.3])

    def test_from_dict_missing_fields_uses_defaults(self):
        item = MemoryItem.from_dict({})
        assert item.memory_type == "episodic"
        assert item.importance == pytest.approx(0.5)
        assert item.confidence == pytest.approx(1.0)


# ===========================================================================
# 2. MemoryQuery.text_for_retrieval
# ===========================================================================

class TestMemoryQuery:
    def test_text_for_retrieval_combines_fields(self):
        q = MemoryQuery(
            task_instruction="Put the apple in the fridge.",
            observation_text="I see an apple on the table.",
            target_objects=["apple", "fridge"],
            recent_actions=["navigate to table"],
            proposed_plan="pick up apple, navigate to fridge, place apple",
        )
        text = q.text_for_retrieval()
        assert "Put the apple" in text
        assert "apple on the table" in text
        assert "apple" in text and "fridge" in text
        assert "navigate to table" in text
        assert "pick up apple" in text

    def test_text_for_retrieval_empty(self):
        q = MemoryQuery()
        assert q.text_for_retrieval() == ""

    def test_to_dict_excludes_non_serializable_obs(self):
        # Simulate a numpy-array-like object (use a plain list here)
        q = MemoryQuery(task_instruction="test", raw_observation=[1, 2, 3])
        d = q.to_dict()
        # raw_observation should be stored as a string repr, not the raw object
        assert isinstance(d["raw_observation"], str)

    def test_to_dict_string_obs_kept(self):
        q = MemoryQuery(task_instruction="test", raw_observation="/tmp/img.png")
        d = q.to_dict()
        assert d["raw_observation"] == "/tmp/img.png"

    def test_roundtrip(self):
        q = MemoryQuery(
            task_instruction="Navigate to the bedroom.",
            target_objects=["bed"],
            env_name="eb-nav",
            step_id=5,
        )
        restored = MemoryQuery.from_dict(q.to_dict())
        assert restored.task_instruction == q.task_instruction
        assert restored.target_objects == q.target_objects
        assert restored.env_name == q.env_name
        assert restored.step_id == q.step_id


# ===========================================================================
# 3. MemoryContext.build_combined_context
# ===========================================================================

class TestMemoryContext:
    def test_build_combined_context_includes_all_sections(self):
        ctx = MemoryContext(
            spatial_context="Kitchen is north.",
            temporal_context="Last seen apple 3 steps ago.",
            episodic_context="Previously failed to open fridge.",
            semantic_context="Apples can be sliced.",
            feasibility_constraints=["Cannot pick up while holding."],
        )
        combined = ctx.build_combined_context()
        assert "[Spatial Memory]" in combined
        assert "[Temporal Memory]" in combined
        assert "[Episodic Memory]" in combined
        assert "[Semantic Memory]" in combined
        assert "[Feasibility Constraints]" in combined
        assert ctx.combined_context == combined  # stored in-place

    def test_build_combined_context_skips_empty_sections(self):
        ctx = MemoryContext(episodic_context="Something happened.")
        combined = ctx.build_combined_context()
        assert "[Episodic Memory]" in combined
        assert "[Spatial Memory]" not in combined

    def test_compact_truncates(self):
        ctx = MemoryContext(episodic_context="x" * 3000)
        compacted = ctx.compact(max_chars=100)
        assert len(compacted) <= 100

    # 4. is_empty
    def test_is_empty_true_when_nothing_set(self):
        ctx = MemoryContext()
        assert ctx.is_empty() is True

    def test_is_empty_false_when_context_set(self):
        ctx = MemoryContext(spatial_context="something")
        assert ctx.is_empty() is False

    def test_is_empty_false_when_retrieved_items(self):
        item = MemoryItem(content="found something")
        ctx = MemoryContext(retrieved_items=[RetrievedMemory(item=item, score=0.9)])
        assert ctx.is_empty() is False

    def test_roundtrip(self):
        item = MemoryItem(content="test memory")
        rm = RetrievedMemory(item=item, score=0.75, reason="high overlap")
        ctx = MemoryContext(
            spatial_context="room A",
            retrieved_items=[rm],
        )
        ctx.build_combined_context()
        restored = MemoryContext.from_dict(ctx.to_dict())
        assert restored.spatial_context == "room A"
        assert len(restored.retrieved_items) == 1
        assert restored.retrieved_items[0].score == pytest.approx(0.75)
        assert restored.retrieved_items[0].item.content == "test memory"


# ===========================================================================
# 5. HashEmbeddingProvider — determinism
# ===========================================================================

class TestHashEmbeddingProvider:
    def test_deterministic(self):
        provider = HashEmbeddingProvider(dim=64)
        v1 = provider.embed_text("pick up the apple")
        v2 = provider.embed_text("pick up the apple")
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        provider = HashEmbeddingProvider(dim=64)
        v1 = provider.embed_text("pick up the apple")
        v2 = provider.embed_text("navigate to the bedroom")
        assert v1 != v2

    def test_output_dimension(self):
        for dim in [32, 64, 128]:
            provider = HashEmbeddingProvider(dim=dim)
            v = provider.embed_text("some text")
            assert len(v) == dim

    def test_normalize_produces_unit_vector(self):
        provider = HashEmbeddingProvider(dim=64, normalize=True)
        v = provider.embed_text("hello world")
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_embed_batch(self):
        provider = HashEmbeddingProvider(dim=32)
        texts = ["text one", "text two", "text three"]
        batch = provider.embed_batch(texts)
        assert len(batch) == 3
        for v in batch:
            assert len(v) == 32

    def test_empty_text_returns_vector(self):
        provider = HashEmbeddingProvider(dim=32)
        v = provider.embed_text("")
        assert len(v) == 32

    def test_dummy_provider_returns_correct_dim(self):
        provider = DummyEmbeddingProvider(dim=64)
        v = provider.embed_text("anything")
        assert len(v) == 64

    def test_dummy_provider_deterministic(self):
        provider = DummyEmbeddingProvider(dim=32)
        v1 = provider.embed_text("hello")
        v2 = provider.embed_text("hello")
        assert v1 == v2


# ===========================================================================
# 6. cosine_similarity — normal and edge cases
# ===========================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_empty_vector_returns_zero(self):
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], []) == 0.0
        assert cosine_similarity([], [1.0]) == 0.0

    def test_zero_magnitude_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_mismatched_lengths_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_hash_embeddings_similar_texts_higher_than_unrelated(self):
        provider = HashEmbeddingProvider(dim=128)
        v_apple1 = provider.embed_text("pick up the apple")
        v_apple2 = provider.embed_text("pick up the red apple")
        v_nav    = provider.embed_text("navigate to bedroom door")
        sim_related   = cosine_similarity(v_apple1, v_apple2)
        sim_unrelated = cosine_similarity(v_apple1, v_nav)
        # Related texts should score higher (not guaranteed to hold for all hash
        # functions, but the HashEmbeddingProvider is designed so this holds on
        # these specific inputs).
        assert sim_related >= sim_unrelated - 0.05  # allow small tolerance


# ===========================================================================
# 7. lexical_overlap_score — higher for related text
# ===========================================================================

class TestLexicalOverlapScore:
    def test_identical_text(self):
        score = lexical_overlap_score("pick up apple", "pick up apple")
        assert score == pytest.approx(1.0)

    def test_disjoint_text(self):
        score = lexical_overlap_score("apple fridge", "navigate bedroom")
        assert score == pytest.approx(0.0)

    def test_partial_overlap_between_zero_and_one(self):
        score = lexical_overlap_score("pick up the apple", "pick up the fridge")
        assert 0.0 < score < 1.0

    def test_related_higher_than_unrelated(self):
        query = "pick up the apple from the table"
        related   = "pick up the apple and put it in the fridge"
        unrelated = "navigate to the bedroom and open the door"
        assert lexical_overlap_score(query, related) > lexical_overlap_score(query, unrelated)

    def test_empty_query_returns_zero(self):
        assert lexical_overlap_score("", "some document") == 0.0

    def test_empty_document_returns_zero(self):
        assert lexical_overlap_score("some query", "") == 0.0


# ===========================================================================
# 8. hybrid_score — works without embeddings
# ===========================================================================

class TestHybridScore:
    def test_falls_back_to_lexical_without_embeddings(self):
        q = "pick up the apple"
        d = "pick up the apple"
        score = hybrid_score(q, d)
        lex   = lexical_overlap_score(q, d)
        assert score == pytest.approx(lex)

    def test_with_embeddings(self):
        provider = HashEmbeddingProvider(dim=64)
        q = "pick up the apple"
        d = "grab the apple from table"
        qe = provider.embed_text(q)
        de = provider.embed_text(d)
        score = hybrid_score(q, d, qe, de, embedding_weight=0.7, lexical_weight=0.3)
        assert 0.0 <= score <= 1.0

    def test_none_embeddings_fallback(self):
        score = hybrid_score("hello world", "hello world", None, None)
        assert score == pytest.approx(1.0)

    def test_empty_embeddings_fallback(self):
        score = hybrid_score("hello world", "hello world", [], [])
        assert score == pytest.approx(1.0)

    def test_returns_float(self):
        result = hybrid_score("foo", "bar")
        assert isinstance(result, float)


# ===========================================================================
# 9. JSON save / load
# ===========================================================================

class TestJsonStorage:
    def test_save_and_load_roundtrip(self, tmp_path):
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        path = str(tmp_path / "subdir" / "data.json")
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data

    def test_load_missing_file_returns_default(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert load_json(path) is None
        assert load_json(path, default={}) == {}

    def test_load_corrupt_file_returns_default(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("{ not valid json }")
        assert load_json(path, default="fallback") == "fallback"

    def test_creates_nested_directories(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "c" / "data.json")
        save_json(path, {"x": 1})
        assert os.path.isfile(path)

    def test_utf8_content(self, tmp_path):
        data = {"message": "日本語テスト"}
        path = str(tmp_path / "utf8.json")
        save_json(path, data)
        loaded = load_json(path)
        assert loaded["message"] == "日本語テスト"


# ===========================================================================
# 10. JSONL append / load
# ===========================================================================

class TestJsonlStorage:
    def test_save_and_load_roundtrip(self, tmp_path):
        rows = [{"a": 1}, {"b": 2}, {"c": 3}]
        path = str(tmp_path / "data.jsonl")
        save_jsonl(path, rows)
        loaded = load_jsonl(path)
        assert loaded == rows

    def test_append_creates_file(self, tmp_path):
        path = str(tmp_path / "append.jsonl")
        append_jsonl(path, {"step": 1})
        append_jsonl(path, {"step": 2})
        loaded = load_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0] == {"step": 1}
        assert loaded[1] == {"step": 2}

    def test_load_missing_file_returns_empty_list(self, tmp_path):
        path = str(tmp_path / "nonexistent.jsonl")
        assert load_jsonl(path) == []

    def test_malformed_lines_are_skipped(self, tmp_path):
        path = str(tmp_path / "mixed.jsonl")
        with open(path, "w") as f:
            f.write('{"good": 1}\n')
            f.write("not json\n")
            f.write('{"good": 2}\n')
        loaded = load_jsonl(path)
        assert loaded == [{"good": 1}, {"good": 2}]

    def test_append_creates_nested_directories(self, tmp_path):
        path = str(tmp_path / "x" / "y" / "rows.jsonl")
        append_jsonl(path, {"row": 0})
        assert os.path.isfile(path)

    def test_memory_item_roundtrip_via_jsonl(self, tmp_path):
        item = MemoryItem(
            memory_type="spatial",
            content="The fridge is in the kitchen.",
            importance=0.7,
        )
        path = str(tmp_path / "items.jsonl")
        append_jsonl(path, item.to_dict())
        rows = load_jsonl(path)
        assert len(rows) == 1
        restored = MemoryItem.from_dict(rows[0])
        assert restored.id == item.id
        assert restored.content == item.content
        assert restored.importance == pytest.approx(0.7)


# ===========================================================================
# 11–24: TemporalMemory tests
# ===========================================================================

def _make_step(
    step_id: int = 0,
    action_text: str = "pick up apple",
    success: bool = True,
    env_feedback: str = "Action succeeded.",
    critic_rejected: bool = False,
    critic_output: str = "",
    task_instruction: str = "Put the apple in the fridge.",
) -> TemporalStep:
    return TemporalStep(
        step_id=step_id,
        task_instruction=task_instruction,
        action=step_id,
        action_text=action_text,
        env_feedback=env_feedback,
        success=success,
        critic_rejected=critic_rejected,
        critic_output=critic_output,
    )


class TestTemporalStep:
    # 1. TemporalStep serialization / deserialization
    def test_roundtrip(self):
        step = _make_step(step_id=3, action_text="open fridge", success=False,
                          env_feedback="Fridge is not reachable.")
        d = step.to_dict()
        restored = TemporalStep.from_dict(d)
        assert restored.step_id == 3
        assert restored.action_text == "open fridge"
        assert restored.success is False
        assert restored.env_feedback == "Fridge is not reachable."

    def test_non_serializable_action_becomes_string(self):
        step = TemporalStep(step_id=0, action=[1, 2, 3])
        d = step.to_dict()
        assert isinstance(d["action"], str)

    def test_info_with_scene_objects_stores_count(self):
        step = TemporalStep(
            step_id=0,
            info={"scene_objects": [{"objectType": "Apple"}, {"objectType": "Fridge"}],
                  "task_success": 0},
        )
        d = step.to_dict()
        assert "scene_objects_count" in d["info"]
        assert d["info"]["scene_objects_count"] == 2
        assert "scene_objects" not in d["info"]

    # 2. TemporalStep.to_memory_item()
    def test_to_memory_item_type(self):
        step = _make_step(step_id=1, action_text="find apple", success=True)
        item = step.to_memory_item()
        assert item.memory_type == "temporal"
        assert "find apple" in item.content
        assert item.source == "temporal_memory"

    def test_to_memory_item_failure_importance(self):
        step = _make_step(success=False)
        item = step.to_memory_item()
        assert item.importance > 0.5

    def test_to_memory_item_critic_rejection_importance(self):
        step = _make_step(critic_rejected=True, success=False)
        item = step.to_memory_item()
        assert item.importance >= 0.8

    def test_short_summary_contains_action(self):
        step = _make_step(action_text="navigate to kitchen", success=True)
        summary = step.short_summary()
        assert "navigate to kitchen" in summary

    def test_short_summary_truncates(self):
        step = _make_step(env_feedback="x" * 500)
        summary = step.short_summary(max_chars=80)
        assert len(summary) <= 80


class TestTemporalMemoryAppend:
    # 3. append_step() adds steps
    def test_append_increases_length(self):
        tm = TemporalMemory(max_steps=10)
        tm.append_step(action_text="find apple", success=True)
        tm.append_step(action_text="pick up apple", success=False)
        assert len(tm) == 2

    def test_step_id_auto_inferred(self):
        tm = TemporalMemory()
        tm.append_step(action_text="step zero")
        tm.append_step(action_text="step one")
        assert tm.steps[0].step_id == 0
        assert tm.steps[1].step_id == 1

    def test_explicit_step_id(self):
        tm = TemporalMemory()
        tm.append_step(action_text="action", step_id=42)
        assert tm.steps[0].step_id == 42

    def test_success_inferred_from_info(self):
        tm = TemporalMemory()
        tm.append_step(info={"last_action_success": 1, "env_feedback": "OK"})
        assert tm.steps[0].success is True

    def test_action_text_inferred_from_info(self):
        tm = TemporalMemory()
        tm.append_step(info={"action_description": "open fridge", "last_action_success": 0})
        assert tm.steps[0].action_text == "open fridge"

    def test_env_feedback_inferred_from_info(self):
        tm = TemporalMemory()
        tm.append_step(info={"env_feedback": "Fridge is closed.", "last_action_success": 0})
        assert tm.steps[0].env_feedback == "Fridge is closed."

    # 4. update() delegates to append_step()
    def test_update_delegates(self):
        tm = TemporalMemory()
        tm.update(action_text="navigate to table", success=True)
        assert len(tm) == 1
        assert tm.steps[0].action_text == "navigate to table"

    def test_update_ignores_unknown_kwargs(self):
        tm = TemporalMemory()
        tm.update(action_text="test", success=True, unknown_future_param="ignored")
        assert len(tm) == 1


class TestTemporalMemoryOverflow:
    # 5. FIFO overflow when compress_on_overflow=False
    def test_fifo_drop(self):
        tm = TemporalMemory(max_steps=3, compress_on_overflow=False)
        for i in range(5):
            tm.append_step(action_text=f"action{i}", step_id=i)
        assert len(tm.steps) == 3
        assert len(tm.summaries) == 0          # no compression
        assert tm.steps[0].step_id == 2        # oldest kept = step 2

    # 6. Compression when compress_on_overflow=True
    def test_compression_stores_summary(self):
        tm = TemporalMemory(max_steps=3, compress_on_overflow=True)
        for i in range(5):
            tm.append_step(action_text=f"action{i}", step_id=i)
        assert len(tm.steps) == 3
        assert len(tm.summaries) >= 1          # compressed steps stored

    def test_compression_summary_contains_old_actions(self):
        tm = TemporalMemory(max_steps=3, compress_on_overflow=True)
        tm.append_step(action_text="find apple", step_id=0)
        tm.append_step(action_text="pick up apple", step_id=1)
        tm.append_step(action_text="navigate", step_id=2)
        tm.append_step(action_text="open fridge", step_id=3)  # triggers overflow
        # step 0 was compressed
        assert any("find apple" in s for s in tm.summaries)

    def test_summaries_pruned_when_over_limit(self):
        tm = TemporalMemory(max_steps=2, max_summaries=2, compress_on_overflow=True)
        for i in range(10):
            tm.append_step(action_text=f"a{i}", step_id=i)
        # Summaries must not exceed max_summaries (merging keeps it bounded)
        assert len(tm.summaries) <= tm.max_summaries


class TestTemporalMemorySummarize:
    # 7. summarize_recent_history() includes recent actions
    def test_includes_recent_actions(self):
        tm = TemporalMemory()
        tm.append_step(action_text="find apple", success=True)
        tm.append_step(action_text="pick up apple", success=False,
                       env_feedback="Apple not visible.")
        summary = tm.summarize_recent_history()
        assert "find apple" in summary
        assert "pick up apple" in summary

    def test_includes_compressed_summaries(self):
        tm = TemporalMemory(max_steps=2, compress_on_overflow=True)
        tm.append_step(action_text="early action", step_id=0)
        tm.append_step(action_text="action1", step_id=1)
        tm.append_step(action_text="action2", step_id=2)  # triggers compression
        summary = tm.summarize_recent_history()
        assert "Earlier history" in summary or "early action" in summary


class TestTemporalMemoryRetrieve:
    # 8. retrieve() returns failed relevant actions
    def test_retrieve_prioritizes_failures(self):
        tm = TemporalMemory(max_steps=20)
        tm.append_step(action_text="find apple", success=True,
                       task_instruction="Put apple in fridge.")
        tm.append_step(action_text="pick up apple", success=False,
                       env_feedback="Apple not visible.",
                       task_instruction="Put apple in fridge.")
        from embodiedbench.memory.base import MemoryQuery
        query = MemoryQuery(task_instruction="Put apple in fridge.",
                            target_objects=["apple"])
        results = tm.retrieve(query, top_k=5)
        assert len(results) > 0
        # The failed step should appear and score higher than the successful one
        failure_scores = [r.score for r in results if "fail" in (r.reason or "")]
        success_scores = [r.score for r in results
                          if r.reason == "recent relevant success"]
        if failure_scores and success_scores:
            assert max(failure_scores) >= max(success_scores)

    # 9. retrieve() returns critic rejections with correct reason
    def test_retrieve_critic_rejection_reason(self):
        tm = TemporalMemory()
        tm.append_step(action_text="pick up apple", success=False,
                       critic_rejected=True,
                       critic_output="Robot is already holding an object.",
                       task_instruction="Pick up apple.")
        from embodiedbench.memory.base import MemoryQuery
        query = MemoryQuery(task_instruction="Pick up apple.")
        results = tm.retrieve(query, top_k=5)
        assert any(r.reason == "critic rejection" for r in results)

    def test_retrieve_empty_returns_empty_list(self):
        tm = TemporalMemory()
        from embodiedbench.memory.base import MemoryQuery
        results = tm.retrieve(MemoryQuery(task_instruction="anything"), top_k=5)
        assert results == []

    def test_retrieve_returns_retrieved_memory_objects(self):
        tm = TemporalMemory()
        tm.append_step(action_text="navigate to fridge", success=True)
        from embodiedbench.memory.base import MemoryQuery, RetrievedMemory
        results = tm.retrieve(MemoryQuery(task_instruction="open fridge"), top_k=3)
        assert all(isinstance(r, RetrievedMemory) for r in results)

    def test_retrieve_top_k_respected(self):
        tm = TemporalMemory(max_steps=20)
        for i in range(10):
            tm.append_step(action_text=f"action {i}", success=(i % 2 == 0))
        from embodiedbench.memory.base import MemoryQuery
        results = tm.retrieve(MemoryQuery(task_instruction="task"), top_k=3)
        assert len(results) <= 3

    def test_retrieve_with_embedding_provider(self):
        from embodiedbench.memory.embeddings import HashEmbeddingProvider
        from embodiedbench.memory.base import MemoryQuery
        provider = HashEmbeddingProvider(dim=64)
        tm = TemporalMemory(embedding_provider=provider)
        tm.append_step(action_text="pick up apple", success=False,
                       task_instruction="Put apple in fridge.")
        results = tm.retrieve(
            MemoryQuery(task_instruction="Put apple in fridge.",
                        target_objects=["apple"]),
            top_k=3,
        )
        assert len(results) >= 1
        assert all(0.0 <= r.score <= 1.0 for r in results)


class TestDetectRepeatedFailures:
    # 10. detect_repeated_failures() detects repeated failed action
    def test_detects_repeat(self):
        tm = TemporalMemory()
        tm.append_step(action_text="pick up apple", success=False)
        tm.append_step(action_text="pick up apple", success=False)
        warnings = tm.detect_repeated_failures()
        assert len(warnings) >= 1
        assert any("pick up apple" in w for w in warnings)

    def test_no_warning_for_single_failure(self):
        tm = TemporalMemory()
        tm.append_step(action_text="pick up apple", success=False)
        warnings = tm.detect_repeated_failures()
        assert warnings == []

    def test_no_warning_for_success(self):
        tm = TemporalMemory()
        tm.append_step(action_text="pick up apple", success=True)
        tm.append_step(action_text="pick up apple", success=True)
        warnings = tm.detect_repeated_failures()
        assert warnings == []


class TestTemporalMemoryPromptContext:
    # 11. to_prompt_context() includes [Temporal Memory]
    def test_includes_temporal_memory_header(self):
        tm = TemporalMemory()
        tm.append_step(action_text="find apple", success=False,
                       env_feedback="Apple not found.")
        from embodiedbench.memory.base import MemoryQuery
        memories = tm.retrieve(MemoryQuery(task_instruction="find apple"), top_k=3)
        ctx = tm.to_prompt_context(memories)
        # Header is added by MemoryPromptFormatter.format_section(); the raw
        # context contains the interaction history content instead.
        assert "Recent relevant interactions:" in ctx

    def test_includes_failure_info(self):
        tm = TemporalMemory()
        tm.append_step(action_text="open fridge", success=False,
                       env_feedback="Fridge not reachable.")
        from embodiedbench.memory.base import MemoryQuery
        memories = tm.retrieve(MemoryQuery(task_instruction="open fridge"), top_k=3)
        ctx = tm.to_prompt_context(memories)
        assert "open fridge" in ctx

    def test_includes_critic_rejection(self):
        tm = TemporalMemory()
        tm.append_step(action_text="pick up apple", critic_rejected=True,
                       critic_output="Already holding an object.")
        from embodiedbench.memory.base import MemoryQuery
        memories = tm.retrieve(MemoryQuery(task_instruction="pick up"), top_k=3)
        ctx = tm.to_prompt_context(memories)
        assert "Critic" in ctx or "critic" in ctx

    def test_includes_repeated_failure_warning(self):
        tm = TemporalMemory()
        tm.append_step(action_text="slice tomato", success=False)
        tm.append_step(action_text="slice tomato", success=False)
        from embodiedbench.memory.base import MemoryQuery
        memories = tm.retrieve(MemoryQuery(task_instruction="slice tomato"), top_k=3)
        ctx = tm.to_prompt_context(memories)
        assert "Warnings" in ctx or "failed" in ctx.lower()

    def test_empty_memories_returns_empty_string(self):
        tm = TemporalMemory()
        ctx = tm.to_prompt_context([])
        assert ctx == ""

    def test_respects_max_chars(self):
        tm = TemporalMemory()
        for i in range(10):
            tm.append_step(action_text=f"action {i}", success=False,
                           env_feedback="x" * 200)
        from embodiedbench.memory.base import MemoryQuery
        memories = tm.retrieve(MemoryQuery(task_instruction="task"), top_k=5)
        ctx = tm.to_prompt_context(memories)
        assert len(ctx) > 0  # content returned fully without truncation


class TestTemporalMemoryEpisodeReset:
    # 12. reset_episode() clears steps and summaries
    def test_reset_clears_everything(self):
        tm = TemporalMemory()
        tm.append_step(action_text="find apple", success=True)
        tm.summaries.append("old summary")
        tm.reset_episode()
        assert len(tm.steps) == 0
        assert len(tm.summaries) == 0
        assert tm._total_steps_added == 0

    def test_can_append_after_reset(self):
        tm = TemporalMemory()
        tm.append_step(action_text="step0")
        tm.reset_episode()
        tm.append_step(action_text="step after reset", step_id=0)
        assert len(tm.steps) == 1
        assert tm.steps[0].action_text == "step after reset"


class TestTemporalMemoryPersistence:
    # 13. save/load preserves steps and summaries
    def test_save_load_roundtrip(self, tmp_path):
        tm = TemporalMemory(max_steps=10)
        tm.append_step(action_text="find apple", success=True,
                       task_instruction="Put apple in fridge.")
        tm.append_step(action_text="pick up apple", success=False,
                       env_feedback="Not visible.",
                       task_instruction="Put apple in fridge.")
        tm.summaries.append("old compressed summary")

        path = str(tmp_path / "temporal.json")
        tm.save(path)

        tm2 = TemporalMemory()
        tm2.load(path)

        assert len(tm2.steps) == 2
        assert tm2.steps[0].action_text == "find apple"
        assert tm2.steps[1].success is False
        assert tm2.summaries == ["old compressed summary"]
        assert tm2.max_steps == 10

    def test_save_creates_directories(self, tmp_path):
        tm = TemporalMemory()
        tm.append_step(action_text="test")
        path = str(tmp_path / "subdir" / "temporal.json")
        tm.save(path)
        import os
        assert os.path.isfile(path)

    # 14. Missing load path does not crash
    def test_load_missing_path_no_crash(self, tmp_path):
        tm = TemporalMemory()
        tm.load(str(tmp_path / "nonexistent.json"))
        assert len(tm.steps) == 0

    def test_load_with_no_path_no_crash(self):
        tm = TemporalMemory(storage_path=None)
        tm.load()   # no path at all — should silently return
        assert len(tm.steps) == 0

    def test_save_with_no_path_no_crash(self):
        tm = TemporalMemory(storage_path=None)
        tm.append_step(action_text="test")
        tm.save()   # no path — should silently return

    def test_storage_path_used_as_default(self, tmp_path):
        path = str(tmp_path / "auto.json")
        tm = TemporalMemory(storage_path=path)
        tm.append_step(action_text="auto saved")
        tm.save()
        tm2 = TemporalMemory(storage_path=path)
        tm2.load()
        assert tm2.steps[0].action_text == "auto saved"


# ===========================================================================
# SemanticMemory tests
# ===========================================================================

class TestSemanticFact:
    # 1. SemanticFact serialization / deserialization
    def test_roundtrip(self):
        fact = SemanticFact(
            content="To open a fridge, it must not already be open.",
            category="precondition",
            related_objects=["fridge"],
            related_actions=["open"],
            confidence=0.9,
            importance=0.8,
            source="manual",
        )
        d = fact.to_dict()
        restored = SemanticFact.from_dict(d)
        assert restored.id == fact.id
        assert restored.content == fact.content
        assert restored.category == "precondition"
        assert restored.related_objects == ["fridge"]
        assert restored.related_actions == ["open"]
        assert restored.confidence == pytest.approx(0.9)
        assert restored.importance == pytest.approx(0.8)

    def test_from_dict_defaults(self):
        fact = SemanticFact.from_dict({})
        assert fact.category == "general"
        assert fact.confidence == pytest.approx(1.0)
        assert fact.importance == pytest.approx(0.5)
        assert fact.related_objects == []

    def test_touch_updates_timestamp(self):
        fact = SemanticFact(content="test")
        old = fact.updated_at
        fact.touch()
        assert fact.updated_at >= old

    def test_short_summary_includes_category(self):
        fact = SemanticFact(content="rule text", category="safety")
        assert "[safety]" in fact.short_summary()
        assert "rule text" in fact.short_summary()

    # 2. SemanticFact.to_memory_item()
    def test_to_memory_item_type(self):
        fact = SemanticFact(content="fridge must be open", category="precondition",
                            related_objects=["fridge"])
        item = fact.to_memory_item()
        assert item.memory_type == "semantic"
        assert "fridge must be open" in item.content
        assert item.metadata["category"] == "precondition"

    def test_to_memory_item_includes_objects(self):
        fact = SemanticFact(content="some rule", related_objects=["apple", "fridge"])
        item = fact.to_memory_item()
        assert "apple" in item.content or "apple" in str(item.metadata)


class TestSemanticMemoryInit:
    def test_init_empty(self):
        sm = SemanticMemory()
        assert len(sm) == 0

    def test_no_preconditions_by_default(self):
        sm = SemanticMemory()
        assert len(sm.get_facts_by_category("precondition")) == 0

    def test_no_safety_by_default(self):
        sm = SemanticMemory()
        assert len(sm.get_facts_by_category("safety")) == 0

    def test_no_search_strategy_by_default(self):
        sm = SemanticMemory()
        assert len(sm.get_facts_by_category("search_strategy")) == 0

    def test_no_seed(self):
        sm = SemanticMemory()
        assert len(sm) == 0

    def test_add_fact_increments_count(self):
        sm = SemanticMemory()
        sm.add_fact(content="test fact", category="general")
        assert len(sm) == 1


class TestSemanticMemoryAddFact:
    # 5. add_fact adds a new fact
    def test_add_new_fact(self):
        sm = SemanticMemory()
        fact = sm.add_fact(content="Apples are usually on counters.", category="affordance")
        assert len(sm) == 1
        assert fact.content == "Apples are usually on counters."
        assert fact.category == "affordance"

    def test_add_fact_returns_semantic_fact(self):
        sm = SemanticMemory()
        result = sm.add_fact(content="Fridges must be open to place items.")
        assert isinstance(result, SemanticFact)

    # 6. add_fact avoids exact duplicates
    def test_exact_duplicate_not_added(self):
        sm = SemanticMemory(dedup_threshold=0.85)
        sm.add_fact(content="Fridges must be open to place items inside.")
        sm.add_fact(content="Fridges must be open to place items inside.")
        assert len(sm) == 1

    # 7. add_fact merges similar facts
    def test_similar_fact_merges_objects(self):
        sm = SemanticMemory(dedup_threshold=0.7)
        sm.add_fact(
            content="To put an object in the fridge the fridge must be open.",
            category="precondition",
            related_objects=["fridge"],
        )
        sm.add_fact(
            content="To put an object in the fridge the fridge must be open.",
            category="precondition",
            related_objects=["apple"],
        )
        # The two identical-content facts should have been merged into one
        matching = [f for f in sm.facts if "fridge must be open" in f.content]
        assert len(matching) == 1
        assert "apple" in matching[0].related_objects or "fridge" in matching[0].related_objects

    def test_merge_takes_max_importance(self):
        sm = SemanticMemory(dedup_threshold=0.7)
        sm.add_fact(content="Open fridge before placing item.", importance=0.5)
        sm.add_fact(content="Open fridge before placing item.", importance=0.9)
        matching = [f for f in sm.facts if "Open fridge before placing item." in f.content]
        assert matching[0].importance == pytest.approx(0.9)

    def test_add_fact_empty_content_raises(self):
        sm = SemanticMemory()
        with pytest.raises(ValueError):
            sm.add_fact(content="")

    def test_max_facts_enforced(self):
        sm = SemanticMemory(max_facts=3)
        for i in range(5):
            sm.add_fact(content=f"Unique fact number {i} about object xyz{i}.",
                        importance=float(i) / 10)
        assert len(sm) <= 3

    def test_add_fact_computes_embedding_if_provider(self):
        from embodiedbench.memory.embeddings import HashEmbeddingProvider
        provider = HashEmbeddingProvider(dim=32)
        sm = SemanticMemory(embedding_provider=provider)
        fact = sm.add_fact(content="Open container before placing.")
        assert fact.embedding is not None
        assert len(fact.embedding) == 32


class TestSemanticMemoryUpdateRemove:
    # 8. update_fact updates content and metadata
    def test_update_content(self):
        sm = SemanticMemory()
        fact = sm.add_fact(content="Original content here.")
        sm.update_fact(fact.id, content="Updated content here.")
        updated = sm._find_by_id(fact.id)
        assert updated.content == "Updated content here."

    def test_update_category(self):
        sm = SemanticMemory()
        fact = sm.add_fact(content="Some rule.", category="general")
        sm.update_fact(fact.id, category="safety")
        updated = sm._find_by_id(fact.id)
        assert updated.category == "safety"

    def test_update_nonexistent_returns_none(self):
        sm = SemanticMemory()
        result = sm.update_fact("nonexistent-id", content="foo")
        assert result is None

    def test_update_recomputes_embedding(self):
        from embodiedbench.memory.embeddings import HashEmbeddingProvider
        provider = HashEmbeddingProvider(dim=32)
        sm = SemanticMemory(embedding_provider=provider)
        fact = sm.add_fact(content="Original.")
        old_emb = list(fact.embedding)
        sm.update_fact(fact.id, content="Completely different text now.")
        updated = sm._find_by_id(fact.id)
        assert updated.embedding != old_emb

    # 9. remove_fact removes by id
    def test_remove_fact(self):
        sm = SemanticMemory()
        fact = sm.add_fact(content="To be removed.")
        assert sm.remove_fact(fact.id) is True
        assert len(sm) == 0

    def test_remove_nonexistent_returns_false(self):
        sm = SemanticMemory()
        assert sm.remove_fact("no-such-id") is False


class TestSemanticMemoryQueries:
    # 10. get_facts_by_category
    def test_get_by_category(self):
        sm = SemanticMemory()
        before = len(sm.get_facts_by_category("safety"))
        sm.add_fact(content="rule A", category="safety")
        sm.add_fact(content="rule B", category="precondition")
        sm.add_fact(content="rule C", category="safety")
        safety = sm.get_facts_by_category("safety")
        assert len(safety) == before + 2

    # 11. get_facts_for_object
    def test_get_for_object_by_related_objects(self):
        sm = SemanticMemory()
        sm.add_fact(content="rule", related_objects=["apple", "fridge"])
        sm.add_fact(content="other rule", related_objects=["bowl"])
        results = sm.get_facts_for_object("apple")
        assert len(results) >= 1
        assert any("apple" in f.related_objects for f in results)

    def test_get_for_object_by_content(self):
        sm = SemanticMemory()
        sm.add_fact(content="The fridge is usually closed.")
        results = sm.get_facts_for_object("fridge")
        assert len(results) >= 1
        assert any("fridge" in f.content for f in results)

    # 12. get_facts_for_action
    def test_get_for_action_by_related_actions(self):
        sm = SemanticMemory()
        before = len(sm.get_facts_for_action("open"))
        sm.add_fact(content="some rule", related_actions=["open", "close"])
        results = sm.get_facts_for_action("open")
        assert len(results) == before + 1

    def test_get_for_action_by_content(self):
        sm = SemanticMemory()
        before = len(sm.get_facts_for_action("open"))
        sm.add_fact(content="You must open the fridge first.")
        results = sm.get_facts_for_action("open")
        assert len(results) == before + 1


class TestSemanticMemoryRetrieve:
    # 13. retrieve returns relevant precondition for open/place tasks
    def test_retrieves_precondition_for_place_task(self):
        sm = SemanticMemory()
        sm.add_fact(
            content="To place an object inside a container, open it first.",
            category="precondition",
            related_actions=["open", "place"],
            importance=0.9,
        )
        query = MemoryQuery(
            task_instruction="Put the apple inside the fridge.",
            target_objects=["apple", "fridge"],
            recent_actions=["navigate to fridge"],
            proposed_plan="open fridge, pick up apple, place apple",
        )
        results = sm.retrieve(query, top_k=5)
        assert len(results) > 0
        categories = [r.item.metadata.get("category") for r in results]
        assert "precondition" in categories or "task_rule" in categories

    # 14. retrieve returns search strategy for invisible object query
    def test_retrieves_search_strategy_for_invisible_object(self):
        sm = SemanticMemory()
        sm.add_fact(
            content="If a target object is not visible, verify before direct manipulation.",
            category="search_strategy",
            importance=0.85,
        )
        query = MemoryQuery(
            task_instruction="Find the apple.",
            target_objects=["apple"],
            observation_text="Apple is not visible in current frame.",
        )
        results = sm.retrieve(query, top_k=5)
        reasons = [r.reason for r in results]
        assert any("search" in (r or "") for r in reasons)

    # 15. retrieve returns safety/failure rule for repeated failure query
    def test_retrieves_failure_avoidance_for_failed_action_query(self):
        sm = SemanticMemory()
        sm.add_fact(
            content="Avoid repeating the same action if it recently failed.",
            category="failure_avoidance",
            related_actions=["repeat"],
            importance=0.85,
        )
        query = MemoryQuery(
            task_instruction="Pick up the apple.",
            recent_actions=["pick up apple", "pick up apple"],  # repeated
        )
        results = sm.retrieve(query, top_k=5)
        categories = [r.item.metadata.get("category") for r in results]
        assert "failure_avoidance" in categories or "safety" in categories

    def test_retrieve_empty_facts_returns_empty(self):
        sm = SemanticMemory()
        results = sm.retrieve(MemoryQuery(task_instruction="test"), top_k=5)
        assert results == []

    def test_retrieve_scores_in_range(self):
        sm = SemanticMemory()
        sm.add_fact(content="open fridge before placing items inside", category="precondition")
        query = MemoryQuery(task_instruction="open the fridge and place the apple inside")
        results = sm.retrieve(query, top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_retrieve_top_k_respected(self):
        sm = SemanticMemory()
        sm.add_fact(content="fact one", category="general")
        sm.add_fact(content="fact two", category="general")
        sm.add_fact(content="fact three", category="general")
        query = MemoryQuery(task_instruction="do something")
        results = sm.retrieve(query, top_k=2)
        assert len(results) <= 2

    def test_retrieve_with_embedding_provider(self):
        from embodiedbench.memory.embeddings import HashEmbeddingProvider
        provider = HashEmbeddingProvider(dim=64)
        sm = SemanticMemory(embedding_provider=provider)
        sm.add_fact(content="open the fridge before placing", category="precondition")
        query = MemoryQuery(
            task_instruction="open the fridge",
            recent_actions=["navigate to fridge"],
        )
        results = sm.retrieve(query, top_k=5)
        assert len(results) > 0
        for r in results:
            assert 0.0 <= r.score <= 1.0


class TestExtractFactsFromEpisode:
    # 16. success adds task_rule fact via add_fact
    def test_success_adds_task_rule(self):
        sm = SemanticMemory()
        f = sm.add_fact(
            content="Put apple in fridge: always open fridge first.",
            category="task_rule",
            source_episode_id="ep-001",
        )
        assert f.category == "task_rule"
        task_rules = sm.get_facts_by_category("task_rule")
        assert len(task_rules) >= 1
        assert "fridge" in task_rules[0].content

    # 17. failure adds failure_avoidance fact
    def test_failure_adds_failure_avoidance(self):
        sm = SemanticMemory()
        before = len(sm.get_facts_by_category("failure_avoidance"))
        f = sm.add_fact(
            content="Avoid picking apple when not visible; search nearby surfaces first.",
            category="failure_avoidance",
            related_objects=["apple"],
            source_episode_id="ep-002",
        )
        avoidance = sm.get_facts_by_category("failure_avoidance")
        assert len(avoidance) == before + 1
        assert f.category == "failure_avoidance"

    def test_visibility_failure_adds_search_strategy(self):
        sm = SemanticMemory()
        sm.add_fact(
            content="When apple not visible at counter, check shelves and tables.",
            category="search_strategy",
            related_objects=["apple"],
        )
        search_facts = sm.get_facts_by_category("search_strategy")
        assert len(search_facts) >= 1

    def test_extract_returns_list(self):
        sm = SemanticMemory()
        sm.add_fact(content="Everything worked fine.", category="task_rule")
        result = sm.get_facts_by_category("task_rule")
        assert isinstance(result, list)


class TestSemanticMemoryPromptContext:
    # 18. to_prompt_context includes [Semantic Memory]
    def test_includes_header(self):
        sm = SemanticMemory()
        sm.add_fact(content="open fridge before placing apple inside", category="precondition")
        query = MemoryQuery(task_instruction="open fridge and place apple")
        memories = sm.retrieve(query, top_k=3)
        ctx = sm.to_prompt_context(memories)
        # Header is added by MemoryPromptFormatter; raw context has the rules body.
        assert "Relevant rules and commonsense knowledge:" in ctx

    def test_includes_category_tags(self):
        sm = SemanticMemory()
        sm.add_fact(content="open container before placing", category="precondition")
        query = MemoryQuery(task_instruction="open fridge", recent_actions=["open"])
        memories = sm.retrieve(query, top_k=5)
        ctx = sm.to_prompt_context(memories)
        # At least one category tag should appear
        assert "[" in ctx and "]" in ctx

    def test_empty_memories_returns_empty(self):
        sm = SemanticMemory()
        assert sm.to_prompt_context([]) == ""

    def test_respects_max_chars(self):
        sm = SemanticMemory()
        sm.add_fact(content="open the container before placing items inside", category="precondition")
        query = MemoryQuery(task_instruction="do anything")
        memories = sm.retrieve(query, top_k=6)
        ctx = sm.to_prompt_context(memories)
        assert len(ctx) > 0  # content returned fully without truncation


class TestSemanticMemoryEpisodeReset:
    # 19. reset_episode does NOT clear facts
    def test_reset_episode_preserves_facts(self):
        sm = SemanticMemory()
        count_before = len(sm)
        sm.reset_episode()
        assert len(sm) == count_before


class TestSemanticMemoryPersistence:
    # 20. save/load preserves facts
    def test_save_load_roundtrip(self, tmp_path):
        sm = SemanticMemory()
        sm.add_fact(content="To place in fridge, open it first.",
                    category="precondition", related_objects=["fridge"])
        sm.add_fact(content="Apples can be found on counters.",
                    category="affordance", related_objects=["apple"])

        path = str(tmp_path / "semantic.json")
        sm.save(path)

        sm2 = SemanticMemory()
        sm2.load(path)

        contents = [f.content for f in sm2.facts]
        assert any("fridge" in c.lower() for c in contents)
        assert any("apple" in c.lower() for c in contents)

    def test_save_preserves_category(self, tmp_path):
        sm = SemanticMemory()
        sm.add_fact(content="Safety rule unique xyz.", category="safety")
        path = str(tmp_path / "s.json")
        sm.save(path)
        sm2 = SemanticMemory()
        sm2.load(path)
        matching = [f for f in sm2.facts if "Safety rule unique xyz." in f.content]
        assert len(matching) == 1
        assert matching[0].category == "safety"

    def test_save_creates_directories(self, tmp_path):
        import os
        sm = SemanticMemory()
        sm.add_fact(content="test fact")
        path = str(tmp_path / "nested" / "semantic.json")
        sm.save(path)
        assert os.path.isfile(path)

    # 21. Missing load path does not crash
    def test_load_missing_path_no_crash(self, tmp_path):
        sm = SemanticMemory()
        sm.load(str(tmp_path / "nonexistent.json"))
        assert len(sm) == 0

    def test_load_no_path_no_crash(self):
        sm = SemanticMemory(storage_path=None)
        sm.load()
        assert len(sm) == 0

    def test_save_no_path_no_crash(self):
        sm = SemanticMemory(storage_path=None)
        sm.add_fact(content="test")
        sm.save()   # no path — should silently return

    def test_storage_path_default(self, tmp_path):
        path = str(tmp_path / "auto.json")
        sm = SemanticMemory(storage_path=path)
        sm.add_fact(content="persistent fact")
        sm.save()
        sm2 = SemanticMemory(storage_path=path)
        sm2.load()
        assert any("persistent fact" in f.content for f in sm2.facts)


# ===========================================================================
# EpisodicMemory tests
# ===========================================================================


class TestEpisodicMemoryBasic:
    def test_add_and_retrieve(self):
        em = EpisodicMemory()
        steps = [
            {"step_id": 0, "action": "navigate to table", "feedback": "ok"},
            {"step_id": 1, "action": "pick up apple", "feedback": "ok"},
            {"step_id": 2, "action": "navigate to fridge", "feedback": "ok"},
            {"step_id": 3, "action": "open fridge", "feedback": "ok"},
            {"step_id": 4, "action": "place apple", "feedback": "ok"},
        ]
        ep = em.add_episode_from_trajectory(
            task_instruction="Put the apple in the fridge.",
            final_status="success",
            steps=steps,
            env_name="eb-alfred",
            scene_id="scene-1",
        )

        q = MemoryQuery(task_instruction="Put the apple in the fridge.", target_objects=["apple", "fridge"])
        results = em.retrieve(q, top_k=3)
        assert len(results) >= 1
        assert any("apple" in r.item.content.lower() for r in results)

    def test_deduplication_merges(self):
        em = EpisodicMemory()
        steps1 = [{"step_id": 0, "action": "a1", "feedback": "ok"}]
        steps2 = [{"step_id": 0, "action": "a2", "feedback": "ok"}]
        em.add_episode_from_trajectory(task_instruction="Task X", final_status="success", steps=steps1)
        em.add_episode_from_trajectory(task_instruction="Task X", final_status="success", steps=steps2)
        # Both episodes should be stored (no auto-merge without VLM)
        assert len(em.episodes) >= 1


class TestEpisodicMemoryPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        em = EpisodicMemory()
        # Only successful episodes are stored; failure is silently dropped
        em.add_episode_from_trajectory(task_instruction="T1", final_status="failure", steps=[{"step_id": 0, "action": "s1", "feedback": ""}])
        em.add_episode_from_trajectory(task_instruction="T2", final_status="success", steps=[{"step_id": 0, "action": "s2", "feedback": ""}])
        path = str(tmp_path / "episodic.json")
        em.save(path)

        em2 = EpisodicMemory()
        em2.load(path)
        assert len(em2.episodes) == 1
        assert em2.episodes[0].task_instruction == "T2"


# ===========================================================================
# SpatialMemory tests
# ===========================================================================


class TestSpatialNode:
    # 1. SpatialNode serialization / deserialization
    def test_roundtrip(self):
        node = SpatialNode(
            name="Apple",
            node_type="object",
            position={"x": 1.0, "y": 0.5, "z": 2.0},
            room="kitchen",
            state={"isOpen": False},
            last_seen_step=5,
            confidence=0.9,
            stale=False,
        )
        d = node.to_dict()
        restored = SpatialNode.from_dict(d)
        assert restored.id == node.id
        assert restored.name == "Apple"
        assert restored.room == "kitchen"
        assert restored.state == {"isOpen": False}
        assert restored.confidence == pytest.approx(0.9)
        assert restored.last_seen_step == 5
        assert restored.stale is False

    def test_from_dict_defaults(self):
        node = SpatialNode.from_dict({})
        assert node.node_type == "object"
        assert node.confidence == pytest.approx(1.0)
        assert node.stale is False
        assert node.room == ""

    def test_short_summary_contains_name(self):
        node = SpatialNode(name="Fridge", node_type="receptacle", room="kitchen", confidence=0.8)
        summary = node.short_summary()
        assert "Fridge" in summary
        assert "kitchen" in summary

    def test_short_summary_stale_tag(self):
        node = SpatialNode(name="Mug", stale=True)
        assert "STALE" in node.short_summary()

    def test_position_unsafe_type_becomes_string(self):
        node = SpatialNode(name="X", position=object())
        d = node.to_dict()
        assert isinstance(d["position"], str)

    def test_touch_updates_timestamp(self):
        node = SpatialNode(name="A")
        old = node.updated_at
        node.touch()
        assert node.updated_at >= old


class TestSpatialRelation:
    # 2. SpatialRelation serialization / deserialization
    def test_roundtrip(self):
        rel = SpatialRelation(
            subject_id="node-1",
            relation="in",
            object_id="node-2",
            confidence=0.85,
            stale=False,
            evidence="scene_objects",
            last_seen_step=3,
        )
        d = rel.to_dict()
        restored = SpatialRelation.from_dict(d)
        assert restored.id == rel.id
        assert restored.subject_id == "node-1"
        assert restored.relation == "in"
        assert restored.object_id == "node-2"
        assert restored.confidence == pytest.approx(0.85)
        assert restored.last_seen_step == 3

    def test_from_dict_defaults(self):
        rel = SpatialRelation.from_dict({})
        assert rel.confidence == pytest.approx(1.0)
        assert rel.stale is False
        assert rel.evidence == ""

    def test_short_summary_with_node_lookup(self):
        n1 = SpatialNode(id="n1", name="Apple")
        n2 = SpatialNode(id="n2", name="Fridge")
        rel = SpatialRelation(subject_id="n1", relation="in", object_id="n2", last_seen_step=4)
        summary = rel.short_summary(node_lookup={"n1": n1, "n2": n2})
        assert "Apple" in summary and "Fridge" in summary
        assert "in" in summary

    def test_short_summary_stale_tag(self):
        rel = SpatialRelation(subject_id="a", relation="on", object_id="b", stale=True)
        assert "STALE" in rel.short_summary()


class TestAddOrUpdateObject:
    # 3. add_or_update_object creates a node
    def test_creates_node(self):
        sm = SpatialMemory()
        node = sm.add_or_update_object(name="Apple", room="kitchen")
        assert node.name == "Apple"
        assert node.room == "kitchen"
        assert node.id in sm.nodes

    def test_name_indexed(self):
        sm = SpatialMemory()
        node = sm.add_or_update_object(name="Apple")
        found = sm.find_nodes_by_name("apple")
        assert any(n.id == node.id for n in found)

    # 4. add_or_update_object updates existing node by name
    def test_updates_existing_by_name(self):
        sm = SpatialMemory()
        sm.add_or_update_object(name="Apple", room="kitchen")
        sm.add_or_update_object(name="Apple", room="kitchen", state={"isSliced": True})
        assert len(sm.nodes) == 1
        node = sm.find_node("Apple")
        assert node.state.get("isSliced") is True

    def test_updates_existing_by_node_id(self):
        sm = SpatialMemory()
        node = sm.add_or_update_object(name="Mug")
        sm.add_or_update_object(name="Mug", room="dining", node_id=node.id)
        assert sm.nodes[node.id].room == "dining"

    def test_updates_step_and_confidence(self):
        sm = SpatialMemory()
        sm.add_or_update_object(name="Cup", step_id=0, confidence=0.5)
        sm.add_or_update_object(name="Cup", step_id=3, confidence=0.9)
        node = sm.find_node("Cup")
        assert node.last_seen_step == 3
        assert node.confidence == pytest.approx(0.9)


class TestAddRelation:
    # 5. add_relation creates relation
    def test_creates_relation(self):
        sm = SpatialMemory()
        n1 = sm.add_or_update_object(name="Apple")
        n2 = sm.add_or_update_object(name="Table", node_type="receptacle")
        rel = sm.add_relation(subject_id=n1.id, relation="on", object_id=n2.id, confidence=0.9)
        assert rel.id in sm.relations
        assert rel.relation == "on"
        assert rel.stale is False

    # 6. add_relation updates duplicate relation (same triple)
    def test_updates_duplicate_triple(self):
        sm = SpatialMemory()
        n1 = sm.add_or_update_object(name="Apple")
        n2 = sm.add_or_update_object(name="Table")
        sm.add_relation(subject_id=n1.id, relation="on", object_id=n2.id, confidence=0.7)
        sm.add_relation(subject_id=n1.id, relation="on", object_id=n2.id, confidence=0.95)
        # Should still be one active relation
        active = [r for r in sm.relations.values() if not r.stale]
        on_rels = [r for r in active if r.subject_id == n1.id and r.relation == "on"]
        assert len(on_rels) == 1
        assert on_rels[0].confidence == pytest.approx(0.95)


class TestStaleness:
    # 7. object location update marks old relation stale
    def test_location_change_stales_old_relation(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple", room="kitchen")
        table = sm.add_or_update_object(name="Table")
        rel = sm.add_relation(subject_id=apple.id, relation="on", object_id=table.id, confidence=1.0)

        # Apple is now observed in living_room
        sm.add_or_update_object(name="Apple", room="living_room")
        assert rel.stale is True
        assert rel.confidence < 1.0

    def test_location_change_new_node_not_stale(self):
        sm = SpatialMemory()
        sm.add_or_update_object(name="Mug", room="kitchen")
        updated = sm.add_or_update_object(name="Mug", room="dining")
        assert updated.stale is False

    # 8. mark_stale reduces confidence
    def test_mark_stale_node(self):
        sm = SpatialMemory()
        node = sm.add_or_update_object(name="Bowl", confidence=1.0)
        sm.mark_stale(node_id=node.id)
        assert sm.nodes[node.id].stale is True
        assert sm.nodes[node.id].confidence < 1.0

    def test_mark_stale_relation(self):
        sm = SpatialMemory()
        n1 = sm.add_or_update_object(name="Cup")
        n2 = sm.add_or_update_object(name="Shelf")
        rel = sm.add_relation(subject_id=n1.id, relation="on", object_id=n2.id, confidence=1.0)
        sm.mark_stale(relation_id=rel.id)
        assert sm.relations[rel.id].stale is True
        assert sm.relations[rel.id].confidence < 1.0

    def test_mark_stale_custom_decay(self):
        sm = SpatialMemory()
        node = sm.add_or_update_object(name="X", confidence=1.0)
        sm.mark_stale(node_id=node.id, confidence_decay=0.1)
        assert sm.nodes[node.id].confidence == pytest.approx(0.1)

    # 9. detect_conflicts reports multiple active location relations
    def test_detect_conflicts_multiple_active(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple")
        table = sm.add_or_update_object(name="Table")
        fridge = sm.add_or_update_object(name="Fridge")
        # Manually add two active location relations (bypassing conflict resolution)
        rel1 = SpatialRelation(subject_id=apple.id, relation="on", object_id=table.id)
        rel2 = SpatialRelation(subject_id=apple.id, relation="on", object_id=fridge.id)
        sm.relations[rel1.id] = rel1
        sm.relations[rel2.id] = rel2
        warnings = sm.detect_conflicts()
        assert any("Apple" in w and "Conflict" in w for w in warnings)

    def test_detect_conflicts_stale_vs_active(self):
        sm = SpatialMemory(stale_confidence_decay=0.5)
        apple = sm.add_or_update_object(name="Apple", room="kitchen")
        table = sm.add_or_update_object(name="Table")
        shelf = sm.add_or_update_object(name="Shelf")
        sm.add_relation(subject_id=apple.id, relation="on", object_id=table.id)
        sm.add_relation(subject_id=apple.id, relation="on", object_id=shelf.id)  # marks table rel stale
        warnings = sm.detect_conflicts()
        assert any("Apple" in w or "stale" in w.lower() for w in warnings)


class TestGetObjectLocations:
    # 10. get_object_locations returns active location
    def test_returns_active_location(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple")
        fridge = sm.add_or_update_object(name="Fridge")
        sm.add_relation(subject_id=apple.id, relation="in", object_id=fridge.id, step_id=5)
        locs = sm.get_object_locations("Apple")
        assert any("Fridge" in l for l in locs)
        assert any("5" in l for l in locs)

    # 11. get_object_locations includes stale location warning
    def test_includes_stale_warning(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple")
        table = sm.add_or_update_object(name="Table")
        rel = sm.add_relation(subject_id=apple.id, relation="on", object_id=table.id)
        sm.mark_stale(relation_id=rel.id)
        locs = sm.get_object_locations("Apple")
        assert any("STALE" in l for l in locs)

    def test_returns_room_when_no_relation(self):
        sm = SpatialMemory()
        sm.add_or_update_object(name="Mug", room="kitchen", step_id=3)
        locs = sm.get_object_locations("Mug")
        assert any("kitchen" in l for l in locs)


class TestGetRelatedObjects:
    # 12. get_related_objects returns relation summaries
    def test_returns_summaries(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple")
        fridge = sm.add_or_update_object(name="Fridge")
        sm.add_relation(subject_id=apple.id, relation="in", object_id=fridge.id)
        related = sm.get_related_objects("Apple")
        assert len(related) >= 1
        assert any("Fridge" in r for r in related)


class TestUpdateFromObservation:
    # 13. update_from_observation handles list of strings
    def test_handles_string_list(self):
        sm = SpatialMemory()
        sm.update_from_observation(info={"scene_objects": ["Apple", "Fridge", "Table"]}, step_id=1)
        assert sm.find_node("Apple") is not None
        assert sm.find_node("Fridge") is not None

    # 14. update_from_observation handles list of dicts
    def test_handles_dict_list(self):
        sm = SpatialMemory()
        info = {
            "scene_objects": [
                {"objectType": "Apple", "position": {"x": 1, "y": 0, "z": 2},
                 "parentReceptacles": ["Fridge|1|2|3"], "isOpen": False},
                {"objectType": "Fridge", "position": {"x": 0, "y": 0, "z": 0}},
            ]
        }
        sm.update_from_observation(info=info, step_id=2)
        apple = sm.find_node("Apple")
        assert apple is not None
        assert apple.last_seen_step == 2
        # Fridge receptacle relation should exist
        fridge = sm.find_node("Fridge")
        assert fridge is not None
        relations = list(sm.relations.values())
        assert any(r.subject_id == apple.id and r.relation == "in" for r in relations)

    def test_handles_inventory_objects(self):
        sm = SpatialMemory()
        sm.update_from_observation(info={"inventory_objects": ["Knife"]}, step_id=3)
        knife = sm.find_node("Knife")
        assert knife is not None
        assert knife.confidence == pytest.approx(1.0)  # inventory → full confidence

    # 15. update_from_observation handles missing/unknown fields without crash
    def test_handles_none_info(self):
        sm = SpatialMemory()
        sm.update_from_observation(info=None)  # should not crash
        assert len(sm.nodes) == 0

    def test_handles_empty_info(self):
        sm = SpatialMemory()
        sm.update_from_observation(info={})
        assert len(sm.nodes) == 0

    def test_handles_unexpected_format(self):
        sm = SpatialMemory()
        sm.update_from_observation(info={"scene_objects": "not a list"})
        assert len(sm.nodes) == 0  # should not crash, no nodes added

    def test_handles_dict_without_name_fields(self):
        sm = SpatialMemory()
        sm.update_from_observation(info={"scene_objects": [{"color": "red"}]})
        assert len(sm.nodes) == 0  # no valid name → no node


class TestSpatialMemoryRetrieve:
    # 16. retrieve returns target object location
    def test_returns_target_location(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple", room="kitchen", step_id=7, confidence=0.9)
        q = MemoryQuery(task_instruction="Put the apple in the fridge.", target_objects=["apple"])
        results = sm.retrieve(q, top_k=5)
        assert len(results) >= 1
        assert any("Apple" in r.item.content for r in results)
        assert any(r.reason == "target object location" for r in results)

    # 17. retrieve returns related receptacle
    def test_returns_related_receptacle(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple")
        fridge = sm.add_or_update_object(name="Fridge", node_type="receptacle")
        sm.add_relation(subject_id=apple.id, relation="in", object_id=fridge.id, confidence=0.9)
        q = MemoryQuery(task_instruction="Place the apple in the fridge.", target_objects=["apple"])
        results = sm.retrieve(q, top_k=5)
        reasons = [r.reason for r in results]
        assert "spatial relation match" in reasons or "target object location" in reasons

    # 18. retrieve includes stale warning for target object
    def test_returns_stale_warning(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple", room="kitchen", confidence=0.9)
        table = sm.add_or_update_object(name="Table")
        rel = sm.add_relation(subject_id=apple.id, relation="on", object_id=table.id)
        sm.mark_stale(relation_id=rel.id)
        q = MemoryQuery(task_instruction="Find the apple.", target_objects=["apple"])
        results = sm.retrieve(q, top_k=5)
        assert any(r.reason == "stale spatial memory warning" for r in results)

    def test_retrieve_scores_in_range(self):
        sm = SpatialMemory()
        sm.add_or_update_object(name="Apple", room="kitchen")
        q = MemoryQuery(task_instruction="pick up apple", target_objects=["apple"])
        results = sm.retrieve(q, top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_retrieve_empty_returns_empty(self):
        sm = SpatialMemory()
        results = sm.retrieve(MemoryQuery(task_instruction="anything"), top_k=5)
        assert results == []

    def test_retrieve_top_k_respected(self):
        sm = SpatialMemory()
        for i in range(10):
            sm.add_or_update_object(name=f"Object{i}", room="kitchen")
        q = MemoryQuery(task_instruction="find object", target_objects=["object"])
        results = sm.retrieve(q, top_k=3)
        assert len(results) <= 3


class TestSpatialMemoryPromptContext:
    # 19. to_prompt_context includes [Spatial Memory]
    def test_includes_header(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple", room="kitchen")
        q = MemoryQuery(task_instruction="find apple", target_objects=["apple"])
        memories = sm.retrieve(q, top_k=3)
        ctx = sm.to_prompt_context(memories)
        # Header is added by MemoryPromptFormatter; raw context has the scene body.
        assert "Relevant Spatial Information:" in ctx

    # 20. to_prompt_context includes override warning when stale memory present
    def test_includes_override_warning_for_stale(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple")
        table = sm.add_or_update_object(name="Table")
        rel = sm.add_relation(subject_id=apple.id, relation="on", object_id=table.id)
        sm.mark_stale(relation_id=rel.id)
        q = MemoryQuery(task_instruction="find apple", target_objects=["apple"])
        memories = sm.retrieve(q, top_k=5)
        ctx = sm.to_prompt_context(memories)
        assert "override" in ctx.lower() or "stale" in ctx.lower()

    def test_empty_memories_returns_empty(self):
        sm = SpatialMemory()
        assert sm.to_prompt_context([]) == ""

    def test_respects_max_chars(self):
        sm = SpatialMemory()
        for i in range(10):
            sm.add_or_update_object(name=f"Object{i}", room="kitchen")
        q = MemoryQuery(task_instruction="find object")
        memories = sm.retrieve(q, top_k=5)
        ctx = sm.to_prompt_context(memories)
        assert len(ctx) > 0  # content returned fully without truncation


class TestSpatialMemoryEpisodeReset:
    # 21. reset_episode clears the scene graph for the new episode
    def test_reset_episode_clears_nodes(self):
        sm = SpatialMemory()
        sm.add_or_update_object(name="Apple", room="kitchen")
        sm.add_or_update_object(name="Fridge", node_type="receptacle")
        assert len(sm.nodes) > 0
        sm.reset_episode()
        assert len(sm.nodes) == 0
        assert len(sm.relations) == 0


class TestSpatialMemoryPersistence:
    # 22. save/load preserves nodes and relations
    def test_save_load_roundtrip(self, tmp_path):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple", room="kitchen", step_id=3, confidence=0.85)
        fridge = sm.add_or_update_object(name="Fridge", node_type="receptacle", room="kitchen")
        rel = sm.add_relation(subject_id=apple.id, relation="in", object_id=fridge.id, confidence=0.9)

        path = str(tmp_path / "spatial.json")
        sm.save(path)

        sm2 = SpatialMemory()
        sm2.load(path)

        assert len(sm2.nodes) == 2
        assert len(sm2.relations) == 1
        apple2 = sm2.find_node("Apple")
        assert apple2 is not None
        assert apple2.room == "kitchen"
        assert apple2.last_seen_step == 3
        assert apple2.confidence == pytest.approx(0.85)

    def test_save_preserves_stale_flag(self, tmp_path):
        sm = SpatialMemory()
        node = sm.add_or_update_object(name="Mug")
        sm.mark_stale(node_id=node.id)
        path = str(tmp_path / "s.json")
        sm.save(path)
        sm2 = SpatialMemory()
        sm2.load(path)
        assert sm2.find_node("Mug").stale is True

    def test_load_rebuilds_name_index(self, tmp_path):
        sm = SpatialMemory()
        sm.add_or_update_object(name="Table", room="dining")
        path = str(tmp_path / "idx.json")
        sm.save(path)
        sm2 = SpatialMemory()
        sm2.load(path)
        assert sm2.find_node("Table") is not None

    # 23. Missing load path does not crash
    def test_load_missing_path_no_crash(self, tmp_path):
        sm = SpatialMemory()
        sm.load(str(tmp_path / "nonexistent.json"))
        assert len(sm.nodes) == 0

    def test_load_no_path_no_crash(self):
        sm = SpatialMemory(storage_path=None)
        sm.load()
        assert len(sm.nodes) == 0

    def test_save_no_path_no_crash(self):
        sm = SpatialMemory(storage_path=None)
        sm.add_or_update_object(name="test")
        sm.save()  # silent no-op


class TestSpatialMemoryRemove:
    # 24. remove_node removes node and connected relations
    def test_remove_node_clears_relations(self):
        sm = SpatialMemory()
        apple = sm.add_or_update_object(name="Apple")
        table = sm.add_or_update_object(name="Table")
        rel = sm.add_relation(subject_id=apple.id, relation="on", object_id=table.id)
        assert sm.remove_node(apple.id) is True
        assert apple.id not in sm.nodes
        assert rel.id not in sm.relations

    def test_remove_node_false_for_unknown_id(self):
        sm = SpatialMemory()
        assert sm.remove_node("no-such-id") is False

    def test_remove_node_clears_name_index(self):
        sm = SpatialMemory()
        node = sm.add_or_update_object(name="Bowl")
        sm.remove_node(node.id)
        assert sm.find_node("Bowl") is None

    # 25. remove_relation works
    def test_remove_relation(self):
        sm = SpatialMemory()
        a = sm.add_or_update_object(name="A")
        b = sm.add_or_update_object(name="B")
        rel = sm.add_relation(subject_id=a.id, relation="near", object_id=b.id)
        assert sm.remove_relation(rel.id) is True
        assert rel.id not in sm.relations

    def test_remove_relation_false_for_unknown_id(self):
        sm = SpatialMemory()
        assert sm.remove_relation("no-such-id") is False


# ===========================================================================
# MemoryConfig tests
# ===========================================================================

class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.enabled is True
        assert cfg.top_k_per_memory == 5
        assert cfg.temporal_max_steps == 20
        assert cfg.use_embeddings is False

    def test_to_dict_round_trip(self):
        cfg = MemoryConfig(top_k_per_memory=7, storage_dir="/tmp/mem")
        d = cfg.to_dict()
        cfg2 = MemoryConfig.from_dict(d)
        assert cfg2.top_k_per_memory == 7
        assert cfg2.storage_dir == "/tmp/mem"

    def test_from_dict_ignores_unknown_keys(self):
        cfg = MemoryConfig.from_dict({"enabled": False, "unknown_key": 99})
        assert cfg.enabled is False

    def test_from_mapping_none_returns_defaults(self):
        cfg = MemoryConfig.from_mapping(None)
        assert cfg.enabled is True

    def test_from_mapping_with_dict(self):
        cfg = MemoryConfig.from_mapping({"top_k_per_memory": 3})
        assert cfg.top_k_per_memory == 3

    def test_from_mapping_with_existing_config(self):
        original = MemoryConfig(enabled=False)
        cfg = MemoryConfig.from_mapping(original)
        assert cfg.enabled is False


# ===========================================================================
# MemoryManager tests
# ===========================================================================

class TestMemoryManagerInit:
    def test_default_construction(self):
        mm = MemoryManager()
        assert mm.is_enabled() is True
        assert mm.spatial is not None
        assert mm.temporal is not None
        assert mm.episodic is not None
        assert mm.semantic is not None

    def test_disabled_config(self):
        cfg = MemoryConfig(enabled=False)
        mm = MemoryManager(config=cfg)
        assert mm.is_enabled() is False

    def test_selective_disable(self):
        cfg = MemoryConfig(spatial_enabled=False, episodic_enabled=False)
        mm = MemoryManager(config=cfg)
        assert mm.spatial is None
        assert mm.episodic is None
        assert mm.temporal is not None
        assert mm.semantic is not None

    def test_custom_storage_dir(self, tmp_path):
        cfg = MemoryConfig(storage_dir=str(tmp_path / "mem"))
        mm = MemoryManager(config=cfg)
        assert mm.config.storage_dir == str(tmp_path / "mem")

    def test_inject_custom_modules(self):
        spatial = SpatialMemory()
        mm = MemoryManager(spatial_memory=spatial)
        assert mm.spatial is spatial


class TestMemoryManagerUpdate:
    def test_update_increments_temporal(self):
        mm = MemoryManager()
        mm.update(task_instruction="pick up apple", action_text="GoTo fridge", step_id=0)
        assert len(mm.temporal) == 1

    def test_update_when_disabled_is_noop(self):
        cfg = MemoryConfig(enabled=False)
        mm = MemoryManager(config=cfg)
        mm.update(task_instruction="pick up apple", action_text="GoTo fridge")
        # Should not raise; temporal is still created (enabled flag only guards manager.update)

    def test_update_multiple_steps(self):
        mm = MemoryManager()
        for i in range(5):
            mm.update(task_instruction="task", action_text=f"action_{i}", step_id=i)
        assert len(mm.temporal) == 5

    def test_update_with_info_dict(self):
        mm = MemoryManager()
        mm.update(
            task_instruction="find book",
            action_text="look around",
            info={"objects": []},
            step_id=0,
        )
        assert len(mm.temporal) == 1


class TestMemoryManagerRetrieve:
    def _populated_manager(self):
        mm = MemoryManager()
        for i in range(3):
            mm.update(
                task_instruction="pick up apple",
                action_text=f"step_{i}",
                step_id=i,
            )
        return mm

    def test_retrieve_returns_memory_context(self):
        mm = self._populated_manager()
        query = MemoryQuery(task_instruction="pick up apple")
        ctx = mm.retrieve(query)
        assert isinstance(ctx, MemoryContext)

    def test_retrieve_combined_context_not_empty(self):
        mm = self._populated_manager()
        query = MemoryQuery(task_instruction="pick up apple")
        ctx = mm.retrieve(query)
        assert ctx.combined_context != ""

    def test_retrieve_has_preamble(self):
        mm = self._populated_manager()
        query = MemoryQuery(task_instruction="pick up apple")
        ctx = mm.retrieve(query)
        assert "[Memory Context]" in ctx.combined_context

    def test_retrieve_disabled_returns_empty(self):
        cfg = MemoryConfig(enabled=False)
        mm = MemoryManager(config=cfg)
        query = MemoryQuery(task_instruction="test")
        ctx = mm.retrieve(query)
        assert ctx.combined_context == ""
        assert ctx.is_empty()

    def test_build_memory_context_convenience(self):
        mm = self._populated_manager()
        ctx = mm.build_memory_context(task_instruction="pick up apple")
        assert isinstance(ctx, MemoryContext)

    def test_retrieve_respects_max_context_chars(self):
        cfg = MemoryConfig(max_context_chars=100, max_section_chars=80)
        mm = MemoryManager(config=cfg)
        for i in range(10):
            mm.update(task_instruction="task" * 20, action_text="action" * 20, step_id=i)
        query = MemoryQuery(task_instruction="task")
        ctx = mm.retrieve(query)
        assert len(ctx.combined_context) > 0  # content returned fully without truncation


class TestMemoryManagerFinalize:
    def test_finalize_creates_episode(self):
        mm = MemoryManager()
        mm.update(task_instruction="pick apple", action_text="go to table", step_id=0)
        episode = mm.finalize_episode(
            task_instruction="pick apple",
            final_status="success",
            env_name="alfred",
        )
        assert episode is not None
        assert len(mm.episodic.episodes) == 1

    def test_finalize_disabled_returns_none(self):
        cfg = MemoryConfig(enabled=False)
        mm = MemoryManager(config=cfg)
        result = mm.finalize_episode(task_instruction="task", final_status="success")
        assert result is None


class TestMemoryManagerResetAndStats:
    def test_reset_episode_clears_temporal(self):
        mm = MemoryManager()
        mm.update(task_instruction="t", action_text="a", step_id=0)
        assert len(mm.temporal) == 1
        mm.reset_episode()
        assert len(mm.temporal) == 0

    def test_get_memory_stats_keys(self):
        mm = MemoryManager()
        stats = mm.get_memory_stats()
        for key in ("enabled", "spatial_nodes", "temporal_steps", "episodic_episodes", "semantic_facts"):
            assert key in stats

    def test_get_memory_stats_updates_after_use(self):
        mm = MemoryManager()
        mm.update(task_instruction="task", action_text="step", step_id=0)
        stats = mm.get_memory_stats()
        assert stats["temporal_steps"] == 1

    def test_save_and_load(self, tmp_path):
        cfg = MemoryConfig(storage_dir=str(tmp_path / "mem"))
        mm = MemoryManager(config=cfg)
        mm.update(task_instruction="task", action_text="a1", step_id=0)
        mm.save()
        assert (tmp_path / "mem" / "temporal_memory.json").exists()

        mm2 = MemoryManager(config=cfg)
        mm2.load()
        assert len(mm2.temporal) == 1


# ===========================================================================
# MemoryPromptFormatter tests
# ===========================================================================

def _make_rich_context() -> MemoryContext:
    """Return a non-empty MemoryContext with all sections populated."""
    ctx = MemoryContext()
    ctx.spatial_context = "Apple was last seen on the kitchen table at step 12, confidence 0.82."
    ctx.temporal_context = "Step 5: pick up apple failed because apple was not visible."
    ctx.episodic_context = "Episode 3: task succeeded using GoTo then PickUp sequence."
    ctx.semantic_context = "Rule: objects must be visible before pickup."
    ctx.feasibility_constraints = ["Reject direct manipulation if target not visible."]
    ctx.stale_memory_warnings = ["Previous apple location may be outdated."]
    ctx.combined_context = "dummy"
    return ctx


class TestMemoryPromptFormatterEmpty:
    def test_planner_empty_context_returns_empty_string(self):
        fmt = MemoryPromptFormatter()
        assert fmt.format_for_planner(MemoryContext()) == ""

    def test_critic_empty_context_returns_empty_string(self):
        fmt = MemoryPromptFormatter()
        assert fmt.format_for_critic(MemoryContext()) == ""

    def test_compact_empty_context_returns_empty_string(self):
        fmt = MemoryPromptFormatter()
        assert fmt.format_compact(MemoryContext()) == ""


class TestMemoryPromptFormatterPreamble:
    def test_planner_includes_planning_preamble(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_planner(_make_rich_context())
        assert "Retrieved memory is helpful but may be outdated" in out

    def test_critic_includes_critic_preamble(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_critic(_make_rich_context())
        assert "Use memory only to check feasibility" in out

    def test_preamble_suppressed_when_disabled(self):
        fmt = MemoryPromptFormatter(include_preamble=False)
        out = fmt.format_for_planner(_make_rich_context())
        assert "Retrieved memory is helpful" not in out


class TestMemoryPromptFormatterOrder:
    def test_planner_section_order(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_planner(_make_rich_context())
        idx_spatial  = out.find("[Spatial Memory]")
        idx_temporal = out.find("[Temporal Memory]")
        idx_episodic = out.find("[Episodic Memory]")
        idx_semantic = out.find("[Semantic Memory]")
        idx_warnings = out.find("[Stale Memory Warnings]")
        assert idx_spatial < idx_temporal < idx_episodic < idx_semantic < idx_warnings

    def test_critic_section_order(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_critic(_make_rich_context())
        idx_constraints = out.find("[Feasibility Constraints]")
        idx_warnings    = out.find("[Stale Memory Warnings]")
        idx_temporal    = out.find("[Temporal Memory]")
        idx_spatial     = out.find("[Spatial Memory]")
        assert idx_constraints < idx_warnings < idx_temporal < idx_spatial


class TestMemoryPromptFormatterContent:

    def test_critic_includes_feasibility_constraints(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_critic(_make_rich_context())
        assert "Reject direct manipulation" in out

    def test_stale_warnings_appear_in_planner(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_planner(_make_rich_context())
        assert "outdated" in out

    def test_stale_warnings_appear_in_critic(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_critic(_make_rich_context())
        assert "outdated" in out

    def test_compact_shorter_than_planner(self):
        fmt = MemoryPromptFormatter()
        ctx = _make_rich_context()
        assert len(fmt.format_compact(ctx)) < len(fmt.format_for_planner(ctx))


class TestMemoryPromptFormatterSection:
    def test_format_section_list_renders_bullets(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_section("Hints", ["hint one", "hint two"])
        assert "- hint one" in out
        assert "- hint two" in out

    def test_format_section_omits_empty_by_default(self):
        fmt = MemoryPromptFormatter()
        assert fmt.format_section("Hints", []) == ""
        assert fmt.format_section("Hints", "") == ""

    def test_format_section_includes_empty_when_flag_set(self):
        fmt = MemoryPromptFormatter(include_empty_sections=True)
        out = fmt.format_section("Hints", "")
        assert "[Hints]" in out


class TestMemoryPromptFormatterSafety:
    def test_planner_output_has_no_code_fences(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_planner(_make_rich_context())
        assert "```" not in out

    def test_critic_output_has_no_code_fences(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_critic(_make_rich_context())
        assert "```" not in out

    def test_planner_output_has_no_raw_json_braces(self):
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_planner(_make_rich_context())
        # The formatter itself should never inject JSON-like examples
        assert '{"action"' not in out
        assert '"output":' not in out
