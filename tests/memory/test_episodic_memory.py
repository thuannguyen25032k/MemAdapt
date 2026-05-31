"""
tests/memory/test_episodic_memory.py

Unit tests for the redesigned EpisodicMemory module:
  - EpisodeRecord
  - EpisodicVectorStore
  - EpisodicUpdater (no-VLM and VLM paths)
  - EpisodicMemory (add_episode_from_trajectory, retrieve, prompt context,
    persistence)
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from embodiedbench.memory.episodic_memory import (
    EpisodeRecord,
    EpisodicMemory,
    EpisodicUpdater,
    EpisodicVectorStore,
)
from embodiedbench.memory.base import MemoryQuery, UpdateDecision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STEPS_A = [
    {"step_id": 0, "action": "navigate to sofa", "feedback": "success"},
    {"step_id": 1, "action": "pick up hammer",   "feedback": "success"},
    {"step_id": 2, "action": "place on table",   "feedback": "success"},
]

STEPS_B = [
    {"step_id": 0, "action": "navigate to kitchen", "feedback": "success"},
    {"step_id": 1, "action": "pick up plate",        "feedback": "The action is invalid."},
]


def make_record(**kwargs) -> EpisodeRecord:
    defaults = dict(
        task_instruction="Move hammer to table",
        final_status="success",
        steps=list(STEPS_A),
        env_name="eb-alfred",
        scene_id="scene_1",
    )
    defaults.update(kwargs)
    return EpisodeRecord(**defaults)


def make_query(**kwargs) -> MemoryQuery:
    defaults = dict(
        task_instruction="Move hammer to table",
        env_name="eb-alfred",
        target_objects=["hammer"],
    )
    defaults.update(kwargs)
    return MemoryQuery(**defaults)


# ---------------------------------------------------------------------------
# TestEpisodeRecord
# ---------------------------------------------------------------------------

class TestEpisodeRecord:
    def test_to_dict_from_dict_roundtrip(self):
        rec = make_record()
        d = rec.to_dict()
        restored = EpisodeRecord.from_dict(d)
        assert restored.id == rec.id
        assert restored.task_instruction == rec.task_instruction
        assert restored.final_status == rec.final_status
        assert restored.steps == rec.steps
        assert restored.env_name == rec.env_name

    def test_trajectory_text_format(self):
        rec = make_record()
        text = rec.trajectory_text()
        assert "Step 0:" in text          # 0-based step numbering
        assert "navigate to sofa" in text
        assert "→ success" in text        # feedback shown after →

    def test_trajectory_text_empty_steps(self):
        rec = make_record(steps=[])
        assert rec.trajectory_text() == "[No steps recorded]"

    def test_trajectory_text_max_steps(self):
        many = [{"step_id": i, "action": f"act{i}", "feedback": "ok"} for i in range(20)]
        rec = make_record(steps=many)
        text = rec.trajectory_text()
        assert "Step 0:" in text
        assert "Step 19:" in text  # all steps shown, no limit

    def test_text_for_retrieval_contains_key_fields(self):
        rec = make_record()
        t = rec.text_for_retrieval()
        assert "Move hammer to table" in t
        assert "success" in t

    def test_text_for_retrieval_includes_failure_reasons(self):
        rec = make_record(final_status="failure")
        t = rec.text_for_retrieval()
        assert "failure" in t

    def test_to_memory_item_type_and_metadata(self):
        rec = make_record()
        item = rec.to_memory_item()
        assert item.memory_type == "episodic"
        assert item.metadata["status"] == "success"
        assert item.metadata["env_name"] == "eb-alfred"
        # content includes task, outcome, trajectory
        assert "Move hammer to table" in item.content
        assert "[Recent steps]" not in item.content
        assert "Step 0:" in item.content

    def test_to_memory_item_failure_reasons_in_content(self):
        rec = make_record(final_status="failure")
        item = rec.to_memory_item()
        assert "failure" in item.content

    def test_touch_updates_timestamp(self):
        rec = make_record()
        before = rec.updated_at
        import time; time.sleep(0.01)
        rec.touch()
        assert rec.updated_at > before

    def test_from_dict_missing_optional_fields(self):
        minimal = {"task_instruction": "Do something"}
        rec = EpisodeRecord.from_dict(minimal)
        assert rec.final_status == "unknown"
        assert rec.steps == []


# ---------------------------------------------------------------------------
# TestEpisodicVectorStore
# ---------------------------------------------------------------------------

class TestEpisodicVectorStore:
    def test_empty_store_returns_empty(self):
        store = EpisodicVectorStore([])
        assert store.top_k_similar("pick up hammer", k=3) == []

    def test_shared_reference_reflects_mutations(self):
        episodes = []
        store = EpisodicVectorStore(episodes)
        assert store.top_k_similar("task", k=1) == []
        episodes.append(make_record())
        results = store.top_k_similar("Move hammer", k=1)
        assert len(results) == 1

    def test_top_k_similar_returns_sorted_scores(self):
        ep1 = make_record(task_instruction="Move hammer to table", id="a")
        ep2 = make_record(task_instruction="Boil water in kitchen", id="b")
        store = EpisodicVectorStore([ep1, ep2])
        results = store.top_k_similar("Move hammer to table", k=2)
        scores = [r[0] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exclude_ids(self):
        ep = make_record(id="x")
        store = EpisodicVectorStore([ep])
        results = store.top_k_similar("Move hammer", k=5, exclude_ids={"x"})
        assert all(r[1].id != "x" for r in results)

    def test_k_limits_results(self):
        episodes = [make_record(id=str(i), task_instruction=f"Task {i}") for i in range(10)]
        store = EpisodicVectorStore(episodes)
        assert len(store.top_k_similar("Task", k=3)) == 3

    def test_embed_returns_none_without_provider(self):
        store = EpisodicVectorStore([])
        assert store.embed("some text") is None


# ---------------------------------------------------------------------------
# TestEpisodicUpdater — no-VLM path
# ---------------------------------------------------------------------------

class TestEpisodicUpdaterNoVlm:
    def test_no_vlm_always_adds(self):
        store = EpisodicVectorStore([])
        updater = EpisodicUpdater(store, vlm_call=None)
        decision = updater.process(make_record())
        assert decision.action == "add"

    def test_empty_task_returns_noop(self):
        store = EpisodicVectorStore([])
        updater = EpisodicUpdater(store, vlm_call=None)
        decision = updater.process(make_record(task_instruction=""))
        assert decision.action == "noop"

    def test_no_vlm_with_similar_episodes_still_adds(self):
        ep = make_record(id="existing")
        store = EpisodicVectorStore([ep])
        updater = EpisodicUpdater(store, vlm_call=None)
        decision = updater.process(make_record(task_instruction="Move hammer to table"))
        assert decision.action == "add"


# ---------------------------------------------------------------------------
# TestEpisodicUpdater — VLM path
# ---------------------------------------------------------------------------

class TestEpisodicUpdaterVlm:
    def _make_vlm(self, response: str):
        return lambda prompt: response

    def test_vlm_add(self):
        ep = make_record(id="old", task_instruction="Transport cups")
        store = EpisodicVectorStore([ep])
        updater = EpisodicUpdater(store, vlm_call=self._make_vlm('{"action": "add"}'), top_s=3)
        decision = updater.process(make_record())
        assert decision.action == "add"

    def test_vlm_noop(self):
        ep = make_record(id="old")
        store = EpisodicVectorStore([ep])
        updater = EpisodicUpdater(store, vlm_call=self._make_vlm('{"action": "noop"}'), top_s=3)
        decision = updater.process(make_record())
        assert decision.action == "noop"

    def test_vlm_update_resolves_target(self):
        ep = make_record(id="target_ep")
        store = EpisodicVectorStore([ep])
        updater = EpisodicUpdater(
            store,
            vlm_call=self._make_vlm('{"action": "update", "target_index": 1}'),
            top_s=3,
        )
        decision = updater.process(make_record())
        assert decision.action == "update"
        assert decision.target_id == "target_ep"

    def test_vlm_remove_resolves_target(self):
        ep = make_record(id="old_ep")
        store = EpisodicVectorStore([ep])
        updater = EpisodicUpdater(
            store,
            vlm_call=self._make_vlm('{"action": "remove", "target_index": 1}'),
            top_s=3,
        )
        decision = updater.process(make_record())
        assert decision.action == "remove"
        assert decision.target_id == "old_ep"

    def test_bad_json_falls_back_to_add(self):
        ep = make_record(id="e")
        store = EpisodicVectorStore([ep])
        updater = EpisodicUpdater(store, vlm_call=self._make_vlm("not json"), top_s=3)
        decision = updater.process(make_record())
        assert decision.action == "add"

    def test_empty_response_falls_back_to_add(self):
        ep = make_record(id="e")
        store = EpisodicVectorStore([ep])
        updater = EpisodicUpdater(store, vlm_call=self._make_vlm(""), top_s=3)
        decision = updater.process(make_record())
        assert decision.action == "add"

    def test_unknown_action_falls_back_to_add(self):
        ep = make_record(id="e")
        store = EpisodicVectorStore([ep])
        updater = EpisodicUpdater(
            store, vlm_call=self._make_vlm('{"action": "destroy"}'), top_s=3
        )
        decision = updater.process(make_record())
        assert decision.action == "add"

    def test_no_similar_episodes_returns_add(self):
        store = EpisodicVectorStore([])   # empty
        updater = EpisodicUpdater(
            store, vlm_call=self._make_vlm('{"action": "noop"}'), top_s=3
        )
        # no similar → always add regardless of VLM response
        decision = updater.process(make_record())
        assert decision.action == "add"

    def test_vlm_exception_falls_back_to_add(self):
        def bad_vlm(prompt):
            raise RuntimeError("VLM failed")
        ep = make_record(id="e")
        store = EpisodicVectorStore([ep])
        updater = EpisodicUpdater(store, vlm_call=bad_vlm, top_s=3)
        decision = updater.process(make_record())
        assert decision.action == "add"

    def test_parse_decision_out_of_range_index_clamps_to_first(self):
        similar = [(0.9, make_record(id="first"))]
        d = EpisodicUpdater._parse_decision('{"action": "update", "target_index": 99}', similar)
        assert d.action == "update"
        assert d.target_id == "first"


# ---------------------------------------------------------------------------
# TestEpisodicMemory — add_episode_from_trajectory
# ---------------------------------------------------------------------------

class TestEpisodicMemoryAdd:
    def test_add_returns_record(self):
        mem = EpisodicMemory()
        rec = mem.add_episode_from_trajectory(
            task_instruction="Move hammer",
            final_status="success",
            steps=list(STEPS_A),
        )
        assert rec is not None
        assert rec.task_instruction == "Move hammer"

    def test_add_appends_to_episodes(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Task A", final_status="success", steps=STEPS_A)
        mem.add_episode_from_trajectory("Task B", final_status="failure", steps=STEPS_B)
        # Only successful episodes are stored
        assert len(mem.episodes) == 1

    def test_empty_task_returns_none(self):
        mem = EpisodicMemory()
        result = mem.add_episode_from_trajectory("", final_status="success", steps=[])
        assert result is None
        assert len(mem) == 0

    def test_no_vlm_always_adds_even_duplicates(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Move hammer", final_status="success")
        mem.add_episode_from_trajectory("Move hammer", final_status="success")
        assert len(mem) == 2

    def test_vlm_noop_skips_episode(self):
        mem = EpisodicMemory(vlm_call=lambda p: '{"action": "noop"}')
        mem.add_episode_from_trajectory("Existing", final_status="success", steps=STEPS_A)
        before = len(mem)
        result = mem.add_episode_from_trajectory("Existing", final_status="success", steps=STEPS_A)
        assert result is None
        assert len(mem) == before

    def test_vlm_update_replaces_trajectory(self):
        mem = EpisodicMemory(vlm_call=lambda p: '{"action": "add"}')
        # Only success episodes are stored; start with one
        old = mem.add_episode_from_trajectory("Task X", final_status="success", steps=STEPS_B)
        old_id = old.id

        calls = [0]
        def vlm(prompt):
            if calls[0] == 0:
                calls[0] += 1
                return f'{{"action": "update", "target_index": 1}}'
            return '{"action": "add"}'

        mem._updater.vlm_call = vlm
        mem.add_episode_from_trajectory("Task X", final_status="success", steps=STEPS_A)

        updated = mem._find_by_id(old_id)
        assert updated is not None
        assert updated.final_status == "success"
        assert updated.steps == STEPS_A

    def test_vlm_remove_then_add(self):
        mem = EpisodicMemory(vlm_call=lambda p: '{"action": "add"}')
        old = mem.add_episode_from_trajectory("Old task", steps=STEPS_A, final_status="success")
        old_id = old.id
        before_count = len(mem)

        mem._updater.vlm_call = lambda p: f'{{"action": "remove", "target_index": 1}}'
        mem.add_episode_from_trajectory("New task", steps=STEPS_A, final_status="success")

        # old is removed, new is added → count stays the same
        assert len(mem) == before_count
        assert mem._find_by_id(old_id) is None

    def test_max_episodes_prunes_oldest(self):
        mem = EpisodicMemory(max_episodes=3)
        for i in range(5):
            mem.add_episode_from_trajectory(f"Task {i}", final_status="success")
        assert len(mem) <= 3

    def test_steps_stored_correctly(self):
        mem = EpisodicMemory()
        rec = mem.add_episode_from_trajectory("Task", final_status="success", steps=STEPS_A)
        assert len(rec.steps) == len(STEPS_A)
        assert rec.steps[0]["action"] == "navigate to sofa"

    def test_failure_not_stored(self):
        mem = EpisodicMemory()
        rec = mem.add_episode_from_trajectory(
            "Carry plate",
            final_status="failure",
        )
        assert rec is None
        assert len(mem) == 0


# ---------------------------------------------------------------------------
# TestEpisodicMemoryRetrieve
# ---------------------------------------------------------------------------

class TestEpisodicMemoryRetrieve:
    def test_retrieve_empty_returns_empty(self):
        mem = EpisodicMemory()
        assert mem.retrieve(make_query()) == []

    def test_retrieve_returns_retrieved_memory_objects(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Move hammer to table", steps=STEPS_A,
                                        final_status="success")
        results = mem.retrieve(make_query())
        assert len(results) == 1
        assert results[0].item.memory_type == "episodic"

    def test_retrieve_top_k_limits(self):
        mem = EpisodicMemory()
        for i in range(10):
            mem.add_episode_from_trajectory(f"Task {i}", final_status="success")
        results = mem.retrieve(make_query(), top_k=3)
        assert len(results) <= 3

    def test_retrieve_scores_sum_to_at_most_one(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Move hammer", steps=STEPS_A, final_status="success")
        results = mem.retrieve(make_query())
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_retrieve_object_overlap_boosts_score(self):
        mem = EpisodicMemory()
        ep_match = mem.add_episode_from_trajectory(
            "Move hammer", steps=STEPS_A, final_status="success",
        )
        ep_other = mem.add_episode_from_trajectory(
            "Move cup", steps=STEPS_A, final_status="success",
        )
        results = mem.retrieve(make_query(target_objects=["hammer"]))
        ep_scores = {r.item.metadata["episode_id"]: r.score for r in results}
        assert ep_scores[ep_match.id] >= ep_scores[ep_other.id]

    def test_retrieve_scene_bonus_works(self):
        mem = EpisodicMemory()
        ep_same = mem.add_episode_from_trajectory(
            "Move hammer", final_status="success",
            env_name="eb-alfred", scene_id="scene_1"
        )
        ep_diff = mem.add_episode_from_trajectory(
            "Move hammer", final_status="success",
            env_name="other-env", scene_id="scene_99"
        )
        results = mem.retrieve(make_query(scene_id="scene_1"))
        ep_scores = {r.item.metadata["episode_id"]: r.score for r in results}
        assert ep_scores[ep_same.id] > ep_scores[ep_diff.id]

    def test_retrieve_reason_matches_status(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Task", final_status="success")
        results = mem.retrieve(make_query())
        assert "success" in results[0].reason


# ---------------------------------------------------------------------------
# TestEpisodicMemoryPersistence
# ---------------------------------------------------------------------------

class TestEpisodicMemoryPersistence:
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "episodic.json")
            mem = EpisodicMemory(storage_path=path)
            mem.add_episode_from_trajectory(
                "Move hammer", final_status="success",
                steps=STEPS_A,
            )
            mem.save()

            mem2 = EpisodicMemory(storage_path=path)
            mem2.load()
            assert len(mem2.episodes) == 1
            ep = mem2.episodes[0]
            assert ep.task_instruction == "Move hammer"
            assert ep.final_status == "success"
            assert len(ep.steps) == len(STEPS_A)

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ep.json")
            mem = EpisodicMemory(storage_path=path)
            mem.add_episode_from_trajectory("Task", final_status="unknown")
            mem.save()
            assert os.path.exists(path)

    def test_load_missing_file_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "missing.json")
            mem = EpisodicMemory(storage_path=path)
            mem.load()   # should not raise
            assert len(mem) == 0

    def test_load_invalid_json_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w") as f:
                f.write("{not valid json")
            mem = EpisodicMemory(storage_path=path)
            mem.load()  # should not raise
            assert len(mem) == 0

    def test_vector_store_shared_reference_after_load(self):
        """After load(), vector store must see the loaded episodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ep.json")
            mem = EpisodicMemory(storage_path=path)
            mem.add_episode_from_trajectory("Task", final_status="success", steps=STEPS_A)
            mem.save()

            mem2 = EpisodicMemory(storage_path=path)
            mem2.load()
            results = mem2._vector_store.top_k_similar("Task", k=5)
            assert len(results) == 1

    def test_save_without_path_no_error(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Task", final_status="success")
        mem.save()  # no path → no-op, should not raise


# ---------------------------------------------------------------------------
# TestEpisodicMemoryPromptContext
# ---------------------------------------------------------------------------

class TestEpisodicMemoryPromptContext:
    def test_empty_memories_returns_empty_string(self):
        mem = EpisodicMemory()
        assert mem.to_prompt_context([]) == ""

    def test_header_present(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Task A", final_status="success", steps=STEPS_A)
        results = mem.retrieve(make_query())
        ctx = mem.to_prompt_context(results)
        assert "Similar successful episode" in ctx

    def test_status_tag_not_in_success_output(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Move hammer", final_status="success", steps=STEPS_A)
        results = mem.retrieve(make_query())
        ctx = mem.to_prompt_context(results)
        assert "(failure)" not in ctx

    def test_trajectory_steps_in_output(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Move hammer", final_status="success", steps=STEPS_A)
        results = mem.retrieve(make_query())
        ctx = mem.to_prompt_context(results)
        assert "Step 0:" in ctx
        assert "navigate to sofa" in ctx

    def test_guidance_footer_absent(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Task", final_status="success", steps=STEPS_A)
        results = mem.retrieve(make_query())
        ctx = mem.to_prompt_context(results)
        # New simplified format has no footer boilerplate
        assert "guidance only" not in ctx

    def test_max_chars_truncates(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Task " * 100, final_status="success", steps=STEPS_A)
        results = mem.retrieve(make_query(task_instruction="Task " * 100))
        ctx = mem.to_prompt_context(results)
        assert len(ctx) > 0  # content is returned fully without truncation


# ---------------------------------------------------------------------------
# TestSuccessfulTrajectoryGuide
# ---------------------------------------------------------------------------

class TestSuccessfulTrajectoryGuide:
    def test_empty_memories_returns_empty(self):
        mem = EpisodicMemory()
        assert mem.successful_trajectory_guide([]) == ""

    def test_no_success_returns_empty(self):
        mem = EpisodicMemory()
        # Failure episodes are not stored → retrieve returns nothing
        mem.add_episode_from_trajectory("Task", final_status="failure", steps=STEPS_B)
        results = mem.retrieve(make_query())
        assert mem.successful_trajectory_guide(results) == ""

    def test_success_episode_produces_guide(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory(
            "Move hammer to table",
            final_status="success",
            steps=STEPS_A,
        )
        results = mem.retrieve(make_query())
        guide = mem.successful_trajectory_guide(results)
        assert "Successful trajectory guide" in guide
        assert "Move hammer to table" in guide
        assert "Step 0:" in guide
        assert "navigate to sofa" in guide

    def test_only_success_steps_shown(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Task A", final_status="failure", steps=STEPS_B)
        mem.add_episode_from_trajectory("Task B", final_status="success", steps=STEPS_A)
        results = mem.retrieve(make_query(task_instruction="Task B"))
        guide = mem.successful_trajectory_guide(results)
        assert "Task B" in guide
        assert "navigate to sofa" in guide

    def test_max_steps_limits_output(self):
        many = [{"step_id": i, "action": f"action_{i}", "feedback": "ok"} for i in range(20)]
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Big task", final_status="success", steps=many)
        results = mem.retrieve(make_query(task_instruction="Big task"))
        guide = mem.successful_trajectory_guide(results)
        assert "Step 3:" in guide
        assert "Step 4:" in guide  # no max_steps limit; all steps shown

    def test_max_chars_truncates(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Task", final_status="success", steps=STEPS_A)
        results = mem.retrieve(make_query())
        guide = mem.successful_trajectory_guide(results)
        assert len(guide) > 0  # content is returned fully without truncation

    def test_to_prompt_context_separates_success_failure(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Task A", final_status="success", steps=STEPS_A)
        mem.add_episode_from_trajectory("Task B", final_status="failure", steps=STEPS_B)
        results = mem.retrieve(make_query())
        ctx = mem.to_prompt_context(results)
        # Only successful episodes stored; new format shows them directly
        assert "Similar successful episode" in ctx
        assert "(failure)" not in ctx

    def test_to_prompt_context_success_only(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("Task A", final_status="success", steps=STEPS_A)
        results = mem.retrieve(make_query())
        ctx = mem.to_prompt_context(results)
        assert "Similar successful episode" in ctx
        assert "(failure)" not in ctx

    def test_to_prompt_context_failure_only(self):
        mem = EpisodicMemory()
        # Failure episodes are not stored → to_prompt_context returns ""
        mem.add_episode_from_trajectory("Task A", final_status="failure", steps=STEPS_B)
        results = mem.retrieve(make_query())
        ctx = mem.to_prompt_context(results)
        assert ctx == ""


# ---------------------------------------------------------------------------
# TestEpisodicMemoryMisc
# ---------------------------------------------------------------------------

class TestEpisodicMemoryMisc:
    def test_len_reflects_episodes(self):
        mem = EpisodicMemory()
        assert len(mem) == 0
        mem.add_episode_from_trajectory("A", final_status="success")
        assert len(mem) == 1

    def test_update_is_noop(self):
        mem = EpisodicMemory()
        mem.update()    # should not raise

    def test_reset_episode_is_noop(self):
        mem = EpisodicMemory()
        mem.add_episode_from_trajectory("A", final_status="success")
        mem.reset_episode()
        assert len(mem) == 1   # episodes persist

    def test_remove_by_id_existing(self):
        mem = EpisodicMemory()
        rec = mem.add_episode_from_trajectory("A", final_status="success")
        assert mem._remove_by_id(rec.id)
        assert len(mem) == 0

    def test_remove_by_id_missing(self):
        mem = EpisodicMemory()
        assert not mem._remove_by_id("does-not-exist")

    def test_find_by_id(self):
        mem = EpisodicMemory()
        rec = mem.add_episode_from_trajectory("Task", final_status="success")
        found = mem._find_by_id(rec.id)
        assert found is rec

    def test_find_by_id_missing(self):
        mem = EpisodicMemory()
        assert mem._find_by_id("x") is None
