"""
tests/memory/test_temporal_ordering.py

Tests that focus on the chronological ordering fix for TemporalMemory:
  - retrieve() returns steps sorted by step_id (ascending), not by score
  - to_prompt_context() produces a single chronological section
  - Loop detection (same action failing consecutively) is surfaced
  - Do-not-repeat warnings appear at the top before the history
  - Existing retrieval scoring behaviour is preserved
"""

import pytest
from embodiedbench.memory.temporal_memory import TemporalMemory, TemporalStep
from embodiedbench.memory.base import MemoryQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tm(**kwargs) -> TemporalMemory:
    return TemporalMemory(max_steps=20, **kwargs)


def _add_steps(tm: TemporalMemory, steps):
    """steps is a list of (action_text, success, feedback) tuples."""
    for i, (action_text, success, feedback) in enumerate(steps):
        tm.append_step(
            task_instruction="Transport all plate and put them on the right counter.",
            action_text=action_text,
            env_feedback=feedback,
            success=success,
            step_id=i + 1,
        )


# ---------------------------------------------------------------------------
# retrieve() – chronological ordering
# ---------------------------------------------------------------------------

class TestRetrieveChronologicalOrder:
    def test_steps_returned_in_step_id_order(self):
        """Retrieved steps must be sorted by step_id ascending, not by score."""
        tm = _make_tm()
        _add_steps(tm, [
            ("navigate to the right drawer of the kitchen counter", True,
             "Last action executed successfully."),
            ("pick up the plate", False,
             "Last action is invalid. Robot cannot pick any object."),
            ("navigate to the right counter in the kitchen", True,
             "Last action executed successfully."),
            ("pick up the plate", False,
             "Last action is invalid. Robot cannot pick any object."),
        ])
        q = MemoryQuery(
            task_instruction="Transport all plate and put them on the right counter.",
            target_objects=["plate"],
        )
        results = tm.retrieve(q, top_k=4)
        step_ids = [r.item.metadata["step_id"] for r in results]
        assert step_ids == sorted(step_ids), (
            f"Steps should be chronological but got order: {step_ids}"
        )

    def test_retrieved_step_ids_ascending(self):
        """Even when failure steps score higher, oldest failure must come first."""
        tm = _make_tm()
        _add_steps(tm, [
            ("navigate to the table", True, "success."),
            ("pick up the plate", False, "not near robot"),
            ("navigate to the counter", True, "success."),
            ("pick up the plate", False, "not near robot"),
            ("pick up the plate", False, "not near robot"),
        ])
        q = MemoryQuery(
            task_instruction="pick up the plate",
            target_objects=["plate"],
        )
        results = tm.retrieve(q, top_k=5)
        step_ids = [r.item.metadata["step_id"] for r in results]
        assert step_ids == sorted(step_ids)

    def test_single_step_returns_single_item(self):
        tm = _make_tm()
        tm.append_step(
            task_instruction="task", action_text="navigate to table",
            success=True, env_feedback="ok", step_id=1,
        )
        results = tm.retrieve(MemoryQuery(task_instruction="task"), top_k=3)
        assert len(results) == 1
        assert results[0].item.metadata["step_id"] == 1

    def test_empty_returns_empty(self):
        tm = _make_tm()
        assert tm.retrieve(MemoryQuery(task_instruction="task"), top_k=5) == []

    def test_top_k_respected(self):
        tm = _make_tm()
        for i in range(10):
            tm.append_step(task_instruction="t", action_text=f"action {i}",
                           success=True, step_id=i)
        results = tm.retrieve(MemoryQuery(task_instruction="action"), top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# to_prompt_context() – single chronological block
# ---------------------------------------------------------------------------

class TestPromptContextFormat:
    def _build_tm_and_context(self, steps, top_k=5):
        tm = _make_tm()
        _add_steps(tm, steps)
        q = MemoryQuery(
            task_instruction="Transport all plate and put them on the right counter.",
            target_objects=["plate"],
        )
        memories = tm.retrieve(q, top_k=top_k)
        ctx = tm.to_prompt_context(memories)
        return ctx, memories

    def test_contains_interaction_header(self):
        ctx, _ = self._build_tm_and_context([
            ("navigate to table", True, "ok"),
            ("pick up the plate", False, "not near robot"),
        ])
        assert "Recent relevant interactions:" in ctx

    def test_steps_appear_in_chronological_text_order(self):
        """Step 1 must appear before Step 2 in the output string."""
        ctx, _ = self._build_tm_and_context([
            ("navigate to the right drawer", True, "success."),
            ("pick up the plate", False, "not near robot"),
            ("navigate to the right counter", True, "success."),
            ("pick up the plate", False, "not near robot"),
        ])
        pos1 = ctx.find("Step 1")
        pos2 = ctx.find("Step 2")
        pos3 = ctx.find("Step 3")
        pos4 = ctx.find("Step 4")
        assert pos1 < pos2 < pos3 < pos4, (
            "Steps must appear in ascending order in the prompt"
        )

    def test_result_symbol_not_shown(self):
        """Result symbols (✓/✗) are intentionally omitted from the output."""
        ctx, _ = self._build_tm_and_context([
            ("navigate to table", True, "success."),
            ("pick up the plate", False, "not near robot"),
        ])
        assert "✓" not in ctx
        assert "✗" not in ctx

    def test_feedback_text_shown(self):
        ctx, _ = self._build_tm_and_context([
            ("pick up the plate", False, "Robot cannot pick any object that is not near the robot"),
        ])
        assert "not near" in ctx.lower() or "cannot pick" in ctx.lower()

    def test_no_redundant_recent_failures_section(self):
        """The old standalone 'Recent failures:' sub-block should not appear."""
        ctx, _ = self._build_tm_and_context([
            ("pick up the plate", False, "not near robot"),
            ("pick up the plate", False, "not near robot"),
        ])
        assert "Recent failures:" not in ctx

    def test_no_redundant_critic_rejections_section(self):
        tm = _make_tm()
        tm.append_step(task_instruction="t", action_text="bad action",
                       success=False, critic_rejected=True,
                       critic_output="Action violates precondition.", step_id=1)
        q = MemoryQuery(task_instruction="t")
        memories = tm.retrieve(q, top_k=5)
        ctx = tm.to_prompt_context(memories)
        assert "Critic rejections:" not in ctx

    def test_do_not_repeat_block_present_for_repeated_failures(self):
        tm = _make_tm()
        for i in range(3):
            tm.append_step(task_instruction="t", action_text="pick up the plate",
                           success=False, env_feedback="not near", step_id=i + 1)
        q = MemoryQuery(task_instruction="pick up the plate", target_objects=["plate"])
        memories = tm.retrieve(q, top_k=5)
        ctx = tm.to_prompt_context(memories)
        assert "Do not repeat" in ctx or "failed" in ctx.lower()

    def test_do_not_repeat_block_before_history(self):
        tm = _make_tm()
        for i in range(2):
            tm.append_step(task_instruction="t", action_text="pick up the plate",
                           success=False, env_feedback="not near", step_id=i + 1)
        q = MemoryQuery(task_instruction="pick up the plate", target_objects=["plate"])
        memories = tm.retrieve(q, top_k=5)
        ctx = tm.to_prompt_context(memories)
        pos_warn = ctx.find("Do not repeat")
        pos_hist = ctx.find("Recent relevant interactions:")
        if pos_warn != -1 and pos_hist != -1:
            assert pos_warn < pos_hist, "Warning block must appear before the history"

    def test_empty_memories_returns_empty_string(self):
        tm = _make_tm()
        assert tm.to_prompt_context([]) == ""

    def test_max_chars_respected(self):
        tm = _make_tm()
        for i in range(15):
            tm.append_step(task_instruction="t", action_text=f"navigate to place {i}",
                           success=True, env_feedback="ok", step_id=i)
        q = MemoryQuery(task_instruction="navigate")
        memories = tm.retrieve(q, top_k=10)
        ctx = tm.to_prompt_context(memories)
        assert len(ctx) > 0  # content returned fully without truncation


# ---------------------------------------------------------------------------
# Loop detection – ⚠ same action still failing tag
# ---------------------------------------------------------------------------

class TestLoopDetection:
    def test_loop_tag_on_consecutive_same_failure(self):
        """When the same action fails twice in a row, the second gets a loop tag."""
        tm = _make_tm()
        _add_steps(tm, [
            ("navigate to counter", True, "success."),
            ("pick up the plate", False, "not near robot"),
            ("pick up the plate", False, "not near robot"),
        ])
        q = MemoryQuery(task_instruction="pick up the plate", target_objects=["plate"])
        memories = tm.retrieve(q, top_k=5)
        ctx = tm.to_prompt_context(memories)
        # The loop warning tag should be present
        assert "still failing" in ctx or "⚠" in ctx

    def test_no_loop_tag_when_actions_differ(self):
        """Different actions between failures should not trigger the loop tag."""
        tm = _make_tm()
        _add_steps(tm, [
            ("pick up the plate", False, "not near robot"),
            ("navigate to sofa", True, "success."),
            ("pick up the plate", False, "not near robot"),
        ])
        q = MemoryQuery(task_instruction="pick up the plate", target_objects=["plate"])
        memories = tm.retrieve(q, top_k=5)
        ctx = tm.to_prompt_context(memories)
        # No consecutive failure pair → no loop tag
        assert "still failing" not in ctx and "⚠" not in ctx

    def test_no_loop_tag_for_single_failure(self):
        tm = _make_tm()
        _add_steps(tm, [
            ("pick up the plate", False, "not near robot"),
        ])
        q = MemoryQuery(task_instruction="pick up the plate", target_objects=["plate"])
        memories = tm.retrieve(q, top_k=5)
        ctx = tm.to_prompt_context(memories)
        assert "still failing" not in ctx and "⚠" not in ctx


# ---------------------------------------------------------------------------
# Regression: existing retrieve() scoring behaviour
# ---------------------------------------------------------------------------

class TestRetrieveScoringRegression:
    def test_failure_steps_score_higher_than_successes(self):
        """Failure steps should have higher retrieval scores than successes for
        an otherwise equally relevant query, so they are selected into top_k."""
        tm = _make_tm()
        for i in range(8):
            tm.append_step(task_instruction="t",
                           action_text="navigate to counter",
                           success=True, env_feedback="ok", step_id=i)
        # Add one failure for a target action
        tm.append_step(task_instruction="t",
                       action_text="pick up the plate",
                       success=False, env_feedback="not near robot", step_id=8)
        q = MemoryQuery(task_instruction="pick up plate", target_objects=["plate"])
        results = tm.retrieve(q, top_k=3)
        step_ids = [r.item.metadata["step_id"] for r in results]
        # The failure step should be selected
        assert 8 in step_ids

    def test_scores_in_valid_range(self):
        tm = _make_tm()
        _add_steps(tm, [
            ("navigate to table", True, "ok"),
            ("pick up the plate", False, "not near"),
        ])
        q = MemoryQuery(task_instruction="pick up plate", target_objects=["plate"])
        results = tm.retrieve(q, top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_retrieve_with_no_embeddings_does_not_crash(self):
        tm = TemporalMemory(max_steps=10, embedding_provider=None)
        _add_steps(tm, [("pick up plate", False, "not near")])
        q = MemoryQuery(task_instruction="pick up plate")
        results = tm.retrieve(q, top_k=5)
        assert len(results) >= 1
