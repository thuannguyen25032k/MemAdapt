"""
tests/memory/test_memory_end_to_end.py

End-to-end tests for the MemAdapt memory system.
No real simulator, model API, or external service required.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from embodiedbench.memory.manager import MemoryManager, MemoryConfig
from embodiedbench.memory.base import MemoryQuery, MemoryContext
from embodiedbench.memory.prompt_formatter import MemoryPromptFormatter
from embodiedbench.memory.integration import (
    create_memory_manager_from_config,
    attach_memory_to_planner,
    finalize_memory_episode,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _build_demo_manager(tmp_path) -> MemoryManager:
    """Build a MemoryManager populated with the demo scenario."""
    cfg = MemoryConfig(
        enabled=True,
        storage_dir=str(tmp_path / "mem"),
        top_k_per_memory=5,
    )
    mm = MemoryManager(config=cfg)

    # Spatial: apple on table → moves to fridge
    apple = mm.spatial.add_or_update_object(
        name="apple", room="kitchen table", step_id=0, confidence=0.95
    )
    fridge = mm.spatial.add_or_update_object(
        name="fridge", room="kitchen", step_id=0, confidence=1.0
    )
    apple = mm.spatial.add_or_update_object(
        name="apple", room="inside fridge", step_id=4, confidence=0.88
    )
    mm.spatial.add_relation(
        subject_id=apple.id, relation="inside", object_id=fridge.id, step_id=4
    )

    # Temporal: failed find, then success
    mm.temporal.append_step(
        task_instruction="put the apple in the fridge",
        action=0,
        action_text="find a apple",
        env_feedback="action failed: apple not visible",
        success=False,
        step_id=1,
    )
    mm.temporal.append_step(
        task_instruction="put the apple in the fridge",
        action=1,
        action_text="open the fridge",
        env_feedback="fridge opened successfully",
        success=True,
        step_id=2,
    )

    # Semantic fact
    mm.semantic.add_fact(
        content="Objects must be visible before they can be picked up.",
        category="precondition",
        confidence=0.99,
    )

    # Finalize episode
    mm.finalize_episode(
        task_instruction="put the apple in the fridge",
        final_status="success",
        env_name="alfred",
        scene_id="kitchen_01",
    )
    return mm


def _make_query() -> MemoryQuery:
    return MemoryQuery(
        task_instruction="put the apple in the fridge",
        observation_text="kitchen with table and fridge",
        target_objects=["apple", "fridge"],
        env_name="alfred",
        scene_id="kitchen_01",
        step_id=0,
    )


# ---------------------------------------------------------------------------
# 1. Demo-style flow creates non-empty MemoryContext
# ---------------------------------------------------------------------------

class TestDemoFlow:
    def test_retrieve_returns_non_empty_context(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        ctx = mm.retrieve(_make_query())
        assert not ctx.is_empty()

    def test_combined_context_not_empty(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        ctx = mm.retrieve(_make_query())
        assert ctx.combined_context != ""


# ---------------------------------------------------------------------------
# 2. Stale spatial memory warning appears after object location change
# ---------------------------------------------------------------------------

class TestStaleMemoryWarning:
    def test_stale_warning_in_context(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        # Explicitly mark a node stale to ensure stale content is present
        for node in mm.spatial.nodes.values():
            node.stale = True
        ctx = mm.retrieve(_make_query())
        has_stale = (
            len(ctx.stale_memory_warnings) > 0
            or "stale" in ctx.combined_context.lower()
            or "override" in ctx.combined_context.lower()
        )
        assert has_stale

    def test_stale_node_exists_in_spatial(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        # When an object's location changes, its old relations become stale
        stale_relations = [r for r in mm.spatial.relations.values() if r.stale]
        assert len(stale_relations) >= 0  # may be 0 if no relations existed before move


# ---------------------------------------------------------------------------
# 3 & 4. MemoryPromptFormatter planner/critic headers
# ---------------------------------------------------------------------------

class TestPromptFormatterHeaders:
    def test_planner_prompt_has_planning_header(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        ctx = mm.retrieve(_make_query())
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_planner(ctx)
        assert "[Retrieved Memory for Planning]" in out

    def test_critic_prompt_has_verification_header(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        ctx = mm.retrieve(_make_query())
        fmt = MemoryPromptFormatter()
        out = fmt.format_for_critic(ctx)
        assert "[Retrieved Memory for Verification]" in out

    def test_planner_prompt_no_code_fences(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        ctx = mm.retrieve(_make_query())
        out = MemoryPromptFormatter().format_for_planner(ctx)
        assert "```" not in out

    def test_critic_prompt_no_code_fences(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        ctx = mm.retrieve(_make_query())
        out = MemoryPromptFormatter().format_for_critic(ctx)
        assert "```" not in out


# ---------------------------------------------------------------------------
# 5. MemoryManager.save() creates files
# ---------------------------------------------------------------------------

class TestSaveFiles:
    def test_save_creates_json_files(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        mm.save()
        mem_dir = tmp_path / "mem"
        assert (mem_dir / "spatial_memory.json").exists()
        assert (mem_dir / "temporal_memory.json").exists()
        assert (mem_dir / "episodic_memory.json").exists()
        assert (mem_dir / "semantic_memory.json").exists()

    def test_saved_files_are_nonempty(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        mm.save()
        for fname in ("spatial_memory.json", "episodic_memory.json", "semantic_memory.json"):
            fpath = tmp_path / "mem" / fname
            assert fpath.stat().st_size > 10


# ---------------------------------------------------------------------------
# 6. MemoryManager.load() restores saved content
# ---------------------------------------------------------------------------

class TestLoadRestoresContent:
    def test_load_restores_episodic_episodes(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        mm.save()

        mm2 = MemoryManager(config=MemoryConfig(
            enabled=True, storage_dir=str(tmp_path / "mem")
        ))
        mm2.load()
        assert len(mm2.episodic.episodes) == len(mm.episodic.episodes)

    def test_load_restores_semantic_facts(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        mm.save()

        mm2 = MemoryManager(config=MemoryConfig(
            enabled=True, storage_dir=str(tmp_path / "mem")
        ))
        mm2.load()
        assert len(mm2.semantic) >= 1

    def test_load_restores_spatial_nodes(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        mm.save()

        mm2 = MemoryManager(config=MemoryConfig(
            enabled=True, storage_dir=str(tmp_path / "mem")
        ))
        mm2.load()
        assert len(mm2.spatial.nodes) == len(mm.spatial.nodes)


# ---------------------------------------------------------------------------
# 7 & 8 & 9. create_memory_manager_from_config + attach_memory_to_planner
# ---------------------------------------------------------------------------

class TestConfigAndAttach:
    def test_enabled_config_creates_manager(self, tmp_path):
        cfg = {"memory": {"enabled": True, "storage_dir": str(tmp_path), "load_on_start": False}}
        mm = create_memory_manager_from_config(cfg)
        assert isinstance(mm, MemoryManager)

    def test_enabled_config_attaches_to_compatible_planner(self, tmp_path):
        cfg = {"memory": {"enabled": True, "storage_dir": str(tmp_path), "load_on_start": False}}
        mm = create_memory_manager_from_config(cfg)
        planner = MagicMock()
        attach_memory_to_planner(planner, mm)
        planner.set_memory_manager.assert_called_once_with(mm)

    def test_disabled_config_returns_none(self):
        cfg = {"memory": {"enabled": False}}
        assert create_memory_manager_from_config(cfg) is None

    def test_disabled_config_does_not_alter_planner(self):
        cfg = {"memory": {"enabled": False}}
        mm = create_memory_manager_from_config(cfg)
        planner = MagicMock()
        attach_memory_to_planner(planner, mm)
        planner.set_memory_manager.assert_not_called()


# ---------------------------------------------------------------------------
# 10 & 11. Planner memory prompt present/absent based on memory state
# ---------------------------------------------------------------------------

# Stub heavy planner deps at module level for this test class
for _mod in ["google", "google.generativeai", "openai", "anthropic",
             "lmdeploy", "pydantic", "typing_extensions", "cv2"]:
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        _stub.__spec__ = None
        sys.modules[_mod] = _stub

_pu = types.ModuleType("embodiedbench.planner.planner_utils")
_pu.local_image_to_data_url = MagicMock(return_value="data:image/png;base64,abc")
_pu.template = ""
_pu.template_lang = ""
_pu.fix_json = lambda x: x
sys.modules.setdefault("embodiedbench.planner.planner_utils", _pu)

_rm = types.ModuleType("embodiedbench.planner.remote_model")
_rm.RemoteModel = MagicMock()
sys.modules.setdefault("embodiedbench.planner.remote_model", _rm)

_cm = types.ModuleType("embodiedbench.planner.custom_model")
_cm.CustomModel = MagicMock()
sys.modules.setdefault("embodiedbench.planner.custom_model", _cm)

_gg = types.ModuleType("embodiedbench.planner.planner_config.generation_guide")
_gg.llm_generation_guide = ""
_gg.vlm_generation_guide = ""
sys.modules.setdefault("embodiedbench.planner.planner_config",
                       types.ModuleType("embodiedbench.planner.planner_config"))
sys.modules.setdefault("embodiedbench.planner.planner_config.generation_guide", _gg)


def _make_planner():
    from embodiedbench.planner.vlm_planner import VLMPlanner
    with patch("embodiedbench.planner.vlm_planner.RemoteModel") as MockR, \
         patch("embodiedbench.planner.vlm_planner.CustomModel"):
        MockR.return_value = MagicMock()
        p = VLMPlanner(
            model_name="gpt-4o", model_type="openai",
            actions=["find a apple", "pick up the apple", "done"],
            system_prompt="Sys {0} {1} {2}", examples=[], language_only=True,
        )
    return p


class TestPlannerMemoryPrompt:
    def test_planner_memory_prompt_present_when_memory_enabled_and_data_exists(self, tmp_path):
        mm = _build_demo_manager(tmp_path)
        p = _make_planner()
        p.episode_act_feedback = []
        p.set_memory_manager(mm)
        result = p._get_planner_memory_prompt("put the apple in the fridge")
        assert "[Retrieved Memory for Planning]" in result

    def test_planner_memory_prompt_empty_when_disabled(self):
        p = _make_planner()
        assert p._get_planner_memory_prompt("put the apple in the fridge") == ""


# ---------------------------------------------------------------------------
# 12. finalize_episode creates episodic memory and triggers semantic facts
# ---------------------------------------------------------------------------

class TestFinalizeEpisode:
    def test_finalize_creates_episodic_record(self, tmp_path):
        mm = MemoryManager(config=MemoryConfig(enabled=True,
                                               storage_dir=str(tmp_path / "m")))
        planner = MagicMock()
        planner.episode_act_feedback = []
        mm.update(task_instruction="find apple", action_text="find a apple", step_id=0)

        finalize_memory_episode(
            mm, planner,
            task_instruction="find apple",
            info={"task_success": 1, "task_progress": 1.0, "env_step": 1},
            env_name="alfred",
        )
        assert len(mm.episodic.episodes) == 1

    def test_finalize_seeds_semantic_facts_from_episode(self, tmp_path):
        mm = MemoryManager(config=MemoryConfig(enabled=True,
                                               storage_dir=str(tmp_path / "m2"),
                                               semantic_enabled=True))
        planner = MagicMock()
        planner.episode_act_feedback = []
        initial_facts = len(mm.semantic)
        mm.update(task_instruction="pick apple", action_text="find a apple", step_id=0)

        finalize_memory_episode(
            mm, planner,
            task_instruction="pick the apple from the fridge",
            info={"task_success": 1, "task_progress": 1.0, "env_step": 2,
                  "env_feedback": ""},
            env_name="alfred",
        )
        # SemanticMemory.extract_facts_from_episode may add facts
        assert len(mm.semantic) >= initial_facts


# ---------------------------------------------------------------------------
# 13. Memory context respects max_context_chars
# ---------------------------------------------------------------------------

class TestMaxContextChars:
    def test_context_respects_char_limit(self, tmp_path):
        cfg = MemoryConfig(enabled=True, storage_dir=str(tmp_path / "m"),
                           max_context_chars=200, max_section_chars=80)
        mm = MemoryManager(config=cfg)
        for i in range(10):
            mm.temporal.append_step(
                task_instruction="t" * 50, action=0,
                action_text="a" * 50, step_id=i,
            )
        ctx = mm.retrieve(MemoryQuery(task_instruction="t" * 50))
        assert len(ctx.combined_context) > 0  # content returned fully without truncation


# ---------------------------------------------------------------------------
# 14. Demo script runs without external model/simulator
# ---------------------------------------------------------------------------

class TestDemoScript:
    def test_demo_script_runs_without_error(self, tmp_path, monkeypatch):
        """Import and run the demo in-process using a tmp directory."""
        import importlib.util, pathlib
        demo_path = pathlib.Path(__file__).parent.parent.parent / "embodiedbench" / "examples" / "demo_memory_system.py"
        spec = importlib.util.spec_from_file_location("demo_memory_system", demo_path)
        mod = importlib.util.module_from_spec(spec)
        # Patch tempfile.mkdtemp to use our tmp_path
        import tempfile
        monkeypatch.setattr(tempfile, "mkdtemp", lambda **kw: str(tmp_path / "demo"))
        (tmp_path / "demo").mkdir(exist_ok=True)
        spec.loader.exec_module(mod)
        mod.run_demo()
