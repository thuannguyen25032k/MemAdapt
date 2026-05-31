"""
memory/manager.py

MemoryManager — top-level coordinator for all four memory modules.

Initialises SpatialMemory, TemporalMemory, EpisodicMemory, SemanticMemory
from a single ``MemoryConfig``; updates all enabled memories every step;
retrieves and combines results into a ``MemoryContext``; finalises episodes;
saves/loads all modules; and exposes a lightweight stats dict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

from embodiedbench.memory.base import (
    MemoryContext,
    MemoryQuery,
)
from embodiedbench.memory.embeddings import EmbeddingProvider, resolve_embedding_provider
from embodiedbench.memory.spatial_memory import SpatialMemory
from embodiedbench.memory.temporal_memory import TemporalMemory
from embodiedbench.memory.episodic_memory import EpisodicMemory
from embodiedbench.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MemoryConfig
# ---------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    enabled: bool = True
    spatial_enabled: bool = True
    temporal_enabled: bool = True
    episodic_enabled: bool = True
    semantic_enabled: bool = True
    storage_dir: str = "./memory_store"
    top_k_per_memory: int = 5
    temporal_max_steps: int = 20
    use_embeddings: bool = False
    embedding_model: str = ""   # e.g. "bge-large-en-v1.5" or "nomic-embed-text-v1.5"
    auto_save: bool = False

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_mapping(cls, mapping: Any) -> "MemoryConfig":
        """
        Construct from:
        - dict
        - Hydra/OmegaConf DictConfig-like object (has __iter__ + __getitem__)
        - dataclass-like object (has __dataclass_fields__)
        - None → return defaults

        Unknown keys are silently ignored.
        """
        if mapping is None:
            return cls()
        if isinstance(mapping, cls):
            return mapping
        if isinstance(mapping, dict):
            return cls.from_dict(mapping)

        # Hydra DictConfig or similar mapping-like object
        try:
            as_dict = dict(mapping)
            return cls.from_dict(as_dict)
        except (TypeError, ValueError):
            pass

        # dataclass-like object
        try:
            as_dict = {f.name: getattr(mapping, f.name) for f in fields(mapping)}
            return cls.from_dict(as_dict)
        except Exception:
            pass

        # attribute-based object (generic cfg)
        valid = {f.name for f in fields(cls)}
        kwargs = {}
        for name in valid:
            if hasattr(mapping, name):
                kwargs[name] = getattr(mapping, name)
        return cls(**kwargs) if kwargs else cls()


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

# Minimum retrieval score for an episodic result to be included in context.
# Irrelevant trajectories anchor the planner to wrong locations.
_EPISODIC_MIN_SCORE = 0.30


class MemoryManager:
    """
    Coordinates SpatialMemory, TemporalMemory, EpisodicMemory, SemanticMemory.

    Typical per-episode usage::

        manager.load()
        manager.reset_episode()
        for step in episode:
            manager.update(...)
            ctx = manager.retrieve(query)
        manager.finalize_episode(...)
        manager.save()
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        spatial_memory: Optional[SpatialMemory] = None,
        temporal_memory: Optional[TemporalMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        vlm_call=None,
    ):
        self.config: MemoryConfig = MemoryConfig.from_mapping(config)
        cfg = self.config
        self._vlm_call = vlm_call

        # Embedding provider
        if embedding_provider is not None:
            self.embedding_provider: Optional[EmbeddingProvider] = embedding_provider
        else:
            self.embedding_provider = resolve_embedding_provider(
                model_name=cfg.embedding_model if cfg.use_embeddings else None,
                use_embeddings=cfg.use_embeddings,
            )

        storage = Path(cfg.storage_dir)

        # Spatial
        if spatial_memory is not None:
            self.spatial = spatial_memory
        elif cfg.spatial_enabled:
            self.spatial = SpatialMemory(
                embedding_provider=self.embedding_provider,
                storage_path=str(storage / "spatial_memory.json"),
            )
        else:
            self.spatial = None

        # Temporal
        if temporal_memory is not None:
            self.temporal = temporal_memory
        elif cfg.temporal_enabled:
            self.temporal = TemporalMemory(
                max_steps=cfg.temporal_max_steps,
                embedding_provider=self.embedding_provider,
                storage_path=str(storage / "temporal_memory.json"),
            )
        else:
            self.temporal = None

        # Episodic
        if episodic_memory is not None:
            self.episodic = episodic_memory
        elif cfg.episodic_enabled:
            self.episodic = EpisodicMemory(
                embedding_provider=self.embedding_provider,
                storage_path=str(storage / "episodic_memory.json"),
                vlm_call=self._vlm_call,
            )
        else:
            self.episodic = None

        # Semantic
        if semantic_memory is not None:
            self.semantic = semantic_memory
        elif cfg.semantic_enabled:
            self.semantic = SemanticMemory(
                embedding_provider=self.embedding_provider,
                storage_path=str(storage / "semantic_memory.json"),
                vlm_call=self._vlm_call,
            )
        else:
            self.semantic = None

    # ------------------------------------------------------------------
    # is_enabled
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(
        self,
        task_instruction: str = "",
        info: Optional[dict] = None,
        metadata: Optional[dict] = None,
        action: Optional[Any] = None,
        action_text: str = "",
        env_feedback: str = "",
        success: Optional[bool] = None,
        planner_output: Optional[str] = None,
        critic_output: Optional[str] = None,
        critic_rejected: bool = False,
        step_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        if not self.config.enabled:
            return

        # Temporal memory — every step
        if self.temporal is not None:
            self.temporal.append_step(
                task_instruction=task_instruction,
                action=action,
                action_text=action_text,
                env_feedback=env_feedback,
                success=success,
                planner_output=planner_output,
                critic_output=critic_output,
                critic_rejected=critic_rejected,
                info=info or {},
                step_id=step_id,
            )

        if self.spatial is not None:
            self.spatial.update(metadata=metadata)

        if self.config.auto_save:
            self.save()

    # ------------------------------------------------------------------
    # retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: MemoryQuery,
        top_k_per_memory: Optional[int] = None,
    ) -> MemoryContext:
        if not self.config.enabled:
            return MemoryContext()
        k = top_k_per_memory if top_k_per_memory is not None else self.config.top_k_per_memory
        ctx = MemoryContext()
        all_items: list = []

        # --- Spatial ---
        if self.spatial is not None:
            spatial_results = self.spatial.retrieve(query, top_k=15)
            if spatial_results:
                ctx.spatial_context = self.spatial.to_prompt_context(
                    spatial_results
                )
                all_items.extend(spatial_results)

        # --- Temporal ---
        if self.temporal is not None:
            temporal_results = self.temporal.retrieve(query, top_k=k)
            if temporal_results:
                ctx.temporal_context = self.temporal.to_prompt_context(
                    temporal_results
                )
                all_items.extend(temporal_results)

        # --- Episodic ---
        # Only inject episodes above the relevance threshold; irrelevant
        # trajectories anchor the planner to wrong locations.
        if self.episodic is not None:
            episodic_results = self.episodic.retrieve(query, top_k=k)
            episodic_results = [r for r in episodic_results if r.score >= _EPISODIC_MIN_SCORE]
            if episodic_results:
                ctx.episodic_context = self.episodic.to_prompt_context(
                    episodic_results
                )
                all_items.extend(episodic_results)

        # --- Semantic ---
        if self.semantic is not None:
            semantic_results = self.semantic.retrieve(query, top_k=k)
            if semantic_results:
                ctx.semantic_context = self.semantic.to_prompt_context(
                    semantic_results
                )
                all_items.extend(semantic_results)

        ctx.retrieved_items = all_items

        return ctx

    # ------------------------------------------------------------------
    # build_memory_context — convenience wrapper
    # ------------------------------------------------------------------

    def build_memory_context(
        self,
        query: Optional[MemoryQuery] = None,
        task_instruction: str = "",
        recent_actions: Optional[list] = None,
        env_name: Optional[str] = None,
        task_type: Optional[str] = None,
        scene_name: Optional[str] = None,
        top_k_per_memory: Optional[int] = None,
    ) -> MemoryContext:
        if query is None:
            query = MemoryQuery(
                task_instruction=task_instruction,
                recent_actions=list(recent_actions or []),
                env_name=env_name,
                task_type=task_type,
                scene_name=scene_name,
            )
        return self.retrieve(query, top_k_per_memory=top_k_per_memory)

    # ------------------------------------------------------------------
    # finalize_episode
    # ------------------------------------------------------------------

    def finalize_episode(
        self,
        task_instruction: str,
        final_status: str = "unknown",
        env_name: str = "",
        task_type: str = "",
    ) -> Optional[Any]:
        if not self.config.enabled:
            return None

        raw_steps: list = []
        trajectory_summary = ""
        if self.temporal is not None:
            for s in self.temporal.steps:
                feedback = s.env_feedback or ""
                # if "invalid" in feedback.lower():
                #     continue   # skip invalid actions
                raw_steps.append({
                    "step_id": s.step_id,
                    "action":  s.action_text or str(s.action or ""),
                    "feedback": feedback,
                })
            trajectory_summary = self.temporal.summarize_recent_history()

        episode = None

        # Add to episodic memory
        if self.episodic is not None:
            # Derive scene_name from spatial memory (AI2-THOR sceneName) if available.
            scene_name = ""
            if self.spatial is not None:
                names = {n.scene for n in self.spatial.nodes.values() if n.scene}
                scene_name = next(iter(names), "")
            episode = self.episodic.add_episode_from_trajectory(
                task_instruction=task_instruction,
                final_status=final_status,
                steps=raw_steps,
                env_name=env_name,
                task_type=task_type,
                scene_name=scene_name,
            )

        # Extract semantic facts via VLM
        if self.semantic is not None:
            self.semantic.update_from_episode(
                episode_summary=trajectory_summary or task_instruction,
                task_instruction=task_instruction,
                success=(final_status == "success"),
                episode_id=episode.id if episode is not None else None,
            )

        if self.config.auto_save:
            self.save()

        return episode

    # ------------------------------------------------------------------
    # reset_episode
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Clear per-episode state ready for the next episode."""
        if self.spatial is not None:
            self.spatial.reset_episode()
        if self.temporal is not None:
            self.temporal.reset_episode()
        # episodic / semantic are long-lived — no reset

    def initialize_episode(self, metadata: Optional[dict] = None) -> None:
        """Seed the spatial graph at the start of an episode.

        Call this immediately after ``env.reset()`` with the dict returned by
        ``env.get_metadata()``.  It calls ``SpatialMemory.build_from_metadata``
        to register all known objects and receptacles so that
        ``update`` can build containment relations as the episode
        progresses.  Safe no-op when spatial memory is disabled or metadata is
        None/empty.
        """
        if self.spatial is None:
            return
        if not metadata:
            return
        try:
            self.spatial.build_from_metadata(metadata)
        except Exception as exc:
            logger.warning(
                "MemoryManager.initialize_episode: build_from_metadata failed: %s", exc
            )

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self) -> None:
        Path(self.config.storage_dir).mkdir(parents=True, exist_ok=True)
        for mem in (self.spatial, self.temporal, self.episodic, self.semantic):
            if mem is not None:
                try:
                    mem.save()
                except Exception:
                    pass

    def load(self) -> None:
        for mem in (self.spatial, self.temporal, self.episodic, self.semantic):
            if mem is not None:
                try:
                    mem.load()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "spatial_enabled": self.spatial is not None,
            "temporal_enabled": self.temporal is not None,
            "episodic_enabled": self.episodic is not None,
            "semantic_enabled": self.semantic is not None,
            "spatial_nodes": len(self.spatial.nodes) if self.spatial else 0,
            "spatial_relations": len(self.spatial.relations) if self.spatial else 0,
            "temporal_steps": len(self.temporal) if self.temporal else 0,
            "temporal_summaries": len(self.temporal.summaries) if self.temporal else 0,
            "episodic_episodes": len(self.episodic.episodes) if self.episodic else 0,
            "semantic_facts": len(self.semantic) if self.semantic else 0,
        }
