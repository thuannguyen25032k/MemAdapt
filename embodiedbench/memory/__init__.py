"""
embodiedbench/memory
====================

MemAdapt memory foundation package.

Public API
----------
Data models:
    MemoryItem, MemoryQuery, RetrievedMemory, MemoryContext

Abstract base:
    BaseMemory

Memory modules:
    SpatialMemory, TemporalMemory, EpisodicMemory, SemanticMemory

Manager:
    MemoryManager, MemoryConfig

Embedding providers:
    EmbeddingProvider, HashEmbeddingProvider, DummyEmbeddingProvider

Integration helpers:
    setup_memory_experiment, finalize_memory_episode, save_memory_if_configured, …
"""

from embodiedbench.memory.base import (
    BaseMemory,
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
    EmbeddingProvider,
    HashEmbeddingProvider,
    cosine_similarity,
    hybrid_score,
    lexical_overlap_score,
)
from embodiedbench.memory.temporal_memory import TemporalMemory, TemporalStep
from embodiedbench.memory.semantic_memory import SemanticMemory, SemanticFact
from embodiedbench.memory.episodic_memory import EpisodicMemory, EpisodeRecord
from embodiedbench.memory.spatial_memory import SpatialMemory, SpatialNode, SpatialRelation
from embodiedbench.memory.manager import MemoryManager, MemoryConfig
from embodiedbench.memory.prompt_formatter import MemoryPromptFormatter

# Integration helpers depend on ML libraries (transformers/torch).
# Guard so core-only imports (e.g. unit tests) never break.
try:
    from embodiedbench.memory.integration import (
        create_memory_manager_from_config,
        attach_memory_to_planner,
        attach_memory_to_critic,
        finalize_memory_episode,
        save_memory_if_configured,
        compute_final_status,
        create_memory_adapter_from_config,
        attach_memory_adapter_to_planner,
        attach_memory_adapter_to_critic,
        unload_memory_adapter,
        setup_memory_experiment,
        create_metrics_from_config,
        attach_metrics_to_planner,
        attach_metrics_to_critic,
        collect_episode_metrics,
        create_logger_from_config,
    )
    from embodiedbench.memory.metrics import MemoryExperimentMetrics
    from embodiedbench.memory.logging import MemoryEpisodeLog, MemoryExperimentLogger
except Exception:  # pragma: no cover
    pass

__all__ = [
    # data models
    "MemoryItem",
    "MemoryQuery",
    "RetrievedMemory",
    "MemoryContext",
    # abstract base
    "BaseMemory",
    # embedding providers
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "DummyEmbeddingProvider",
    # scoring utilities
    "cosine_similarity",
    "lexical_overlap_score",
    "hybrid_score",
    # memory modules
    "TemporalMemory",
    "TemporalStep",
    "SemanticMemory",
    "SemanticFact",
    "EpisodicMemory",
    "EpisodeRecord",
    "SpatialMemory",
    "SpatialNode",
    "SpatialRelation",
    # memory manager
    "MemoryManager",
    "MemoryConfig",
    # prompt formatter
    "MemoryPromptFormatter",
    # helper utilities
    "now_ts",
    "normalize_text",
    "safe_json_dumps",
    "truncate_text",
    # integration helpers
    "create_memory_manager_from_config",
    "attach_memory_to_planner",
    "attach_memory_to_critic",
    "finalize_memory_episode",
    "save_memory_if_configured",
    "compute_final_status",
    # memory adapter lifecycle helpers
    "create_memory_adapter_from_config",
    "attach_memory_adapter_to_planner",
    "attach_memory_adapter_to_critic",
    "unload_memory_adapter",
    "setup_memory_experiment",
    # metrics helpers
    "MemoryExperimentMetrics",
    "create_metrics_from_config",
    "attach_metrics_to_planner",
    "attach_metrics_to_critic",
    "collect_episode_metrics",
    # logging helpers
    "MemoryEpisodeLog",
    "MemoryExperimentLogger",
    "create_logger_from_config",
]
