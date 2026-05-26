"""
memory/embeddings.py

Embedding providers and similarity utilities for the MemAdapt memory system.

Providers
---------
- HashEmbeddingProvider       — deterministic, offline, no GPU (CI/testing)
- DummyEmbeddingProvider      — zero-like vectors, placeholder only
- SentenceTransformerProvider — real semantic embeddings via sentence-transformers

Factory
-------
- resolve_embedding_provider(model_name)  — create the right provider by name
"""

from __future__ import annotations

import abc
import hashlib
import re
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity in [−1, 1]. Returns 0.0 for empty/mismatched vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(float(va @ vb) / (na * nb), -1.0, 1.0))


def lexical_overlap_score(query: str, document: str) -> float:
    """Jaccard token-overlap similarity in [0, 1]."""
    def _tokens(t: str) -> set:
        return {w for w in re.split(r'\W+', t.lower()) if w}
    q, d = _tokens(query), _tokens(document)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def hybrid_score(
    query_text: str,
    document_text: str,
    query_embedding: Optional[list] = None,
    document_embedding: Optional[list] = None,
    embedding_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> float:
    """
    Weighted blend of cosine similarity and lexical overlap.
    Falls back to pure lexical when embeddings are absent.
    """
    lexical = lexical_overlap_score(query_text, document_text)
    has_emb = query_embedding and document_embedding
    if not has_emb:
        return lexical
    total = embedding_weight + lexical_weight or 1.0
    return (embedding_weight / total) * cosine_similarity(query_embedding, document_embedding) \
         + (lexical_weight   / total) * lexical


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EmbeddingProvider(abc.ABC):
    """Interface for text embedding providers."""

    @abc.abstractmethod
    def embed_text(self, text: str) -> list:
        """Return a float vector for *text*."""

    def embed_batch(self, texts: list) -> list:
        """Embed a list of texts. Override for batched providers."""
        return [self.embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# HashEmbeddingProvider — deterministic, no dependencies
# ---------------------------------------------------------------------------

class HashEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic token-hash embedding. No GPU or network required.
    Semantically meaningless — suitable for CI/testing only.
    """

    def __init__(self, dim: int = 128, normalize: bool = True):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.normalize = normalize
        self._indices = np.arange(32, dtype=np.intp) % dim
        self._cache: dict[str, list] = {}

    def _token_vector(self, token: str) -> np.ndarray:
        digest = hashlib.sha256(token.encode()).digest()
        arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float64) / 127.5 - 1.0
        vec = np.zeros(self.dim, dtype=np.float64)
        np.add.at(vec, self._indices, arr)
        return vec

    def embed_text(self, text: str) -> list:
        if text in self._cache:
            return self._cache[text]
        tokens = [t for t in re.split(r'\W+', text.lower()) if t] or ["<empty>"]
        avg = np.stack([self._token_vector(t) for t in tokens]).mean(axis=0)
        if self.normalize:
            n = float(np.linalg.norm(avg))
            if n > 0.0:
                avg = avg / n
        result = avg.tolist()
        self._cache[text] = result
        return result


# ---------------------------------------------------------------------------
# DummyEmbeddingProvider — placeholder / testing
# ---------------------------------------------------------------------------

class DummyEmbeddingProvider(EmbeddingProvider):
    """Near-zero vectors. Use only as a placeholder when embeddings are unused."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed_text(self, text: str) -> list:
        vec = [0.0] * self.dim
        if text:
            vec[0] = float(len(text) % 256) / 255.0
        return vec


# ---------------------------------------------------------------------------
# SentenceTransformerProvider — real semantic embeddings
# ---------------------------------------------------------------------------

# Short alias → full HuggingFace model ID
_MODEL_ALIASES: dict[str, str] = {
    "bge-large-en-v1.5":     "BAAI/bge-large-en-v1.5",
    "nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
    "bge-m3":                "BAAI/bge-m3",
    "all-MiniLM-L6-v2":     "sentence-transformers/all-MiniLM-L6-v2",
    "e5-large-v2":           "intfloat/e5-large-v2",
}


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Semantic embeddings via sentence-transformers.

    Accepts short aliases (e.g. ``"bge-large-en-v1.5"``) or full HuggingFace
    model IDs; see ``_MODEL_ALIASES`` for the mapping table.
    Install: ``pip install sentence-transformers``
    """

    def __init__(
        self,
        model_name: str = "bge-large-en-v1.5",
        device: str = "cuda",
        batch_size: int = 32,
        normalize: bool = True,
        trust_remote_code: bool = True,
    ):
        # Suppress the HuggingFace tokenizers fork/deadlock warning that fires
        # when DataLoader workers are active alongside sentence-transformers.
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install: pip install sentence-transformers"
            ) from exc

        resolved = _MODEL_ALIASES.get(model_name, model_name)
        self.model_name = resolved
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = SentenceTransformer(resolved, device=device, trust_remote_code=trust_remote_code)
        self._cache: dict[str, list] = {}

    def embed_text(self, text: str) -> list:
        if text not in self._cache:
            self._cache[text] = self._model.encode(
                text, normalize_embeddings=self.normalize, batch_size=1,
                show_progress_bar=False,
            ).tolist()
        return self._cache[text]

    def embed_batch(self, texts: list) -> list:
        results: list = [None] * len(texts)
        uncached = [(i, t) for i, t in enumerate(texts) if t not in self._cache]

        if uncached:
            indices, raw = zip(*uncached)
            vecs = self._model.encode(
                list(raw), normalize_embeddings=self.normalize, batch_size=self.batch_size,
                show_progress_bar=False,
            )
            for idx, text, vec in zip(indices, raw, vecs):
                self._cache[text] = vec.tolist()

        for i, t in enumerate(texts):
            results[i] = self._cache[t]
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def resolve_embedding_provider(
    model_name: Optional[str],
    use_embeddings: bool = True,
    **kwargs,
) -> Optional[EmbeddingProvider]:
    """
    Create an ``EmbeddingProvider`` from a name string.

    ``None``/``"hash"`` → ``HashEmbeddingProvider``; ``"dummy"`` →
    ``DummyEmbeddingProvider``; any other string (short alias or full
    HuggingFace ID) → ``SentenceTransformerProvider``.
    Returns ``None`` when ``use_embeddings=False``.
    """
    if not use_embeddings:
        return None
    name = (model_name or "").strip()
    if not name or name == "hash":
        return HashEmbeddingProvider()
    if name == "dummy":
        return DummyEmbeddingProvider()
    return SentenceTransformerProvider(model_name=name, **kwargs)
