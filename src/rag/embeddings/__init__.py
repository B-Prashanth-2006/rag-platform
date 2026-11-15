"""Embeddings package: models, caching, fake, and factory."""

from __future__ import annotations

from rag.embeddings.cache import CachedEmbedder
from rag.embeddings.factory import EmbedderFactory, get_model_family
from rag.embeddings.fake import FakeEmbedder
from rag.embeddings.hf import HuggingFaceEmbedder, resolve_embedding_device

__all__ = [
    "CachedEmbedder",
    "EmbedderFactory",
    "FakeEmbedder",
    "HuggingFaceEmbedder",
    "get_model_family",
    "resolve_embedding_device",
]
