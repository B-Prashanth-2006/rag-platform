"""Factory for instantiating embedders from configuration settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.domain.interfaces import Embedder
from rag.embeddings.cache import CachedEmbedder
from rag.embeddings.hf import HuggingFaceEmbedder

if TYPE_CHECKING:
    from rag.config import Settings


def get_model_family(model_name: str) -> str:
    """Identify the model family from the model name or raise ValueError."""
    lower = model_name.lower()
    if "bge" in lower:
        return "bge"
    if "e5" in lower:
        return "e5"
    raise ValueError(f"Unknown model family for model: {model_name}")


class EmbedderFactory:
    """Instantiates and optionally wraps an Embedder with caching."""

    @staticmethod
    def create(
        settings: Settings,
        inner_embedder: Embedder | None = None,
    ) -> Embedder:
        """Create an Embedder matching ``settings``."""
        if inner_embedder is None:
            # Validates model family (raises ValueError on unknown)
            get_model_family(settings.embedding_model)

            embedder: Embedder = HuggingFaceEmbedder(
                model_name=settings.embedding_model,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
            )
        else:
            embedder = inner_embedder

        if settings.embedding_cache_enabled:
            return CachedEmbedder(
                embedder=embedder,
                cache_path=settings.embedding_cache_path,
            )

        return embedder
