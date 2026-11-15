"""Factory for creating chunkers based on application settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.chunking.fixed import FixedWindowChunker
from rag.chunking.structure import StructureAwareChunker
from rag.chunking.tokenizer import HuggingFaceTokenCounter
from rag.domain.interfaces import Chunker, TokenCounter

if TYPE_CHECKING:
    from rag.config import Settings


class ChunkerFactory:
    """Instantiates a Chunker according to configuration settings."""

    @staticmethod
    def create(settings: Settings, token_counter: TokenCounter | None = None) -> Chunker:
        """Create and return a Chunker matching ``settings.chunker_strategy``."""
        counter = token_counter or HuggingFaceTokenCounter(model_name=settings.tokenizer_model)

        if settings.chunker_strategy == "fixed":
            return FixedWindowChunker(
                token_counter=counter,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        if settings.chunker_strategy == "structure":
            return StructureAwareChunker(
                token_counter=counter,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        raise ValueError(f"Unknown chunker strategy: {settings.chunker_strategy}")
