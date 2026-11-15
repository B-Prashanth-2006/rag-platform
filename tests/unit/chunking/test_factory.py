"""Tests for ChunkerFactory."""

from __future__ import annotations

import pytest

from rag.chunking.factory import ChunkerFactory
from rag.chunking.fixed import FixedWindowChunker
from rag.chunking.structure import StructureAwareChunker
from rag.chunking.tokenizer import WhitespaceTokenCounter
from rag.config import Settings


def test_chunker_factory_fixed() -> None:
    settings = Settings(chunker_strategy="fixed", chunk_size=128, chunk_overlap=16)
    counter = WhitespaceTokenCounter()
    chunker = ChunkerFactory.create(settings=settings, token_counter=counter)

    assert isinstance(chunker, FixedWindowChunker)
    assert chunker.chunk_size == 128
    assert chunker.chunk_overlap == 16


def test_chunker_factory_structure() -> None:
    settings = Settings(chunker_strategy="structure", chunk_size=300, chunk_overlap=30)
    counter = WhitespaceTokenCounter()
    chunker = ChunkerFactory.create(settings=settings, token_counter=counter)

    assert isinstance(chunker, StructureAwareChunker)
    assert chunker.chunk_size == 300
    assert chunker.chunk_overlap == 30


def test_chunker_factory_unknown_strategy_raises_value_error() -> None:
    settings = Settings(chunker_strategy="semantic_magic")
    counter = WhitespaceTokenCounter()
    with pytest.raises(ValueError, match="Unknown chunker strategy"):
        ChunkerFactory.create(settings=settings, token_counter=counter)
