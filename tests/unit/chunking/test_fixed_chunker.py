"""Tests for FixedWindowChunker."""

from __future__ import annotations

import pytest

from rag.chunking.fixed import FixedWindowChunker
from rag.chunking.tokenizer import WhitespaceTokenCounter
from rag.domain.entities import ChunkDraft, ExtractedDocument, ExtractedPage
from rag.domain.interfaces import Chunker


def test_fixed_chunker_satisfies_protocol() -> None:
    counter = WhitespaceTokenCounter()
    chunker = FixedWindowChunker(token_counter=counter, chunk_size=10, chunk_overlap=2)
    assert isinstance(chunker, Chunker)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [
        (0, 0),
        (-5, 0),
        (10, 10),
        (10, 12),
        (10, -1),
    ],
)
def test_fixed_chunker_invalid_params_raise_value_error(size: int, overlap: int) -> None:
    counter = WhitespaceTokenCounter()
    with pytest.raises(ValueError):
        FixedWindowChunker(token_counter=counter, chunk_size=size, chunk_overlap=overlap)


def test_fixed_chunker_empty_document() -> None:
    counter = WhitespaceTokenCounter()
    chunker = FixedWindowChunker(token_counter=counter, chunk_size=10, chunk_overlap=2)

    doc_no_pages = ExtractedDocument(pages=[], metadata={})
    assert chunker.chunk(doc_no_pages) == []

    doc_empty_pages = ExtractedDocument(
        pages=[ExtractedPage(number=1, text="", headings=[], tables=[])],
        metadata={},
    )
    assert chunker.chunk(doc_empty_pages) == []


def test_fixed_chunker_single_chunk() -> None:
    counter = WhitespaceTokenCounter()
    chunker = FixedWindowChunker(token_counter=counter, chunk_size=10, chunk_overlap=2)

    doc = ExtractedDocument(
        pages=[ExtractedPage(number=1, text="one two three four five", headings=[], tables=[])],
        metadata={},
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].token_count == 5
    assert chunks[0].page == 1
    assert chunks[0].section is None
    assert chunks[0].text == "one two three four five"


def test_fixed_chunker_multi_page_attribution() -> None:
    counter = WhitespaceTokenCounter()
    # chunk_size=5, overlap=1 -> step=4
    chunker = FixedWindowChunker(token_counter=counter, chunk_size=5, chunk_overlap=1)

    # Page 1 has 6 words (tokens 0..5)
    # Page 2 has 6 words (tokens 6..11)
    doc = ExtractedDocument(
        pages=[
            ExtractedPage(
                number=1, text="p1_w0 p1_w1 p1_w2 p1_w3 p1_w4 p1_w5", headings=[], tables=[]
            ),
            ExtractedPage(
                number=2, text="p2_w6 p2_w7 p2_w8 p2_w9 p2_w10 p2_w11", headings=[], tables=[]
            ),
        ],
        metadata={},
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    # First chunk starts on page 1
    assert chunks[0].page == 1
    assert chunks[0].index == 0

    # Chunks are contiguous
    for i, c in enumerate(chunks):
        assert isinstance(c, ChunkDraft)
        assert c.index == i
        assert 0 < c.token_count <= 5

    # Later chunk starting after page 1 words should be attributed to page 2
    last_chunk = chunks[-1]
    assert last_chunk.page == 2


def test_fixed_chunker_deterministic() -> None:
    counter = WhitespaceTokenCounter()
    chunker = FixedWindowChunker(token_counter=counter, chunk_size=8, chunk_overlap=2)
    words = " ".join(f"word{i}" for i in range(50))
    doc = ExtractedDocument(
        pages=[ExtractedPage(number=1, text=words, headings=[], tables=[])],
        metadata={},
    )

    chunks1 = chunker.chunk(doc)
    chunks2 = chunker.chunk(doc)
    assert chunks1 == chunks2
