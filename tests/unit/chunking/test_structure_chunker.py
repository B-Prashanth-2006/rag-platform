"""Tests for StructureAwareChunker."""

from __future__ import annotations

import pytest

from rag.chunking.structure import StructureAwareChunker
from rag.chunking.tokenizer import WhitespaceTokenCounter
from rag.domain.entities import ChunkDraft, ExtractedDocument, ExtractedPage
from rag.domain.interfaces import Chunker


def test_structure_chunker_satisfies_protocol() -> None:
    counter = WhitespaceTokenCounter()
    chunker = StructureAwareChunker(token_counter=counter, chunk_size=50, chunk_overlap=10)
    assert isinstance(chunker, Chunker)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [
        (0, 0),
        (-5, 0),
        (10, 10),
        (10, 15),
        (10, -1),
    ],
)
def test_structure_chunker_invalid_params_raise_value_error(size: int, overlap: int) -> None:
    counter = WhitespaceTokenCounter()
    with pytest.raises(ValueError):
        StructureAwareChunker(token_counter=counter, chunk_size=size, chunk_overlap=overlap)


def test_structure_chunker_empty_document() -> None:
    counter = WhitespaceTokenCounter()
    chunker = StructureAwareChunker(token_counter=counter, chunk_size=20, chunk_overlap=5)

    assert chunker.chunk(ExtractedDocument(pages=[], metadata={})) == []
    assert (
        chunker.chunk(
            ExtractedDocument(
                pages=[ExtractedPage(number=1, text="", headings=[], tables=[])],
                metadata={},
            )
        )
        == []
    )


def test_structure_chunker_heading_section_path() -> None:
    counter = WhitespaceTokenCounter()
    chunker = StructureAwareChunker(token_counter=counter, chunk_size=50, chunk_overlap=5)

    text = (
        "# Introduction\n\n"
        "This is the opening text.\n\n"
        "## Background\n\n"
        "Historical context and overview.\n\n"
        "### Details\n\n"
        "Deep technical nuances.\n\n"
        "# Conclusion\n\n"
        "Final remarks."
    )
    doc = ExtractedDocument(
        pages=[
            ExtractedPage(
                number=1,
                text=text,
                headings=["Introduction", "Background", "Details", "Conclusion"],
                tables=[],
            )
        ],
        metadata={},
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3
    # Check section paths
    sections = [c.section for c in chunks]
    assert "Introduction" in sections
    assert (
        "Introduction > Background" in sections or "Introduction > Background > Details" in sections
    )
    assert "Conclusion" in sections


def test_structure_chunker_never_merges_across_top_level_headings() -> None:
    counter = WhitespaceTokenCounter()
    # Large chunk_size: if permitted, all text would fit in 1 chunk
    chunker = StructureAwareChunker(token_counter=counter, chunk_size=200, chunk_overlap=10)

    text = "# Chapter 1\n\nShort paragraph 1.\n\n# Chapter 2\n\nShort paragraph 2."
    doc = ExtractedDocument(
        pages=[ExtractedPage(number=1, text=text, headings=["Chapter 1", "Chapter 2"], tables=[])],
        metadata={},
    )
    chunks = chunker.chunk(doc)

    # Must NOT merge across Chapter 1 and Chapter 2
    assert len(chunks) == 2
    assert chunks[0].section == "Chapter 1"
    assert "Chapter 2" not in chunks[0].text
    assert chunks[1].section == "Chapter 2"
    assert "Chapter 1" not in chunks[1].text


def test_structure_chunker_oversized_paragraph_fallback() -> None:
    counter = WhitespaceTokenCounter()
    # Small chunk_size: 10 words, overlap 2
    chunker = StructureAwareChunker(token_counter=counter, chunk_size=10, chunk_overlap=2)

    # Paragraph alone has 25 words -> exceeds chunk_size 10
    oversized_para = " ".join(f"word{i}" for i in range(25))
    text = f"# Big Section\n\n{oversized_para}\n\nNormal paragraph."

    doc = ExtractedDocument(
        pages=[ExtractedPage(number=1, text=text, headings=["Big Section"], tables=[])],
        metadata={},
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) > 2
    for i, c in enumerate(chunks):
        assert isinstance(c, ChunkDraft)
        assert c.index == i
        assert 0 < c.token_count <= 10
        assert c.section == "Big Section"
        assert c.page == 1
