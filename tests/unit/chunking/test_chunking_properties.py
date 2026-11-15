"""Property-style randomized tests for chunking invariants (fixed seed, no hypothesis)."""

from __future__ import annotations

import random

import pytest

from rag.chunking.fixed import FixedWindowChunker
from rag.chunking.structure import StructureAwareChunker
from rag.chunking.tokenizer import WhitespaceTokenCounter
from rag.domain.entities import ExtractedDocument, ExtractedPage


def _generate_random_text(rng: random.Random, num_words: int) -> str:
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    return " ".join(rng.choice(vocab) for _ in range(num_words))


@pytest.mark.parametrize("seed", [42, 101, 2024, 777, 9999])
def test_fixed_chunker_invariants_and_properties(seed: int) -> None:
    rng = random.Random(seed)
    counter = WhitespaceTokenCounter()

    chunk_size = rng.randint(10, 50)
    chunk_overlap = rng.randint(1, chunk_size - 1)
    chunker = FixedWindowChunker(
        token_counter=counter, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    num_words = rng.randint(5, 200)
    text = _generate_random_text(rng, num_words)
    doc = ExtractedDocument(
        pages=[ExtractedPage(number=1, text=text, headings=[], tables=[])],
        metadata={},
    )

    chunks = chunker.chunk(doc)

    # 1. Contiguous indexing 0..n-1
    assert [c.index for c in chunks] == list(range(len(chunks)))

    # 2. Token size constraints
    for c in chunks:
        assert 0 < c.token_count <= chunk_size

    # 3. Determinism
    chunks_again = chunker.chunk(doc)
    assert chunks == chunks_again

    # 4. Overlap between consecutive chunks is exactly chunk_overlap
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            curr_tokens = counter.encode(chunks[i].text)
            next_tokens = counter.encode(chunks[i + 1].text)
            overlap_tail = curr_tokens[-chunk_overlap:]
            overlap_head = next_tokens[:chunk_overlap]
            assert overlap_tail == overlap_head, f"Overlap mismatch between chunk {i} and {i + 1}"

    # 5. No text lost: joining chunks with overlaps removed reproduces the full normalized text
    if chunks:
        reconstructed_tokens = counter.encode(chunks[0].text)
        for c in chunks[1:]:
            c_tokens = counter.encode(c.text)
            reconstructed_tokens.extend(c_tokens[chunk_overlap:])
        reconstructed_text = counter.decode(reconstructed_tokens)
        assert reconstructed_text == text


@pytest.mark.parametrize("seed", [42, 101, 2024, 777, 9999])
def test_structure_chunker_invariants_and_properties(seed: int) -> None:
    rng = random.Random(seed)
    counter = WhitespaceTokenCounter()

    chunk_size = rng.randint(15, 60)
    chunk_overlap = rng.randint(1, min(10, chunk_size - 1))
    chunker = StructureAwareChunker(
        token_counter=counter, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Generate document with multiple headings and paragraphs
    paragraphs = [_generate_random_text(rng, rng.randint(3, 40)) for _ in range(5)]
    text = (
        f"# Heading One\n\n{paragraphs[0]}\n\n"
        f"## Subheading\n\n{paragraphs[1]}\n\n{paragraphs[2]}\n\n"
        f"# Heading Two\n\n{paragraphs[3]}\n\n{paragraphs[4]}"
    )

    doc = ExtractedDocument(
        pages=[
            ExtractedPage(
                number=1,
                text=text,
                headings=["Heading One", "Subheading", "Heading Two"],
                tables=[],
            )
        ],
        metadata={},
    )

    chunks = chunker.chunk(doc)

    # 1. Contiguous indexing 0..n-1
    assert [c.index for c in chunks] == list(range(len(chunks)))

    # 2. Token size constraints
    for c in chunks:
        assert 0 < c.token_count <= chunk_size

    # 3. Determinism
    chunks_again = chunker.chunk(doc)
    assert chunks == chunks_again

    # 4. No text lost: every word of input appears in some chunk
    all_chunk_words = set()
    for c in chunks:
        all_chunk_words.update(c.text.split())

    for p in paragraphs:
        for word in p.split():
            assert word in all_chunk_words, f"Word {word!r} missing from chunks"
