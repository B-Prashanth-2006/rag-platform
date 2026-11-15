"""Tests for HuggingFaceEmbedder."""

from __future__ import annotations

import math

import pytest

from rag.domain.interfaces import Embedder
from rag.embeddings.hf import HuggingFaceEmbedder


def _dot(v1: list[float], v2: list[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2, strict=True))


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _get_hf_embedder() -> HuggingFaceEmbedder:
    try:
        return HuggingFaceEmbedder(model_name="BAAI/bge-small-en-v1.5", device="cpu", batch_size=4)
    except (OSError, Exception) as exc:
        pytest.skip(f"HuggingFace model download failed or offline: {exc}")


def test_hf_embedder_satisfies_protocol() -> None:
    embedder = _get_hf_embedder()
    assert isinstance(embedder, Embedder)
    assert embedder.info.name == "BAAI/bge-small-en-v1.5"
    assert embedder.info.dimension == 384
    assert embedder.info.max_tokens > 0
    assert embedder.info.version != ""


def test_hf_embedder_semantic_sanity() -> None:
    embedder = _get_hf_embedder()
    query = "A cat sits on the mat"
    passage_similar = "A kitten rests on a rug"
    passage_unrelated = "Quarterly revenue grew by 8 percent"

    q_vec = embedder.embed_query(query)
    similar_vec = embedder.embed_passages([passage_similar])[0]
    unrelated_vec = embedder.embed_passages([passage_unrelated])[0]

    sim_score = _dot(q_vec, similar_vec)
    unrelated_score = _dot(q_vec, unrelated_vec)

    assert sim_score > unrelated_score


def test_hf_embedder_unit_norm() -> None:
    embedder = _get_hf_embedder()
    texts = ["First passage", "Second passage", ""]
    vectors = embedder.embed_passages(texts)
    for vec in vectors:
        assert math.isclose(_norm(vec), 1.0, abs_tol=1e-5)

    q_vec = embedder.embed_query("query text")
    assert math.isclose(_norm(q_vec), 1.0, abs_tol=1e-5)


def test_hf_embedder_batching_matches_one_by_one() -> None:
    embedder = _get_hf_embedder()
    texts = [
        "Machine learning models need evaluation.",
        "Retrieval Augmented Generation combines search and LLMs.",
        "Deep learning architectures are powerful.",
        "Vector databases store dense embeddings.",
        "Python is a popular programming language.",
    ]
    batched_vectors = embedder.embed_passages(texts)
    one_by_one_vectors = [embedder.embed_passages([t])[0] for t in texts]

    assert len(batched_vectors) == len(one_by_one_vectors)
    for b_vec, o_vec in zip(batched_vectors, one_by_one_vectors, strict=True):
        assert len(b_vec) == len(o_vec)
        for b_val, o_val in zip(b_vec, o_vec, strict=True):
            assert math.isclose(b_val, o_val, abs_tol=1e-4)


def test_hf_embedder_query_and_passage_differ() -> None:
    embedder = _get_hf_embedder()
    text = "Artificial intelligence safety research"
    q_vec = embedder.embed_query(text)
    p_vec = embedder.embed_passages([text])[0]
    assert q_vec != p_vec


def test_hf_embedder_empty_list_and_empty_string() -> None:
    embedder = _get_hf_embedder()
    assert embedder.embed_passages([]) == []

    # Empty string should not crash and should return a unit norm vector
    vectors = embedder.embed_passages(["", "   "])
    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 384
        assert math.isclose(_norm(vec), 1.0, abs_tol=1e-5)
