"""Tests for FakeEmbedder."""

from __future__ import annotations

import math

from rag.domain.entities import EmbeddingModelInfo
from rag.domain.interfaces import Embedder
from rag.embeddings.fake import FakeEmbedder


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def test_fake_embedder_satisfies_protocol() -> None:
    embedder = FakeEmbedder()
    assert isinstance(embedder, Embedder)
    assert isinstance(embedder.info, EmbeddingModelInfo)
    assert embedder.info.dimension == 384
    assert embedder.info.max_tokens > 0


def test_fake_embedder_unit_norm() -> None:
    embedder = FakeEmbedder()
    passages = ["hello world", "test sentence", ""]
    vectors = embedder.embed_passages(passages)
    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 384
        assert math.isclose(_norm(vec), 1.0, abs_tol=1e-5)

    query_vec = embedder.embed_query("search query")
    assert len(query_vec) == 384
    assert math.isclose(_norm(query_vec), 1.0, abs_tol=1e-5)


def test_fake_embedder_empty_list() -> None:
    embedder = FakeEmbedder()
    assert embedder.embed_passages([]) == []


def test_fake_embedder_determinism() -> None:
    embedder1 = FakeEmbedder()
    embedder2 = FakeEmbedder()
    text = "deterministic embedding text"
    assert embedder1.embed_passages([text]) == embedder2.embed_passages([text])
    assert embedder1.embed_query(text) == embedder2.embed_query(text)


def test_fake_embedder_query_differs_from_passage() -> None:
    embedder = FakeEmbedder()
    text = "same text for query and passage"
    passage_vec = embedder.embed_passages([text])[0]
    query_vec = embedder.embed_query(text)
    assert passage_vec != query_vec
