"""Tests for CachedEmbedder."""

from __future__ import annotations

from pathlib import Path

from rag.domain.entities import EmbeddingModelInfo
from rag.domain.interfaces import Embedder
from rag.embeddings.cache import CachedEmbedder
from rag.embeddings.fake import FakeEmbedder


class SpyEmbedder:
    """Wraps an embedder and records call counts and texts."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.passages_call_count = 0
        self.passages_called_with: list[list[str]] = []
        self.query_call_count = 0
        self.query_called_with: list[str] = []

    @property
    def info(self) -> EmbeddingModelInfo:
        return self.inner.info

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.passages_call_count += 1
        self.passages_called_with.append(texts)
        return self.inner.embed_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_call_count += 1
        self.query_called_with.append(text)
        return self.inner.embed_query(text)


def test_cached_embedder_satisfies_protocol(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.db"
    inner = FakeEmbedder()
    cached = CachedEmbedder(embedder=inner, cache_path=cache_path)
    assert isinstance(cached, Embedder)
    assert cached.info == inner.info


def test_cached_embedder_miss_then_hit(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.db"
    spy = SpyEmbedder(FakeEmbedder())
    cached = CachedEmbedder(embedder=spy, cache_path=cache_path)

    texts = ["apple", "banana"]
    res1 = cached.embed_passages(texts)
    assert spy.passages_call_count == 1
    assert spy.passages_called_with[0] == ["apple", "banana"]

    # Second call should be 100% cache hits
    res2 = cached.embed_passages(texts)
    assert spy.passages_call_count == 1  # not called again!
    assert res1 == res2


def test_cached_embedder_mixed_hit_miss_preserves_order(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.db"
    spy = SpyEmbedder(FakeEmbedder())
    cached = CachedEmbedder(embedder=spy, cache_path=cache_path)

    # Prime cache with "apple" and "cherry"
    cached.embed_passages(["apple", "cherry"])
    assert spy.passages_call_count == 1
    assert spy.passages_called_with[0] == ["apple", "cherry"]

    # Now request ["apple", "banana", "cherry", "date"]
    # "apple" and "cherry" are hits; "banana" and "date" are misses
    mixed = ["apple", "banana", "cherry", "date"]
    res = cached.embed_passages(mixed)
    assert spy.passages_call_count == 2
    # Only misses sent in one batch
    assert spy.passages_called_with[1] == ["banana", "date"]

    # Verify results preserve original order and match direct computation
    direct = FakeEmbedder().embed_passages(mixed)
    assert res == direct


def test_cached_embedder_different_model_version_is_miss(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.db"

    # Embed with v1.0
    spy1 = SpyEmbedder(FakeEmbedder(version="1.0"))
    cached1 = CachedEmbedder(embedder=spy1, cache_path=cache_path)
    cached1.embed_passages(["text"])
    assert spy1.passages_call_count == 1

    # Embed same text with v2.0
    spy2 = SpyEmbedder(FakeEmbedder(version="2.0"))
    cached2 = CachedEmbedder(embedder=spy2, cache_path=cache_path)
    cached2.embed_passages(["text"])
    assert spy2.passages_call_count == 1  # Cache miss because version differs


def test_cached_embedder_query_caching(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.db"
    spy = SpyEmbedder(FakeEmbedder())
    cached = CachedEmbedder(embedder=spy, cache_path=cache_path)

    q1 = cached.embed_query("search term")
    assert spy.query_call_count == 1

    q2 = cached.embed_query("search term")
    assert spy.query_call_count == 1  # hit!
    assert q1 == q2


def test_cached_embedder_empty_list(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.db"
    spy = SpyEmbedder(FakeEmbedder())
    cached = CachedEmbedder(embedder=spy, cache_path=cache_path)

    assert cached.embed_passages([]) == []
    assert spy.passages_call_count == 0
