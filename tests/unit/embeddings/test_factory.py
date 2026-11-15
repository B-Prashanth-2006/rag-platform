"""Tests for EmbedderFactory."""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.embeddings.cache import CachedEmbedder
from rag.embeddings.factory import EmbedderFactory, get_model_family
from rag.embeddings.fake import FakeEmbedder


def test_get_model_family() -> None:
    assert get_model_family("BAAI/bge-small-en-v1.5") == "bge"
    assert get_model_family("BAAI/bge-large-en") == "bge"
    assert get_model_family("intfloat/e5-small-v2") == "e5"
    assert get_model_family("intfloat/multilingual-e5-base") == "e5"

    with pytest.raises(ValueError, match="Unknown model family"):
        get_model_family("custom/unknown-model")


def test_embedder_factory_unknown_family_raises() -> None:
    settings = Settings(embedding_model="custom/unknown-model")
    with pytest.raises(ValueError, match="Unknown model family"):
        EmbedderFactory.create(settings=settings)


def test_embedder_factory_with_fake(tmp_path: pytest.TempPathFactory) -> None:
    settings = Settings(
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_cache_enabled=True,
    )
    fake = FakeEmbedder()
    embedder = EmbedderFactory.create(settings=settings, inner_embedder=fake)
    assert isinstance(embedder, CachedEmbedder)
    assert embedder.info.name == fake.info.name


def test_embedder_factory_without_cache() -> None:
    settings = Settings(
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_cache_enabled=False,
    )
    fake = FakeEmbedder()
    embedder = EmbedderFactory.create(settings=settings, inner_embedder=fake)
    assert not isinstance(embedder, CachedEmbedder)
    assert embedder is fake
