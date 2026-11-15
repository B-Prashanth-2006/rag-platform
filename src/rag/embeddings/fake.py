"""Fake deterministic hash-based embedder for testing."""

from __future__ import annotations

import hashlib
import math
import random

from rag.domain.entities import EmbeddingModelInfo


class FakeEmbedder:
    """Deterministic hash-based embedder that requires no downloads or heavy compute."""

    def __init__(
        self,
        name: str = "fake-model",
        version: str = "1.0",
        dimension: int = 384,
        max_tokens: int = 512,
    ) -> None:
        self._info = EmbeddingModelInfo(
            name=name,
            version=version,
            dimension=dimension,
            max_tokens=max_tokens,
        )

    @property
    def info(self) -> EmbeddingModelInfo:
        """Return model metadata."""
        return self._info

    def _generate_vector(self, prefix: str, text: str) -> list[float]:
        digest = hashlib.sha256(f"{prefix}:{text}".encode()).hexdigest()
        seed = int(digest[:16], 16)
        rng = random.Random(seed)

        # Generate pseudorandom values centered at 0
        raw = [rng.gauss(0.0, 1.0) for _ in range(self._info.dimension)]
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0.0:
            raw[0] = 1.0
            norm = 1.0
        return [x / norm for x in raw]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embedding vectors for passages."""
        return [self._generate_vector("passage", t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Generate a normalized embedding vector for a query."""
        return self._generate_vector("query", text)
