"""Tests for HuggingFaceTokenCounter.

Automatically skips if downloading the model fails (e.g. offline / network restricted).
"""

from __future__ import annotations

import pytest

from rag.chunking.tokenizer import HuggingFaceTokenCounter
from rag.domain.interfaces import TokenCounter


def test_huggingface_token_counter() -> None:
    try:
        counter = HuggingFaceTokenCounter(model_name="BAAI/bge-small-en-v1.5")
    except (OSError, Exception) as exc:
        pytest.skip(f"HuggingFace model download failed or offline: {exc}")

    assert isinstance(counter, TokenCounter)

    text = "Retrieval Augmented Generation with dense embeddings"
    count = counter.count(text)
    assert count > 0

    tokens = counter.encode(text)
    assert len(tokens) == count
    assert all(isinstance(t, int) for t in tokens)

    decoded = counter.decode(tokens)
    assert "Retrieval" in decoded
    assert "embeddings" in decoded
