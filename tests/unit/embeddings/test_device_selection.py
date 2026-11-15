"""Tests for embedding device selection."""

from __future__ import annotations

import pytest
import torch

from rag.config import Settings
from rag.embeddings.hf import resolve_embedding_device


def test_device_selection_explicit_override() -> None:
    settings = Settings(embedding_device="cpu")
    assert resolve_embedding_device(settings=settings) == "cpu"


def test_device_selection_cuda_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    settings = Settings(embedding_device=None)
    assert resolve_embedding_device(settings=settings) == "cuda"


def test_device_selection_mps_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    settings = Settings(embedding_device=None)
    assert resolve_embedding_device(settings=settings) == "mps"


def test_device_selection_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    settings = Settings(embedding_device=None)
    assert resolve_embedding_device(settings=settings) == "cpu"
