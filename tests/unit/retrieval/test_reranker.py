"""Unit tests for CrossEncoderReranker and FakeReranker — written first (TDD)."""

from __future__ import annotations

import uuid

import pytest
import torch

from rag.config import Settings
from rag.domain.entities import RetrievedChunk
from rag.domain.interfaces import Reranker
from rag.retrieval.fakes import FakeReranker
from rag.retrieval.reranker import CrossEncoderReranker, resolve_reranker_device


def _make_chunk(
    text: str,
    score: float = 0.5,
    source: str = "dense",
    chunk_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        page=1,
        section="Section",
        score=score,
        source=source,
    )


class TestRerankerDeviceSelection:
    def test_device_selection_explicit_override(self) -> None:
        settings = Settings(reranker_device="cpu")
        assert resolve_reranker_device(settings=settings) == "cpu"
        assert resolve_reranker_device(explicit_device="cpu") == "cpu"

    def test_device_selection_cuda_preferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        settings = Settings(reranker_device=None)
        assert resolve_reranker_device(settings=settings) == "cuda"

    def test_device_selection_mps_when_cuda_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        settings = Settings(reranker_device=None)
        assert resolve_reranker_device(settings=settings) == "mps"

    def test_device_selection_cpu_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        settings = Settings(reranker_device=None)
        assert resolve_reranker_device(settings=settings) == "cpu"


class TestFakeReranker:
    def test_fake_reranker_satisfies_protocol(self) -> None:
        reranker = FakeReranker()
        assert isinstance(reranker, Reranker)

    def test_fake_reranker_scores_and_limits_top_k(self) -> None:
        c1 = _make_chunk(text="Paris is the capital of France.", score=0.1)
        c2 = _make_chunk(text="Apples and oranges are fruits.", score=0.9)

        # Keyword match scoring function
        reranker = FakeReranker(score_fn=lambda q, c: 1.0 if q.lower() in c.text.lower() else 0.0)

        reranked = reranker.rerank(query="France", chunks=[c2, c1], top_k=1)
        assert len(reranked) == 1
        assert reranked[0].chunk_id == c1.chunk_id
        assert reranked[0].source == "reranked"
        assert reranked[0].score == 1.0

    def test_fake_reranker_empty_input(self) -> None:
        reranker = FakeReranker()
        assert reranker.rerank(query="", chunks=[], top_k=5) == []
        c = _make_chunk(text="Something")
        assert reranker.rerank(query="", chunks=[c], top_k=5) == []
        assert reranker.rerank(query="test", chunks=[c], top_k=0) == []


class TestCrossEncoderRerankerRealModel:
    @pytest.fixture(scope="class")
    def reranker(self) -> CrossEncoderReranker:
        try:
            return CrossEncoderReranker(
                model_name="BAAI/bge-reranker-base",
                device="cpu",
                batch_size=2,
                rerank_candidates=10,
            )
        except OSError as exc:
            pytest.skip(f"Skipping real model reranker test due to OSError: {exc}")
        # Offline HuggingFace errors may wrap OSError in a different exception.
        except Exception as exc:
            if "offline" in str(exc).lower() or "connection" in str(exc).lower():
                pytest.skip(f"Skipping real model reranker test due to network failure: {exc}")
            raise

    def test_real_model_satisfies_protocol(self, reranker: CrossEncoderReranker) -> None:
        assert isinstance(reranker, Reranker)

    def test_relevant_passage_outranks_irrelevant(self, reranker: CrossEncoderReranker) -> None:
        c_relevant = _make_chunk(
            text="Paris is the capital and most populous city of France.",
            score=0.2,
        )
        c_irrelevant = _make_chunk(
            text="Photosynthesis is the biological process used by plants to synthesize nutrients.",
            score=0.9,
        )

        query = "What is the capital city of France?"
        # Irrelevant chunk initially has higher retriever score (0.9 vs 0.2)
        reranked = reranker.rerank(query=query, chunks=[c_irrelevant, c_relevant], top_k=2)

        assert len(reranked) == 2
        # Cross-encoder should outrank irrelevant
        assert reranked[0].chunk_id == c_relevant.chunk_id
        assert reranked[0].source == "reranked"
        assert reranked[1].chunk_id == c_irrelevant.chunk_id
        assert reranked[1].source == "reranked"

        # Scores must be in 0..1 (from sigmoid)
        for rc in reranked:
            assert 0.0 <= rc.score <= 1.0

        assert reranked[0].score > reranked[1].score

    def test_real_model_empty_and_zero_k(self, reranker: CrossEncoderReranker) -> None:
        assert reranker.rerank(query="test", chunks=[], top_k=5) == []
        c = _make_chunk(text="test passage")
        assert reranker.rerank(query="", chunks=[c], top_k=5) == []
        assert reranker.rerank(query="test", chunks=[c], top_k=0) == []
