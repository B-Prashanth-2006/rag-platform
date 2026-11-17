"""Tests for grounded answer generation and streaming."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator

import pytest

from rag.chunking.tokenizer import WhitespaceTokenCounter
from rag.config import Settings
from rag.domain.entities import (
    LLMMessage,
    RetrievalFilter,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)
from rag.generation.context import ContextBuilder
from rag.generation.fake import FakeLLM
from rag.generation.fakes import InMemoryDocumentLookup
from rag.generation.service import GenerationService


class _FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int, RetrievalFilter | None]] = []

    def retrieve(
        self,
        query: str,
        k: int,
        filter: RetrievalFilter | None = None,
    ) -> RetrievalResult:
        self.calls.append((query, k, filter))
        return RetrievalResult(
            chunks=self.chunks[:k],
            trace=[StageTrace(stage="dense", latency_ms=1.0, results=[])],
        )


class _FailingStreamLLM:
    def generate(self, messages: list[LLMMessage]) -> str:
        raise AssertionError("streaming should not call generate")

    def stream(self, messages: list[LLMMessage]) -> Iterator[str]:
        yield "partial"
        raise RuntimeError("stream failed")


def _chunk(
    text: str,
    *,
    source: str = "dense",
    score: float = 0.8,
    document_id: uuid.UUID | None = None,
    chunk_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        text=text,
        page=2,
        section="Overview",
        score=score,
        source=source,
    )


def _service(
    chunks: list[RetrievedChunk],
    llm: FakeLLM | _FailingStreamLLM,
    *,
    names: dict[uuid.UUID, str] | None = None,
    settings: Settings | None = None,
) -> tuple[GenerationService, _FakeRetriever]:
    retriever = _FakeRetriever(chunks)
    service = GenerationService(
        retriever=retriever,
        llm=llm,
        context_builder=ContextBuilder(WhitespaceTokenCounter(), token_budget=100),
        document_lookup=InMemoryDocumentLookup(names),
        settings=settings or Settings(),
    )
    return service, retriever


def test_refuses_empty_retrieval_without_calling_llm() -> None:
    llm = FakeLLM()
    service, _ = _service([], llm)

    answer = service.answer("question", k=3)

    assert answer.refused is True
    assert answer.citations == []
    assert answer.text == "I could not find this in the provided documents."
    assert answer.top_score is None
    assert llm.messages_received == []


def test_refusal_text_is_configurable() -> None:
    llm = FakeLLM()
    service, _ = _service(
        [],
        llm,
        settings=Settings(generation_refusal_text="No supporting document was found."),
    )

    answer = service.answer("question", k=3)

    assert answer.text == "No supporting document was found."


@pytest.mark.parametrize(
    ("source", "score", "settings"),
    [
        ("reranked", 0.2, Settings(min_score_reranked=0.5)),
        ("dense", 0.1, Settings(min_score_dense=0.2)),
    ],
)
def test_refuses_low_score_without_calling_llm(
    source: str,
    score: float,
    settings: Settings,
) -> None:
    llm = FakeLLM()
    service, _ = _service([_chunk("evidence", source=source, score=score)], llm, settings=settings)

    answer = service.answer("question", k=3)

    assert answer.refused is True
    assert answer.citations == []
    assert llm.messages_received == []


def test_rrf_score_is_not_compared_with_dense_or_reranked_thresholds() -> None:
    chunk = _chunk("evidence", source="fused", score=0.001)
    llm = FakeLLM(responses=["Answer [1]."])
    service, _ = _service(
        [chunk],
        llm,
        settings=Settings(min_score_dense=0.9, min_score_reranked=0.9),
    )

    answer = service.answer("question", k=3)

    assert answer.refused is False
    assert len(llm.messages_received) == 1


def test_resolves_citations_in_first_appearance_order_and_counts_invalid_markers() -> None:
    first = _chunk("first source")
    second = _chunk("second source")
    llm = FakeLLM(responses=["Second [2], first [1], again [2], invalid [9]."])
    service, _ = _service(
        [first, second],
        llm,
        names={first.document_id: "first.pdf", second.document_id: "second.pdf"},
    )

    answer = service.answer("question", k=3)

    assert [citation.index for citation in answer.citations] == [2, 1]
    assert [citation.chunk_id for citation in answer.citations] == [
        second.chunk_id,
        first.chunk_id,
    ]
    assert [citation.document_name for citation in answer.citations] == [
        "second.pdf",
        "first.pdf",
    ]
    citation_trace = next(trace for trace in answer.trace if trace.stage == "citations")
    assert citation_trace.invalid_citations == 1
    assert {trace.stage for trace in answer.trace} >= {"dense", "context", "generate"}


def test_answer_without_citations_is_returned_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLLM(responses=["An unsupported answer."])
    service, _ = _service([_chunk("evidence")], llm)

    with caplog.at_level(logging.WARNING):
        answer = service.answer("question", k=3)

    assert answer.text == "An unsupported answer."
    assert answer.citations == []
    assert answer.refused is False
    assert "no valid citations" in caplog.text.lower()


def test_answer_rejects_empty_question_before_retrieval() -> None:
    llm = FakeLLM()
    service, retriever = _service([], llm)

    with pytest.raises(ValueError, match="question"):
        service.answer(" \n ", k=3)

    assert retriever.calls == []


def test_answer_stream_yields_tokens_then_citations_and_done() -> None:
    chunk = _chunk("evidence")
    llm = FakeLLM(responses=["Answer [1]."], stream_chunks=[["Answer ", "[1]."]])
    service, _ = _service([chunk], llm)

    events = list(service.answer_stream("question", k=3))

    assert [event.kind for event in events] == ["token", "token", "citations", "done"]
    assert [event.payload for event in events[:2]] == ["Answer ", "[1]."]
    citations = events[2].payload
    assert isinstance(citations, list)
    assert len(citations) == 1


def test_answer_stream_refusal_yields_one_token_then_citations_and_done() -> None:
    llm = FakeLLM()
    service, _ = _service([], llm)

    events = list(service.answer_stream("question", k=3))

    assert [event.kind for event in events] == ["token", "citations", "done"]
    assert events[0].payload == "I could not find this in the provided documents."
    assert events[1].payload == []
    assert llm.messages_received == []


def test_answer_stream_turns_midstream_failure_into_error_event() -> None:
    service, _ = _service([_chunk("evidence")], _FailingStreamLLM())

    events = list(service.answer_stream("question", k=3))

    assert [event.kind for event in events] == ["token", "error"]
    assert events[0].payload == "partial"
    assert events[1].payload == "stream failed"


def test_answer_stream_rejects_empty_question_before_retrieval() -> None:
    llm = FakeLLM()
    service, retriever = _service([], llm)

    with pytest.raises(ValueError, match="question"):
        list(service.answer_stream("", k=3))

    assert retriever.calls == []
