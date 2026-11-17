"""Tests for generation domain types and contracts."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

from rag.domain.entities import Answer, AnswerEvent, Citation, LLMMessage, StageTrace
from rag.domain.interfaces import LLM, DocumentLookup


class _FakeLLMContract:
    def generate(self, messages: list[LLMMessage]) -> str:
        return "answer"

    def stream(self, messages: list[LLMMessage]) -> Iterator[str]:
        yield "answer"


class _FakeDocumentLookupContract:
    def get_names(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, str]:
        return {doc_id: str(doc_id) for doc_id in ids}


def test_generation_domain_types_and_protocols() -> None:
    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    citation = Citation(
        index=1,
        chunk_id=chunk_id,
        document_id=document_id,
        document_name="guide.txt",
        page=2,
        section="Intro",
        text="Source text",
    )
    trace = [StageTrace(stage="retrieve", latency_ms=1.5, results=[])]
    answer = Answer(
        text="answer [1]",
        citations=[citation],
        refused=False,
        top_score=0.8,
        trace=trace,
        prompt_version="v1",
    )
    event = AnswerEvent(kind="citations", payload=[citation])

    assert answer.citations == [citation]
    assert answer.trace == trace
    assert event.payload == [citation]
    assert isinstance(_FakeLLMContract(), LLM)
    assert isinstance(_FakeDocumentLookupContract(), DocumentLookup)
