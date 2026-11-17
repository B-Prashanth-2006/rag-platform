"""Grounded answer generation with refusal and citation handling."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from rag.config import Settings, get_settings
from rag.domain.entities import (
    Answer,
    AnswerEvent,
    Citation,
    LLMMessage,
    RetrievalFilter,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)
from rag.domain.interfaces import LLM, DocumentLookup, Retriever
from rag.generation.context import ContextBuilder
from rag.generation.prompts import PROMPT_VERSION, build_messages

logger = logging.getLogger(__name__)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass
class _PreparedGeneration:
    retrieval: RetrievalResult
    trace: list[StageTrace]
    included: list[RetrievedChunk]
    document_names: dict[uuid.UUID, str]
    messages: list[LLMMessage]
    refused: bool


class GenerationService:
    """Generate answers from retrieved evidence using injected dependencies."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        context_builder: ContextBuilder,
        document_lookup: DocumentLookup,
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.context_builder = context_builder
        self.document_lookup = document_lookup
        self.settings = settings or get_settings()

    def answer(
        self,
        question: str,
        k: int,
        filter: RetrievalFilter | None = None,
    ) -> Answer:
        """Return a grounded answer or a configured refusal."""
        prepared = self._prepare(question, k, filter)
        if prepared.refused:
            return self._refusal(prepared)

        started = time.perf_counter()
        text = self.llm.generate(prepared.messages)
        generate_latency_ms = (time.perf_counter() - started) * 1000.0
        return self._finish(prepared, text, generate_latency_ms)

    def answer_stream(
        self,
        question: str,
        k: int,
        filter: RetrievalFilter | None = None,
    ) -> Iterator[AnswerEvent]:
        """Stream response tokens, followed by resolved citations and completion."""
        prepared = self._prepare(question, k, filter)
        if prepared.refused:
            yield AnswerEvent(kind="token", payload=self.settings.generation_refusal_text)
            yield AnswerEvent(kind="citations", payload=[])
            yield AnswerEvent(kind="done")
            return

        started = time.perf_counter()
        response_parts: list[str] = []
        try:
            for token in self.llm.stream(prepared.messages):
                response_parts.append(token)
                yield AnswerEvent(kind="token", payload=token)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            yield AnswerEvent(kind="error", payload=error)
            return

        generate_latency_ms = (time.perf_counter() - started) * 1000.0
        answer = self._finish(prepared, "".join(response_parts), generate_latency_ms)
        yield AnswerEvent(kind="citations", payload=answer.citations)
        yield AnswerEvent(kind="done")

    def _prepare(
        self,
        question: str,
        k: int,
        filter: RetrievalFilter | None,
    ) -> _PreparedGeneration:
        if not question.strip():
            raise ValueError("question must not be empty or whitespace-only.")

        retrieval = self.retriever.retrieve(query=question, k=k, filter=filter)
        trace = list(retrieval.trace)
        if self._should_refuse(retrieval):
            trace.extend(
                [
                    StageTrace(stage="context", latency_ms=0.0, results=[]),
                    StageTrace(stage="generate", latency_ms=0.0, results=[]),
                ]
            )
            return _PreparedGeneration(
                retrieval=retrieval,
                trace=trace,
                included=[],
                document_names={},
                messages=[],
                refused=True,
            )

        context_started = time.perf_counter()
        document_ids = list(dict.fromkeys(chunk.document_id for chunk in retrieval.chunks))
        document_names = self.document_lookup.get_names(document_ids)
        context, included = self.context_builder.build(retrieval.chunks, document_names)
        context_latency_ms = (time.perf_counter() - context_started) * 1000.0
        trace.append(
            StageTrace(
                stage="context",
                latency_ms=context_latency_ms,
                results=[(chunk.chunk_id, chunk.score) for chunk in included],
            )
        )
        return _PreparedGeneration(
            retrieval=retrieval,
            trace=trace,
            included=included,
            document_names=document_names,
            messages=build_messages(question, context),
            refused=False,
        )

    def _should_refuse(self, retrieval: RetrievalResult) -> bool:
        if not retrieval.chunks:
            return True
        top_chunk = retrieval.chunks[0]
        if top_chunk.source == "reranked":
            return top_chunk.score < self.settings.min_score_reranked
        if top_chunk.source == "dense":
            return top_chunk.score < self.settings.min_score_dense
        return False

    def _refusal(self, prepared: _PreparedGeneration) -> Answer:
        return Answer(
            text=self.settings.generation_refusal_text,
            citations=[],
            refused=True,
            top_score=prepared.retrieval.top_score,
            trace=prepared.trace,
            prompt_version=PROMPT_VERSION,
        )

    def _finish(
        self,
        prepared: _PreparedGeneration,
        text: str,
        generate_latency_ms: float,
    ) -> Answer:
        prepared.trace.append(
            StageTrace(stage="generate", latency_ms=generate_latency_ms, results=[])
        )
        citation_started = time.perf_counter()
        citations, invalid_citations = self._resolve_citations(
            text,
            prepared.included,
            prepared.document_names,
        )
        citation_latency_ms = (time.perf_counter() - citation_started) * 1000.0
        prepared.trace.append(
            StageTrace(
                stage="citations",
                latency_ms=citation_latency_ms,
                results=[(citation.chunk_id, float(citation.index)) for citation in citations],
                invalid_citations=invalid_citations,
            )
        )
        if not citations:
            logger.warning("Generated answer contains no valid citations.")
        return Answer(
            text=text,
            citations=citations,
            refused=False,
            top_score=prepared.retrieval.top_score,
            trace=prepared.trace,
            prompt_version=PROMPT_VERSION,
        )

    @staticmethod
    def _resolve_citations(
        text: str,
        included: list[RetrievedChunk],
        document_names: dict[uuid.UUID, str],
    ) -> tuple[list[Citation], int]:
        chunks_by_index = dict(enumerate(included, start=1))
        citations: list[Citation] = []
        seen_indices: set[int] = set()
        invalid_count = 0
        for marker in _CITATION_PATTERN.finditer(text):
            index = int(marker.group(1))
            chunk = chunks_by_index.get(index)
            if chunk is None:
                invalid_count += 1
                continue
            if index in seen_indices:
                continue
            seen_indices.add(index)
            citations.append(
                Citation(
                    index=index,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=document_names.get(chunk.document_id, str(chunk.document_id)),
                    page=chunk.page,
                    section=chunk.section,
                    text=chunk.text,
                )
            )
        return citations, invalid_count
