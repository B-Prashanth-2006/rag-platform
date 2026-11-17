"""Tests for the deterministic fake language model."""

from __future__ import annotations

import uuid

import pytest

from rag.domain.entities import LLMMessage
from rag.generation.fake import FakeLLM
from rag.generation.fakes import InMemoryDocumentLookup


def test_fake_llm_scripts_responses_and_records_messages() -> None:
    messages = [LLMMessage(role="user", content="question")]
    llm = FakeLLM(responses=["first", "second"], stream_chunks=[["one", " two"]])

    assert llm.generate(messages) == "first"
    assert list(llm.stream(messages)) == ["one", " two"]
    assert llm.messages_received == [messages, messages]


def test_fake_llm_raises_configured_error() -> None:
    llm = FakeLLM(error=RuntimeError("scripted failure"))

    with pytest.raises(RuntimeError, match="scripted failure"):
        llm.generate([LLMMessage(role="user", content="question")])


def test_fake_llm_satisfies_llm_protocol() -> None:
    from rag.domain.interfaces import LLM

    assert isinstance(FakeLLM(), LLM)


def test_in_memory_document_lookup_returns_only_known_names() -> None:
    first_id = uuid.uuid4()
    missing_id = uuid.uuid4()
    lookup = InMemoryDocumentLookup({first_id: "first.pdf"})

    assert lookup.get_names([missing_id, first_id]) == {first_id: "first.pdf"}
