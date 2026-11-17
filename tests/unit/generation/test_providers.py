"""Provider tests using httpx.MockTransport; no network access is used."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from rag.domain.entities import LLMMessage
from rag.domain.errors import LLMTimeoutError, LLMUnavailableError
from rag.generation.providers.ollama import OllamaLLM
from rag.generation.providers.openai_compatible import OpenAICompatibleLLM


def test_ollama_non_streaming_request_and_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "question"}]
        assert body["options"]["temperature"] == 0.3
        assert body["stream"] is False
        return httpx.Response(200, json={"message": {"content": "answer"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = OllamaLLM(
            model="test-model",
            base_url="http://ollama.test/",
            temperature=0.3,
            client=client,
        )
        answer = llm.generate([LLMMessage(role="user", content="question")])

    assert answer == "answer"


def test_ollama_ndjson_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["stream"] is True
        return httpx.Response(
            200,
            content=(
                b'{"message":{"content":"Hello"},"done":false}\n'
                b'{"message":{"content":" world"},"done":false}\n'
                b'{"message":{"content":""},"done":true}\n'
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = OllamaLLM(client=client)
        tokens = list(llm.stream([LLMMessage(role="user", content="question")]))

    assert tokens == ["Hello", " world"]


def test_openai_compatible_non_streaming_sends_api_key() -> None:
    api_key = "private-test-key"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {api_key}"
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = OpenAICompatibleLLM(
            base_url="http://openai.test/v1",
            model="test-model",
            api_key=api_key,
            client=client,
        )
        assert llm.generate([LLMMessage(role="user", content="question")]) == "answer"


def test_openai_compatible_sse_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = OpenAICompatibleLLM(client=client)
        tokens = list(llm.stream([LLMMessage(role="user", content="question")]))

    assert tokens == ["Hello", " world"]


@pytest.mark.parametrize(
    ("exception", "expected_error"),
    [
        (httpx.ReadTimeout("timed out"), LLMTimeoutError),
        (httpx.ConnectError("unavailable"), LLMUnavailableError),
    ],
)
def test_provider_maps_transport_errors(
    exception: Exception,
    expected_error: type[Exception],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = OllamaLLM(client=client)
        with pytest.raises(expected_error):
            llm.generate([LLMMessage(role="user", content="question")])


def test_provider_maps_non_success_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = OpenAICompatibleLLM(api_key="private-test-key", client=client)
        with pytest.raises(LLMUnavailableError):
            llm.generate([LLMMessage(role="user", content="question")])


def test_api_key_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    api_key = "private-test-key"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {api_key}"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = OpenAICompatibleLLM(api_key=api_key, client=client)
        with caplog.at_level(logging.DEBUG):
            llm.generate([LLMMessage(role="user", content="question")])

    assert api_key not in caplog.text
