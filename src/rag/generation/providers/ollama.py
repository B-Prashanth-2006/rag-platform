"""Ollama chat-completion provider."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from rag.domain.entities import LLMMessage
from rag.domain.errors import LLMUnavailableError
from rag.generation.providers._http import post_json, stream_response


class OllamaLLM:
    """Generate text using Ollama's ``/api/chat`` endpoint."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self._owns_client = client is None
        self._client = client or httpx.Client()

    def generate(self, messages: list[LLMMessage]) -> str:
        """Return the complete Ollama response."""
        response = post_json(
            self._client,
            f"{self.base_url}/api/chat",
            self._payload(messages, stream=False),
            {},
            self.timeout,
        )
        try:
            content = response.json()["message"]["content"]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMUnavailableError(
                "The language-model provider returned an invalid response."
            ) from exc
        if not isinstance(content, str):
            raise LLMUnavailableError("The language-model provider returned an invalid response.")
        return content

    def stream(self, messages: list[LLMMessage]) -> Iterator[str]:
        """Yield content from Ollama's newline-delimited JSON response."""
        with stream_response(
            self._client,
            f"{self.base_url}/api/chat",
            self._payload(messages, stream=True),
            {},
            self.timeout,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    event: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LLMUnavailableError(
                        "The language-model provider returned an invalid stream."
                    ) from exc
                if not isinstance(event, dict):
                    raise LLMUnavailableError(
                        "The language-model provider returned an invalid stream."
                    )
                message = event.get("message", {})
                if not isinstance(message, dict):
                    raise LLMUnavailableError(
                        "The language-model provider returned an invalid stream."
                    )
                content = message.get("content", "")
                if not isinstance(content, str):
                    raise LLMUnavailableError(
                        "The language-model provider returned an invalid stream."
                    )
                if content:
                    yield content
                if event.get("done") is True:
                    break

    def close(self) -> None:
        """Close the internally created HTTP client, if any."""
        if self._owns_client:
            self._client.close()

    def _payload(self, messages: list[LLMMessage], *, stream: bool) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "stream": stream,
            "options": {"temperature": self.temperature},
        }
