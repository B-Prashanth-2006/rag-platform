"""Provider for OpenAI-compatible chat-completion endpoints."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from rag.domain.entities import LLMMessage
from rag.domain.errors import LLMUnavailableError
from rag.generation.providers._http import post_json, stream_response


class OpenAICompatibleLLM:
    """Generate text using an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self._owns_client = client is None
        self._client = client or httpx.Client()

    def generate(self, messages: list[LLMMessage]) -> str:
        """Return the complete chat-completion response."""
        response = post_json(
            self._client,
            f"{self.base_url}/chat/completions",
            self._payload(messages, stream=False),
            self._headers(),
            self.timeout,
        )
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise LLMUnavailableError(
                "The language-model provider returned an invalid response."
            ) from exc
        return self._extract_content(payload)

    def stream(self, messages: list[LLMMessage]) -> Iterator[str]:
        """Yield content from the endpoint's server-sent events."""
        with stream_response(
            self._client,
            f"{self.base_url}/chat/completions",
            self._payload(messages, stream=True),
            self._headers(),
            self.timeout,
        ) as response:
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event: Any = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise LLMUnavailableError(
                        "The language-model provider returned an invalid stream."
                    ) from exc
                if not isinstance(event, dict):
                    raise LLMUnavailableError(
                        "The language-model provider returned an invalid stream."
                    )
                choices = event.get("choices", [])
                if not isinstance(choices, list):
                    raise LLMUnavailableError(
                        "The language-model provider returned an invalid stream."
                    )
                if not choices:
                    continue
                if not isinstance(choices[0], dict):
                    raise LLMUnavailableError(
                        "The language-model provider returned an invalid stream."
                    )
                delta = choices[0].get("delta", {})
                if not isinstance(delta, dict):
                    raise LLMUnavailableError(
                        "The language-model provider returned an invalid stream."
                    )
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        raise LLMUnavailableError(
                            "The language-model provider returned an invalid stream."
                        )
                    if content:
                        yield content

    def close(self) -> None:
        """Close the internally created HTTP client, if any."""
        if self._owns_client:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        if self.api_key is None:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _payload(self, messages: list[LLMMessage], *, stream: bool) -> dict[str, object]:
        return {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "stream": stream,
            "temperature": self.temperature,
        }

    @staticmethod
    def _extract_content(response: Any) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise LLMUnavailableError(
                "The language-model provider returned an invalid response."
            ) from exc
        if not isinstance(content, str):
            raise LLMUnavailableError("The language-model provider returned an invalid response.")
        return content
