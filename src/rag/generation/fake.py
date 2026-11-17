"""Deterministic fake language model for tests and local development."""

from __future__ import annotations

from collections.abc import Iterator

from rag.domain.entities import LLMMessage


class FakeLLM:
    """Return scripted text and retain each message list received."""

    def __init__(
        self,
        responses: list[str] | None = None,
        stream_chunks: list[list[str]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = list(responses) if responses is not None else ["Fake LLM response."]
        self.stream_chunks = stream_chunks
        self.error = error
        self.messages_received: list[list[LLMMessage]] = []
        self._call_index = 0
        self._stream_call_index = 0

    def generate(self, messages: list[LLMMessage]) -> str:
        """Return the next scripted response."""
        self.messages_received.append(list(messages))
        if self.error is not None:
            raise self.error
        response = self._next_response()
        return response

    def stream(self, messages: list[LLMMessage]) -> Iterator[str]:
        """Yield configured chunks or one complete scripted response."""
        self.messages_received.append(list(messages))
        if self.error is not None:
            raise self.error
        stream_call_index = self._stream_call_index
        self._stream_call_index += 1
        response = self._next_response()
        if self.stream_chunks is not None and stream_call_index < len(self.stream_chunks):
            yield from self.stream_chunks[stream_call_index]
            return
        yield response

    def _next_response(self) -> str:
        if self._call_index >= len(self.responses):
            raise RuntimeError("No scripted LLM response remains.")
        response = self.responses[self._call_index]
        self._call_index += 1
        return response
