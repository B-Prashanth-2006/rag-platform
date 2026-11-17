"""Shared HTTP error handling for language-model providers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import httpx

from rag.domain.errors import LLMTimeoutError, LLMUnavailableError


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    timeout: float,
) -> httpx.Response:
    """POST JSON and map transport and status failures to domain errors."""
    try:
        response = client.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError("The language-model request timed out.") from exc
    except httpx.RequestError as exc:
        raise LLMUnavailableError("The language-model provider is unavailable.") from exc
    _check_status(response)
    return response


@contextmanager
def stream_response(
    client: httpx.Client,
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    timeout: float,
) -> Iterator[httpx.Response]:
    """Open a streaming POST and translate provider transport failures."""
    try:
        with client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as response:
            _check_status(response)
            yield response
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError("The language-model request timed out.") from exc
    except httpx.RequestError as exc:
        raise LLMUnavailableError("The language-model provider is unavailable.") from exc


def _check_status(response: httpx.Response) -> None:
    if not response.is_success:
        raise LLMUnavailableError(
            f"The language-model provider returned HTTP {response.status_code}."
        )
