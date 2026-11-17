"""Tests for settings-driven language-model provider selection."""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.domain.interfaces import LLM
from rag.generation.factory import LLMFactory
from rag.generation.fake import FakeLLM
from rag.generation.providers.ollama import OllamaLLM
from rag.generation.providers.openai_compatible import OpenAICompatibleLLM


def test_factory_builds_ollama_from_settings() -> None:
    llm = LLMFactory.create(
        Settings(
            llm_provider="ollama",
            ollama_base_url="http://ollama.test",
            ollama_model="test-model",
            llm_timeout=12.0,
            llm_temperature=0.25,
        )
    )

    assert isinstance(llm, OllamaLLM)
    assert llm.base_url == "http://ollama.test"
    assert llm.model == "test-model"
    assert llm.timeout == 12.0
    assert llm.temperature == 0.25


def test_factory_builds_openai_compatible_from_settings() -> None:
    llm = LLMFactory.create(
        Settings(
            llm_provider="openai_compatible",
            openai_base_url="http://openai.test/v1",
            openai_model="test-model",
            openai_api_key="test-key",
        )
    )

    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.base_url == "http://openai.test/v1"
    assert llm.model == "test-model"
    assert llm.api_key == "test-key"


def test_factory_builds_fake() -> None:
    llm = LLMFactory.create(Settings(llm_provider="fake"))

    assert isinstance(llm, FakeLLM)
    assert isinstance(llm, LLM)
    assert Settings().context_token_budget == 3000


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.create(Settings(llm_provider="unknown"))
