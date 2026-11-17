"""Settings-driven language-model provider factory."""

from __future__ import annotations

from rag.config import Settings, get_settings
from rag.domain.interfaces import LLM
from rag.generation.fake import FakeLLM
from rag.generation.providers.ollama import OllamaLLM
from rag.generation.providers.openai_compatible import OpenAICompatibleLLM


class LLMFactory:
    """Create configured LLM providers without binding callers to a vendor."""

    @staticmethod
    def create(settings: Settings | None = None) -> LLM:
        """Create an LLM implementation selected by settings."""
        config = settings or get_settings()
        provider = config.llm_provider.lower()
        if provider == "ollama":
            return OllamaLLM(
                model=config.ollama_model,
                base_url=config.ollama_base_url,
                timeout=config.llm_timeout,
                temperature=config.llm_temperature,
            )
        if provider in {"openai", "openai_compatible"}:
            return OpenAICompatibleLLM(
                base_url=config.openai_base_url,
                model=config.openai_model,
                api_key=config.openai_api_key,
                timeout=config.llm_timeout,
                temperature=config.llm_temperature,
            )
        if provider == "fake":
            return FakeLLM()
        raise ValueError(
            f"Unknown LLM provider '{config.llm_provider}'. "
            "Must be 'ollama', 'openai', 'openai_compatible', or 'fake'."
        )
