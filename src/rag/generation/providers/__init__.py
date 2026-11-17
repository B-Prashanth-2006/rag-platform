"""HTTP-based language-model providers."""

from rag.generation.providers.ollama import OllamaLLM
from rag.generation.providers.openai_compatible import OpenAICompatibleLLM

__all__ = ["OllamaLLM", "OpenAICompatibleLLM"]
