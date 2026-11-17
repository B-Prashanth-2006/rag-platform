"""Grounded language-model generation components."""

from rag.generation.context import ContextBuilder
from rag.generation.factory import LLMFactory
from rag.generation.fake import FakeLLM
from rag.generation.fakes import InMemoryDocumentLookup
from rag.generation.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_messages
from rag.generation.providers import OllamaLLM, OpenAICompatibleLLM

__all__ = [
    "ContextBuilder",
    "FakeLLM",
    "InMemoryDocumentLookup",
    "LLMFactory",
    "OllamaLLM",
    "OpenAICompatibleLLM",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "build_messages",
]
