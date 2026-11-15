"""Token counters for chunking strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class WhitespaceTokenCounter:
    """Whitespace-based token counter for fast, offline unit and property tests."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._inv_vocab: dict[int, str] = {}

    def count(self, text: str) -> int:
        """Return the number of whitespace-delimited tokens in *text*."""
        return len(text.split())

    def encode(self, text: str) -> list[int]:
        """Convert whitespace-delimited words into token IDs."""
        words = text.split()
        tokens: list[int] = []
        for word in words:
            if word not in self._vocab:
                token_id = len(self._vocab)
                self._vocab[word] = token_id
                self._inv_vocab[token_id] = word
            tokens.append(self._vocab[word])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Reconstruct whitespace-delimited text from token IDs."""
        if not tokens:
            return ""
        words: list[str] = []
        for token in tokens:
            if token not in self._inv_vocab:
                raise KeyError(f"Unknown token ID: {token}")
            words.append(self._inv_vocab[token])
        return " ".join(words)


class HuggingFaceTokenCounter:
    """Token counter wrapping HuggingFace AutoTokenizer with add_special_tokens=False."""

    def __init__(self, model_name: str | None = None) -> None:
        from transformers import AutoTokenizer

        from rag.config import get_settings

        self.model_name = model_name or get_settings().tokenizer_model
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            self.model_name
        )
        self._tokenizer = tokenizer
        self._cache: dict[tuple[int, ...], str] = {}

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self.encode(text))

    def encode(self, text: str) -> list[int]:
        """Convert *text* into token IDs without special tokens."""
        if not text:
            return []
        tokens: list[int] = self._tokenizer.encode(text, add_special_tokens=False)
        self._cache[tuple(tokens)] = text
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Reconstruct text from token IDs."""
        if not tokens:
            return ""
        tokens_tuple = tuple(tokens)
        if tokens_tuple in self._cache:
            return self._cache[tokens_tuple]
        return str(self._tokenizer.decode(tokens, skip_special_tokens=True))
