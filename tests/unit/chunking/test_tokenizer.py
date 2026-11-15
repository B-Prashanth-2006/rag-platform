"""Tests for WhitespaceTokenCounter."""

from __future__ import annotations

from rag.chunking.tokenizer import WhitespaceTokenCounter
from rag.domain.interfaces import TokenCounter


def test_whitespace_token_counter_satisfies_protocol() -> None:
    counter = WhitespaceTokenCounter()
    assert isinstance(counter, TokenCounter)


def test_whitespace_token_counter_empty() -> None:
    counter = WhitespaceTokenCounter()
    assert counter.count("") == 0
    assert counter.count("   \n\t  ") == 0
    assert counter.encode("") == []
    assert counter.decode([]) == ""


def test_whitespace_token_counter_count_and_roundtrip() -> None:
    counter = WhitespaceTokenCounter()
    text = "The quick brown fox jumps over the lazy dog"
    assert counter.count(text) == 9

    tokens = counter.encode(text)
    assert len(tokens) == 9
    assert all(isinstance(t, int) for t in tokens)

    decoded = counter.decode(tokens)
    assert decoded == text


def test_whitespace_token_counter_deterministic() -> None:
    counter1 = WhitespaceTokenCounter()
    counter2 = WhitespaceTokenCounter()
    text = "deterministic tokenization test with repeated words repeated words"

    tokens1 = counter1.encode(text)
    tokens2 = counter2.encode(text)
    assert tokens1 == tokens2
    assert counter1.decode(tokens1) == counter2.decode(tokens2)
