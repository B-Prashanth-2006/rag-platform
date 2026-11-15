"""Fixed-window token chunker implementation."""

from __future__ import annotations

from rag.domain.entities import ChunkDraft, ExtractedDocument
from rag.domain.interfaces import TokenCounter


class FixedWindowChunker:
    """Chunks documents into fixed-size windows of tokens with a configurable overlap."""

    def __init__(
        self,
        token_counter: TokenCounter,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size, "
                f"got overlap={chunk_overlap}, chunk_size={chunk_size}"
            )

        self.token_counter = token_counter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: ExtractedDocument) -> list[ChunkDraft]:
        """Split *doc* into fixed-window chunk drafts with overlap."""
        all_tokens: list[int] = []
        token_pages: list[int] = []

        for page in doc.pages:
            tokens = self.token_counter.encode(page.text)
            if not tokens:
                continue
            all_tokens.extend(tokens)
            token_pages.extend([page.number] * len(tokens))

        if not all_tokens:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[ChunkDraft] = []
        start = 0

        while start < len(all_tokens):
            end = min(start + self.chunk_size, len(all_tokens))
            window_tokens = all_tokens[start:end]
            chunk_text = self.token_counter.decode(window_tokens)
            start_page = token_pages[start]

            chunks.append(
                ChunkDraft(
                    index=len(chunks),
                    text=chunk_text,
                    page=start_page,
                    section=None,
                    token_count=len(window_tokens),
                )
            )

            if end >= len(all_tokens):
                break
            start += step

        return chunks
