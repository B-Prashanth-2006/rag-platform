"""Build token-budgeted, injection-safe source context."""

from __future__ import annotations

import html
import re
import uuid
from collections.abc import Mapping
from dataclasses import replace

from rag.domain.entities import RetrievedChunk
from rag.domain.interfaces import TokenCounter


class ContextBuilder:
    """Render ranked chunks as numbered and escaped source blocks."""

    def __init__(self, token_counter: TokenCounter, token_budget: int) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be greater than zero.")
        self.token_counter = token_counter
        self.token_budget = token_budget

    def build(
        self,
        chunks: list[RetrievedChunk],
        document_names: Mapping[uuid.UUID, str] | None = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        """Render sources in rank order, stopping when the context budget is reached."""
        names = document_names or {}
        blocks: list[str] = []
        included: list[RetrievedChunk] = []
        seen_ids: set[uuid.UUID] = set()

        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            source_id = len(included) + 1
            document_name = names.get(chunk.document_id, str(chunk.document_id))
            block = self._render_source(source_id, document_name, chunk.page, chunk.text)
            candidate_context = "\n".join([*blocks, block])
            if self.token_counter.count(candidate_context) <= self.token_budget:
                blocks.append(block)
                included.append(chunk)
                continue

            if not included:
                truncated = self._truncate_first_chunk(chunk, document_name, source_id)
                if truncated is not None:
                    truncated_chunk, truncated_block = truncated
                    blocks.append(truncated_block)
                    included.append(truncated_chunk)
            break

        return "\n".join(blocks), included

    def _truncate_first_chunk(
        self,
        chunk: RetrievedChunk,
        document_name: str,
        source_id: int,
    ) -> tuple[RetrievedChunk, str] | None:
        encoded = self.token_counter.encode(chunk.text)
        for token_count in range(len(encoded) - 1, -1, -1):
            text = self.token_counter.decode(encoded[:token_count])
            block = self._render_source(source_id, document_name, chunk.page, text)
            if self.token_counter.count(block) <= self.token_budget:
                return replace(chunk, text=text), block
        return None

    @staticmethod
    def _render_source(
        source_id: int,
        document_name: str,
        page: int | None,
        text: str,
    ) -> str:
        safe_name = _escape_untrusted_text(document_name, quote=True)
        safe_text = _escape_untrusted_text(text, quote=False)
        page_attribute = f' page="{page}"' if page is not None else ""
        return f'<source id="{source_id}" doc="{safe_name}"{page_attribute}>{safe_text}</source>'


def _escape_untrusted_text(text: str, *, quote: bool) -> str:
    escaped = html.escape(text, quote=quote)
    return re.sub(
        r"\b(system|assistant|user)\s*:",
        lambda match: f"{match.group(1)}&#58;",
        escaped,
        flags=re.IGNORECASE,
    )
