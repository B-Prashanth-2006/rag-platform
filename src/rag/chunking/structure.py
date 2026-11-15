"""Structure-aware chunker implementation."""

from __future__ import annotations

import re

from rag.domain.entities import ChunkDraft, ExtractedDocument
from rag.domain.interfaces import TokenCounter


class StructureAwareChunker:
    """Chunks documents respecting headings and paragraph boundaries."""

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
        """Split *doc* into structure-aware chunk drafts."""
        if not doc.pages:
            return []

        chunks: list[ChunkDraft] = []
        heading_stack: list[tuple[int, str]] = []
        current_section_path: str | None = None

        current_paragraphs: list[str] = []
        current_page: int | None = None

        def get_section_path() -> str | None:
            if not heading_stack:
                return None
            return " > ".join(title for _, title in heading_stack)

        def flush_current_chunk() -> None:
            nonlocal current_paragraphs, current_page
            if not current_paragraphs or current_page is None:
                current_paragraphs = []
                current_page = None
                return

            text = "\n\n".join(current_paragraphs)
            tokens = self.token_counter.encode(text)
            if tokens:
                chunks.append(
                    ChunkDraft(
                        index=len(chunks),
                        text=text,
                        page=current_page,
                        section=current_section_path,
                        token_count=len(tokens),
                    )
                )
            current_paragraphs = []
            current_page = None

        def handle_heading(level: int, title: str) -> None:
            nonlocal current_section_path
            # Headings create a section boundary -> flush pending paragraphs
            flush_current_chunk()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_section_path = get_section_path()

        def add_paragraph(para_text: str, page_num: int) -> None:
            nonlocal current_paragraphs, current_page
            stripped = para_text.strip()
            if not stripped:
                return

            para_tokens = self.token_counter.encode(stripped)
            if not para_tokens:
                return

            if len(para_tokens) > self.chunk_size:
                # Oversized paragraph falls back to fixed window chunking
                flush_current_chunk()
                step = self.chunk_size - self.chunk_overlap
                start = 0
                while start < len(para_tokens):
                    end = min(start + self.chunk_size, len(para_tokens))
                    window_tokens = para_tokens[start:end]
                    chunk_text = self.token_counter.decode(window_tokens)
                    chunks.append(
                        ChunkDraft(
                            index=len(chunks),
                            text=chunk_text,
                            page=page_num,
                            section=current_section_path,
                            token_count=len(window_tokens),
                        )
                    )
                    if end >= len(para_tokens):
                        break
                    start += step
                return

            # Check if paragraph fits into current chunk buffer
            candidate_paragraphs = current_paragraphs + [stripped]
            candidate_text = "\n\n".join(candidate_paragraphs)
            candidate_tokens = self.token_counter.encode(candidate_text)

            if len(candidate_tokens) <= self.chunk_size:
                current_paragraphs.append(stripped)
                if current_page is None:
                    current_page = page_num
            else:
                flush_current_chunk()
                current_paragraphs = [stripped]
                current_page = page_num

        for page in doc.pages:
            if not page.text.strip():
                continue

            lines = page.text.splitlines()
            current_lines: list[str] = []

            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    if current_lines:
                        add_paragraph("\n".join(current_lines), page.number)
                        current_lines = []
                    continue

                heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped_line)
                if heading_match:
                    if current_lines:
                        add_paragraph("\n".join(current_lines), page.number)
                        current_lines = []
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()
                    handle_heading(level, title)
                    continue

                if stripped_line in page.headings:
                    if current_lines:
                        add_paragraph("\n".join(current_lines), page.number)
                        current_lines = []
                    handle_heading(1, stripped_line)
                    continue

                current_lines.append(stripped_line)

            if current_lines:
                add_paragraph("\n".join(current_lines), page.number)
                current_lines = []

        flush_current_chunk()
        return chunks
