# Module 3: Chunking

## Objective
Split extracted documents into retrieval-sized chunks with metadata. Chunking is the
biggest lever on RAG quality, so every strategy sits behind one interface and is
measured later in Module 9. The semantic strategy is NOT part of this module (it needs
embeddings, Module 4).

## Scope (may touch)
src/rag/domain/interfaces.py (add the Chunker and TokenCounter Protocols only, do not
modify existing ones), src/rag/domain/entities.py (add new dataclasses only),
src/rag/domain/errors.py (new errors only), src/rag/chunking/, src/rag/config.py
(chunking settings only), pyproject.toml (listed deps only), tests/unit/chunking/

## Requirements
1. **New domain types** (plain dataclasses, no third-party imports):
   - ChunkDraft(index: int, text: str, page: int | None, section: str | None,
     token_count: int). A draft has no ids; ids are assigned when persisting.
   - TokenCounter Protocol: `count(text: str) -> int` and
     `encode(text: str) -> list[int]` / `decode(tokens: list[int]) -> str`
   - Chunker Protocol: `chunk(doc: ExtractedDocument) -> list[ChunkDraft]`
2. **Token counter** (chunking/tokenizer.py): HuggingFaceTokenCounter wrapping an
   AutoTokenizer, model name from Settings (default BAAI/bge-small-en-v1.5), with
   add_special_tokens=False. A WhitespaceTokenCounter (words = tokens) is provided for
   fast unit tests and needs no downloads.
3. **FixedWindowChunker**: windows of `chunk_size` tokens with `chunk_overlap` tokens of
   overlap, built over the concatenated page text. Each chunk records the page where it
   starts. Validation: chunk_size > 0, 0 <= overlap < chunk_size, else ValueError.
4. **StructureAwareChunker**: splits on headings and paragraphs first, then packs
   paragraphs into chunks up to `chunk_size` tokens without splitting a paragraph unless
   it alone exceeds the limit (then fall back to the fixed window for that paragraph).
   Each chunk records its page and its section path (for example
   "Introduction > Background") built from the headings seen so far. Never merge text
   across a top-level heading boundary.
5. **ChunkerFactory**: builds a chunker from Settings (`chunker_strategy` in
   {"fixed", "structure"}, `chunk_size` default 256, `chunk_overlap` default 32); an
   unknown strategy raises ValueError.
6. **Invariants** for every chunker:
   - chunk indexes are 0..n-1 with no gaps
   - every chunk has token_count > 0 and token_count <= chunk_size
   - empty documents produce an empty list
   - deterministic: same input gives identical output
   - no text is lost: for the fixed chunker, joining the chunks with overlaps removed
     reproduces the full normalized text; for the structure chunker, every word of the
     input appears in some chunk

## Acceptance criteria
- Unit tests use WhitespaceTokenCounter (fast, offline). One test file uses the real
  HuggingFaceTokenCounter and is skipped automatically when the tokenizer cannot be
  downloaded (pytest skip on OSError).
- Property-style tests (hypothesis is NOT allowed; use parametrized randomized inputs
  with a fixed seed): size limit holds, overlap is exactly chunk_overlap tokens between
  consecutive fixed chunks, indexes are contiguous, no text lost, determinism.
- Tests cover: multi-page documents (page attribution), headings producing section
  paths, an oversized paragraph falling back to the window, an empty document, and
  invalid parameters.
- Run this exact sequence and it passes: uv run ruff format . && uv run ruff check . &&
  uv run ruff format --check . && uv run mypy src tests && uv run pytest
- No imports from rag.db anywhere in src/rag/chunking/.

## Constraints
- Allowed new dependencies: transformers, tokenizers. Nothing else (no torch yet).
- Do not change existing Protocols, entities, models or repositories.
- When editing pyproject.toml, add to existing config; never replace entries.
- Write tests first, then implementation.
- Finish with a summary of design decisions and anything unfinished.