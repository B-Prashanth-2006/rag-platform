# Module 4: Embeddings

## Objective
Turn text into normalized vectors with a Hugging Face model, fast and reproducibly,
and record which model produced every vector so stale indexes can be detected.

## Scope (may touch)
src/rag/domain/interfaces.py (add the Embedder Protocol only, do not modify existing
ones), src/rag/domain/entities.py (add new dataclasses only), src/rag/domain/errors.py
(new errors only), src/rag/embeddings/, src/rag/config.py (embedding settings only),
pyproject.toml (listed deps only), tests/unit/embeddings/, scripts/benchmark_embeddings.py

## Requirements
1. **Domain types** (no third-party imports):
   - EmbeddingModelInfo(name: str, version: str, dimension: int, max_tokens: int)
   - Embedder Protocol: `info` property, `embed_passages(texts: list[str]) -> list[list[float]]`,
     `embed_query(text: str) -> list[float]`
2. **HuggingFaceEmbedder** (embeddings/hf.py), default model BAAI/bge-small-en-v1.5
   (name from Settings):
   - device selection: cuda, then mps, then cpu, overridable by Settings.embedding_device
   - batching (Settings.embedding_batch_size, default 32), inference under torch.no_grad()
   - truncation to the model max length; mean or CLS pooling as the model requires
     (bge uses CLS), then L2 normalization so cosine similarity equals dot product
   - correct prefixes: bge queries use "Represent this sentence for searching relevant
     passages: ", passages use no prefix. Prefixes are configurable per model family
     (e5 uses "query: " and "passage: ")
   - empty input list returns an empty list; empty strings do not crash
   - `info.version` is the model revision/commit hash when available, otherwise "unknown"
3. **CachedEmbedder** (embeddings/cache.py): wraps any Embedder with an on-disk cache
   (SQLite file under data/, standard library only) keyed by (model name, model version,
   kind, SHA-256 of text). Only cache misses are sent to the wrapped embedder, in one
   batch. Preserves input order. Thread-safe enough for a single process.
4. **FakeEmbedder** (embeddings/fake.py): deterministic hash-based vectors, for tests of
   other modules; no downloads.
5. **EmbedderFactory** from Settings; unknown model family raises ValueError.
6. **Benchmark script** (scripts/benchmark_embeddings.py): embeds N synthetic passages,
   prints device, batch size, texts per second and p95 batch latency as a markdown table.
   Not run in CI.

## Acceptance criteria
- Sanity test with the real model: "A cat sits on the mat" scores higher with
  "A kitten rests on a rug" than with "Quarterly revenue grew by 8 percent".
  Skipped automatically if the model cannot be downloaded (skip on OSError).
- Tests: output vectors have unit norm (within 1e-5); batching gives the same vectors as
  one-by-one embedding (within 1e-4); query and passage embeddings differ for the same
  text; empty list and empty string handled; device selection logic tested with
  monkeypatching (cuda, mps, cpu paths).
- Cache tests (use FakeEmbedder): second call hits the cache and the wrapped embedder is
  not called; mixed hit/miss batch preserves order; different model version means a miss.
- Run this exact sequence and it passes: uv run ruff format . && uv run ruff check . &&
  uv run ruff format --check . && uv run mypy src tests && uv run pytest
- No imports from rag.db in src/rag/embeddings/.

## Constraints
- Allowed new dependencies: torch (already added). Nothing else.
- Do not change existing Protocols, entities, models or repositories.
- When editing pyproject.toml, add to existing config; never replace entries.
- Write tests first, then implementation.
- Finish with a summary of design decisions and anything unfinished.
