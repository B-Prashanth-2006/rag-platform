# Architecture

## Goal
A document intelligence platform: ingest documents, retrieve relevant passages with
hybrid search and reranking, and generate cited answers. Quality is measured, not assumed.

## Pipelines

**Ingestion:** Upload -> Extract -> Clean -> Chunk -> Embed -> Index (vector + keyword)
Metadata, chunks and job state are stored in PostgreSQL.

**Query:** Query -> (Rewrite) -> Hybrid Retrieve -> Rerank -> Context Builder -> LLM -> Answer + Citations

## Layering rules (strict)
1. `domain/` has entities and Protocols only. It imports nothing from other `rag` packages.
2. Implementations depend on `domain/`, never on each other directly.
3. `api/` routes are thin: parse request, call a service, return a response. No business logic.
4. Only `db/` repositories touch SQLAlchemy. Services receive repositories through DI.
5. All configuration comes from `config.py` (pydantic-settings). No hardcoded values.
6. Every pipeline stage sits behind an interface so it can be swapped and tested.

## Key interfaces (domain/interfaces.py)
- `Extractor`: file bytes -> structured document (pages, headings, tables)
- `Chunker`: document -> list of chunks with metadata
- `Embedder`: texts -> normalized vectors, with model name and version
- `VectorStore`: add / search / delete / filter
- `Retriever`: query -> ranked chunks (dense, sparse, hybrid and rerank all implement this)
- `LLM`: prompt -> text or token stream

## Data model (PostgreSQL)
- documents(id, filename, content_hash UNIQUE, status, metadata JSONB, created_at, updated_at)
- chunks(id, document_id, index, text, page, section, token_count, embedding_model_version)
- ingestion_jobs(id, document_id, status, error, started_at, finished_at)
- query_logs(id, query, retrieved_chunk_ids, latency_ms, answer, feedback, created_at)

## Principles
- Idempotent ingestion: the same file uploaded twice does nothing.
- Every embedding records its model version so stale indexes can be detected.
- Every retrieval or chunking decision is justified by the evaluation harness (Module 9).
- Failures are explicit: corrupt files, LLM timeouts and empty indexes have defined behavior.

## Decisions
Short ADRs live in `docs/decisions/`. Module specs live in `docs/modules/`.