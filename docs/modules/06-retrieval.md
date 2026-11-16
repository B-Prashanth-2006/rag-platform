# Module 6: Retrieval

## Objective
Given a query, return the best chunks using hybrid search (dense + BM25), reciprocal
rank fusion and cross-encoder reranking. Every stage is a composable component behind
one interface, toggleable from config, and every query can return a per-stage debug
trace. Quality is measured in Module 9, so nothing here is tuned by guesswork.

## Scope (may touch)
src/rag/domain/interfaces.py (add Protocols only, do not modify existing ones),
src/rag/domain/entities.py (add new dataclasses only), src/rag/domain/errors.py (new
errors only), src/rag/retrieval/, src/rag/db/repositories/chunk_lookup.py (new file
only), src/rag/config.py (retrieval settings only), pyproject.toml (listed deps only),
tests/unit/retrieval/, tests/integration/

## Requirements
1. **Domain types** (no third-party imports):
   - RetrievalFilter(document_ids: list[uuid.UUID] | None = None)
   - RetrievedChunk(chunk_id, document_id, text, page: int | None, section: str | None,
     score: float, source: str)  # source in {"dense","sparse","fused","reranked"}
   - StageTrace(stage: str, latency_ms: float, results: list[tuple[uuid.UUID, float]])
   - RetrievalResult(chunks: list[RetrievedChunk], trace: list[StageTrace])
   - Retriever Protocol: `retrieve(query: str, k: int, filter: RetrievalFilter | None = None) -> RetrievalResult`
   - ChunkLookup Protocol: `get_many(chunk_ids: list[uuid.UUID]) -> list[Chunk]`
     (returns found chunks in the requested order, silently skipping missing ids)
   - SparseIndex Protocol: `add(chunks: list[Chunk]) -> None`,
     `search(query: str, k: int, filter: RetrievalFilter | None = None) -> list[tuple[uuid.UUID, float]]`,
     `delete_document(document_id: uuid.UUID) -> None`, `count() -> int`
   - Reranker Protocol: `rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]`
2. **SqlChunkLookup** (db/repositories/chunk_lookup.py): implements ChunkLookup with the
   existing ORM; never commits. Also an InMemoryChunkLookup (retrieval/fakes.py) for tests.
3. **DenseRetriever**: embeds the query with Embedder.embed_query, searches the
   VectorStore (applying the filter as document_id metadata), hydrates text through
   ChunkLookup. Scores are cosine similarity.
4. **Bm25Index** (retrieval/bm25.py), implements SparseIndex using rank_bm25 (in memory):
   lowercase tokenization, simple stopword list, filter by document_ids, deletion by
   document, deterministic ordering (score desc, chunk_id tiebreak). Empty index or empty
   query returns [].
5. **SparseRetriever**: Bm25Index plus ChunkLookup, same Retriever interface.
6. **ReciprocalRankFusion** (retrieval/fusion.py), pure function:
   score(d) = sum over lists of 1 / (k + rank), k = 60 default (rank starts at 1),
   weights optional per list. Deterministic tie-break by chunk_id. Output source "fused".
7. **HybridRetriever**: runs dense and sparse (each fetching `candidate_k`, default 50),
   fuses with RRF, returns the top `fusion_k`.
8. **CrossEncoderReranker** (retrieval/reranker.py): transformers
   AutoModelForSequenceClassification, default BAAI/bge-reranker-base, device cuda, then
   mps, then cpu (same logic as the embedder), batched, torch.no_grad(), raw logits
   converted with a sigmoid to a 0..1 relevance score. Reranks only the top
   `rerank_candidates` (default 20), returns `top_k`. Empty input returns [].
9. **RetrievalPipeline** (retrieval/pipeline.py), built by `build_retriever(settings, ...)`:
   - Settings toggles: retrieval_mode in {"dense","sparse","hybrid"}, use_reranker (bool),
     candidate_k, fusion_k, rerank_candidates, rrf_k
   - every stage records a StageTrace with latency in milliseconds
   - the final RetrievalResult always carries the trace
   - unknown mode raises ValueError
10. **Confidence**: RetrievalResult exposes `top_score` (the best final score, or None
    when empty). Module 7 uses it for the refusal path.
11. **Metadata filter** is applied in every stage (dense, sparse, hybrid).

## Acceptance criteria
- Unit tests use FakeEmbedder, an in-memory vector store backend (FAISS flat on tmp_path),
  InMemoryChunkLookup and a FakeReranker; they need no network.
- RRF tests: hand-computed expected scores on small lists; a chunk present in both lists
  outranks one present in one; weights work; ties are deterministic.
- BM25 tests: exact-term query ranks the matching chunk first (including an identifier
  like "ERR-4521" that dense search would miss); filter by document; delete_document.
- Hybrid test: a chunk found only by dense and a chunk found only by sparse both appear
  in the fused result.
- Reranker test with the real model: a relevant passage outranks an irrelevant one;
  skipped automatically on OSError (download failure). Device selection tested with
  monkeypatching.
- Pipeline tests: each mode and the reranker toggle change the stages in the trace;
  filter respected; empty index returns an empty result with top_score None.
- One integration test: SqlChunkLookup on testcontainers Postgres returns chunks in the
  requested order and skips missing ids.
- Run this exact sequence and it passes: uv run ruff format . && uv run ruff check . &&
  uv run ruff format --check . && uv run mypy src tests && uv run pytest
- No imports from rag.db in src/rag/retrieval/.

## Constraints
- Allowed new dependencies: rank-bm25 (already added). Nothing else.
- Do not change existing Protocols, entities, models or repositories.
- When editing pyproject.toml, add to existing config; never replace entries.
- Write tests first, then implementation.
- Finish with a summary of design decisions and anything unfinished.