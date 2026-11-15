# Module 5: Vector Store

## Objective
Store and search chunk embeddings behind one interface, with two interchangeable
backends (FAISS and ChromaDB), proven equivalent by a shared contract test suite and
compared with a recall-vs-latency benchmark.

## Scope (may touch)
src/rag/domain/interfaces.py (add the VectorStore Protocol only, do not modify existing
ones), src/rag/domain/entities.py (add new dataclasses only), src/rag/domain/errors.py
(new errors only), src/rag/vectorstore/, src/rag/config.py (vector store settings only),
pyproject.toml (listed deps only), tests/unit/vectorstore/,
scripts/benchmark_vectorstore.py, .gitignore (index files only)

## Requirements
1. **Domain types** (no third-party imports):
   - VectorRecord(chunk_id: uuid.UUID, vector: list[float], document_id: uuid.UUID,
     metadata: dict[str, str | int | float | bool])
   - SearchHit(chunk_id: uuid.UUID, score: float)  (higher score = more similar)
   - VectorStore Protocol: `add(records: list[VectorRecord]) -> None`,
     `search(vector: list[float], k: int, filter: dict[str, ...] | None = None) -> list[SearchHit]`,
     `delete_chunks(chunk_ids: list[uuid.UUID]) -> None`,
     `delete_document(document_id: uuid.UUID) -> None`,
     `count() -> int`, `persist() -> None`
2. **Similarity**: vectors are L2-normalized (Module 4), so use inner product. Scores
   are cosine similarity in [-1, 1], identical across backends.
3. **FaissVectorStore** (vectorstore/faiss_store.py), faiss-cpu:
   - index types via Settings.faiss_index_type: "flat" (IndexFlatIP) and "hnsw"
     (IndexHNSWFlat with inner product, M and efSearch configurable)
   - FAISS ids are int64, so keep an id-map between int64 and chunk UUIDs, plus a
     per-record metadata store for filtering
   - deletion: for flat use remove_ids; for HNSW (no true delete) mark ids as
     tombstoned and exclude them from results, and rebuild the index on demand
     (`compact()`)
   - persistence: `persist()` writes the index and the id/metadata map under
     Settings.vector_index_dir, and the constructor loads them if they exist. Writes are
     atomic (write to a temp file, then rename)
   - filtering: over-fetch then post-filter, so a filter still returns up to k hits when
     enough matching records exist
   - dimension is fixed at creation; adding a different dimension raises
     DimensionMismatchError
4. **ChromaVectorStore** (vectorstore/chroma_store.py): chromadb PersistentClient under
   Settings.vector_index_dir, collection configured with cosine space, scores converted
   to cosine similarity (1 - distance), metadata filtering via the `where` clause.
5. **Behavior rules** (both backends):
   - adding a record whose chunk_id already exists replaces it (upsert semantics)
   - searching an empty store returns []
   - k larger than the store size returns all matching records
   - results are sorted by score descending, ties broken by chunk_id
   - deleting a missing id is a no-op
6. **VectorStoreFactory** from Settings (`vector_store_backend` in {"faiss", "chroma"});
   unknown value raises ValueError.
7. **Benchmark script** (scripts/benchmark_vectorstore.py): random normalized vectors
   (N=20000, dim=384, fixed seed), 200 queries, k=10. Flat is the ground truth. Prints a
   markdown table of backend / index type, build time, mean and p95 query latency, and
   Recall@10 against flat. Not run in CI.

## Acceptance criteria
- ONE contract test suite (parametrized over a fixture that yields each backend) that
  both FAISS-flat, FAISS-HNSW and Chroma pass. It covers every behavior rule above plus:
  add/search roundtrip returns the nearest vector first with score close to 1.0 for an
  identical vector; metadata filter by document_id; delete_document removes only that
  document's vectors; replace on re-add; persist then reload gives identical search
  results; DimensionMismatchError for FAISS.
- Additional FAISS tests: HNSW tombstoned ids never appear in results; compact() rebuilds
  and keeps live records; atomic persistence (a leftover temp file does not corrupt
  loading).
- Cross-backend test: on the same random data, FAISS-flat and Chroma return the same top-1
  for 50 random queries.
- Tests use tmp_path for index directories and never write into data/.
- Run this exact sequence and it passes: uv run ruff format . && uv run ruff check . &&
  uv run ruff format --check . && uv run mypy src tests && uv run pytest
- No imports from rag.db in src/rag/vectorstore/.

## Constraints
- Allowed new dependencies: faiss-cpu, chromadb (already added). Nothing else.
- Do not change existing Protocols, entities, models or repositories.
- When editing pyproject.toml, add to existing config; never replace entries.
- Write tests first, then implementation.
- Finish with a summary of design decisions and anything unfinished.