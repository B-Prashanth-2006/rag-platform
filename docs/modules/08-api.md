# Module 8: API

## Objective
Expose the pipeline over HTTP with FastAPI. Routes are thin: parse, call a service,
return. No business logic, no ORM in routes. Gives a working end-to-end system.

## Scope (may touch)
src/rag/api/, src/rag/main.py (router registration only), src/rag/config.py (API
settings only), src/rag/domain/entities.py and errors.py (additions only),
src/rag/db/repositories/ (new files only), pyproject.toml (listed deps only),
tests/unit/api/, tests/integration/

## Requirements
1. Dependency wiring (api/deps.py): FastAPI dependencies that build, per request, a DB
   session (get_session), repositories, IngestionService and a GenerationService. Heavy
   objects (embedder, vector store, reranker, BM25 index, LLM) are created once at
   startup (lifespan) and stored on app.state. Tests override dependencies.
2. Auth: header X-API-Key checked against Settings.api_keys (constant-time compare).
   Missing or wrong key returns 401. /health stays public.
3. Endpoints (all under /v1):
   - POST /documents (multipart upload): validates size and type, stores the file under
     data/uploads, creates the job, returns 202 with {job_id, document_id, duplicate}.
     Ingestion runs in a FastAPI BackgroundTask; after extraction the background task
     also chunks, embeds, writes chunks to Postgres and vectors to the vector store and
     BM25 index (a PipelineIndexer service, no logic in the route).
   - GET /jobs/{id}: status, error, timings; 404 if unknown.
   - GET /documents?limit&offset&status: paginated {items, total, limit, offset}.
   - DELETE /documents/{id}: removes DB rows, vectors, BM25 entries and the file; 204;
     an unknown id returns 404.
   - POST /query: body {question, k=5, document_ids?, stream=false}. Non-streaming
     returns {answer, refused, citations[{index, document_id, document_name, page,
     section, text}], trace}. stream=true returns text/event-stream with events token,
     citations, done, error.
   - POST /feedback: {query_log_id, feedback: "up"|"down", comment?}.
   - Every query writes a query_logs row (question, chunk ids, latency, answer); the
     response includes query_log_id.
4. Errors: one handler maps domain errors to a consistent JSON body
   {error: {code, message, request_id}}. Mapping: validation 422, unsupported type 415,
   too large 413, not found 404, LLM unavailable 503, LLM timeout 504. Internal error
   text and stack traces are never returned; SSE error events carry a generic message
   and the real one is logged.
5. Rate limiting: in-memory sliding window per API key, Settings.rate_limit_per_minute
   (default 60); exceeding returns 429 with Retry-After.
6. CORS: configurable origins from Settings.
7. OpenAPI: titles, descriptions, response models for every route; tags per resource.

## Acceptance criteria
- Tests use TestClient with dependency overrides, FakeEmbedder, FakeLLM, FakeReranker and
  in-memory repositories where possible; no network, no downloads.
- Tests cover: 401 without a key; upload returns 202 and the job reaches succeeded;
  re-upload of the same bytes returns duplicate=true; oversized file 413; wrong type 415;
  list pagination; delete removes vectors (a later query cannot cite the document); query
  returns citations with document name, page and text; refused answer; SSE event order
  token, citations, done; LLM failure gives an error event with a generic message; 429
  after the limit; 404s; error body shape and request_id.
- One integration test: upload then query against testcontainers Postgres with fake
  models.
- Run this exact sequence and it passes: uv run ruff format . && uv run ruff check . &&
  uv run ruff format --check . && uv run mypy src tests && uv run pytest
- Routes contain no business logic and no sqlalchemy imports (grep check).

## Constraints
- Allowed new dependency: python-multipart (already added). Nothing else.
- Do not change existing Protocols, entities, models or repositories.
- When editing pyproject.toml, add to existing config; never replace entries.
- Write tests first, then implementation.
- Finish with a summary of design decisions and anything unfinished.
