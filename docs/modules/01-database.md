# Module 1: Domain Models and Database

## Objective
A PostgreSQL persistence layer with migrations and repositories. Services never touch
the ORM; they depend on repository Protocols.

## Scope (may touch)
src/rag/domain/entities.py, src/rag/domain/interfaces.py (add repository Protocols only,
do not modify existing ones), src/rag/db/, src/rag/config.py (database settings only),
docker-compose.yml (add postgres service), pyproject.toml (listed deps only),
alembic.ini, tests/unit/db/, tests/integration/, Makefile (add targets)

## Requirements
1. **Entities** (domain/entities.py): plain dataclasses (no SQLAlchemy imports):
   Document, Chunk, IngestionJob, QueryLog, plus enums DocumentStatus
   (pending, processing, ready, failed) and JobStatus (queued, running, succeeded, failed).
2. **Repository Protocols** (domain/interfaces.py): DocumentRepository, ChunkRepository,
   JobRepository, QueryLogRepository. Methods return domain entities, never ORM objects.
3. **ORM models** (db/models.py): SQLAlchemy 2.0 typed style (Mapped, mapped_column).
   Tables per ARCHITECTURE.md. Requirements:
   - documents.content_hash UNIQUE
   - chunks.document_id FK with ON DELETE CASCADE, UNIQUE(document_id, index)
   - indexes on documents.status, chunks.document_id, query_logs.created_at
   - metadata as JSONB, timestamps timezone-aware (UTC)
4. **Repositories** (db/repositories/): implementations of each Protocol, with mappers
   between ORM models and entities. Needed operations:
   - Documents: add, get, get_by_hash, list (paginated, filter by status), update_status, delete
   - Chunks: add_many (bulk), list_by_document, delete_by_document, count_by_model_version
   - Jobs: create, get, mark_running, mark_succeeded, mark_failed(error)
   - QueryLogs: add, add_feedback, list_recent
5. **Session handling** (db/session.py): engine and session factory from Settings,
   a unit-of-work or session-per-request pattern with commit/rollback.
   Use sync SQLAlchemy 2.0 with psycopg 3.
6. **Alembic**: configured to read the URL from Settings. One initial migration
   generated from the models. `make migrate` (upgrade head) and
   `make migration m="message"` targets.
7. **docker-compose**: postgres service with healthcheck, named volume, env from .env.
   Update .env.example.

## Acceptance criteria
- `alembic upgrade head` then `alembic downgrade base` both run cleanly on an empty DB.
- Repository tests run against a real Postgres (testcontainers-python or a dedicated
  test database in compose), not SQLite.
- Tests cover: duplicate content_hash is rejected, cascade delete removes chunks,
  pagination, status transitions for jobs, rollback on error.
- No SQLAlchemy imports outside src/rag/db/ (verify with a grep or an import-linter test).
- make lint, make typecheck, make test pass.

## Constraints
- Allowed new dependencies: sqlalchemy, alembic, psycopg[binary]; dev: testcontainers.
- Do not change existing Protocols. Only add new ones.
- Write tests first, then implementation.
- Finish with a summary of design decisions and anything unfinished.