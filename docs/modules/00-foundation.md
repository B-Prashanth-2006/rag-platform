# Module 0: Foundation

## Objective
A production-style project skeleton: tooling, config, logging, CI, container. No RAG logic yet.

## Scope (may touch)
pyproject.toml, Makefile, Dockerfile (docker/), docker-compose.yml, .github/workflows/,
.pre-commit-config.yaml, src/rag/config.py, src/rag/main.py, src/rag/observability/logging.py,
tests/

## Requirements
1. **pyproject.toml**
   - src layout, package `rag`, Python >=3.11
   - Runtime deps: fastapi, uvicorn, pydantic, pydantic-settings, structlog
   - Dev deps (uv dependency group): pytest, pytest-cov, httpx, ruff, mypy, pre-commit
   - Configure ruff (lint + format), mypy (strict), pytest (testpaths, coverage)
2. **config.py**: `Settings` via pydantic-settings, loaded from env and `.env`.
   Fields: app_env, log_level, database_url, api_key. Provide a cached `get_settings()`.
3. **Logging**: structured JSON logs (structlog) with a request-id middleware.
   Console-friendly output when app_env=dev.
4. **main.py**: FastAPI app factory `create_app()`, `GET /health` returning
   `{"status": "ok", "version": ...}`.
5. **Makefile** targets: `install`, `lint`, `format`, `typecheck`, `test`, `up`, `down`.
6. **Docker**: multi-stage Dockerfile (non-root user, uv for installs) and a
   docker-compose service for the API with a healthcheck.
7. **CI** (GitHub Actions): on push and PR, run ruff, mypy and pytest.
8. **pre-commit**: ruff and mypy hooks.
9. **.env.example** listing every setting with safe placeholder values.

## Acceptance criteria
- `make lint`, `make typecheck` and `make test` all pass locally with zero errors.
- Tests cover: health endpoint, settings loading from env, request-id appears in logs.
- `make up` starts the API in Docker and `curl localhost:8000/health` returns 200.
- CI workflow file is valid and runs the same three checks.

## Constraints
- No dependencies beyond those listed without asking.
- No changes to `src/rag/domain/`.
- Write tests first, then implementation.
- Finish with a short summary of design decisions.