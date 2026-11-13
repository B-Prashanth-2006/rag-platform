.PHONY: install lint format typecheck test up down migrate migrate-down migrate-history

install:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src tests

test:
	uv run pytest

up:
	docker compose up -d --build

down:
	docker compose down

# ── Alembic helpers ────────────────────────────────────────────────────────────

migrate:
	uv run alembic upgrade head

migrate-down:
	uv run alembic downgrade base

migrate-history:
	uv run alembic history --verbose
