.PHONY: install lint format typecheck test up down

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

