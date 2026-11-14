---
trigger: always_on
---

# Project rules
- Read ARCHITECTURE.md and the relevant docs/modules/*.md before coding.
- Full type hints on all code; mypy strict must pass.
- No business logic in API routes. No ORM use outside db/ repositories.
- No hardcoded config; use Settings.
- Write failing tests first, then the implementation.
- Do not add dependencies or edit files outside the module scope without asking.
- Never change Protocols in domain/interfaces.py without explicit approval.
- After each task, summarize design decisions and anything left unfinished.
- Target platform is macOS on Apple Silicon (arm64). Dev runs natively; Docker is for Postgres and the API.
- Only use Docker images that support linux/arm64.
- Device selection for PyTorch must try cuda, then mps, then cpu.
- Do not run Ollama or GPU workloads in Docker; reach the host via host.docker.internal.
- Use named volumes, not bind mounts, for database and index data.
- Before finishing any task, run `uv run ruff format .` first, then the exact CI sequence: uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest. Show the full output. Never claim checks passed without running them.
- When editing pyproject.toml, add to existing config; never replace or remove existing entries without asking.