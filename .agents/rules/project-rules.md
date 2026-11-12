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