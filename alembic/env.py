"""Alembic sync environment.

Reads DATABASE_URL at runtime from :func:`rag.config.get_settings` so no
credentials need to be stored in ``alembic.ini``.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

import rag.db.models  # noqa: F401 — registers ORM classes with Base.metadata
from rag.config import get_settings

# Import Base *and* all models so their tables are registered in metadata.
from rag.db.base import Base

alembic_config = context.config

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Run migrations without an active DB connection (SQL emitted to stdout)."""
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a live DB connection (normal mode)."""
    configuration = dict(alembic_config.get_section(alembic_config.config_ini_section) or {})
    configuration["sqlalchemy.url"] = _get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
