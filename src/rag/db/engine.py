"""Sync SQLAlchemy engine and session-maker factory.

Both ``engine`` and ``SessionLocal`` are module-level singletons created
once at import time.  All settings come from :func:`rag.config.get_settings`;
no URLs or pool parameters are hard-coded here.
"""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag.config import get_settings


def _make_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=(settings.app_env == "dev"),
        future=True,
    )


engine: Engine = _make_engine()

SessionLocal: sessionmaker[Session] = sessionmaker(
    engine,
    expire_on_commit=False,
)
