"""Sync session dependency for use in FastAPI routes and services.

Usage as a FastAPI dependency::

    def my_route(session: Session = Depends(get_session)) -> ...:
        ...

Usage as a plain context manager::

    with get_session() as session:
        ...

The context manager commits on success and rolls back on any exception.
"""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from rag.db.engine import SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional ``Session``, commit on success, rollback on error."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
