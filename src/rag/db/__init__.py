"""Database package: ORM models, engine, session factory, and repositories.

Public surface re-exported here for convenience:

    from rag.db import Base, engine, SessionLocal, get_session
"""

from rag.db.base import Base
from rag.db.engine import SessionLocal, engine
from rag.db.session import get_session

__all__ = ["Base", "SessionLocal", "engine", "get_session"]
