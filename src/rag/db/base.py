"""SQLAlchemy declarative base — the single metadata registry for all ORM models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """All ORM models inherit from this class so their tables share one MetaData."""

