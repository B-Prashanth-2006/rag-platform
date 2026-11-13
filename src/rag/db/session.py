"""Async session dependency for use in FastAPI routes and services.

Usage as a FastAPI dependency::

    async def my_route(session: AsyncSession = Depends(get_session)) -> ...:
        ...

Usage as a plain async context manager::

    async with get_session() as session:
        ...
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from rag.db.engine import AsyncSessionLocal


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional ``AsyncSession`` and close it when done."""
    async with AsyncSessionLocal() as session:
        yield session

