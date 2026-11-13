"""Shared SQLAlchemy Enum type objects for PostgreSQL native enums.

Defined here (not in ``models.py``) so ``alembic/env.py`` can import them
without pulling in the full model graph, and to avoid circular imports.
"""

from sqlalchemy import Enum as SAEnum

from rag.domain.entities import DocumentStatus, JobStatus

DOCUMENT_STATUS_ENUM: SAEnum = SAEnum(
    DocumentStatus,
    name="document_status",
    create_constraint=True,
    validate_strings=True,
)

JOB_STATUS_ENUM: SAEnum = SAEnum(
    JobStatus,
    name="job_status",
    create_constraint=True,
    validate_strings=True,
)
