"""Ingestion package: document extraction, normalization, and service coordination."""

from __future__ import annotations

from rag.ingestion.service import IngestionResult, IngestionService

__all__ = [
    "IngestionResult",
    "IngestionService",
]
