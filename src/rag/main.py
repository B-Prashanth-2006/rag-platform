"""FastAPI application factory and core routes."""

from importlib import metadata

from fastapi import FastAPI

from rag.config import Settings, get_settings
from rag.observability.logging import RequestIdMiddleware, setup_logging


def get_app_version() -> str:
    """Retrieve package version from metadata or fallback."""
    try:
        return metadata.version("rag")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    setup_logging(app_env=settings.app_env, log_level=settings.log_level)

    app = FastAPI(
        title="RAG Platform API",
        version=get_app_version(),
    )

    app.add_middleware(RequestIdMiddleware)

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "version": get_app_version(),
        }

    return app
