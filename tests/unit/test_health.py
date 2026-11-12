"""Tests for the health check endpoint and application factory."""

from importlib import metadata
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag.main import create_app, get_app_version


def test_health_endpoint_success() -> None:
    """GET /health should return 200 with status ok and version."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["version"] == "0.1.0"


def test_health_endpoint_generates_request_id() -> None:
    """Response should include an X-Request-ID header generated if not provided."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    assert len(request_id) > 0


def test_health_endpoint_propagates_request_id() -> None:
    """Response should propagate the incoming X-Request-ID header."""
    app = create_app()
    client = TestClient(app)

    custom_id = "test-request-id-98765"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


def test_get_app_version_fallback() -> None:
    """get_app_version should fallback to 0.1.0 if package metadata is not found."""
    with patch("importlib.metadata.version", side_effect=metadata.PackageNotFoundError):
        assert get_app_version() == "0.1.0"
