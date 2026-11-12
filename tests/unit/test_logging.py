"""Tests for structured logging and request-id middleware."""

import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.observability.logging import RequestIdMiddleware, setup_logging


def test_setup_logging_dev() -> None:
    """setup_logging should configure structlog with ConsoleRenderer in dev."""
    setup_logging(app_env="dev", log_level="DEBUG")
    logger = structlog.get_logger()
    assert logger is not None


def test_setup_logging_prod() -> None:
    """setup_logging should configure structlog with JSONRenderer in non-dev environments."""
    setup_logging(app_env="prod", log_level="INFO")
    logger = structlog.get_logger()
    assert logger is not None


def test_request_id_middleware_appears_in_logs() -> None:
    """RequestIdMiddleware should attach the request-id to logs and contextvars."""
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        logger = structlog.get_logger("test_ping")
        logger.info("ping_executed")
        return {"ping": "pong"}

    client = TestClient(app)

    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as captured_logs:
        response = client.get("/ping", headers={"X-Request-ID": "custom-uuid-42"})
        assert response.status_code == 200

        # Verify that captured logs contain request_id matching the header
        request_ids = [log.get("request_id") for log in captured_logs if "request_id" in log]
        assert "custom-uuid-42" in request_ids

        # Verify both handler log and middleware access log have request_id
        ping_events = [log for log in captured_logs if log.get("event") == "ping_executed"]
        assert len(ping_events) == 1
        assert ping_events[0].get("request_id") == "custom-uuid-42"

        access_events = [log for log in captured_logs if log.get("event") == "request_completed"]
        assert len(access_events) == 1
        assert access_events[0].get("request_id") == "custom-uuid-42"
        assert access_events[0].get("status_code") == 200


def test_request_id_middleware_logs_on_exception() -> None:
    """RequestIdMiddleware should log failure and re-raise on unhandled errors."""
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/error")
    def fail() -> None:
        raise RuntimeError("Simulated crash")

    client = TestClient(app, raise_server_exceptions=False)

    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as captured_logs:
        response = client.get("/error", headers={"X-Request-ID": "err-uuid-99"})
        assert response.status_code == 500

        error_events = [log for log in captured_logs if log.get("event") == "request_failed"]
        assert len(error_events) == 1
        assert error_events[0].get("request_id") == "err-uuid-99"
        assert error_events[0].get("error") == "Simulated crash"
