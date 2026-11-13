"""Tests for configuration management."""

import pytest

from rag.config import Settings, get_settings


def test_settings_defaults() -> None:
    """Settings should have default values specified in the requirements."""
    settings = Settings()
    assert settings.app_env == "dev"
    assert settings.log_level == "INFO"
    assert settings.database_url == "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
    assert settings.api_key == "dev-secret-key"


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should properly load values overridden by environment variables."""
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@custom-host:5432/custom_db")
    monkeypatch.setenv("API_KEY", "prod-api-key-12345")

    settings = Settings()
    assert settings.app_env == "prod"
    assert settings.log_level == "WARNING"
    assert settings.database_url == "postgresql+psycopg://user:pass@custom-host:5432/custom_db"
    assert settings.api_key == "prod-api-key-12345"


def test_get_settings_cached() -> None:
    """get_settings should return a cached singleton instance."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
