"""
Pytest configuration and shared fixtures for opensearch-agent-server tests.
"""

import os
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.opensearch_client import OpenSearchClientManager

import pytest

# ---------------------------------------------------------------------------
# Environment variables — set before any imports that read config at module level
# ---------------------------------------------------------------------------
os.environ["OPENSEARCH_URL"] = os.getenv("OPENSEARCH_URL", "http://localhost:9200")

# Mock AWS credentials for tests (real calls are patched out in unit tests)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "test-key")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv(
    "AWS_SECRET_ACCESS_KEY", "test-secret"
)
os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_env(monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict[str, str]]:
    """Patch environment variables for the duration of a test.

    Usage:
        def test_something(patch_env):
            patch_env(OPENSEARCH_URL="http://other:9200")
    """

    def _patch(clear: bool = False, **kwargs: str) -> dict[str, str]:
        if clear:
            for key in dict(os.environ):
                monkeypatch.delenv(key, raising=False)
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)
        return kwargs

    return _patch


@pytest.fixture
def test_opensearch_url() -> str:
    """Returns the test OpenSearch URL (TEST_OPENSEARCH_URL env var, default localhost:9200)."""
    return os.getenv("TEST_OPENSEARCH_URL", "http://localhost:9200")


@pytest.fixture
def opensearch_client(
    test_opensearch_url: str,
) -> Generator["OpenSearchClientManager", None, None]:
    """Provide an OpenSearch client connected to a live instance.

    Skips the test automatically if OpenSearch is not reachable.
    Start OpenSearch locally with:
        docker compose -f docker-compose.yml up -d
    """
    from utils.opensearch_client import OpenSearchClientManager

    username = os.getenv("OPENSEARCH_USERNAME", "")
    password = os.getenv("OPENSEARCH_PASSWORD", "")

    try:
        client_manager = OpenSearchClientManager(
            opensearch_url=test_opensearch_url,
            username=username,
            password=password,
            verify_certs=False,
        )
        client = client_manager.connect()
        try:
            client.info()
        except Exception as conn_error:
            pytest.skip(
                f"OpenSearch not running at {test_opensearch_url}: "
                f"{type(conn_error).__name__}: {conn_error}"
            )
        yield client_manager
    except (ConnectionError, TimeoutError, OSError) as e:
        pytest.skip(
            f"OpenSearch not running at {test_opensearch_url}: "
            f"{type(e).__name__}: {e}"
        )
    except Exception as e:
        pytest.fail(
            f"Failed to connect to OpenSearch at {test_opensearch_url}: "
            f"{type(e).__name__}: {e}"
        )
