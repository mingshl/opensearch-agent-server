"""
Unit tests for utils.opensearch_client.

Covers OpenSearchClientManager, SearchRelevanceClient, and get_client_manager:
- Client initialization (params, env fallback for URL)
- Connection management (connect, get_client, get_search_relevance_client, disconnect)
- URL parsing and OpenSearch constructor args (pool_maxsize, timeout, use_ssl, http_auth)
- URL construction and path normalization (trailing slash, path segments, different ports)
- Error handling (exceptions from OpenSearch or close() propagate)
- get_client_manager singleton behavior
- SearchRelevanceClient path building (smoke + put/delete/post)

Pragmatic scope: critical paths and contract; no exhaustive URL or
SearchRelevanceClient method coverage.
"""

from unittest.mock import MagicMock, patch

import pytest

from utils.opensearch_client import (
    OpenSearchClientManager,
    SearchRelevanceClient,
    get_client_manager,
)

pytestmark = pytest.mark.unit


class TestOpenSearchClientManagerInit:
    """Client initialization: param storage and URL defaulting."""

    def test_init_stores_explicit_params(self):
        """Explicit URL, auth, pool_maxsize, timeout, and SSL flags are stored."""
        mgr = OpenSearchClientManager(
            opensearch_url="https://host:9300",
            username="u",
            password="p",
            verify_certs=True,
            ssl_show_warn=True,
            pool_maxsize=10,
            timeout=60,
        )
        assert mgr.opensearch_url == "https://host:9300"
        assert mgr.username == "u"
        assert mgr.password == "p"
        assert mgr.verify_certs is True
        assert mgr.ssl_show_warn is True
        assert mgr.pool_maxsize == 10
        assert mgr.timeout == 60
        assert mgr.client is None
        assert mgr._search_relevance_client is None

    def test_init_url_defaults_to_env_when_url_none(self, patch_env):
        """When opensearch_url is None, OPENSEARCH_URL is used if set."""
        patch_env(OPENSEARCH_URL="http://from-env:9200")
        mgr = OpenSearchClientManager(opensearch_url=None)
        assert mgr.opensearch_url == "http://from-env:9200"

    def test_init_url_defaults_to_localhost_when_env_unset(self, monkeypatch):
        """When opensearch_url is None and OPENSEARCH_URL is unset, default to localhost:9200."""
        monkeypatch.delenv("OPENSEARCH_URL", raising=False)
        mgr = OpenSearchClientManager(opensearch_url=None)
        assert mgr.opensearch_url == "http://localhost:9200"

    def test_init_default_pool_and_timeout(self):
        """Default pool_maxsize=20 and timeout=30 when not provided."""
        mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
        assert mgr.pool_maxsize == 20
        assert mgr.timeout == 30


class TestOpenSearchClientManagerConnect:
    """connect(): OpenSearch construction, URL parsing, idempotency, SearchRelevanceClient."""

    def test_connect_passes_pool_maxsize_timeout_and_auth_to_opensearch(self):
        """connect() forwards pool_maxsize, timeout, http_auth, and SSL options to OpenSearch."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(
                opensearch_url="http://myhost:9200",
                username="u",
                password="p",
                pool_maxsize=12,
                timeout=45,
                verify_certs=True,
                ssl_show_warn=True,
            )
            mgr.connect()
            mock_os.assert_called_once()
            kwargs = mock_os.call_args[1]
            assert kwargs["http_auth"] == ("u", "p")
            assert kwargs["pool_maxsize"] == 12
            assert kwargs["timeout"] == 45
            assert kwargs["verify_certs"] is True
            assert kwargs["ssl_show_warn"] is True
            assert kwargs["use_ssl"] is False

    def test_connect_idempotent_returns_same_client(self):
        """Second connect() returns the same client instance."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ):
            mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
            c1 = mgr.connect()
            c2 = mgr.connect()
            assert c1 is c2

    def test_connect_parses_http_url_host_and_port(self):
        """http://host:9200 produces host=host, port=9200, use_ssl=False."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="http://myhost:9200")
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "myhost", "port": 9200}]
            assert call_kw["use_ssl"] is False

    def test_connect_http_without_port_uses_9200(self):
        """http://host with no port uses default 9200 and use_ssl=False."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="http://myhost")
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "myhost", "port": 9200}]
            assert call_kw["use_ssl"] is False

    def test_connect_https_without_port_uses_443(self):
        """https://host with no port uses 443 and use_ssl=True."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="https://myhost")
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "myhost", "port": 443}]
            assert call_kw["use_ssl"] is True

    def test_connect_parses_https_url_uses_443_when_no_port(self):
        """https://host with no port uses 443 and use_ssl=True."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="https://securehost")
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "securehost", "port": 443}]
            assert call_kw["use_ssl"] is True

    def test_connect_parses_url_with_path_strips_path_for_port(self):
        """URL like http://host:9300/path uses port 9300 (path stripped for port)."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="http://h:9300/some/path")
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "h", "port": 9300}]

    def test_connect_url_with_multiple_path_segments_parses_port_correctly(self):
        """URL with multiple path segments (e.g. /a/b/c) still parses host and port correctly."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(
                opensearch_url="http://node1:9200/opensearch/dashboards/path"
            )
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "node1", "port": 9200}]
            assert call_kw["use_ssl"] is False


class TestOpenSearchClientManagerUrlAndPathBehavior:
    """URL construction, path normalization, and connection-parameter handling.

    Covers OPENSEARCH_URL with/without trailing slash, path segments, and
    different ports so that connect() host/port/use_ssl behavior is well-defined.
    """

    def test_connect_url_with_trailing_slash_http(self):
        """http://host:9200/ parses to host, port 9200, use_ssl False."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="http://myhost:9200/")
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "myhost", "port": 9200}]
            assert call_kw["use_ssl"] is False

    def test_connect_url_with_trailing_slash_https(self):
        """https://host:443/ parses to host, port 443, use_ssl True."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="https://securehost:443/")
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "securehost", "port": 443}]
            assert call_kw["use_ssl"] is True

    def test_connect_url_with_path_segment_normalizes_port(self):
        """URL with path (e.g. /opensearch) strips path for port; host/port correct."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="http://node1:9200/opensearch")
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "node1", "port": 9200}]
            assert call_kw["use_ssl"] is False

    def test_init_stores_url_with_trailing_slash_from_env(self, patch_env):
        """OPENSEARCH_URL with trailing slash is stored as-is; connect still parses."""
        patch_env(OPENSEARCH_URL="https://envhost:9200/")
        mgr = OpenSearchClientManager(opensearch_url=None)
        assert mgr.opensearch_url == "https://envhost:9200/"
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "envhost", "port": 9200}]
            assert call_kw["use_ssl"] is True

    def test_connect_http_custom_port(self):
        """http://host:9201 uses port 9201 and use_ssl False."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="http://custom:9201")
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "custom", "port": 9201}]
            assert call_kw["use_ssl"] is False

    def test_connect_https_custom_port(self):
        """https://host:8443 uses port 8443 and use_ssl True."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="https://secure:8443")
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "secure", "port": 8443}]
            assert call_kw["use_ssl"] is True

    def test_connect_connection_params_passed_through(self):
        """pool_maxsize, timeout, verify_certs, ssl_show_warn passed to OpenSearch for any URL."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(
                opensearch_url="http://localhost:9200/",
                pool_maxsize=8,
                timeout=15,
                verify_certs=True,
                ssl_show_warn=True,
            )
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "localhost", "port": 9200}]
            assert call_kw["pool_maxsize"] == 8
            assert call_kw["timeout"] == 15
            assert call_kw["verify_certs"] is True
            assert call_kw["ssl_show_warn"] is True

    def test_connect_sets_use_ssl_for_https(self):
        """https URL sets use_ssl=True."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="https://h:9200")
            mgr.connect()
            assert mock_os.call_args[1]["use_ssl"] is True

    def test_connect_creates_search_relevance_client(self):
        """After connect(), _search_relevance_client wraps the same client."""
        mock_client = MagicMock()
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=mock_client,
        ):
            mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
            mgr.connect()
            assert mgr._search_relevance_client is not None
            assert isinstance(mgr._search_relevance_client, SearchRelevanceClient)
            assert mgr._search_relevance_client._client is mock_client

    def test_connect_passes_connection_params_when_url_has_path(self):
        """pool_maxsize, timeout, and auth passed to OpenSearch when URL has path segment."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(
                opensearch_url="http://host:9200/opensearch",
                username="u",
                password="p",
                pool_maxsize=5,
                timeout=10,
            )
            mgr.connect()
            call_kw = mock_os.call_args[1]
            assert call_kw["hosts"] == [{"host": "host", "port": 9200}]
            assert call_kw["http_auth"] == ("u", "p")
            assert call_kw["pool_maxsize"] == 5
            assert call_kw["timeout"] == 10


class TestOpenSearchClientManagerGetClient:
    """get_client(): lazy connect and caching."""

    def test_get_client_lazy_connect(self):
        """get_client() triggers connect() when client is None."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
            assert mgr.client is None
            _ = mgr.get_client()
            mock_os.assert_called_once()

    def test_get_client_returns_cached_after_connect(self):
        """get_client() returns the same instance as connect() and client."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ):
            mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
            mgr.connect()
            g = mgr.get_client()
            assert g is mgr.client


class TestOpenSearchClientManagerGetSearchRelevanceClient:
    """get_search_relevance_client(): lazy connect and caching."""

    def test_get_search_relevance_client_lazy_connect(self):
        """get_search_relevance_client() triggers connect() when _search_relevance_client is None."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ) as mock_os:
            mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
            _ = mgr.get_search_relevance_client()
            mock_os.assert_called_once()

    def test_get_search_relevance_client_returns_same_after_connect(self):
        """get_search_relevance_client() returns the same instance as after connect()."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=MagicMock(),
        ):
            mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
            mgr.connect()
            sr = mgr.get_search_relevance_client()
            assert sr is mgr._search_relevance_client


class TestOpenSearchClientManagerConnectDisconnectLogging:
    """Security: connect/disconnect log_info_event must never log passwords.

    Do not add tests purely to hit logging line coverage. A single test that
    asserts we never log sensitive substrings (e.g. password) is justified when
    there is an explicit requirement ("we must never log passwords").
    """

    def test_log_info_event_never_includes_password(self):
        """log_info_event in connect() and disconnect() must never receive password."""
        with patch(
            "utils.opensearch_client.log_info_event",
        ) as mock_log:
            with patch(
                "utils.opensearch_client.OpenSearch",
                return_value=MagicMock(),
            ):
                mgr = OpenSearchClientManager(
                    opensearch_url="http://localhost:9200",
                    username="testuser",
                    password="s3cretp@ss",
                )
                mgr.connect()
                mgr.disconnect()
        assert mock_log.called, (
            "connect and disconnect must call log_info_event (sanity check)"
        )
        for call in mock_log.call_args_list:
            args = call[0]
            kwargs = call[1]
            for v in list(args) + list(kwargs.values()):
                s = str(v)
                assert "s3cretp@ss" not in s, (
                    "password must not appear in log_info_event args or kwargs"
                )


class TestOpenSearchClientManagerDisconnect:
    """disconnect(): close, clear state, idempotency."""

    def test_disconnect_closes_client_and_clears_state(self):
        """disconnect() calls client.close() and sets client and _search_relevance_client to None."""
        mock_client = MagicMock()
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=mock_client,
        ):
            mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
            mgr.connect()
            mgr.disconnect()
            mock_client.close.assert_called_once()
            assert mgr.client is None
            assert mgr._search_relevance_client is None

    def test_disconnect_idempotent_when_already_disconnected(self):
        """disconnect() when client is None does not raise."""
        mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
        mgr.disconnect()
        mgr.disconnect()

    def test_disconnect_when_close_raises_propagates(self):
        """disconnect() does not swallow if client.close() raises."""
        mock_client = MagicMock()
        mock_client.close.side_effect = OSError("close failed")
        with patch(
            "utils.opensearch_client.OpenSearch",
            return_value=mock_client,
        ):
            mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
            mgr.connect()
            with pytest.raises(OSError, match="close failed"):
                mgr.disconnect()


class TestOpenSearchClientManagerErrorHandling:
    """Error handling: exceptions from OpenSearch constructor or close propagate."""

    def test_connect_when_opensearch_raises_propagates(self):
        """connect() propagates if OpenSearch constructor raises."""
        with patch(
            "utils.opensearch_client.OpenSearch",
            side_effect=ConnectionError("cannot connect"),
        ):
            mgr = OpenSearchClientManager(opensearch_url="http://localhost:9200")
            with pytest.raises(ConnectionError, match="cannot connect"):
                mgr.connect()


class TestGetClientManager:
    """get_client_manager(): singleton creation and reuse."""

    def test_get_client_manager_creates_manager_with_args_on_first_call(self):
        """First call creates OpenSearchClientManager with passed URL, username, password."""
        with patch("utils.opensearch_client._client_manager", None):
            mgr = get_client_manager(
                opensearch_url="http://first:9200",
                username="u1",
                password="p1",
            )
            assert mgr.opensearch_url == "http://first:9200"
            assert mgr.username == "u1"
            assert mgr.password == "p1"

    def test_get_client_manager_returns_same_instance_on_subsequent_calls(self):
        """Second call returns the same manager; constructor args are ignored."""
        with patch("utils.opensearch_client._client_manager", None):
            first = get_client_manager(
                opensearch_url="http://first:9200",
                username="u1",
                password="p1",
            )
            second = get_client_manager(
                opensearch_url="http://other:9200",
                username="u2",
                password="p2",
            )
            assert first is second
            assert first.opensearch_url == "http://first:9200"


class TestSearchRelevanceClient:
    """SearchRelevanceClient: path building (smoke tests)."""

    def test_get_search_configurations_with_id_builds_correct_path(self):
        """get_search_configurations(id) calls perform_request with ID in path."""
        mock_transport = MagicMock()
        mock_transport.perform_request.return_value = {}
        mock_client = MagicMock()
        mock_client.transport = mock_transport
        src = SearchRelevanceClient(mock_client)
        src.get_search_configurations(search_configuration_id="cfg-1")
        mock_transport.perform_request.assert_called_once_with(
            "GET",
            "/_plugins/_search_relevance/search_configurations/cfg-1",
            params={},
        )

    def test_get_search_configurations_without_id_builds_list_path(self):
        """get_search_configurations() without ID uses base path."""
        mock_transport = MagicMock()
        mock_transport.perform_request.return_value = {}
        mock_client = MagicMock()
        mock_client.transport = mock_transport
        src = SearchRelevanceClient(mock_client)
        src.get_search_configurations()
        mock_transport.perform_request.assert_called_once_with(
            "GET",
            "/_plugins/_search_relevance/search_configurations",
            params={},
        )

    def test_put_judgments_calls_perform_request_with_put_and_path(self):
        """put_judgments() calls perform_request with PUT, base path, and body."""
        mock_transport = MagicMock()
        mock_transport.perform_request.return_value = {}
        mock_client = MagicMock()
        mock_client.transport = mock_transport
        src = SearchRelevanceClient(mock_client)
        body = {"query_id": "q1", "doc_id": "d1", "grade": 1}
        src.put_judgments(body=body)
        mock_transport.perform_request.assert_called_once_with(
            "PUT",
            "/_plugins/_search_relevance/judgments",
            body=body,
            params={},
        )

    def test_delete_experiments_calls_perform_request_with_delete_and_id_path(self):
        """delete_experiments(experiment_id) calls perform_request with DELETE and ID in path."""
        mock_transport = MagicMock()
        mock_transport.perform_request.return_value = {}
        mock_client = MagicMock()
        mock_client.transport = mock_transport
        src = SearchRelevanceClient(mock_client)
        src.delete_experiments(experiment_id="exp-123")
        mock_transport.perform_request.assert_called_once_with(
            "DELETE",
            "/_plugins/_search_relevance/experiments/exp-123",
            params={},
        )

    def test_post_query_sets_calls_perform_request_with_post_and_path(self):
        """post_query_sets() calls perform_request with POST, base path, and body."""
        mock_transport = MagicMock()
        mock_transport.perform_request.return_value = {}
        mock_client = MagicMock()
        mock_client.transport = mock_transport
        src = SearchRelevanceClient(mock_client)
        body = {"index": "my-index", "size": 100}
        src.post_query_sets(body=body)
        mock_transport.perform_request.assert_called_once_with(
            "POST",
            "/_plugins/_search_relevance/query_sets",
            body=body,
            params={},
        )

