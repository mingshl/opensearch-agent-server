"""
OpenSearch Client Connection Utility
Provides shared access to OpenSearch Python client with search_relevance plugin.
"""

from __future__ import annotations

import os
from typing import Any

from opensearchpy import OpenSearch

from utils.logging_helpers import get_logger, log_info_event

logger = get_logger(__name__)


class SearchRelevanceClient:
    """
    Client for OpenSearch Search Relevance plugin API.
    Provides methods to interact with search relevance endpoints.
    """

    def __init__(self, client: OpenSearch) -> None:
        """
        Initialize the search relevance client.

        Args:
            client: The main OpenSearch client instance
        """
        self._client = client

    def get_search_configurations(
        self,
        search_configuration_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get search configurations."""
        if search_configuration_id:
            path = f"/_plugins/_search_relevance/search_configurations/{search_configuration_id}"
        else:
            path = "/_plugins/_search_relevance/search_configurations"
        return self._client.transport.perform_request("GET", path, params=kwargs)

    def put_search_configurations(self, body: Any | None = None, **kwargs: Any) -> Any:
        """Create or update a search configuration."""
        path = "/_plugins/_search_relevance/search_configurations"
        return self._client.transport.perform_request(
            "PUT", path, body=body, params=kwargs
        )

    def delete_search_configurations(
        self, search_configuration_id: str, **kwargs: Any
    ) -> Any:
        """Delete a search configuration."""
        path = f"/_plugins/_search_relevance/search_configurations/{search_configuration_id}"
        return self._client.transport.perform_request("DELETE", path, params=kwargs)

    def get_judgments(self, judgment_id: str | None = None, **kwargs: Any) -> Any:
        """Get judgments."""
        if judgment_id:
            path = f"/_plugins/_search_relevance/judgments/{judgment_id}"
        else:
            path = "/_plugins/_search_relevance/judgments"
        return self._client.transport.perform_request("GET", path, params=kwargs)

    def put_judgments(self, body: Any | None = None, **kwargs: Any) -> Any:
        """Create or update a judgment."""
        path = "/_plugins/_search_relevance/judgments"
        return self._client.transport.perform_request(
            "PUT", path, body=body, params=kwargs
        )

    def delete_judgments(self, judgment_id: str, **kwargs: Any) -> Any:
        """Delete a judgment."""
        path = f"/_plugins/_search_relevance/judgments/{judgment_id}"
        return self._client.transport.perform_request("DELETE", path, params=kwargs)

    def get_query_sets(self, query_set_id: str | None = None, **kwargs: Any) -> Any:
        """Get query sets."""
        if query_set_id:
            path = f"/_plugins/_search_relevance/query_sets/{query_set_id}"
        else:
            path = "/_plugins/_search_relevance/query_sets"
        return self._client.transport.perform_request("GET", path, params=kwargs)

    def put_query_sets(self, body: Any | None = None, **kwargs: Any) -> Any:
        """Create or update a query set."""
        path = "/_plugins/_search_relevance/query_sets"
        return self._client.transport.perform_request(
            "PUT", path, body=body, params=kwargs
        )

    def post_query_sets(self, body: Any | None = None, **kwargs: Any) -> Any:
        """Create a query set by sampling from UBI data."""
        path = "/_plugins/_search_relevance/query_sets"
        return self._client.transport.perform_request(
            "POST", path, body=body, params=kwargs
        )

    def delete_query_sets(self, query_set_id: str, **kwargs: Any) -> Any:
        """Delete a query set."""
        path = f"/_plugins/_search_relevance/query_sets/{query_set_id}"
        return self._client.transport.perform_request("DELETE", path, params=kwargs)

    def get_experiments(self, experiment_id: str | None = None, **kwargs: Any) -> Any:
        """Get experiments."""
        if experiment_id:
            path = f"/_plugins/_search_relevance/experiments/{experiment_id}"
        else:
            path = "/_plugins/_search_relevance/experiments"
        return self._client.transport.perform_request("GET", path, params=kwargs)

    def put_experiments(self, body: Any | None = None, **kwargs: Any) -> Any:
        """Create or update an experiment."""
        path = "/_plugins/_search_relevance/experiments"
        return self._client.transport.perform_request(
            "PUT", path, body=body, params=kwargs
        )

    def delete_experiments(self, experiment_id: str, **kwargs: Any) -> Any:
        """Delete an experiment."""
        path = f"/_plugins/_search_relevance/experiments/{experiment_id}"
        return self._client.transport.perform_request("DELETE", path, params=kwargs)



class OpenSearchClientManager:
    """Manages OpenSearch client connection with search_relevance plugin support."""

    def __init__(
        self,
        opensearch_url: str | None = None,
        username: str = "admin",
        password: str = "admin",
        verify_certs: bool = False,
        ssl_show_warn: bool = False,
        pool_maxsize: int = 20,
        timeout: int = 30,
    ) -> None:
        """
        Initialize OpenSearch client manager.

        Args:
            opensearch_url: OpenSearch cluster URL
            username: Authentication username
            password: Authentication password
            verify_certs: Whether to verify SSL certificates
            ssl_show_warn: Whether to show SSL warnings
            pool_maxsize: Maximum number of connections in the pool (default: 20)
            timeout: Connection timeout in seconds (default: 30)
        """
        self.opensearch_url = opensearch_url or os.getenv(
            "OPENSEARCH_URL",
            "http://localhost:9200",
        )
        self.username = username
        self.password = password
        self.verify_certs = verify_certs
        self.ssl_show_warn = ssl_show_warn
        self.pool_maxsize = pool_maxsize
        self.timeout = timeout
        self.client: OpenSearch | None = None
        self._search_relevance_client: SearchRelevanceClient | None = None

    def connect(self) -> OpenSearch:
        """
        Connect to OpenSearch and return the client.

        Returns:
            OpenSearch: Configured OpenSearch client instance
        """
        if self.client is None:
            # Parse URL to extract host and port
            url_parts = self.opensearch_url.replace("http://", "").replace(
                "https://", ""
            )
            host_port = url_parts.split(":")
            host = host_port[0]
            port = (
                int(host_port[1].split("/")[0])
                if len(host_port) > 1
                else (443 if "https" in self.opensearch_url else 9200)
            )

            self.client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=(self.username, self.password),
                use_ssl="https" in self.opensearch_url,
                verify_certs=self.verify_certs,
                ssl_show_warn=self.ssl_show_warn,
                pool_maxsize=self.pool_maxsize,
                timeout=self.timeout,
            )
            log_info_event(
                logger,
                f"[OpenSearch] Connected to {self.opensearch_url} (pool_maxsize={self.pool_maxsize}, timeout={self.timeout}s)",
                "opensearch.connected",
                opensearch_url=self.opensearch_url,
                pool_maxsize=self.pool_maxsize,
                timeout=self.timeout,
            )

            # Initialize search_relevance client
            self._search_relevance_client = SearchRelevanceClient(self.client)

        return self.client

    def get_client(self) -> OpenSearch:
        """
        Get the OpenSearch client, connecting if necessary.

        Returns:
            OpenSearch: The OpenSearch client instance
        """
        if self.client is None:
            self.connect()
        return self.client

    def get_search_relevance_client(self) -> SearchRelevanceClient:
        """
        Get the search_relevance plugin client.

        Returns:
            SearchRelevanceClient: The search_relevance plugin client
        """
        if self._search_relevance_client is None:
            self.connect()
        return self._search_relevance_client

    def disconnect(self) -> None:
        """Close the OpenSearch connection."""
        if self.client:
            self.client.close()
            log_info_event(
                logger, "[OpenSearch] Connection closed", "opensearch.connection_closed"
            )
            self.client = None
            self._search_relevance_client = None


# Global client manager instance
_client_manager: OpenSearchClientManager | None = None


def get_client_manager(
    opensearch_url: str | None = None,
    username: str = "admin",
    password: str = "admin",
) -> OpenSearchClientManager:
    """
    Get or create the global OpenSearch client manager.

    Connection options (verify_certs, ssl_show_warn, pool_maxsize, timeout)
    are not accepted here; the created manager uses OpenSearchClientManager
    defaults. Subsequent calls return the same instance; URL/username/password
    are ignored after the first call.

    Args:
        opensearch_url: OpenSearch cluster URL
        username: Authentication username
        password: Authentication password

    Returns:
        OpenSearchClientManager: The client manager instance
    """
    global _client_manager
    if _client_manager is None:
        _client_manager = OpenSearchClientManager(
            opensearch_url=opensearch_url,
            username=username,
            password=password,
        )
    return _client_manager
