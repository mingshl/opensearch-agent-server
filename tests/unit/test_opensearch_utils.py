"""
Unit tests for OpenSearch utility tools.

Tests critical paths for OpenSearch utility operations including:
- Index listing
- Index searching
- Error handling
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.opensearch_utils import (
    list_index,
    search_index,
)

pytestmark = pytest.mark.unit


class TestListIndex:
    """Tests for list_index function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_list_index_success(self):
        """Test successful listing of indexes."""
        mock_response = [
            {
                "index": "test_index",
                "status": "open",
                "docs.count": "100",
                "store.size": "1mb",
            },
            {
                "index": "other_index",
                "status": "open",
                "docs.count": "50",
                "store.size": "500kb",
            },
        ]

        mock_client = Mock()
        mock_client.cat.indices.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.opensearch_utils.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await list_index()
            result_data = json.loads(result)

            assert result_data["found"] is True
            assert result_data["count"] == 2
            assert len(result_data["indexes"]) == 2
            mock_client.cat.indices.assert_called_once_with(
                index="*",
                format="json",
                h="index,status,docs.count,store.size",
            )

    @pytest.mark.asyncio
    async def test_list_index_with_pattern(self):
        """Test listing indexes with specific pattern."""
        mock_response = [
            {
                "index": "test_index",
                "status": "open",
                "docs.count": "100",
                "store.size": "1mb",
            },
        ]

        mock_client = Mock()
        mock_client.cat.indices.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.opensearch_utils.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await list_index("test_*")
            result_data = json.loads(result)

            assert result_data["found"] is True
            assert result_data["count"] == 1
            mock_client.cat.indices.assert_called_once_with(
                index="test_*",
                format="json",
                h="index,status,docs.count,store.size",
            )

    @pytest.mark.asyncio
    async def test_list_index_no_results(self):
        """Test listing when no indexes match pattern."""
        mock_client = Mock()
        mock_client.cat.indices.return_value = []

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.opensearch_utils.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await list_index("nonexistent_*")
            result_data = json.loads(result)

            assert result_data["found"] is False
            assert "No indexes found" in result_data["message"]

    @pytest.mark.asyncio
    async def test_list_index_error(self):
        """Test error handling when listing indexes fails."""
        mock_client = Mock()
        mock_client.cat.indices.side_effect = Exception("Connection error")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.opensearch_utils.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await list_index()
            assert "Error listing indexes" in result

    @pytest.mark.asyncio
    async def test_list_index_not_found_error(self):
        """Test handling of index_not_found exception."""
        mock_client = Mock()
        error = Exception("index_not_found_exception")
        error.error = "index_not_found_exception"
        mock_client.cat.indices.side_effect = error

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.opensearch_utils.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await list_index("nonexistent")
            result_data = json.loads(result)

            assert result_data["found"] is False
            assert "does not exist" in result_data["message"]


class TestSearchIndex:
    """Tests for search_index function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_search_index_success(self):
        """Test successful search in index."""
        mock_response = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_id": "doc1",
                        "_score": 1.5,
                        "_source": {
                            "id": "doc1",
                            "title": "Test Document 1",
                            "attrs.Brand": "Brand1",
                        },
                    },
                    {
                        "_id": "doc2",
                        "_score": 1.2,
                        "_source": {
                            "id": "doc2",
                            "title": "Test Document 2",
                            "attrs.Brand": "Brand2",
                        },
                    },
                ],
            }
        }

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.opensearch_utils.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await search_index(
                query_text="laptop", index="test_index", size=10
            )
            result_data = json.loads(result)

            assert result_data["total_hits"] == 2
            assert result_data["returned"] == 2
            assert len(result_data["documents"]) == 2
            assert result_data["documents"][0]["_id"] == "doc1"
            assert result_data["documents"][0]["_score"] == 1.5

    @pytest.mark.asyncio
    async def test_search_index_with_custom_fields(self):
        """Test search with custom fields."""
        mock_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "doc1",
                        "_score": 1.0,
                        "_source": {"id": "doc1", "title": "Test"},
                    },
                ],
            }
        }

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.opensearch_utils.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await search_index(
                query_text="test",
                index="test_index",
                fields=["id", "title"],
                size=5,
            )
            result_data = json.loads(result)

            assert result_data["total_hits"] == 1
            # Verify fields were used in query
            search_call = mock_client.search.call_args
            query_body = search_call[1]["body"]
            assert query_body["query"]["query_string"]["fields"] == ["id", "title"]
            assert query_body["_source"] == ["id", "title"]

    @pytest.mark.asyncio
    async def test_search_index_with_default_fields(self):
        """Test search with default fields."""
        mock_response = {
            "hits": {
                "total": {"value": 0},
                "hits": [],
            }
        }

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.opensearch_utils.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await search_index(query_text="test", index="test_index")
            json.loads(result)  # Verify it's valid JSON

            # Verify default fields were used
            search_call = mock_client.search.call_args
            query_body = search_call[1]["body"]
            assert query_body["query"]["query_string"]["fields"] == [
                "id",
                "title",
                "attrs.Brand",
            ]
            assert query_body["_source"] == ["id", "title", "attrs.Brand"]

    @pytest.mark.asyncio
    async def test_search_index_error(self):
        """Test error handling when search fails."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("Index not found")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.opensearch_utils.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await search_index(query_text="test", index="nonexistent")
            assert "Error searching index" in result

    @pytest.mark.asyncio
    async def test_search_index_empty_results(self):
        """Test search that returns no results."""
        mock_response = {
            "hits": {
                "total": {"value": 0},
                "hits": [],
            }
        }

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.opensearch_utils.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await search_index(query_text="nonexistent", index="test_index")
            result_data = json.loads(result)

            assert result_data["total_hits"] == 0
            assert result_data["returned"] == 0
            assert len(result_data["documents"]) == 0
            assert result_data["documents"] == []
