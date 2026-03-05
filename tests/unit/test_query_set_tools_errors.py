"""
Unit tests for query set tools error scenarios.

Tests error paths and edge cases for query set operations including:
- delete_query_set error paths
- Query set not found scenarios
- Invalid sample size handling
- Invalid query format handling
- create_query_set query parsing (list of strings, dict with queryText, non-string fallback)
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.query_set_tools import (
    create_query_set,
    delete_query_set,
    get_query_set,
    sample_query_set,
)

pytestmark = pytest.mark.unit


class TestQuerySetToolsErrors:
    """Test query set tool error scenarios."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_query_set_not_found(self):
        """Test get_query_set when query set doesn't exist."""
        mock_response = {"hits": {"total": {"value": 0}, "hits": []}}

        mock_sr_client = Mock()
        mock_sr_client.get_query_sets.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_set("nonexistent_query_set")
            result_data = json.loads(result)

            # Should return valid JSON with empty results, not crash
            assert "hits" in result_data
            assert result_data["hits"]["total"]["value"] == 0

    @pytest.mark.asyncio
    async def test_get_query_set_error(self):
        """Test get_query_set when API call fails."""
        mock_sr_client = Mock()
        mock_sr_client.get_query_sets.side_effect = Exception("Connection error")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_set("query_set1")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error retrieving query set" in result_data["error"]

    @pytest.mark.asyncio
    async def test_delete_query_set_error(self):
        """Test delete_query_set when API call fails."""
        mock_sr_client = Mock()
        mock_sr_client.delete_query_sets.side_effect = Exception("Not found")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await delete_query_set("missing-id")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error deleting query set" in result_data["error"]

    @pytest.mark.asyncio
    async def test_sample_query_set_invalid_size(self):
        """Test sample_query_set with invalid sample size."""
        # Test with zero size
        mock_sr_client = Mock()
        mock_sr_client.post_query_sets.side_effect = Exception(
            "query_set_size must be greater than 0"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await sample_query_set(name="test_set", query_set_size=0)
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error sampling query set" in result_data["error"]

    @pytest.mark.asyncio
    async def test_sample_query_set_negative_size(self):
        """Test sample_query_set with negative sample size."""
        mock_sr_client = Mock()
        mock_sr_client.post_query_sets.side_effect = Exception(
            "query_set_size must be positive"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await sample_query_set(name="test_set", query_set_size=-5)
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error sampling query set" in result_data["error"]

    @pytest.mark.asyncio
    async def test_sample_query_set_connection_error(self):
        """Test sample_query_set when connection error occurs."""
        mock_sr_client = Mock()
        mock_sr_client.post_query_sets.side_effect = ConnectionError(
            "Connection failed"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await sample_query_set(name="test_set", query_set_size=20)
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error sampling query set" in result_data["error"]

    @pytest.mark.asyncio
    async def test_create_query_set_invalid_queries(self):
        """Test create_query_set with invalid query format."""
        # Test with invalid JSON string
        mock_client_manager = Mock()

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_query_set(
                name="test_set",
                queries="invalid json {",  # Invalid JSON
            )
            # Should return error (JSON decode error)
            assert "error" in json.loads(result) or "Error creating query set" in result

    @pytest.mark.asyncio
    async def test_create_query_set_queries_not_a_list(self):
        """Test create_query_set when queries JSON is not an array (e.g. {} or 123)."""
        mock_client_manager = Mock()

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_query_set(
                name="test_set",
                queries="{}",
            )
            result_data = json.loads(result)
            assert "error" in result_data
            assert "queries must be a JSON array" in result_data["error"]

            result2 = await create_query_set(
                name="test_set",
                queries="123",
            )
            result_data2 = json.loads(result2)
            assert "error" in result_data2
            assert "queries must be a JSON array" in result_data2["error"]

    @pytest.mark.asyncio
    async def test_create_query_set_empty_queries(self):
        """Test create_query_set with empty queries list."""
        mock_sr_client = Mock()
        mock_sr_client.put_query_sets.side_effect = Exception(
            "querySetQueries cannot be empty"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_query_set(
                name="test_set",
                queries="[]",  # Empty list
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error creating query set" in result_data["error"]

    @pytest.mark.asyncio
    async def test_create_query_set_invalid_query_structure(self):
        """Test create_query_set with queries that have invalid structure."""
        # The function should handle various query formats, but API may reject invalid ones
        mock_sr_client = Mock()
        mock_sr_client.put_query_sets.side_effect = Exception(
            "Invalid query format in querySetQueries"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            # Test with queries that are not strings or dicts with queryText
            result = await create_query_set(
                name="test_set",
                queries='[{"invalid": "format"}]',
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error creating query set" in result_data["error"]

    @pytest.mark.asyncio
    async def test_create_query_set_duplicate_name(self):
        """Test create_query_set with duplicate name."""
        mock_sr_client = Mock()
        mock_sr_client.put_query_sets.side_effect = Exception(
            "Query set with name 'existing_set' already exists"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_query_set(
                name="existing_set",
                queries='["query1", "query2"]',
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error creating query set" in result_data["error"]
            assert "already exists" in result_data["error"]

    @pytest.mark.asyncio
    async def test_create_query_set_connection_error(self):
        """Test create_query_set when connection error occurs."""
        mock_sr_client = Mock()
        mock_sr_client.put_query_sets.side_effect = ConnectionError("Connection failed")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_query_set(
                name="test_set",
                queries='["query1", "query2"]',
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error creating query set" in result_data["error"]

    @pytest.mark.asyncio
    async def test_create_query_set_queries_as_strings(self):
        """Test create_query_set with JSON list of plain strings -> queryText format."""
        mock_sr_client = Mock()
        mock_sr_client.put_query_sets.return_value = {"query_set_id": "qs-1"}

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_query_set(
                name="parsed_strings",
                queries='["q1", "q2"]',
            )
            result_data = json.loads(result)
            assert "error" not in result_data

            call_args = mock_sr_client.put_query_sets.call_args
            body = call_args.kwargs["body"]
            assert body["querySetQueries"] == [
                {"queryText": "q1"},
                {"queryText": "q2"},
            ]

    @pytest.mark.asyncio
    async def test_create_query_set_queries_dict_with_query_text(self):
        """Test create_query_set with dict entries containing queryText."""
        mock_sr_client = Mock()
        mock_sr_client.put_query_sets.return_value = {"query_set_id": "qs-1"}

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_query_set(
                name="dict_queries",
                queries='[{"queryText": "preformed"}]',
            )
            result_data = json.loads(result)
            assert "error" not in result_data

            call_args = mock_sr_client.put_query_sets.call_args
            body = call_args.kwargs["body"]
            assert body["querySetQueries"] == [{"queryText": "preformed"}]

    @pytest.mark.asyncio
    async def test_create_query_set_queries_non_string_non_dict_fallback(self):
        """Test create_query_set with non-string/non-dict items uses str(q) fallback."""
        mock_sr_client = Mock()
        mock_sr_client.put_query_sets.return_value = {"query_set_id": "qs-1"}

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_query_set(
                name="fallback",
                queries="[123, null]",
            )
            result_data = json.loads(result)
            assert "error" not in result_data

            call_args = mock_sr_client.put_query_sets.call_args
            body = call_args.kwargs["body"]
            assert body["querySetQueries"] == [
                {"queryText": "123"},
                {"queryText": "None"},
            ]

    @pytest.mark.asyncio
    async def test_sample_query_set_no_ubi_data(self):
        """Test sample_query_set when no UBI data is available."""
        mock_sr_client = Mock()
        mock_sr_client.post_query_sets.side_effect = Exception(
            "No UBI data available for sampling"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await sample_query_set(name="test_set", query_set_size=20)
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error sampling query set" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_query_set_timeout(self):
        """Test get_query_set when timeout occurs."""
        mock_sr_client = Mock()
        mock_sr_client.get_query_sets.side_effect = TimeoutError("Request timeout")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.query_set_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_set("query_set1")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error retrieving query set" in result_data["error"]
