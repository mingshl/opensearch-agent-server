"""
Unit tests for UBI analytics tools error scenarios.

Tests error paths and edge cases for UBI analytics operations including:
- No data available scenarios
- Invalid index handling
- Timeout scenarios
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.ubi_analytics_tools import (
    get_document_ctr,
    get_query_ctr,
    get_query_performance_metrics,
)

pytestmark = pytest.mark.unit


class TestUBIAnalyticsToolsErrors:
    """Test UBI analytics tool error scenarios."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_query_ctr_no_data(self):
        """Test get_query_ctr with no data available."""
        mock_response = {
            "aggregations": {
                "total_searches": {"value": 0},
                "searches_with_clicks": {
                    "doc_count": 0,
                    "unique_queries": {"value": 0},
                },
                "total_clicks": {"doc_count": 0},
            }
        }

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_ctr(
                query_text="nonexistent_query", time_range_days=30
            )
            result_data = json.loads(result)

            # Should return valid JSON with zero values, not error
            assert result_data["query_text"] == "nonexistent_query"
            assert result_data["total_searches"] == 0
            assert result_data["searches_with_clicks"] == 0
            assert result_data["total_clicks"] == 0
            assert result_data["ctr_percentage"] == 0
            assert result_data["average_clicks_per_search"] == 0

    @pytest.mark.asyncio
    async def test_get_query_ctr_index_not_found(self):
        """Test get_query_ctr when index doesn't exist."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("index_not_found_exception")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_ctr(
                query_text="test_query", ubi_index="nonexistent_index"
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error calculating query CTR" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_query_ctr_connection_error(self):
        """Test get_query_ctr when connection error occurs."""
        mock_client = Mock()
        mock_client.search.side_effect = ConnectionError("Connection failed")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_ctr(query_text="test_query")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error calculating query CTR" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_document_ctr_invalid_index(self):
        """Test get_document_ctr with invalid index."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("index_not_found_exception")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_document_ctr(
                doc_id="doc123", ubi_index="nonexistent_index"
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error calculating document CTR" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_document_ctr_no_data(self):
        """Test get_document_ctr with no data available."""
        mock_response = {
            "aggregations": {
                "total_impressions": {"doc_count": 0},
                "clicks": {
                    "doc_count": 0,
                    "avg_position": {"value": None},
                },
            }
        }

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_document_ctr(
                doc_id="nonexistent_doc", time_range_days=30
            )
            result_data = json.loads(result)

            # Should return valid JSON with zero values, not error
            assert result_data["document_id"] == "nonexistent_doc"
            assert result_data["total_impressions"] == 0
            assert result_data["total_clicks"] == 0
            assert result_data["ctr_percentage"] == 0
            assert result_data["average_position_when_clicked"] is None

    @pytest.mark.asyncio
    async def test_get_document_ctr_connection_error(self):
        """Test get_document_ctr when connection error occurs."""
        mock_client = Mock()
        mock_client.search.side_effect = ConnectionError("Connection failed")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_document_ctr(doc_id="doc123")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error calculating document CTR" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_query_performance_metrics_timeout(self):
        """Test get_query_performance_metrics timeout."""
        mock_client = Mock()
        mock_client.search.side_effect = TimeoutError("Request timeout")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_performance_metrics(top_n=20, time_range_days=30)
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error getting query performance metrics" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_query_performance_metrics_no_data(self):
        """Test get_query_performance_metrics when no data available."""
        mock_response = {"aggregations": {"top_queries": {"buckets": []}}}

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_performance_metrics(top_n=20, time_range_days=30)
            result_data = json.loads(result)

            # Should return valid JSON with empty results, not error
            assert result_data["time_range_days"] == 30
            assert result_data["total_queries_analyzed"] == 0
            assert len(result_data["queries"]) == 0

    @pytest.mark.asyncio
    async def test_get_query_performance_metrics_invalid_index(self):
        """Test get_query_performance_metrics with invalid index."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("index_not_found_exception")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_performance_metrics(
                top_n=20, ubi_index="nonexistent_index"
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error getting query performance metrics" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_query_performance_metrics_specific_query_error(self):
        """Test get_query_performance_metrics for specific query when error occurs."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("Index error")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with (
            patch(
                "tools.ubi_analytics_tools.get_client_manager"
            ) as mock_get_manager,
            patch("tools.ubi_analytics_tools.get_query_ctr") as mock_get_ctr,
        ):
            mock_get_manager.return_value = mock_client_manager
            mock_get_ctr.side_effect = Exception("Query CTR calculation failed")

            result = await get_query_performance_metrics(query_text="test_query")
            result_data = json.loads(result)

            assert "error" in result_data
            # When get_query_ctr fails, the error is wrapped by get_query_performance_metrics
            assert "Error getting query performance metrics" in result_data["error"]
            assert "Query CTR calculation failed" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_query_ctr_timeout(self):
        """Test get_query_ctr when timeout occurs."""
        mock_client = Mock()
        mock_client.search.side_effect = TimeoutError("Request timeout")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_ctr(query_text="test_query", time_range_days=30)
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error calculating query CTR" in result_data["error"]
            assert "timeout" in result_data["error"].lower()

    @pytest.mark.asyncio
    async def test_get_document_ctr_timeout(self):
        """Test get_document_ctr when timeout occurs."""
        mock_client = Mock()
        mock_client.search.side_effect = TimeoutError("Request timeout")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_document_ctr(doc_id="doc123", time_range_days=30)
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error calculating document CTR" in result_data["error"]
            assert "timeout" in result_data["error"].lower()
