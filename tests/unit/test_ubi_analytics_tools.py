"""
Unit tests for UBI analytics tools.

Tests critical paths for user behavior insights analytics including:
- Query CTR calculation
- Document CTR calculation
- Query performance metrics
- Top queries by engagement
- Top documents by engagement
- Error handling
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.ubi_analytics_tools import (
    get_document_ctr,
    get_query_ctr,
    get_query_performance_metrics,
    get_top_documents_by_engagement,
    get_top_queries_by_engagement,
)

pytestmark = pytest.mark.unit


class TestGetQueryCTR:
    """Tests for get_query_ctr function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_query_ctr_success(self):
        """Test successful calculation of query CTR."""
        mock_response = {
            "aggregations": {
                "total_searches": {"value": 100},
                "searches_with_clicks": {
                    "doc_count": 50,
                    "unique_queries": {"value": 30},
                },
                "total_clicks": {"doc_count": 50},
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

            result = await get_query_ctr(query_text="laptop", time_range_days=30)
            result_data = json.loads(result)

            assert result_data["query_text"] == "laptop"
            assert result_data["total_searches"] == 100
            assert result_data["searches_with_clicks"] == 30
            assert result_data["total_clicks"] == 50
            assert result_data["ctr_percentage"] == 30.0  # 30/100 * 100
            assert result_data["average_clicks_per_search"] == 0.5  # 50/100
            assert (
                result_data["zero_click_rate_percentage"] == 70.0
            )  # (100-30)/100 * 100

    @pytest.mark.asyncio
    async def test_get_query_ctr_no_searches(self):
        """Test query CTR when no searches found."""
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

            result = await get_query_ctr(query_text="nonexistent", time_range_days=30)
            result_data = json.loads(result)

            assert result_data["total_searches"] == 0
            assert result_data["ctr_percentage"] == 0
            assert result_data["average_clicks_per_search"] == 0

    @pytest.mark.asyncio
    async def test_get_query_ctr_custom_time_range(self):
        """Test query CTR with custom time range."""
        mock_response = {
            "aggregations": {
                "total_searches": {"value": 50},
                "searches_with_clicks": {
                    "doc_count": 25,
                    "unique_queries": {"value": 15},
                },
                "total_clicks": {"doc_count": 25},
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

            result = await get_query_ctr(query_text="phone", time_range_days=7)
            result_data = json.loads(result)

            assert result_data["time_range_days"] == 7
            # Verify time range was used in query
            search_call = mock_client.search.call_args
            query_body = search_call[1]["body"]
            assert "range" in query_body["query"]["bool"]["must"][1]
            assert "timestamp" in query_body["query"]["bool"]["must"][1]["range"]

    @pytest.mark.asyncio
    async def test_get_query_ctr_error(self):
        """Test error handling when calculating query CTR fails."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("Index not found")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_ctr(query_text="test")
            assert "Error calculating query CTR" in result


class TestGetDocumentCTR:
    """Tests for get_document_ctr function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_document_ctr_success(self):
        """Test successful calculation of document CTR."""
        mock_response = {
            "aggregations": {
                "total_impressions": {"doc_count": 200},
                "clicks": {
                    "doc_count": 40,
                    "avg_position": {"value": 2.5},
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

            result = await get_document_ctr(doc_id="doc123", time_range_days=30)
            result_data = json.loads(result)

            assert result_data["document_id"] == "doc123"
            assert result_data["total_impressions"] == 200
            assert result_data["total_clicks"] == 40
            assert result_data["ctr_percentage"] == 20.0  # 40/200 * 100
            assert result_data["average_position_when_clicked"] == 2.5

    @pytest.mark.asyncio
    async def test_get_document_ctr_no_impressions(self):
        """Test document CTR when no impressions found."""
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

            result = await get_document_ctr(doc_id="nonexistent")
            result_data = json.loads(result)

            assert result_data["total_impressions"] == 0
            assert result_data["ctr_percentage"] == 0
            assert result_data["average_position_when_clicked"] is None

    @pytest.mark.asyncio
    async def test_get_document_ctr_no_clicks(self):
        """Test document CTR when document has impressions but no clicks."""
        mock_response = {
            "aggregations": {
                "total_impressions": {"doc_count": 100},
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

            result = await get_document_ctr(doc_id="doc123")
            result_data = json.loads(result)

            assert result_data["total_impressions"] == 100
            assert result_data["total_clicks"] == 0
            assert result_data["ctr_percentage"] == 0

    @pytest.mark.asyncio
    async def test_get_document_ctr_error(self):
        """Test error handling when calculating document CTR fails."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("Connection error")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_document_ctr(doc_id="doc123")
            assert "Error calculating document CTR" in result


class TestGetQueryPerformanceMetrics:
    """Tests for get_query_performance_metrics function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_query_performance_metrics_specific_query(self):
        """Test getting metrics for a specific query."""
        mock_response = {
            "aggregations": {
                "total_searches": {"value": 100},
                "searches_with_clicks": {
                    "doc_count": 50,
                    "unique_queries": {"value": 30},
                },
                "total_clicks": {"doc_count": 50},
            }
        }

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with (
            patch(
                "tools.ubi_analytics_tools.get_client_manager"
            ) as mock_get_manager,
            patch("tools.ubi_analytics_tools.get_query_ctr") as mock_get_ctr,
        ):
            mock_get_manager.return_value = mock_client_manager
            mock_get_ctr.return_value = json.dumps(
                {"query_text": "laptop", "ctr_percentage": 30.0}
            )

            result = await get_query_performance_metrics(
                query_text="laptop", time_range_days=30
            )
            result_data = json.loads(result)

            assert result_data["query_text"] == "laptop"
            mock_get_ctr.assert_called_once_with("laptop", 30, "ubi_events")

    @pytest.mark.asyncio
    async def test_get_query_performance_metrics_top_queries(self):
        """Test getting top N queries with metrics."""
        mock_response = {
            "aggregations": {
                "top_queries": {
                    "buckets": [
                        {
                            "key": "laptop",
                            "doc_count": 150,
                            "unique_searches": {"value": 100},
                            "click_events": {
                                "doc_count": 50,
                                "unique_queries_with_clicks": {"value": 30},
                            },
                        },
                        {
                            "key": "phone",
                            "doc_count": 120,
                            "unique_searches": {"value": 80},
                            "click_events": {
                                "doc_count": 40,
                                "unique_queries_with_clicks": {"value": 20},
                            },
                        },
                    ]
                }
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

            result = await get_query_performance_metrics(top_n=20, time_range_days=30)
            result_data = json.loads(result)

            assert result_data["time_range_days"] == 30
            assert result_data["total_queries_analyzed"] == 2
            assert len(result_data["queries"]) == 2
            assert result_data["queries"][0]["query_text"] == "laptop"
            assert result_data["queries"][0]["search_volume"] == 100
            assert result_data["queries"][0]["ctr_percentage"] == 30.0  # 30/100 * 100

    @pytest.mark.asyncio
    async def test_get_query_performance_metrics_empty_results(self):
        """Test getting metrics when no queries found."""
        mock_response = {
            "aggregations": {
                "top_queries": {
                    "buckets": [],
                },
            },
        }

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_performance_metrics(top_n=20)
            result_data = json.loads(result)

            assert result_data["total_queries_analyzed"] == 0
            assert len(result_data["queries"]) == 0

    @pytest.mark.asyncio
    async def test_get_query_performance_metrics_error(self):
        """Test error handling when getting metrics fails."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("Index error")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_query_performance_metrics()
            assert "Error getting query performance metrics" in result


class TestGetTopQueriesByEngagement:
    """Tests for get_top_queries_by_engagement function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_top_queries_by_engagement_success(self):
        """Test getting top queries by engagement."""
        # Mock get_query_performance_metrics response
        mock_metrics_response = {
            "time_range_days": 30,
            "queries": [
                {
                    "query_text": "high_ctr_query",
                    "search_volume": 50,
                    "searches_with_clicks": 30,
                    "total_clicks": 45,
                    "ctr_percentage": 60.0,
                    "average_clicks_per_search": 0.9,
                    "zero_click_rate_percentage": 40.0,
                },
                {
                    "query_text": "low_ctr_query",
                    "search_volume": 100,
                    "searches_with_clicks": 10,
                    "total_clicks": 12,
                    "ctr_percentage": 10.0,
                    "average_clicks_per_search": 0.12,
                    "zero_click_rate_percentage": 90.0,
                },
                {
                    "query_text": "low_volume_query",
                    "search_volume": 3,  # Below min_search_volume
                    "searches_with_clicks": 2,
                    "total_clicks": 3,
                    "ctr_percentage": 66.67,
                    "average_clicks_per_search": 1.0,
                    "zero_click_rate_percentage": 33.33,
                },
            ],
        }

        with patch(
            "tools.ubi_analytics_tools.get_query_performance_metrics"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = json.dumps(mock_metrics_response)

            result = await get_top_queries_by_engagement(
                top_n=20,
                min_search_volume=5,
                time_range_days=30,
            )
            result_data = json.loads(result)

            assert result_data["time_range_days"] == 30
            assert result_data["min_search_volume"] == 5
            assert (
                result_data["total_queries_analyzed"] == 2
            )  # Only 2 meet min_search_volume
            assert len(result_data["queries"]) == 2
            # Should be sorted by CTR descending
            assert result_data["queries"][0]["query_text"] == "high_ctr_query"
            assert result_data["queries"][0]["ctr_percentage"] == 60.0
            assert result_data["queries"][1]["query_text"] == "low_ctr_query"
            assert result_data["queries"][1]["ctr_percentage"] == 10.0

    @pytest.mark.asyncio
    async def test_get_top_queries_by_engagement_respects_top_n(self):
        """Test that top_n limit is respected."""
        # Create mock with more queries than top_n
        mock_queries = [
            {
                "query_text": f"query_{i}",
                "search_volume": 10,
                "searches_with_clicks": 5,
                "total_clicks": 6,
                "ctr_percentage": 50.0 - i,  # Decreasing CTR
                "average_clicks_per_search": 0.6,
                "zero_click_rate_percentage": 50.0,
            }
            for i in range(30)
        ]

        mock_metrics_response = {
            "time_range_days": 30,
            "queries": mock_queries,
        }

        with patch(
            "tools.ubi_analytics_tools.get_query_performance_metrics"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = json.dumps(mock_metrics_response)

            result = await get_top_queries_by_engagement(top_n=5, min_search_volume=5)
            result_data = json.loads(result)

            assert result_data["total_queries_analyzed"] == 5
            assert len(result_data["queries"]) == 5

    @pytest.mark.asyncio
    async def test_get_top_queries_by_engagement_error(self):
        """Test error handling when getting top queries fails."""
        with patch(
            "tools.ubi_analytics_tools.get_query_performance_metrics"
        ) as mock_get_metrics:
            mock_get_metrics.side_effect = Exception("Metrics error")

            result = await get_top_queries_by_engagement()
            assert "Error getting top queries by engagement" in result


class TestGetTopDocumentsByEngagement:
    """Tests for get_top_documents_by_engagement function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_top_documents_by_engagement_success(self):
        """Test getting top documents by engagement."""
        mock_response = {
            "aggregations": {
                "top_documents": {
                    "buckets": [
                        {
                            "key": "doc1",
                            "doc_count": 200,
                            "impressions": {"doc_count": 150},
                            "clicks": {
                                "doc_count": 45,
                                "avg_position": {"value": 2.0},
                            },
                        },
                        {
                            "key": "doc2",
                            "doc_count": 180,
                            "impressions": {"doc_count": 120},
                            "clicks": {
                                "doc_count": 30,
                                "avg_position": {"value": 3.5},
                            },
                        },
                        {
                            "key": "doc3",
                            "doc_count": 50,
                            "impressions": {"doc_count": 3},  # Below min_impressions
                            "clicks": {
                                "doc_count": 2,
                                "avg_position": {"value": 1.0},
                            },
                        },
                    ]
                }
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

            result = await get_top_documents_by_engagement(
                top_n=20,
                min_impressions=5,
                time_range_days=30,
            )
            result_data = json.loads(result)

            assert result_data["time_range_days"] == 30
            assert result_data["min_impressions"] == 5
            assert (
                result_data["total_documents_analyzed"] == 2
            )  # Only 2 meet min_impressions
            assert len(result_data["documents"]) == 2
            # Should be sorted by CTR descending
            assert result_data["documents"][0]["document_id"] == "doc1"
            assert result_data["documents"][0]["ctr_percentage"] == 30.0  # 45/150 * 100
            assert result_data["documents"][1]["document_id"] == "doc2"
            assert result_data["documents"][1]["ctr_percentage"] == 25.0  # 30/120 * 100

    @pytest.mark.asyncio
    async def test_get_top_documents_by_engagement_respects_top_n(self):
        """Test that top_n limit is respected."""
        # Create mock with more documents than top_n
        mock_buckets = [
            {
                "key": f"doc_{i}",
                "doc_count": 100,
                "impressions": {"doc_count": 50},
                "clicks": {
                    "doc_count": 25 - i,  # Decreasing clicks
                    "avg_position": {"value": float(i + 1)},
                },
            }
            for i in range(30)
        ]

        mock_response = {
            "aggregations": {
                "top_documents": {"buckets": mock_buckets},
            },
        }

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_top_documents_by_engagement(top_n=10, min_impressions=5)
            result_data = json.loads(result)

            assert result_data["total_documents_analyzed"] == 10
            assert len(result_data["documents"]) == 10

    @pytest.mark.asyncio
    async def test_get_top_documents_by_engagement_no_documents(self):
        """Test when no documents meet criteria."""
        mock_response = {
            "aggregations": {
                "top_documents": {"buckets": []},
            },
        }

        mock_client = Mock()
        mock_client.search.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_top_documents_by_engagement()
            result_data = json.loads(result)

            assert result_data["total_documents_analyzed"] == 0
            assert len(result_data["documents"]) == 0

    @pytest.mark.asyncio
    async def test_get_top_documents_by_engagement_error(self):
        """Test error handling when getting top documents fails."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("Search error")

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.ubi_analytics_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_top_documents_by_engagement()
            assert "Error getting top documents by engagement" in result
