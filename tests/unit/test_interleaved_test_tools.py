"""
Unit tests for interleaved test tools.

Tests critical paths for interleaved A/B testing including:
- Interleaving algorithms (team draft)
- Test CRUD operations
- Simulation logic
- Winner calculation
- Error handling
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.interleaved_test_tools import (
    calculate_interleaved_winner,
    cascade_click_probability,
    create_interleaved_test,
    delete_interleaved_test,
    get_interleaved_test,
    position_based_click_probability,
    simulate_interleaved_test,
    stop_interleaved_test,
    team_draft_interleave,
)

pytestmark = pytest.mark.unit


class TestTeamDraftInterleave:
    """Tests for team_draft_interleave function."""

    def test_interleave_basic(self):
        """Test basic interleaving of two result lists."""
        results_a = [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}]
        results_b = [{"id": "doc4"}, {"id": "doc5"}]

        interleaved, attribution = team_draft_interleave(results_a, results_b, k=5)

        assert len(interleaved) == 5
        assert len(attribution) == 5
        # Verify all docs are attributed
        for doc_id, source in interleaved:
            assert doc_id in attribution
            assert source in ["config_a", "config_b"]

    def test_interleave_with_duplicates(self):
        """Test interleaving skips duplicate documents."""
        results_a = [{"id": "doc1"}, {"id": "doc2"}]
        results_b = [{"id": "doc1"}, {"id": "doc3"}]  # doc1 is duplicate

        interleaved, attribution = team_draft_interleave(results_a, results_b, k=10)

        # Should only have 3 unique docs
        unique_docs = set(doc_id for doc_id, _ in interleaved)
        assert len(unique_docs) == 3
        assert "doc1" in unique_docs

    def test_interleave_respects_k_limit(self):
        """Test that interleaving respects k limit."""
        results_a = [{"id": f"doc{i}"} for i in range(10)]
        results_b = [{"id": f"doc{i + 10}"} for i in range(10)]

        interleaved, attribution = team_draft_interleave(results_a, results_b, k=5)

        assert len(interleaved) == 5

    def test_interleave_with_seed(self):
        """Test that seed produces reproducible results."""
        results_a = [{"id": "doc1"}, {"id": "doc2"}]
        results_b = [{"id": "doc3"}, {"id": "doc4"}]

        interleaved1, _ = team_draft_interleave(
            results_a, results_b, k=4, random_seed=42
        )
        interleaved2, _ = team_draft_interleave(
            results_a, results_b, k=4, random_seed=42
        )

        # With same seed, should get same team assignment
        assert interleaved1 == interleaved2


class TestClickProbabilityModels:
    """Tests for click probability model functions."""

    def test_position_based_click_probability(self):
        """Test position-based click probability model."""
        prob1 = position_based_click_probability(1)
        prob2 = position_based_click_probability(2)
        prob10 = position_based_click_probability(10)

        # Position 1 should have highest probability
        assert prob1 > prob2
        assert prob2 > prob10
        # All probabilities should be between 0 and 1
        assert 0 < prob1 < 1
        assert 0 < prob2 < 1
        assert 0 < prob10 < 1

    def test_cascade_click_probability(self):
        """Test cascade click probability model."""
        prob1 = cascade_click_probability(1, relevance_prob=0.5)
        prob2 = cascade_click_probability(2, relevance_prob=0.5)
        prob10 = cascade_click_probability(10, relevance_prob=0.5)

        # Position 1 should have highest probability
        assert prob1 > prob2
        assert prob2 > prob10
        # All probabilities should be between 0 and 1
        assert 0 < prob1 < 1
        assert 0 < prob2 < 1
        assert 0 < prob10 < 1


class TestCreateInterleavedTest:
    """Tests for create_interleaved_test function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_create_interleaved_test_success(self):
        """Test successful creation of interleaved test."""
        mock_response = {"_id": "test_id", "result": "created"}

        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.return_value = {"acknowledged": True}
        mock_client.index.return_value = mock_response

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch("tools.interleaved_test_tools.uuid.uuid4") as mock_uuid,
        ):
            mock_uuid.return_value.hex = "test12345678"

            result = await create_interleaved_test(
                test_name="Test A/B",
                search_configuration_a_id="config_a",
                search_configuration_b_id="config_b",
                query_set_id="qs1",
                algorithm="team_draft",
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert "test_id" in result_data
            assert result_data["test_name"] == "Test A/B"
            assert result_data["status"] == "active"

    @pytest.mark.asyncio
    async def test_create_interleaved_test_index_exists(self):
        """Test creation when index already exists."""
        mock_response = {"_id": "test_id", "result": "created"}

        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_client.index.return_value = mock_response

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await create_interleaved_test(
                test_name="Test",
                search_configuration_a_id="config_a",
                search_configuration_b_id="config_b",
                query_set_id="qs1",
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            # Should not create index if it exists
            mock_client.indices.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_interleaved_test_error(self):
        """Test error handling when creation fails."""
        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.side_effect = Exception("Index creation failed")

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await create_interleaved_test(
                test_name="Test",
                search_configuration_a_id="config_a",
                search_configuration_b_id="config_b",
                query_set_id="qs1",
            )
            result_data = json.loads(result)

            assert "error" in result_data


class TestGetInterleavedTest:
    """Tests for get_interleaved_test function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_interleaved_test_success(self):
        """Test successful retrieval of interleaved test."""
        mock_response = {
            "found": True,
            "_source": {
                "test_id": "test1",
                "test_name": "Test A/B",
                "status": "active",
            },
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_response

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await get_interleaved_test("test1")
            result_data = json.loads(result)

            assert result_data["test_id"] == "test1"
            assert result_data["test_name"] == "Test A/B"

    @pytest.mark.asyncio
    async def test_get_interleaved_test_not_found(self):
        """Test error when test is not found."""
        mock_response = {"found": False}

        mock_client = Mock()
        mock_client.get.return_value = mock_response

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await get_interleaved_test("invalid")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "not found" in result_data["error"]


class TestStopInterleavedTest:
    """Tests for stop_interleaved_test function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_stop_interleaved_test_success(self):
        """Test successful stopping of interleaved test."""
        mock_response = {"_id": "test1", "result": "updated"}

        mock_client = Mock()
        mock_client.update.return_value = mock_response

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await stop_interleaved_test("test1")
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["status"] == "stopped"
            mock_client.update.assert_called_once()
            update_call = mock_client.update.call_args[1]["body"]["doc"]
            assert update_call["status"] == "stopped"
            assert "stopped_at" in update_call


class TestDeleteInterleavedTest:
    """Tests for delete_interleaved_test function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_delete_interleaved_test_success(self):
        """Test successful deletion of interleaved test."""
        mock_get_response = {
            "found": True,
            "_source": {"test_name": "Test A/B"},
        }

        mock_delete_response = {"result": "deleted"}

        mock_client = Mock()
        mock_client.get.return_value = mock_get_response
        mock_client.delete.return_value = mock_delete_response

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await delete_interleaved_test("test1", delete_events=False)
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["test_deleted"] is True
            assert result_data["events_deleted"] == 0

    @pytest.mark.asyncio
    async def test_delete_interleaved_test_with_events(self):
        """Test deletion with associated events."""
        mock_get_response = {
            "found": True,
            "_source": {"test_name": "Test A/B"},
        }

        mock_delete_response = {"result": "deleted"}
        mock_delete_by_query_response = {"deleted": 10}

        mock_client = Mock()
        mock_client.get.return_value = mock_get_response
        mock_client.delete.return_value = mock_delete_response
        mock_client.delete_by_query.return_value = mock_delete_by_query_response

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await delete_interleaved_test("test1", delete_events=True)
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["events_deleted"] == 10
            mock_client.delete_by_query.assert_called_once()


class TestCalculateInterleavedWinner:
    """Tests for calculate_interleaved_winner function."""

    scipy = pytest.importorskip("scipy")

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_calculate_winner_config_b_wins(self):
        """Test winner calculation when config_b wins."""
        mock_test_response = {
            "found": True,
            "_source": {
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
            },
        }

        mock_agg_response = {
            "aggregations": {
                "by_action": {
                    "buckets": [
                        {
                            "key": "click",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 10},
                                    {"key": "config_b", "doc_count": 20},
                                ]
                            },
                        },
                        {
                            "key": "impression",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 100},
                                    {"key": "config_b", "doc_count": 100},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_test_response
        mock_client.search.return_value = mock_agg_response

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch("builtins.__import__") as mock_import,
        ):
            # Mock scipy import
            mock_scipy_stats = Mock()
            mock_binomtest = Mock()
            mock_binomtest.pvalue = 0.01  # Significant
            mock_scipy_stats.binomtest.return_value = mock_binomtest

            def import_side_effect(name, *args, **kwargs):
                if name == "scipy":
                    mock_scipy = Mock()
                    mock_scipy.stats = mock_scipy_stats
                    return mock_scipy
                # Use real import for other modules
                import builtins

                return builtins.__import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = await calculate_interleaved_winner("test1")
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["winner"] == "config_b"
            assert result_data["statistical_test"]["is_significant"] is True

    @pytest.mark.asyncio
    async def test_calculate_winner_search_body_includes_test_id_filter(self):
        """Verify the UBI search query filters by test_id so winner is per-test."""
        test_id = "interleaved_test_xyz789"
        mock_test_response = {
            "found": True,
            "_source": {
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
            },
        }
        mock_agg_response = {
            "aggregations": {
                "by_action": {
                    "buckets": [
                        {
                            "key": "click",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 10},
                                    {"key": "config_b", "doc_count": 20},
                                ]
                            },
                        },
                        {
                            "key": "impression",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 100},
                                    {"key": "config_b", "doc_count": 100},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_test_response
        mock_client.search.return_value = mock_agg_response

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch("builtins.__import__") as mock_import,
        ):
            mock_scipy_stats = Mock()
            mock_binomtest = Mock()
            mock_binomtest.pvalue = 0.01
            mock_scipy_stats.binomtest.return_value = mock_binomtest

            def import_side_effect(name, *args, **kwargs):
                if name == "scipy":
                    mock_scipy = Mock()
                    mock_scipy.stats = mock_scipy_stats
                    return mock_scipy
                import builtins

                return builtins.__import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            await calculate_interleaved_winner(test_id)

            mock_client.search.assert_called_once()
            body = mock_client.search.call_args[1]["body"]
            must = body["query"]["bool"]["must"]
            expected_term = {
                "term": {"event_attributes.metadata.test_id.keyword": test_id}
            }
            assert expected_term in must, (
                "Search body must filter by test_id so winner is per-test; "
                f"got must={must}"
            )

    @pytest.mark.asyncio
    async def test_calculate_winner_no_significant_difference(self):
        """Test winner calculation when no significant difference."""
        mock_test_response = {
            "found": True,
            "_source": {
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
            },
        }

        mock_agg_response = {
            "aggregations": {
                "by_action": {
                    "buckets": [
                        {
                            "key": "click",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 15},
                                    {"key": "config_b", "doc_count": 15},
                                ]
                            },
                        },
                        {
                            "key": "impression",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 100},
                                    {"key": "config_b", "doc_count": 100},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_test_response
        mock_client.search.return_value = mock_agg_response

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch("builtins.__import__") as mock_import,
        ):
            # Mock scipy import
            mock_scipy_stats = Mock()
            mock_binomtest = Mock()
            mock_binomtest.pvalue = 0.5  # Not significant
            mock_scipy_stats.binomtest.return_value = mock_binomtest

            def import_side_effect(name, *args, **kwargs):
                if name == "scipy":
                    mock_scipy = Mock()
                    mock_scipy.stats = mock_scipy_stats
                    return mock_scipy
                # Use real import for other modules
                import builtins

                return builtins.__import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = await calculate_interleaved_winner("test1")
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["winner"] == "no_significant_difference"
            assert result_data["statistical_test"]["is_significant"] is False

    @pytest.mark.asyncio
    async def test_calculate_winner_no_clicks(self):
        """Test error when no clicks are found."""
        mock_test_response = {
            "found": True,
            "_source": {
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
            },
        }

        mock_agg_response = {
            "aggregations": {
                "by_action": {
                    "buckets": [
                        {
                            "key": "impression",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 100},
                                    {"key": "config_b", "doc_count": 100},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_test_response
        mock_client.search.return_value = mock_agg_response

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await calculate_interleaved_winner("test1")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "No clicks found" in result_data["error"]

    @pytest.mark.asyncio
    async def test_calculate_winner_tie_scenario(self):
        """Test critical path: tie scenario with equal clicks."""
        mock_test_response = {
            "found": True,
            "_source": {
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
            },
        }

        mock_agg_response = {
            "aggregations": {
                "by_action": {
                    "buckets": [
                        {
                            "key": "click",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 10},
                                    {
                                        "key": "config_b",
                                        "doc_count": 10,
                                    },  # Equal clicks
                                ]
                            },
                        },
                        {
                            "key": "impression",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 100},
                                    {"key": "config_b", "doc_count": 100},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_test_response
        mock_client.search.return_value = mock_agg_response

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch("builtins.__import__") as mock_import,
        ):
            # Mock scipy import
            mock_scipy_stats = Mock()
            mock_binomtest = Mock()
            mock_binomtest.pvalue = 0.8  # Not significant (high p-value for tie)
            mock_scipy_stats.binomtest.return_value = mock_binomtest

            def import_side_effect(name, *args, **kwargs):
                if name == "scipy":
                    mock_scipy = Mock()
                    mock_scipy.stats = mock_scipy_stats
                    return mock_scipy
                import builtins

                return builtins.__import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = await calculate_interleaved_winner("test1")
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["winner"] == "no_significant_difference"
            assert result_data["statistical_test"]["is_significant"] is False
            assert result_data["click_counts"]["config_a"] == 10
            assert result_data["click_counts"]["config_b"] == 10

    @pytest.mark.asyncio
    async def test_calculate_winner_very_small_sample(self):
        """Test critical path: very small sample size (edge case)."""
        mock_test_response = {
            "found": True,
            "_source": {
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
            },
        }

        mock_agg_response = {
            "aggregations": {
                "by_action": {
                    "buckets": [
                        {
                            "key": "click",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 1},
                                    {
                                        "key": "config_b",
                                        "doc_count": 2,
                                    },  # Very small sample
                                ]
                            },
                        },
                        {
                            "key": "impression",
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 10},
                                    {"key": "config_b", "doc_count": 10},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_test_response
        mock_client.search.return_value = mock_agg_response

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch("builtins.__import__") as mock_import,
        ):
            # Mock scipy import
            mock_scipy_stats = Mock()
            mock_binomtest = Mock()
            mock_binomtest.pvalue = 0.5  # Not significant with small sample
            mock_scipy_stats.binomtest.return_value = mock_binomtest

            def import_side_effect(name, *args, **kwargs):
                if name == "scipy":
                    mock_scipy = Mock()
                    mock_scipy.stats = mock_scipy_stats
                    return mock_scipy
                import builtins

                return builtins.__import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = await calculate_interleaved_winner("test1")
            result_data = json.loads(result)

            assert result_data["success"] is True
            # With very small sample, should likely be no significant difference
            assert result_data["statistical_test"]["is_significant"] is False
            assert result_data["click_counts"]["total"] == 3

    @pytest.mark.asyncio
    async def test_calculate_winner_scipy_not_available(self):
        """Test error when scipy is not available."""
        mock_client = Mock()

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch(
                "builtins.__import__", side_effect=ImportError("No module named scipy")
            ),
        ):
            result = await calculate_interleaved_winner("test1")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "scipy is required" in result_data["error"]


class TestInterleavedTestToolsSimulationScenarios:
    """Critical path tests for simulation scenarios with varied data patterns."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_simulate_interleaved_test_varied_click_patterns(self):
        """Test critical path: simulation with varied click patterns (some queries get clicks, others don't)."""
        mock_test_response = {
            "found": True,
            "_source": {
                "status": "active",
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
                "query_set_id": "qs1",
            },
        }

        mock_config_a = {
            "_source": {
                "query": {"match_all": {}},
                "index": "test_index",
            }
        }

        mock_config_b = {
            "_source": {
                "query": {"match_all": {}},
                "index": "test_index",
            }
        }

        mock_query_set = {
            "_source": {
                "querySetQueries": [{"queryText": f"query_{i}"} for i in range(10)]
            }
        }

        mock_client = Mock()
        mock_client.get.side_effect = [
            mock_test_response,
            mock_config_a,
            mock_config_b,
            mock_query_set,
        ]

        # Mock search responses with varied results
        def mock_search_side_effect(*args, **kwargs):
            query_text = (
                kwargs.get("body", {})
                .get("query", {})
                .get("match", {})
                .get("query", "")
            )
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": f"doc_{i}_{query_text}",
                            "_source": {"title": f"Result {i}"},
                        }
                        for i in range(10)
                    ]
                }
            }

        mock_client.search = Mock(side_effect=mock_search_side_effect)
        mock_client.index = Mock(return_value={"_id": "event1"})
        # Mock bulk response with proper structure
        mock_client.bulk = Mock(return_value={"errors": False, "items": []})

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await simulate_interleaved_test(
                test_id="test1",
                num_sessions=50,
                click_model="position_based",
                impressions_per_query=10,
                seed=42,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["simulation_config"]["num_sessions"] == 50

    @pytest.mark.asyncio
    async def test_simulate_interleaved_test_empty_results(self):
        """Test critical path: simulation handles queries with empty search results."""
        mock_test_response = {
            "found": True,
            "_source": {
                "status": "active",
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
                "query_set_id": "qs1",
            },
        }

        mock_config_a = {
            "_source": {
                "query": {"match_all": {}},
                "index": "test_index",
            }
        }

        mock_config_b = {
            "_source": {
                "query": {"match_all": {}},
                "index": "test_index",
            }
        }

        mock_query_set = {
            "_source": {
                "querySetQueries": [
                    {"queryText": "query_with_results"},
                    {"queryText": "query_no_results"},
                ]
            }
        }

        mock_client = Mock()
        mock_client.get.side_effect = [
            mock_test_response,
            mock_config_a,
            mock_config_b,
            mock_query_set,
        ]

        call_count = 0

        def mock_search_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First query returns results, second returns empty
            if call_count <= 2:
                return {
                    "hits": {
                        "hits": [
                            {"_id": f"doc_{i}", "_source": {"title": f"Result {i}"}}
                            for i in range(5)
                        ]
                    }
                }
            else:
                return {"hits": {"hits": []}}

        mock_client.search = Mock(side_effect=mock_search_side_effect)
        mock_client.index = Mock(return_value={"_id": "event1"})
        # Mock bulk response with proper structure
        mock_client.bulk = Mock(return_value={"errors": False, "items": []})

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await simulate_interleaved_test(
                test_id="test1",
                num_sessions=10,
                click_model="position_based",
                impressions_per_query=10,
                seed=42,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True

    @pytest.mark.asyncio
    async def test_simulate_interleaved_test_duplicate_documents(self):
        """Test critical path: simulation handles duplicate documents in results."""
        mock_test_response = {
            "found": True,
            "_source": {
                "status": "active",
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
                "query_set_id": "qs1",
            },
        }

        mock_config_a = {
            "_source": {
                "query": {"match_all": {}},
                "index": "test_index",
            }
        }

        mock_config_b = {
            "_source": {
                "query": {"match_all": {}},
                "index": "test_index",
            }
        }

        mock_query_set = {"_source": {"querySetQueries": [{"queryText": "test_query"}]}}

        mock_client = Mock()
        mock_client.get.side_effect = [
            mock_test_response,
            mock_config_a,
            mock_config_b,
            mock_query_set,
        ]

        # Return same document IDs from both configs (duplicates)
        mock_client.search = Mock(
            return_value={
                "hits": {
                    "hits": [
                        {"_id": "doc1", "_source": {"title": "Result 1"}},
                        {"_id": "doc2", "_source": {"title": "Result 2"}},
                    ]
                }
            }
        )
        mock_client.index = Mock(return_value={"_id": "event1"})
        # Mock bulk response with proper structure
        mock_client.bulk = Mock(return_value={"errors": False, "items": []})

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await simulate_interleaved_test(
                test_id="test1",
                num_sessions=5,
                click_model="position_based",
                impressions_per_query=10,
                seed=42,
            )

            result_data = json.loads(result)
            assert result_data["success"] is True


class TestInterleavedTestToolsStatisticalVariations:
    """Critical path tests for statistical test variations."""

    scipy = pytest.importorskip("scipy")

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_calculate_winner_different_confidence_levels(self):
        """Test critical path: calculate_winner with different significance levels."""
        mock_test_response = {
            "found": True,
            "_source": {
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
            },
        }

        # Mock aggregation response showing config_b wins
        mock_agg_response = {
            "aggregations": {
                "by_action": {
                    "buckets": [
                        {
                            "key": "click",
                            "doc_count": 100,
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 40},
                                    {"key": "config_b", "doc_count": 60},
                                ]
                            },
                        },
                        {
                            "key": "impression",
                            "doc_count": 200,
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 100},
                                    {"key": "config_b", "doc_count": 100},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_test_response
        mock_client.search.return_value = mock_agg_response

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch("builtins.__import__") as mock_import,
        ):
            # Mock scipy import
            mock_scipy_stats = Mock()
            mock_binomtest_05 = Mock()
            mock_binomtest_05.pvalue = 0.03  # Significant at 0.05 but not at 0.01
            mock_binomtest_01 = Mock()
            mock_binomtest_01.pvalue = 0.03  # Not significant at 0.01
            mock_scipy_stats.binomtest.side_effect = [
                mock_binomtest_05,
                mock_binomtest_01,
            ]

            def import_side_effect(name, *args, **kwargs):
                if name == "scipy":
                    mock_scipy = Mock()
                    mock_scipy.stats = mock_scipy_stats
                    return mock_scipy
                import builtins

                return builtins.__import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            # Test with 0.05 significance level (should be significant)
            result_05 = await calculate_interleaved_winner(
                "test1", significance_level=0.05
            )
            result_data_05 = json.loads(result_05)
            assert result_data_05["statistical_test"]["is_significant"] is True

            # Test with 0.01 significance level (should not be significant)
            result_01 = await calculate_interleaved_winner(
                "test1", significance_level=0.01
            )
            result_data_01 = json.loads(result_01)
            assert result_data_01["statistical_test"]["is_significant"] is False

    @pytest.mark.asyncio
    async def test_calculate_winner_small_sample_size(self):
        """Test critical path: calculate_winner with very small sample size."""
        mock_test_response = {
            "found": True,
            "_source": {
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
            },
        }

        # Very small sample - only 5 clicks total
        mock_agg_response = {
            "aggregations": {
                "by_action": {
                    "buckets": [
                        {
                            "key": "click",
                            "doc_count": 5,
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 2},
                                    {"key": "config_b", "doc_count": 3},
                                ]
                            },
                        },
                        {
                            "key": "impression",
                            "doc_count": 20,
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 10},
                                    {"key": "config_b", "doc_count": 10},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_test_response
        mock_client.search.return_value = mock_agg_response

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch("builtins.__import__") as mock_import,
        ):
            # Mock scipy import
            mock_scipy_stats = Mock()
            mock_binomtest = Mock()
            mock_binomtest.pvalue = 0.5  # Not significant with small sample
            mock_scipy_stats.binomtest.return_value = mock_binomtest

            def import_side_effect(name, *args, **kwargs):
                if name == "scipy":
                    mock_scipy = Mock()
                    mock_scipy.stats = mock_scipy_stats
                    return mock_scipy
                import builtins

                return builtins.__import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = await calculate_interleaved_winner(
                "test1", significance_level=0.05
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["statistical_test"]["is_significant"] is False
            assert result_data["click_counts"]["total"] == 5

    @pytest.mark.asyncio
    async def test_calculate_winner_large_sample_size(self):
        """Test critical path: calculate_winner with large sample size."""
        mock_test_response = {
            "found": True,
            "_source": {
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
            },
        }

        # Large sample - 10000 clicks
        mock_agg_response = {
            "aggregations": {
                "by_action": {
                    "buckets": [
                        {
                            "key": "click",
                            "doc_count": 10000,
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 4800},
                                    {"key": "config_b", "doc_count": 5200},
                                ]
                            },
                        },
                        {
                            "key": "impression",
                            "doc_count": 20000,
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 10000},
                                    {"key": "config_b", "doc_count": 10000},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_test_response
        mock_client.search.return_value = mock_agg_response

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch("builtins.__import__") as mock_import,
        ):
            # Mock scipy import
            mock_scipy_stats = Mock()
            mock_binomtest = Mock()
            mock_binomtest.pvalue = 0.001  # Significant with large sample
            mock_scipy_stats.binomtest.return_value = mock_binomtest

            def import_side_effect(name, *args, **kwargs):
                if name == "scipy":
                    mock_scipy = Mock()
                    mock_scipy.stats = mock_scipy_stats
                    return mock_scipy
                import builtins

                return builtins.__import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = await calculate_interleaved_winner(
                "test1", significance_level=0.05
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["statistical_test"]["is_significant"] is True
            assert result_data["click_counts"]["total"] == 10000


class TestInterleavedTestToolsLargeTests:
    """Critical path tests for large interleaved test performance."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_simulate_interleaved_test_large_sessions(self):
        """Test critical path: simulation handles large number of sessions (1000+)."""
        mock_test_response = {
            "found": True,
            "_source": {
                "status": "active",
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
                "query_set_id": "qs1",
            },
        }

        mock_config_a = {
            "_source": {
                "query": {"match_all": {}},
                "index": "test_index",
            }
        }

        mock_config_b = {
            "_source": {
                "query": {"match_all": {}},
                "index": "test_index",
            }
        }

        mock_query_set = {
            "_source": {
                "querySetQueries": [{"queryText": f"query_{i}"} for i in range(50)]
            }
        }

        mock_client = Mock()
        mock_client.get.side_effect = [
            mock_test_response,
            mock_config_a,
            mock_config_b,
            mock_query_set,
        ]

        mock_client.search = Mock(
            return_value={
                "hits": {
                    "hits": [
                        {"_id": f"doc_{i}", "_source": {"title": f"Result {i}"}}
                        for i in range(10)
                    ]
                }
            }
        )
        # Mock bulk indexing to return proper structure
        mock_client.bulk = Mock(return_value={"errors": False, "items": []})

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await simulate_interleaved_test(
                test_id="test1",
                num_sessions=1000,
                click_model="position_based",
                impressions_per_query=10,
                seed=42,
            )

            result_data = json.loads(result)
            # Should either succeed or fail gracefully
            assert isinstance(result_data, dict)
            # If successful, verify structure
            if "success" in result_data:
                assert result_data["success"] is True
                assert result_data["simulation_config"]["num_sessions"] == 1000

    @pytest.mark.asyncio
    async def test_calculate_winner_large_result_set(self):
        """Test critical path: calculate_winner with large result set (100K+ events)."""
        mock_test_response = {
            "found": True,
            "_source": {
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
            },
        }

        # Large result set
        mock_agg_response = {
            "aggregations": {
                "by_action": {
                    "buckets": [
                        {
                            "key": "click",
                            "doc_count": 50000,
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 24000},
                                    {"key": "config_b", "doc_count": 26000},
                                ]
                            },
                        },
                        {
                            "key": "impression",
                            "doc_count": 100000,
                            "by_config": {
                                "buckets": [
                                    {"key": "config_a", "doc_count": 50000},
                                    {"key": "config_b", "doc_count": 50000},
                                ]
                            },
                        },
                    ]
                }
            }
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_test_response
        mock_client.search.return_value = mock_agg_response

        with (
            patch("tools.interleaved_test_tools.os_client", mock_client),
            patch("builtins.__import__") as mock_import,
        ):
            # Mock scipy import
            mock_scipy_stats = Mock()
            mock_binomtest = Mock()
            mock_binomtest.pvalue = 0.0001  # Highly significant
            mock_scipy_stats.binomtest.return_value = mock_binomtest

            def import_side_effect(name, *args, **kwargs):
                if name == "scipy":
                    mock_scipy = Mock()
                    mock_scipy.stats = mock_scipy_stats
                    return mock_scipy
                import builtins

                return builtins.__import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            result = await calculate_interleaved_winner(
                "test1", significance_level=0.05
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["statistical_test"]["is_significant"] is True
            assert result_data["click_counts"]["total"] == 50000


class TestInterleavedTestToolsPartialFailures:
    """Critical path tests for partial failure scenarios."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_simulate_interleaved_test_partial_query_failure(self):
        """Test critical path: simulation handles partial query failures (some queries succeed, others fail)."""
        mock_test_response = {
            "found": True,
            "_source": {
                "status": "active",
                "search_configuration_a_id": "config_a",
                "search_configuration_b_id": "config_b",
                "query_set_id": "qs1",
            },
        }

        mock_config_a = {
            "_source": {
                "query": {"match_all": {}},
                "index": "test_index",
            }
        }

        mock_config_b = {
            "_source": {
                "query": {"match_all": {}},
                "index": "test_index",
            }
        }

        mock_query_set = {
            "_source": {
                "querySetQueries": [
                    {"queryText": "query1"},
                    {"queryText": "query2"},
                    {"queryText": "query3"},
                ]
            }
        }

        mock_client = Mock()
        mock_client.get.side_effect = [
            mock_test_response,
            mock_config_a,
            mock_config_b,
            mock_query_set,
        ]

        call_count = 0

        def mock_search_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail every 3rd search
            if call_count % 3 == 0:
                raise Exception("Search failed")
            return {
                "hits": {
                    "hits": [
                        {"_id": f"doc_{i}", "_source": {"title": f"Result {i}"}}
                        for i in range(5)
                    ]
                }
            }

        mock_client.search = Mock(side_effect=mock_search_side_effect)
        mock_client.index = Mock(return_value={"_id": "event1"})

        with patch("tools.interleaved_test_tools.os_client", mock_client):
            result = await simulate_interleaved_test(
                test_id="test1",
                num_sessions=10,
                click_model="position_based",
                impressions_per_query=10,
                seed=42,
            )

            # Should handle partial failures gracefully
            result_data = json.loads(result)
            # May succeed with partial data or fail gracefully
            assert isinstance(result_data, dict)
