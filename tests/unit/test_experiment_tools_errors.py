"""
Unit tests for experiment tools error scenarios.

Tests error paths and edge cases for experiment operations including:
- Experiment not found scenarios
- Max retries exceeded
- Invalid configuration handling
- Empty results handling
- Error status handling
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.experiment_tools import (
    create_experiment,
    delete_experiment,
    get_experiment,
    get_experiment_results,
)

pytestmark = pytest.mark.unit


class TestExperimentToolsErrors:
    """Test experiment tool error scenarios."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_experiment_not_found(self):
        """Test get_experiment when experiment doesn't exist."""
        # Should return formatted error, not crash
        mock_response = {"hits": {"total": {"value": 0}, "hits": []}}

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with (
            patch(
                "tools.experiment_tools.get_client_manager"
            ) as mock_get_manager,
            patch("tools.experiment_tools.time.sleep") as mock_sleep,
        ):
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment("nonexistent_exp")
            result_data = json.loads(result)

            # Should return valid JSON with empty results, not crash
            assert "hits" in result_data
            assert result_data["hits"]["total"]["value"] == 0
            # Should have retried
            assert mock_sleep.call_count >= 1

    @pytest.mark.asyncio
    async def test_get_experiment_max_retries_exceeded(self):
        """Test get_experiment when max retries exceeded."""
        # Should return error after 10 retries
        mock_response_running = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {"status": "RUNNING"},
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_response_running

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with (
            patch(
                "tools.experiment_tools.get_client_manager"
            ) as mock_get_manager,
            patch("tools.experiment_tools.time.sleep") as mock_sleep,
        ):
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment("exp1")
            result_data = json.loads(result)

            # Should have retried 10 times
            assert mock_sr_client.get_experiments.call_count == 10
            assert mock_sleep.call_count == 9  # 9 sleeps between 10 attempts
            # Should return the last response (still running)
            assert result_data["hits"]["total"]["value"] == 1
            assert result_data["hits"]["hits"][0]["_source"]["status"] == "RUNNING"

    @pytest.mark.asyncio
    async def test_create_experiment_invalid_config(self):
        """Test create_experiment with invalid configuration."""
        # Should return formatted error, validate input
        mock_client_manager = Mock()

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            # Test invalid experiment type
            result = await create_experiment(
                query_set_id="qs1",
                search_configuration_ids='["config1"]',
                experiment_type="INVALID_TYPE",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "experiment_type must be one of" in result_data["error"]

            # Test PAIRWISE_COMPARISON with wrong number of configs
            result = await create_experiment(
                query_set_id="qs1",
                search_configuration_ids='["config1"]',  # Only 1, needs 2
                experiment_type="PAIRWISE_COMPARISON",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "exactly 2 search configurations" in result_data["error"]

            # Test POINTWISE_EVALUATION with wrong number of configs
            result = await create_experiment(
                query_set_id="qs1",
                search_configuration_ids='["config1", "config2"]',  # 2 configs, needs 1
                experiment_type="POINTWISE_EVALUATION",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "exactly 1 search configuration" in result_data["error"]

            # Test invalid JSON in search_configuration_ids
            result = await create_experiment(
                query_set_id="qs1",
                search_configuration_ids="invalid json",
                experiment_type="PAIRWISE_COMPARISON",
            )
            # Should return error (JSON decode error)
            assert (
                "error" in json.loads(result) or "Error creating experiment" in result
            )

    @pytest.mark.asyncio
    async def test_delete_experiment_not_found(self):
        """Test delete_experiment when experiment doesn't exist."""
        # Should handle gracefully, return error
        mock_sr_client = Mock()
        mock_sr_client.delete_experiments.side_effect = Exception("Not found")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await delete_experiment("nonexistent_exp")
            result_data = json.loads(result)

            # Should return formatted error, not crash
            assert "error" in result_data
            assert "Error deleting experiment" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_experiment_results_empty(self):
        """Test get_experiment_results with no results."""
        # Should return empty results, not error
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "results": [],  # Empty results
                            "searchConfigurationList": ["config1", "config2"],
                        },
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            # Should return valid JSON with empty results message
            assert result_data["experiment_id"] == "exp1"
            assert result_data["type"] == "PAIRWISE_COMPARISON"
            assert result_data["total_queries"] == 0
            assert "message" in result_data
            assert "No results found" in result_data["message"]

    @pytest.mark.asyncio
    async def test_get_experiment_results_not_found(self):
        """Test get_experiment_results when experiment doesn't exist."""
        mock_experiment_response = {"hits": {"total": {"value": 0}, "hits": []}}

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("nonexistent_exp")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Experiment not found" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_experiment_results_error_status(self):
        """Test get_experiment_results when experiment has ERROR status."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "ERROR",
                            "errorMessage": "Experiment execution failed",
                        },
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["status"] == "ERROR"
            assert "error_message" in result_data
            assert result_data["error_message"] == "Experiment execution failed"
            assert "message" in result_data
            assert "No results available" in result_data["message"]

    @pytest.mark.asyncio
    async def test_get_experiment_results_pending_status(self):
        """Test get_experiment_results when experiment is still PENDING."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "PENDING",
                        },
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["status"] == "PENDING"
            assert "message" in result_data
            assert "still pending" in result_data["message"].lower()

    @pytest.mark.asyncio
    async def test_get_experiment_results_unknown_status(self):
        """Test get_experiment_results when status is not COMPLETED/ERROR/PENDING/RUNNING."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "CANCELLED",
                        },
                    }
                ],
            }
        }
        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response
        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager
            result = await get_experiment_results("exp1")
            result_data = json.loads(result)
            assert result_data["status"] == "CANCELLED"
            assert "message" in result_data
            assert "Results not available" in result_data["message"]
            assert "CANCELLED" in result_data["message"]

    @pytest.mark.asyncio
    async def test_create_experiment_missing_judgment_lists(self):
        """Test create_experiment when judgment lists are missing for POINTWISE_EVALUATION."""
        mock_client_manager = Mock()

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_experiment(
                query_set_id="qs1",
                search_configuration_ids='["config1"]',
                experiment_type="POINTWISE_EVALUATION",
                judgment_list_ids=None,  # Missing
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "requires judgment_list_ids" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_experiment_connection_error(self):
        """Test get_experiment when connection error occurs."""
        mock_sr_client = Mock()
        mock_sr_client.get_experiments.side_effect = ConnectionError(
            "Connection failed"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment("exp1")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error retrieving experiment" in result_data["error"]
            assert "Connection failed" in result_data["error"]
