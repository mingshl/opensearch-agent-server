"""
Unit tests for search configuration tools error scenarios.

Tests error paths and edge cases for search configuration operations including:
- Configuration not found scenarios
- Duplicate name handling
- Invalid configuration handling
- Timeout scenarios
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.search_configuration_tools import (
    create_search_configuration,
    execute_search_with_configuration,
    get_search_configuration,
)

pytestmark = pytest.mark.unit


class TestSearchConfigurationToolsErrors:
    """Test search configuration tool error scenarios."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_search_configuration_not_found(self):
        """Test get_search_configuration when config doesn't exist."""
        mock_response = {"hits": {"total": {"value": 0}, "hits": []}}

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_search_configuration("nonexistent_config")
            result_data = json.loads(result)

            # Should return valid JSON with empty results, not crash
            assert "hits" in result_data
            assert result_data["hits"]["total"]["value"] == 0

    @pytest.mark.asyncio
    async def test_get_search_configuration_error(self):
        """Test get_search_configuration when API call fails."""
        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.side_effect = Exception(
            "Connection error"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_search_configuration("config1")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error retrieving search configuration" in result_data["error"]

    @pytest.mark.asyncio
    async def test_create_search_configuration_duplicate(self):
        """Test create_search_configuration with duplicate name."""
        # The API may return an error for duplicate names
        mock_sr_client = Mock()
        mock_sr_client.put_search_configurations.side_effect = Exception(
            "Configuration with name 'existing_config' already exists"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_search_configuration(
                name="existing_config",
                index="test_index",
                query='{"match_all": {}}',
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error creating search configuration" in result_data["error"]
            assert "already exists" in result_data["error"]

    @pytest.mark.asyncio
    async def test_create_search_configuration_invalid_query(self):
        """Test create_search_configuration with invalid query format."""
        mock_sr_client = Mock()
        mock_sr_client.put_search_configurations.side_effect = Exception(
            "Invalid query format"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_search_configuration(
                name="test_config",
                index="test_index",
                query="invalid query format",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error creating search configuration" in result_data["error"]

    @pytest.mark.asyncio
    async def test_execute_search_with_invalid_config(self):
        """Test execute_search_with_configuration with invalid config."""
        # Config not found
        mock_config_response = {"hits": {"hits": []}}

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_config_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await execute_search_with_configuration(
                search_configuration_id="invalid_config",
                query_text="test query",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "not found" in result_data["error"]

    @pytest.mark.asyncio
    async def test_execute_search_missing_index_or_query(self):
        """Test execute_search_with_configuration when config is missing index or query."""
        mock_config_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "config1",
                        "_source": {
                            "name": "Test Config",
                            # Missing index or query
                        },
                    }
                ]
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_config_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await execute_search_with_configuration(
                search_configuration_id="config1",
                query_text="test query",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "missing index or query" in result_data["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_search_invalid_query_json(self):
        """Test execute_search_with_configuration with invalid query JSON in config."""
        mock_config_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "config1",
                        "_source": {
                            "name": "Test Config",
                            "index": "test_index",
                            "query": "invalid json {",  # Invalid JSON
                        },
                    }
                ]
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_config_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await execute_search_with_configuration(
                search_configuration_id="config1",
                query_text="test query",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Invalid query JSON" in result_data["error"]

    @pytest.mark.asyncio
    async def test_execute_search_timeout(self):
        """Test execute_search_with_configuration timeout."""
        mock_config_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "config1",
                        "_source": {
                            "name": "Test Config",
                            "index": "test_index",
                            "query": '{"match_all": {}}',
                        },
                    }
                ]
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_config_response

        mock_client = Mock()
        mock_client.search.side_effect = TimeoutError("Request timeout")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await execute_search_with_configuration(
                search_configuration_id="config1",
                query_text="test query",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error executing search" in result_data["error"]
            assert "timeout" in result_data["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_search_connection_error(self):
        """Test execute_search_with_configuration when connection error occurs."""
        mock_config_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "config1",
                        "_source": {
                            "name": "Test Config",
                            "index": "test_index",
                            "query": '{"match_all": {}}',
                        },
                    }
                ]
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_config_response

        mock_client = Mock()
        mock_client.search.side_effect = ConnectionError("Connection failed")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await execute_search_with_configuration(
                search_configuration_id="config1",
                query_text="test query",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error executing search" in result_data["error"]

    @pytest.mark.asyncio
    async def test_execute_search_index_not_found(self):
        """Test execute_search_with_configuration when index doesn't exist."""
        mock_config_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "config1",
                        "_source": {
                            "name": "Test Config",
                            "index": "nonexistent_index",
                            "query": '{"match_all": {}}',
                        },
                    }
                ]
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_config_response

        mock_client = Mock()
        mock_client.search.side_effect = Exception("index_not_found_exception")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await execute_search_with_configuration(
                search_configuration_id="config1",
                query_text="test query",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Error executing search" in result_data["error"]
