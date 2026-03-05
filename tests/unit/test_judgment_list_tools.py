"""
Unit tests for judgment list tools.

Tests critical paths for judgment operations including:
- CRUD operations (list, get, create, delete)
- Error handling
- UBI judgment creation
- LLM judgment generation (with mocked Bedrock)
- Pair extraction from experiments
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.judgment_list_tools import (
    create_ubi_judgment,
    delete_judgment,
    extract_pairs_from_pairwise_experiment,
    generate_llm_judgments,
    get_judgment,
    import_judgment,
)

pytestmark = pytest.mark.unit


class TestGetJudgment:
    """Tests for get_judgment function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_judgment_success(self):
        """Test successful retrieval of a judgment."""
        mock_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [{"_id": "judgment1", "_source": {"name": "Test Judgment"}}],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_judgments.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_judgment("judgment1")
            result_data = json.loads(result)

            assert "hits" in result_data
            mock_sr_client.get_judgments.assert_called_once_with(
                judgment_id="judgment1"
            )

    @pytest.mark.asyncio
    async def test_get_judgment_error(self):
        """Test error handling when getting judgment fails."""
        mock_sr_client = Mock()
        mock_sr_client.get_judgments.side_effect = Exception("Not found")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_judgment("invalid_id")
            assert "Error retrieving judgment" in result


class TestImportJudgment:
    """Tests for import_judgment function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_import_judgment_success(self):
        """Test successful import of a judgment."""
        mock_response = {"_id": "new_judgment_id", "status": "PROCESSING"}

        mock_sr_client = Mock()
        mock_sr_client.put_judgments.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await import_judgment(
                name="Test Judgment",
                query_text="laptop",
                doc_id="doc123",
                rating="3",
            )
            result_data = json.loads(result)

            assert result_data["_id"] == "new_judgment_id"
            assert result_data["status"] == "PROCESSING"
            mock_sr_client.put_judgments.assert_called_once()
            call_args = mock_sr_client.put_judgments.call_args[1]["body"]
            assert call_args["name"] == "Test Judgment"
            assert call_args["type"] == "IMPORT_JUDGMENT"
            assert len(call_args["judgmentRatings"]) == 1

    @pytest.mark.asyncio
    async def test_import_judgment_error(self):
        """Test error handling when importing judgment fails."""
        mock_sr_client = Mock()
        mock_sr_client.put_judgments.side_effect = Exception("Validation error")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await import_judgment(
                name="Test", query_text="query", doc_id="doc1", rating="1"
            )
            assert "Error importing judgment" in result

    @pytest.mark.asyncio
    async def test_import_judgment_name_too_long(self):
        """Test validation rejects judgment list name longer than 50 characters."""
        long_name = "A" * 51
        result = await import_judgment(
            name=long_name,
            query_text="query",
            doc_id="doc1",
            rating="1",
        )
        result_data = json.loads(result)
        assert result_data["error"] == "Invalid judgment list name"
        assert "too long" in result_data["message"]
        assert "50" in result_data["message"]


class TestCreateUBIJudgment:
    """Tests for create_ubi_judgment function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_create_ubi_judgment_success(self):
        """Test successful creation of UBI judgment."""
        mock_response = {"_id": "ubi_judgment_id", "status": "PROCESSING"}

        mock_client = Mock()
        mock_client.indices.exists.return_value = True

        mock_sr_client = Mock()
        mock_sr_client.put_judgments.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_ubi_judgment(
                name="UBI Judgment",
                click_model="coec",
                max_rank=20,
            )
            result_data = json.loads(result)

            assert result_data["_id"] == "ubi_judgment_id"
            mock_client.indices.exists.assert_called_once_with(index="ubi_events")
            mock_sr_client.put_judgments.assert_called_once()
            call_args = mock_sr_client.put_judgments.call_args[1]["body"]
            # Name should start with "UBI Judgment" and end with date suffix (_yyyyMMdd)
            assert call_args["name"].startswith("UBI Judgment_")
            assert (
                len(call_args["name"]) == len("UBI Judgment_") + 8
            )  # 8 digits for date
            assert call_args["name"][
                -8:
            ].isdigit()  # Last 8 characters should be digits
            assert call_args["type"] == "UBI_JUDGMENT"
            assert call_args["clickModel"] == "coec"

    @pytest.mark.asyncio
    async def test_create_ubi_judgment_index_missing(self):
        """Test error when ubi_events index doesn't exist."""
        mock_client = Mock()
        mock_client.indices.exists.return_value = False

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_ubi_judgment(name="UBI Judgment", click_model="coec")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "ubi_events index does not exist" in result_data["error"]

    @pytest.mark.asyncio
    async def test_create_ubi_judgment_with_dates(self):
        """Test UBI judgment creation with date filters."""
        mock_response = {"_id": "ubi_judgment_id", "status": "PROCESSING"}

        mock_client = Mock()
        mock_client.indices.exists.return_value = True

        mock_sr_client = Mock()
        mock_sr_client.put_judgments.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await create_ubi_judgment(
                name="UBI Judgment",
                click_model="coec",
                start_date="2024-01-01",
                end_date="2024-12-31",
            )
            result_data = json.loads(result)

            assert result_data["_id"] == "ubi_judgment_id"
            call_args = mock_sr_client.put_judgments.call_args[1]["body"]
            assert call_args["startDate"] == "2024-01-01"
            assert call_args["endDate"] == "2024-12-31"


class TestGenerateLLMJudgments:
    """Tests for generate_llm_judgments function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_generate_llm_judgments_success(self):
        """Test successful LLM judgment generation."""
        query_doc_pairs = json.dumps(
            [
                {"query": "laptop", "doc_id": "doc1"},
                {"query": "phone", "doc_id": "doc2"},
            ]
        )

        mock_doc_response = {
            "_source": {
                "id": "doc1",
                "title": "Laptop Product",
                "description": "A great laptop",
            }
        }

        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_client.get.return_value = mock_doc_response

        mock_bedrock_response = {"output": {"message": {"content": [{"text": "3"}]}}}

        mock_bedrock_client = Mock()
        mock_bedrock_client.converse = Mock(return_value=mock_bedrock_response)

        mock_sr_client = Mock()
        mock_sr_client.put_judgments.return_value = {"_id": "llm_judgment_id"}

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with (
            patch(
                "tools.judgment_list_tools.get_client_manager"
            ) as mock_get_manager,
            patch("tools.judgment_list_tools.boto3.Session") as mock_session,
            patch(
                "tools.judgment_list_tools.asyncio.to_thread"
            ) as mock_to_thread,
        ):
            mock_get_manager.return_value = mock_client_manager
            mock_session.return_value.client.return_value = mock_bedrock_client
            mock_to_thread.return_value = mock_bedrock_response

            result = await generate_llm_judgments(
                query_doc_pairs=query_doc_pairs,
                index="test_index",
                judgment_list_name="LLM Judgment",
            )
            result_data = json.loads(result)

            assert "judgment_list" in result_data
            assert "statistics" in result_data
            assert result_data["statistics"]["total_pairs"] == 2

    @pytest.mark.asyncio
    async def test_generate_llm_judgments_missing_aws_credentials(self):
        """Test error when AWS credentials are missing."""
        with (
            patch("tools.judgment_list_tools.AWS_ACCESS_KEY_ID", None),
            patch("tools.judgment_list_tools.AWS_SECRET_ACCESS_KEY", None),
        ):
            result = await generate_llm_judgments(
                query_doc_pairs='[{"query": "test", "doc_id": "doc1"}]',
                index="test_index",
                judgment_list_name="Test",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "AWS credentials not found" in result_data["error"]

    @pytest.mark.asyncio
    async def test_generate_llm_judgments_invalid_json(self):
        """Test error handling for invalid JSON input."""
        result = await generate_llm_judgments(
            query_doc_pairs="invalid json",
            index="test_index",
            judgment_list_name="Test",
        )

        assert "Error parsing query_doc_pairs JSON" in result

    @pytest.mark.asyncio
    async def test_generate_llm_judgments_index_not_exists(self):
        """Test error when index doesn't exist."""
        mock_client = Mock()
        mock_client.indices.exists.return_value = False

        mock_client_manager = Mock()
        mock_client_manager.get_client.return_value = mock_client

        with (
            patch(
                "tools.judgment_list_tools.get_client_manager"
            ) as mock_get_manager,
            patch("tools.judgment_list_tools.AWS_ACCESS_KEY_ID", "test-key"),
            patch(
                "tools.judgment_list_tools.AWS_SECRET_ACCESS_KEY", "test-secret"
            ),
        ):
            mock_get_manager.return_value = mock_client_manager

            result = await generate_llm_judgments(
                query_doc_pairs='[{"query": "test", "doc_id": "doc1"}]',
                index="nonexistent_index",
                judgment_list_name="Test",
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "does not exist" in result_data["error"]


class TestExtractPairsFromPairwiseExperiment:
    """Tests for extract_pairs_from_pairwise_experiment function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_extract_pairs_success(self):
        """Test successful extraction of pairs from pairwise experiment."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "results": [
                                {
                                    "query_text": "laptop",
                                    "snapshots": [
                                        {"docIds": ["doc1", "doc2"]},
                                        {"docIds": ["doc3", "doc4"]},
                                    ],
                                }
                            ],
                        }
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await extract_pairs_from_pairwise_experiment(
                experiment_id="exp1",
                max_docs_per_query=10,
            )
            result_data = json.loads(result)

            assert result_data["experiment_id"] == "exp1"
            assert result_data["total_pairs"] == 4
            assert len(result_data["pairs"]) == 4

    @pytest.mark.asyncio
    async def test_extract_pairs_experiment_not_found(self):
        """Test error when experiment is not found."""
        mock_experiment_response = {"hits": {"total": {"value": 0}, "hits": []}}

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await extract_pairs_from_pairwise_experiment(
                experiment_id="invalid"
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Experiment not found" in result_data["error"]

    @pytest.mark.asyncio
    async def test_extract_pairs_wrong_type(self):
        """Test error when experiment is not pairwise."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                        }
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await extract_pairs_from_pairwise_experiment(experiment_id="exp1")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "PAIRWISE_COMPARISON" in result_data["error"]


class TestDeleteJudgment:
    """Tests for delete_judgment function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock get_monitor to avoid Chainlit dependencies."""
        with patch("utils.monitored_tool.get_monitor", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_delete_judgment_success(self):
        """Test successful deletion of a judgment."""
        mock_response = {"result": "deleted"}

        mock_sr_client = Mock()
        mock_sr_client.delete_judgments.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await delete_judgment("judgment1")
            result_data = json.loads(result)

            assert result_data["result"] == "deleted"
            mock_sr_client.delete_judgments.assert_called_once_with(
                judgment_id="judgment1"
            )

    @pytest.mark.asyncio
    async def test_delete_judgment_error(self):
        """Test error handling when deleting judgment fails."""
        mock_sr_client = Mock()
        mock_sr_client.delete_judgments.side_effect = Exception("Not found")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await delete_judgment("invalid_id")
            assert "Error deleting judgment" in result
