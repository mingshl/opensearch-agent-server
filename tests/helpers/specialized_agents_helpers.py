"""
Helper functions for patching dependencies in specialized agent tests.

These functions reduce code duplication across test files by providing
reusable context managers for patching all dependencies needed for testing
specialized agents.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch


def patch_hypothesis_agent_dependencies(mock_agent: MagicMock) -> ExitStack:
    """Helper function to patch all dependencies for hypothesis_agent.

    Args:
        mock_agent: Mock agent instance to be returned by Agent constructor

    Returns:
        ExitStack context manager with all patches applied
    """
    stack = ExitStack()
    stack.enter_context(
        patch("agents.specialized_agents.Agent", return_value=mock_agent)
    )
    stack.enter_context(patch("agents.specialized_agents.BedrockModel"))
    stack.enter_context(patch("agents.specialized_agents.bedrock_session"))
    stack.enter_context(patch("agents.specialized_agents.list_index"))
    stack.enter_context(patch("agents.specialized_agents.search_index"))
    stack.enter_context(patch("tools.ubi_analytics_tools.get_query_ctr"))
    stack.enter_context(patch("tools.ubi_analytics_tools.get_document_ctr"))
    stack.enter_context(
        patch("tools.ubi_analytics_tools.get_query_performance_metrics")
    )
    stack.enter_context(
        patch("tools.ubi_analytics_tools.get_top_queries_by_engagement")
    )
    stack.enter_context(
        patch("tools.ubi_analytics_tools.get_top_documents_by_engagement")
    )
    stack.enter_context(
        patch("tools.search_configuration_tools.get_search_configuration")
    )
    stack.enter_context(
        patch("tools.search_configuration_tools.create_search_configuration")
    )
    stack.enter_context(
        patch("tools.search_configuration_tools.delete_search_configuration")
    )
    stack.enter_context(
        patch(
            "tools.search_configuration_tools.execute_search_with_configuration"
        )
    )
    stack.enter_context(patch("tools.query_set_tools.get_query_set"))
    stack.enter_context(patch("tools.query_set_tools.create_query_set"))
    stack.enter_context(patch("tools.query_set_tools.sample_query_set"))
    stack.enter_context(patch("tools.query_set_tools.delete_query_set"))
    stack.enter_context(patch("tools.experiment_tools.create_experiment"))
    stack.enter_context(patch("tools.experiment_tools.get_experiment_results"))
    return stack


def patch_evaluation_agent_dependencies(mock_agent: MagicMock) -> ExitStack:
    """Helper function to patch all dependencies for evaluation_agent.

    Args:
        mock_agent: Mock agent instance to be returned by Agent constructor

    Returns:
        ExitStack context manager with all patches applied
    """
    stack = ExitStack()
    stack.enter_context(
        patch("agents.specialized_agents.Agent", return_value=mock_agent)
    )
    stack.enter_context(patch("agents.specialized_agents.BedrockModel"))
    stack.enter_context(patch("agents.specialized_agents.bedrock_session"))
    stack.enter_context(patch("agents.specialized_agents.list_index"))
    stack.enter_context(patch("agents.specialized_agents.search_index"))
    stack.enter_context(patch("tools.judgment_list_tools.get_judgment"))
    stack.enter_context(patch("tools.judgment_list_tools.import_judgment"))
    stack.enter_context(
        patch("tools.judgment_list_tools.generate_llm_judgments")
    )
    stack.enter_context(
        patch("tools.judgment_list_tools.extract_pairs_from_pairwise_experiment")
    )
    stack.enter_context(patch("tools.judgment_list_tools.delete_judgment"))
    stack.enter_context(
        patch("tools.search_configuration_tools.get_search_configuration")
    )
    stack.enter_context(
        patch("tools.search_configuration_tools.create_search_configuration")
    )
    stack.enter_context(
        patch("tools.search_configuration_tools.delete_search_configuration")
    )
    stack.enter_context(
        patch(
            "tools.search_configuration_tools.execute_search_with_configuration"
        )
    )
    stack.enter_context(patch("tools.query_set_tools.get_query_set"))
    stack.enter_context(patch("tools.query_set_tools.create_query_set"))
    stack.enter_context(patch("tools.query_set_tools.sample_query_set"))
    stack.enter_context(patch("tools.query_set_tools.delete_query_set"))
    stack.enter_context(patch("tools.experiment_tools.get_experiment"))
    stack.enter_context(patch("tools.experiment_tools.create_experiment"))
    stack.enter_context(patch("tools.experiment_tools.delete_experiment"))
    stack.enter_context(patch("tools.experiment_tools.get_experiment_results"))
    stack.enter_context(patch("tools.ubi_analytics_tools.get_query_ctr"))
    stack.enter_context(patch("tools.ubi_analytics_tools.get_document_ctr"))
    stack.enter_context(
        patch("tools.ubi_analytics_tools.get_query_performance_metrics")
    )
    stack.enter_context(
        patch("tools.ubi_analytics_tools.get_top_queries_by_engagement")
    )
    stack.enter_context(
        patch("tools.ubi_analytics_tools.get_top_documents_by_engagement")
    )
    return stack


def patch_ubi_agent_dependencies(mock_agent: MagicMock) -> ExitStack:
    """Helper function to patch all dependencies for user_behavior_analysis_agent.

    Args:
        mock_agent: Mock agent instance to be returned by Agent constructor

    Returns:
        ExitStack context manager with all patches applied
    """
    stack = ExitStack()
    stack.enter_context(
        patch("agents.specialized_agents.Agent", return_value=mock_agent)
    )
    stack.enter_context(patch("agents.specialized_agents.BedrockModel"))
    stack.enter_context(patch("agents.specialized_agents.bedrock_session"))
    stack.enter_context(patch("agents.specialized_agents.list_index"))
    stack.enter_context(patch("agents.specialized_agents.search_index"))
    stack.enter_context(patch("tools.ubi_analytics_tools.get_query_ctr"))
    stack.enter_context(patch("tools.ubi_analytics_tools.get_document_ctr"))
    stack.enter_context(
        patch("tools.ubi_analytics_tools.get_query_performance_metrics")
    )
    stack.enter_context(
        patch("tools.ubi_analytics_tools.get_top_queries_by_engagement")
    )
    stack.enter_context(
        patch("tools.ubi_analytics_tools.get_top_documents_by_engagement")
    )
    return stack


