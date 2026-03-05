"""
Query Set Tools
Tools for managing query sets using the OpenSearch Python client.
Query sets are collections of search queries used for testing and evaluation.
"""

from __future__ import annotations

import json

from utils.logging_helpers import get_logger
from utils.monitored_tool import monitored_tool
from utils.opensearch_client import get_client_manager
from utils.tool_utils import format_tool_error, log_tool_error

logger = get_logger(__name__)



@monitored_tool(
    name="GetQuerySetTool",
    description="Retrieves a specific query set by ID from OpenSearch",
)
def get_query_set(query_set_id: str) -> str:
    """
    Retrieves a specific query set by ID.

    Args:
        query_set_id: ID of the query set to retrieve

    Returns:
        str: JSON string containing the query set details
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.get_query_sets(query_set_id=query_set_id)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error retrieving query set: {str(e)}")


@monitored_tool(
    name="CreateQuerySetTool",
    description="Creates a new query set in OpenSearch by providing a list of queries",
)
def create_query_set(
    name: str,
    queries: str,
    description: str = "",
) -> str:
    """
    Creates a new query set with a list of queries.

    Args:
        name: Name of the query set
        queries: JSON string containing list of queries, e.g., ["query1", "query2"]
        description: Optional description of the query set

    Returns:
        str: Result of the creation operation with query set ID
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        # Parse queries if it's a string
        if isinstance(queries, str):
            queries = json.loads(queries)

        if not isinstance(queries, list):
            return format_tool_error(
                'queries must be a JSON array of strings or objects with queryText, e.g. ["q1", "q2"]'
            )

        # Convert plain string queries to the required format: [{"queryText": "query"}]
        query_set_queries = []
        for q in queries:
            if isinstance(q, str):
                query_set_queries.append({"queryText": q})
            elif isinstance(q, dict) and "queryText" in q:
                query_set_queries.append(q)
            else:
                query_set_queries.append({"queryText": str(q)})

        # The API expects this structure
        body = {
            "name": name,
            "description": description or f"Query set: {name}",
            "sampling": "manual",  # Required field for manually created query sets
            "querySetQueries": query_set_queries,
        }

        response = sr_client.put_query_sets(body=body)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error creating query set: {str(e)}")


@monitored_tool(
    name="SampleQuerySetTool",
    description="Creates a query set by sampling from user behavior data (UBI indices) in OpenSearch using topn sampling",
)
def sample_query_set(
    name: str,
    query_set_size: int = 20,
    description: str = "",
) -> str:
    """
    Creates a query set by sampling the top N most frequent queries from user behavior data.

    Args:
        name: Name of the query set
        query_set_size: Number of top queries to sample (default: 20)
        description: Optional description of the query set

    Returns:
        str: Result with the created query set ID
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        body = {
            "name": name,
            "description": description or f"Top {query_set_size} most frequent queries",
            "sampling": "topn",
            "querySetSize": query_set_size,
        }

        response = sr_client.post_query_sets(body=body)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error sampling query set: {str(e)}")


@monitored_tool(
    name="DeleteQuerySetTool", description="Deletes a query set by ID from OpenSearch"
)
def delete_query_set(query_set_id: str) -> str:
    """
    Deletes a query set by ID.

    Args:
        query_set_id: ID of the query set to delete

    Returns:
        str: Result of the deletion operation
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.delete_query_sets(query_set_id=query_set_id)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error deleting query set: {str(e)}")
