"""
Search Configuration Tools
Tools for managing search configurations using the OpenSearch Python client.
"""

from __future__ import annotations

import json

from utils.logging_helpers import get_logger
from utils.monitored_tool import monitored_tool
from utils.opensearch_client import get_client_manager
from utils.tool_utils import format_tool_error, log_tool_error

logger = get_logger(__name__)



@monitored_tool(
    name="CreateSearchConfigurationTool",
    description="Creates a new search configuration in OpenSearch",
)
def create_search_configuration(
    name: str,
    index: str,
    query: str,
) -> str:
    """
    Creates a new search configuration.

    Args:
        name: Name of the search configuration
        index: The index to search
        query: The query (OpenSearch DSL as JSON string). Will be JSON-escaped for the API.

    Returns:
        str: Result of the creation operation with the created configuration ID
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        # The query field must be a JSON-escaped string, not an object
        # If the user passes a dict or parsed JSON, convert it back to a string
        if not isinstance(query, str):
            query = json.dumps(query)

        # If the query is already a JSON string (not escaped), we keep it as-is
        # The API expects: "query": "{\"multi_match\":{...}}"
        # So we need to ensure it's a string representation of JSON

        # Create search configuration body
        body = {
            "name": name,
            "index": index,
            "query": query,  # Keep as JSON string
        }

        response = sr_client.put_search_configurations(body=body)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error creating search configuration: {str(e)}")


@monitored_tool(
    name="GetSearchConfigurationTool",
    description="Retrieves a specific search configuration by ID from OpenSearch",
)
def get_search_configuration(search_configuration_id: str) -> str:
    """
    Retrieves a specific search configuration by ID.

    Args:
        search_configuration_id: ID of the search configuration to retrieve

    Returns:
        str: JSON string containing the search configuration details
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.get_search_configurations(
            search_configuration_id=search_configuration_id,
        )

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(
            logger, f"Error retrieving search configuration: {str(e)}"
        )


@monitored_tool(
    name="DeleteSearchConfigurationTool",
    description="Deletes a search configuration by ID from OpenSearch",
)
def delete_search_configuration(search_configuration_id: str) -> str:
    """
    Deletes a search configuration by ID.

    Args:
        search_configuration_id: ID of the search configuration to delete

    Returns:
        str: Result of the deletion operation
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.delete_search_configurations(
            search_configuration_id=search_configuration_id,
        )

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error deleting search configuration: {str(e)}")


@monitored_tool(
    name="DeploySearchConfigurationToProduction",
    description="Deploys a search configuration to Production OpenSearch",
)
def deploy_search_configuration_to_production(search_configuration_id: str) -> str:
    """
    Deploys a search configuration to Production by:
    1. Finding any existing configuration with name "agentic" and deleting it if found
    2. Getting details of the passed-in configuration
    3. Creating a new configuration with name "agentic" and values from the passed-in configuration

    Args:
        search_configuration_id: ID of the search configuration to deploy

    Returns:
        str: Result of the deploy operation
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        # Step 1: Find and delete the "agentic" configuration if it exists
        existing_configs = sr_client.get_search_configurations()
        agentic_config_id = None

        for hit in existing_configs.get("hits", {}).get("hits", []):
            config = hit.get("_source", {})
            if config.get("name") == "agentic":
                agentic_config_id = hit.get("_id")
                break

        if agentic_config_id:
            # Delete the existing "agentic" configuration
            sr_client.delete_search_configurations(
                search_configuration_id=agentic_config_id
            )

        # Step 2: Get the details of the passed-in configuration
        source_config_details = None
        existing_configs = sr_client.get_search_configurations()
        for hit in existing_configs.get("hits", {}).get("hits", []):
            config = hit.get("_source", {})
            if hit.get("_id") == search_configuration_id:
                source_config_details = config
                break

        if not source_config_details:
            return format_tool_error("Source configuration not found")

        # Step 3: Create a new configuration with name "agentic" and values from the source
        new_config_body = {
            "name": "agentic",
            "index": source_config_details.get("index", ""),
            "query": source_config_details.get("query", ""),
        }

        create_response = sr_client.put_search_configurations(body=new_config_body)

        return json.dumps(
            {"result": "success", "new_config": create_response},
            indent=2,
        )
    except Exception as e:
        return log_tool_error(logger, f"Error deploying search configuration: {str(e)}")


def _replace_search_text_placeholder(obj: object, query_text: str) -> object:
    """
    Recursively replace %SearchText% placeholder in a JSON structure.
    This safely handles JSON escaping by working with parsed structures.

    Args:
        obj: The JSON object (dict, list, or primitive)
        query_text: The text to replace %SearchText% with

    Returns:
        The object with %SearchText% replaced in all string values
    """
    if isinstance(obj, dict):
        return {
            key: _replace_search_text_placeholder(value, query_text)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [_replace_search_text_placeholder(item, query_text) for item in obj]
    elif isinstance(obj, str):
        return obj.replace("%SearchText%", query_text)
    else:
        return obj


@monitored_tool(
    name="ExecuteSearchWithConfigurationTool",
    description="Executes a search query using a specific search configuration. Returns the search results from OpenSearch.",
)
def execute_search_with_configuration(
    search_configuration_id: str,
    query_text: str,
    size: int = 10,
    fields: list[str | None] = None,
) -> str:
    """
    Executes a search query using a specific search configuration.

    Args:
        search_configuration_id: ID of the search configuration to use
        query_text: The search query text to execute
        size: Number of results to return (default: 10)
        fields: List of fields to return from documents (default: ["id", "title", "attrs.Brand"])

    Returns:
        str: JSON string with search results from OpenSearch containing only specified fields
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()
        os_client = client_manager.get_client()

        # Use default fields if not specified
        if fields is None:
            fields = ["id", "title", "attrs.Brand"]

        # Step 1: Get the search configuration details
        config_response = sr_client.get_search_configurations(
            search_configuration_id=search_configuration_id,
        )

        # Extract configuration details
        if "hits" not in config_response or len(config_response["hits"]["hits"]) == 0:
            return json.dumps(
                {
                    "error": f"Search configuration '{search_configuration_id}' not found",
                },
                indent=2,
            )

        config_source = config_response["hits"]["hits"][0]["_source"]
        index = config_source.get("index", "")
        query_template = config_source.get("query", "")

        if not index or not query_template:
            return json.dumps(
                {"error": "Search configuration missing index or query"},
                indent=2,
            )

        # Step 2: Parse the template JSON first, then safely replace placeholder
        try:
            query_body = json.loads(query_template)
        except json.JSONDecodeError as e:
            logger.exception("Invalid query JSON in search configuration")
            return json.dumps(
                {"error": f"Invalid query JSON in search configuration: {str(e)}"},
                indent=2,
            )

        # Recursively replace %SearchText% placeholder in the parsed structure
        # This safely handles JSON escaping since we work with parsed objects
        query_body = _replace_search_text_placeholder(query_body, query_text)

        # Step 3: Add size parameter and field filtering
        query_body["size"] = size
        query_body["_source"] = fields

        # Step 4: Execute the search
        search_response = os_client.search(index=index, body=query_body)

        # Step 5: Format the response
        hits = search_response.get("hits", {})
        total_raw = hits.get("total", 0)
        total_hits = (
            total_raw.get("value", 0)
            if isinstance(total_raw, dict)
            else (total_raw if isinstance(total_raw, int) else 0)
        )
        result = {
            "search_configuration_id": search_configuration_id,
            "search_configuration_name": config_source.get("name", ""),
            "query_text": query_text,
            "index": index,
            "total_hits": total_hits,
            "max_score": search_response.get("max_score"),
            "took_ms": search_response.get("took"),
            "results": [],
        }

        # Add results with relevant fields
        for hit in hits.get("hits", []):
            result["results"].append(
                {
                    "_id": hit["_id"],
                    "_score": hit["_score"],
                    "_source": hit["_source"],
                },
            )

        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(logger, f"Error executing search: {str(e)}")
