"""
OpenSearch Utility Tools
Direct OpenSearch API access with minimal verbosity.
"""

import json

from utils.logging_helpers import get_logger
from utils.monitored_tool import monitored_tool
from utils.opensearch_client import get_client_manager
from utils.tool_utils import log_tool_error

logger = get_logger(__name__)


@monitored_tool(
    name="ListIndexTool",
    description="List specific OpenSearch indexes using _cat/indices API. Check if an index exists and see basic stats.",
)
def list_index(index_pattern: str = "*") -> str:
    """
    List indexes matching the pattern.

    Args:
        index_pattern: Index name or pattern (default: "*" for all indexes)

    Returns:
        str: JSON string with index information (name, status, docs count, size)
    """
    try:
        client_manager = get_client_manager()
        client = client_manager.get_client()

        # Use cat.indices API for concise output
        response = client.cat.indices(
            index=index_pattern,
            format="json",
            h="index,status,docs.count,store.size",  # Only essential fields
        )

        if not response:
            return json.dumps(
                {
                    "found": False,
                    "message": f"No indexes found matching pattern '{index_pattern}'",
                },
                indent=2,
            )

        result = {"found": True, "count": len(response), "indexes": response}

        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = str(e)
        if "index_not_found" in error_msg.lower():
            return json.dumps(
                {"found": False, "message": f"Index '{index_pattern}' does not exist"},
                indent=2,
            )
        return log_tool_error(logger, f"Error listing indexes: {error_msg}")


@monitored_tool(
    name="SearchIndexTool",
    description="Search in a specific OpenSearch index with control over fields searched and returned. Defaults to searching in id, title, and attrs.Brand fields.",
)
def search_index(
    query_text: str, index: str, fields: list[str] | None = None, size: int = 10
) -> str:
    """
    Search an index with minimal, controlled output.

    Args:
        query_text: The search query text
        index: The index name to search
        fields: List of fields to search in and return (default: ["id", "title", "attrs.Brand"])
        size: Number of documents to return (default: 10)

    Returns:
        str: JSON string with search results containing only specified fields
    """
    try:
        client_manager = get_client_manager()
        client = client_manager.get_client()

        # Use default fields if not specified
        if fields is None:
            fields = ["id", "title", "attrs.Brand"]

        # Build query body
        query_body = {
            "size": size,
            "query": {
                "query_string": {
                    "query": query_text,
                    "fields": fields,
                },
            },
            "_source": fields,
        }

        response = client.search(index=index, body=query_body)

        # Extract only essential information
        hits = response.get("hits", {})
        total_raw = hits.get("total", 0)
        total = (
            total_raw.get("value", 0)
            if isinstance(total_raw, dict)
            else (total_raw if isinstance(total_raw, int) else 0)
        )
        documents = []

        for hit in hits.get("hits", []):
            doc = {
                "_id": hit.get("_id"),
                "_score": hit.get("_score"),
                "_source": hit.get("_source", {}),
            }
            documents.append(doc)

        result = {
            "total_hits": total,
            "returned": len(documents),
            "documents": documents,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(logger, f"Error searching index: {str(e)}")
