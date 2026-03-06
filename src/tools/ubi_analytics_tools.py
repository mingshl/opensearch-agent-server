"""
UBI Analytics Tools
Tools for analyzing user behavior insights (UBI) events to calculate metrics like CTR.
UBI events are stored in the ubi_events index and track search queries, clicks, and impressions.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from utils.logging_helpers import get_logger
from utils.monitored_tool import monitored_tool
from utils.opensearch_client import get_client_manager
from utils.tool_utils import log_tool_error

logger = get_logger(__name__)


@monitored_tool(
    name="GetQueryCTRTool",
    description="Calculate click-through rate (CTR) for a specific search query based on user behavior events",
)
def get_query_ctr(
    query_text: str,
    time_range_days: int = 30,
    ubi_index: str = "ubi_events",
) -> str:
    """
    Calculate click-through rate for a specific query.

    Args:
        query_text: The search query text to analyze
        time_range_days: Number of days to look back (default: 30)
        ubi_index: Name of the UBI events index (default: "ubi_events")

    Returns:
        str: JSON string with query CTR metrics including total searches,
             searches with clicks, CTR percentage, and average clicks per search
    """
    try:
        client_manager = get_client_manager()
        client = client_manager.get_client()

        # Calculate time range (format without microseconds for OpenSearch)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=time_range_days)

        # Format timestamps without microseconds
        end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Query to get search and click counts for the query
        query_body = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_query": query_text}},
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time_str,
                                    "lte": end_time_str,
                                },
                            },
                        },
                    ],
                },
            },
            "aggs": {
                "total_searches": {
                    "cardinality": {
                        "field": "query_id",
                    },
                },
                "searches_with_clicks": {
                    "filter": {
                        "term": {
                            "action_name": "click",
                        },
                    },
                    "aggs": {
                        "unique_queries": {
                            "cardinality": {
                                "field": "query_id",
                            },
                        },
                    },
                },
            },
        }

        response = client.search(
            index=ubi_index,
            body=query_body,
        )

        total_searches = response["aggregations"]["total_searches"]["value"]
        searches_with_clicks = response["aggregations"]["searches_with_clicks"][
            "unique_queries"
        ]["value"]
        total_clicks = response["aggregations"]["searches_with_clicks"]["doc_count"]

        ctr = (searches_with_clicks / total_searches * 100) if total_searches > 0 else 0
        avg_clicks = (total_clicks / total_searches) if total_searches > 0 else 0
        zero_click_rate = (
            ((total_searches - searches_with_clicks) / total_searches * 100)
            if total_searches > 0
            else 0
        )

        result = {
            "query_text": query_text,
            "time_range_days": time_range_days,
            "total_searches": total_searches,
            "searches_with_clicks": searches_with_clicks,
            "total_clicks": total_clicks,
            "ctr_percentage": round(ctr, 2),
            "average_clicks_per_search": round(avg_clicks, 2),
            "zero_click_rate_percentage": round(zero_click_rate, 2),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(logger, f"Error calculating query CTR: {str(e)}")


@monitored_tool(
    name="GetDocumentCTRTool",
    description="Calculate click-through rate (CTR) for a specific document based on user behavior events",
)
def get_document_ctr(
    doc_id: str,
    time_range_days: int = 30,
    ubi_index: str = "ubi_events",
) -> str:
    """
    Calculate click-through rate for a specific document.

    Args:
        doc_id: The document ID to analyze
        time_range_days: Number of days to look back (default: 30)
        ubi_index: Name of the UBI events index (default: "ubi_events")

    Returns:
        str: JSON string with document CTR metrics including total impressions,
             total clicks, CTR percentage, and average position when clicked
    """
    try:
        client_manager = get_client_manager()
        client = client_manager.get_client()

        # Calculate time range (format without microseconds for OpenSearch)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=time_range_days)

        # Format timestamps without microseconds
        end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Query to get impression and click counts for the document
        query_body = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"event_attributes.object.object_id": doc_id}},
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_time_str,
                                    "lte": end_time_str,
                                },
                            },
                        },
                    ],
                },
            },
            "aggs": {
                "total_impressions": {
                    "filter": {
                        "term": {
                            "action_name": "impression",
                        },
                    },
                },
                "clicks": {
                    "filter": {
                        "term": {
                            "action_name": "click",
                        },
                    },
                    "aggs": {
                        "avg_position": {
                            "avg": {
                                "field": "event_attributes.position.ordinal",
                            },
                        },
                    },
                },
            },
        }

        response = client.search(
            index=ubi_index,
            body=query_body,
        )

        total_impressions = response["aggregations"]["total_impressions"]["doc_count"]
        total_clicks = response["aggregations"]["clicks"]["doc_count"]
        avg_position = response["aggregations"]["clicks"]["avg_position"]["value"]

        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0

        result = {
            "document_id": doc_id,
            "time_range_days": time_range_days,
            "total_impressions": total_impressions,
            "total_clicks": total_clicks,
            "ctr_percentage": round(ctr, 2),
            "average_position_when_clicked": round(avg_position, 2)
            if avg_position
            else None,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(logger, f"Error calculating document CTR: {str(e)}")


@monitored_tool(
    name="GetQueryPerformanceMetricsTool",
    description="Get comprehensive performance metrics for queries. Can analyze a specific query or return top N queries by volume with their metrics.",
)
def get_query_performance_metrics(
    query_text: str | None = None,
    top_n: int = 20,
    time_range_days: int = 30,
    ubi_index: str = "ubi_events",
) -> str:
    """
    Get comprehensive performance metrics for queries.
    If query_text provided: detailed metrics for that query
    If query_text is None: top N queries by volume with their metrics

    Args:
        query_text: Specific query to analyze (optional)
        top_n: Number of top queries to return if query_text not provided (default: 20)
        time_range_days: Number of days to look back (default: 30)
        ubi_index: Name of the UBI events index (default: "ubi_events")

    Returns:
        str: JSON string with performance metrics for query/queries
    """
    try:
        client_manager = get_client_manager()
        client = client_manager.get_client()

        # Calculate time range (format without microseconds for OpenSearch)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=time_range_days)

        # Format timestamps without microseconds
        end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # If specific query provided, use GetQueryCTRTool
        if query_text:
            return get_query_ctr(query_text, time_range_days, ubi_index)

        # Otherwise, get top queries with metrics
        query_body = {
            "size": 0,
            "query": {
                "range": {
                    "timestamp": {
                        "gte": start_time_str,
                        "lte": end_time_str,
                    },
                },
            },
            "aggs": {
                "top_queries": {
                    "terms": {
                        "field": "user_query",
                        "size": top_n,
                        "order": {
                            "_count": "desc",
                        },
                    },
                    "aggs": {
                        "unique_searches": {
                            "cardinality": {
                                "field": "query_id",
                            },
                        },
                        "click_events": {
                            "filter": {
                                "term": {
                                    "action_name": "click",
                                },
                            },
                            "aggs": {
                                "unique_queries_with_clicks": {
                                    "cardinality": {
                                        "field": "query_id",
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

        response = client.search(
            index=ubi_index,
            body=query_body,
        )

        queries = []
        for bucket in response["aggregations"]["top_queries"]["buckets"]:
            query = bucket["key"]
            total_searches = bucket["unique_searches"]["value"]
            searches_with_clicks = bucket["click_events"]["unique_queries_with_clicks"][
                "value"
            ]
            total_clicks = bucket["click_events"]["doc_count"]

            ctr = (
                (searches_with_clicks / total_searches * 100)
                if total_searches > 0
                else 0
            )
            avg_clicks = (total_clicks / total_searches) if total_searches > 0 else 0
            zero_click_rate = (
                ((total_searches - searches_with_clicks) / total_searches * 100)
                if total_searches > 0
                else 0
            )

            queries.append(
                {
                    "query_text": query,
                    "search_volume": total_searches,
                    "searches_with_clicks": searches_with_clicks,
                    "total_clicks": total_clicks,
                    "ctr_percentage": round(ctr, 2),
                    "average_clicks_per_search": round(avg_clicks, 2),
                    "zero_click_rate_percentage": round(zero_click_rate, 2),
                }
            )

        result = {
            "time_range_days": time_range_days,
            "total_queries_analyzed": len(queries),
            "queries": queries,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(
            logger, f"Error getting query performance metrics: {str(e)}"
        )


@monitored_tool(
    name="GetTopQueriesByEngagementTool",
    description="Get queries with highest click-through rates (best engagement)",
)
def get_top_queries_by_engagement(
    top_n: int = 20,
    min_search_volume: int = 5,
    time_range_days: int = 30,
    ubi_index: str = "ubi_events",
) -> str:
    """
    Get queries with highest CTR (best engagement).
    Filters out low-volume queries to ensure statistical significance.

    Args:
        top_n: Number of top queries to return (default: 20)
        min_search_volume: Minimum number of searches required (default: 5)
        time_range_days: Number of days to look back (default: 30)
        ubi_index: Name of the UBI events index (default: "ubi_events")

    Returns:
        str: JSON string with top queries by CTR
    """
    try:
        # Get all query metrics
        metrics_json = get_query_performance_metrics(
            query_text=None,
            top_n=100,  # Get more to filter by volume
            time_range_days=time_range_days,
            ubi_index=ubi_index,
        )

        metrics = json.loads(metrics_json)
        if "error" in metrics:
            return metrics_json

        # Filter by minimum volume and sort by CTR
        filtered_queries = [
            q for q in metrics["queries"] if q["search_volume"] >= min_search_volume
        ]

        sorted_queries = sorted(
            filtered_queries, key=lambda x: x["ctr_percentage"], reverse=True
        )[:top_n]

        result = {
            "time_range_days": time_range_days,
            "min_search_volume": min_search_volume,
            "total_queries_analyzed": len(sorted_queries),
            "queries": sorted_queries,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(
            logger, f"Error getting top queries by engagement: {str(e)}"
        )


@monitored_tool(
    name="GetTopDocumentsByEngagementTool",
    description="Get documents with highest click-through rates (best engagement)",
)
def get_top_documents_by_engagement(
    top_n: int = 20,
    min_impressions: int = 5,
    time_range_days: int = 30,
    ubi_index: str = "ubi_events",
) -> str:
    """
    Get documents with highest CTR (best engagement).
    Filters out low-impression documents to ensure statistical significance.

    Args:
        top_n: Number of top documents to return (default: 20)
        min_impressions: Minimum number of impressions required (default: 5)
        time_range_days: Number of days to look back (default: 30)
        ubi_index: Name of the UBI events index (default: "ubi_events")

    Returns:
        str: JSON string with top documents by CTR
    """
    try:
        client_manager = get_client_manager()
        client = client_manager.get_client()

        # Calculate time range (format without microseconds for OpenSearch)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=time_range_days)

        # Format timestamps without microseconds
        end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Get top documents by impressions first
        query_body = {
            "size": 0,
            "query": {
                "range": {
                    "timestamp": {
                        "gte": start_time_str,
                        "lte": end_time_str,
                    },
                },
            },
            "aggs": {
                "top_documents": {
                    "terms": {
                        "field": "event_attributes.object.object_id",
                        "size": 100,  # Get more to filter
                        "order": {
                            "_count": "desc",
                        },
                    },
                    "aggs": {
                        "impressions": {
                            "filter": {
                                "term": {
                                    "action_name": "impression",
                                },
                            },
                        },
                        "clicks": {
                            "filter": {
                                "term": {
                                    "action_name": "click",
                                },
                            },
                            "aggs": {
                                "avg_position": {
                                    "avg": {
                                        "field": "event_attributes.position.ordinal",
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

        response = client.search(
            index=ubi_index,
            body=query_body,
        )

        documents = []
        for bucket in response["aggregations"]["top_documents"]["buckets"]:
            doc_id = bucket["key"]
            total_impressions = bucket["impressions"]["doc_count"]
            total_clicks = bucket["clicks"]["doc_count"]
            avg_position = bucket["clicks"]["avg_position"]["value"]

            # Filter by minimum impressions
            if total_impressions < min_impressions:
                continue

            ctr = (
                (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
            )

            documents.append(
                {
                    "document_id": doc_id,
                    "total_impressions": total_impressions,
                    "total_clicks": total_clicks,
                    "ctr_percentage": round(ctr, 2),
                    "average_position_when_clicked": round(avg_position, 2)
                    if avg_position
                    else None,
                }
            )

        # Sort by CTR and take top N
        sorted_documents = sorted(
            documents, key=lambda x: x["ctr_percentage"], reverse=True
        )[:top_n]

        result = {
            "time_range_days": time_range_days,
            "min_impressions": min_impressions,
            "total_documents_analyzed": len(sorted_documents),
            "documents": sorted_documents,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return log_tool_error(
            logger, f"Error getting top documents by engagement: {str(e)}"
        )
