"""
Judgment List Tools
Tools for managing judgment lists using the OpenSearch Python client.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import boto3

from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_info_event,
    log_warning_event,
)
from utils.monitored_tool import monitored_tool
from utils.opensearch_client import get_client_manager
from utils.tool_utils import log_tool_error

logger = get_logger(__name__)

# Get AWS credentials and configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
INFERENCE_PROFILE_ARN = os.getenv(
    "BEDROCK_INFERENCE_PROFILE_ARN", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

# Maximum allowed length for judgment list names
MAX_JUDGMENT_NAME_LENGTH = 50
# Reserve space for date suffix (_yyyyMMdd = 9 chars: 1 underscore + 8 date)
DATE_SUFFIX_LENGTH = 9
MAX_NAME_WITH_DATE = MAX_JUDGMENT_NAME_LENGTH - DATE_SUFFIX_LENGTH  # 41 chars


def _validate_judgment_name(name: str) -> tuple[bool, str]:
    """
    Validate judgment list name length.

    Args:
        name: The judgment list name

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(name) > MAX_JUDGMENT_NAME_LENGTH:
        return (
            False,
            f"Judgment list name is too long ({len(name)} characters). Maximum allowed is {MAX_JUDGMENT_NAME_LENGTH} characters. Please provide a shorter name.",
        )
    return True, ""


def _truncate_name_for_date(name: str) -> str:
    """
    Truncate judgment list name to allow room for date suffix (_yyyyMMdd).

    Args:
        name: The judgment list name

    Returns:
        Truncated name (max 41 characters to allow for 9-char date suffix)
    """
    if len(name) > MAX_NAME_WITH_DATE:
        truncated = name[:MAX_NAME_WITH_DATE]
        log_info_event(
            logger,
            "Truncated judgment list name to allow for date suffix.",
            "tools.judgment_list.name_truncated",
            name=name,
            truncated=truncated,
        )
        return truncated
    return name


def _append_date_suffix(name: str) -> str:
    """
    Append current date in yyyyMMdd format to judgment list name.

    Args:
        name: The judgment list name (should already be truncated)

    Returns:
        Name with date suffix
    """
    date_suffix = datetime.now().strftime("%Y%m%d")
    return f"{name}_{date_suffix}"



@monitored_tool(
    name="GetJudgmentTool",
    description="Retrieves a specific judgment list by ID from OpenSearch",
)
def get_judgment(judgment_id: str) -> str:
    """
    Retrieves a specific judgment by ID.

    Args:
        judgment_id: ID of the judgment to retrieve

    Returns:
        str: JSON string containing the judgment details
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.get_judgments(judgment_id=judgment_id)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error retrieving judgment: {str(e)}")


@monitored_tool(
    name="ImportJudgmentTool",
    description="Imports a judgment list (relevance ratings) into OpenSearch for search evaluation. Creates a judgment list with type IMPORT_JUDGMENT. The judgment will be processed asynchronously.",
)
def import_judgment(
    name: str,
    query_text: str,
    doc_id: str,
    rating: str,
) -> str:
    """
    Imports a judgment list for search relevance evaluation.

    Args:
        name: Name of the judgment list
        query_text: The search query text
        doc_id: ID of the document being judged
        rating: Relevance rating

    Returns:
        str: Result of the import operation with judgment ID.
             Status will be "PROCESSING" initially and will be processed asynchronously by OpenSearch.
    """
    try:
        # Validate name length
        is_valid, error_msg = _validate_judgment_name(name)
        if not is_valid:
            return json.dumps(
                {"error": "Invalid judgment list name", "message": error_msg}, indent=2
            )

        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        # Create judgment body following OpenSearch structure
        # type can be: IMPORT_JUDGMENT, CLICK_MODEL, etc.
        body = {
            "name": name,
            "type": "IMPORT_JUDGMENT",  # Standard type for manually created judgments
            "judgmentRatings": [
                {
                    "query": query_text,
                    "ratings": [
                        {
                            "docId": doc_id,
                            "rating": rating,
                        },
                    ],
                },
            ],
        }

        response = sr_client.put_judgments(body=body)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error importing judgment: {str(e)}")


@monitored_tool(
    name="CreateUBIJudgmentTool",
    description="Creates a new judgment list based on user behavior insights (UBI) data. Uses implicit judgments derived from user interactions like clicks. Requires ubi_events index to exist.",
)
def create_ubi_judgment(
    name: str,
    click_model: str,
    max_rank: int = 20,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """
    Creates a judgment list from user behavior insights (UBI) data.

    Args:
        name: Name of the judgment list
        click_model: Click model to use (e.g., "coec" for Clicks Over Expected Clicks)
        max_rank: Maximum rank position to consider in results (default: 20)
        start_date: Optional start date for UBI data (ISO format: YYYY-MM-DD)
        end_date: Optional end date for UBI data (ISO format: YYYY-MM-DD)

    Returns:
        str: Result of the creation operation with judgment ID.
             Status will be "PROCESSING" initially and will be processed asynchronously by OpenSearch.
    """
    try:
        client_manager = get_client_manager()
        client = client_manager.get_client()
        sr_client = client_manager.get_search_relevance_client()

        # Check if ubi_events index exists
        try:
            index_exists = client.indices.exists(index="ubi_events")
            if not index_exists:
                return json.dumps(
                    {
                        "error": "ubi_events index does not exist",
                        "message": "The ubi_events index is required for creating UBI-based judgments. Please ensure UBI data is being collected.",
                    },
                    indent=2,
                )
        except Exception as e:
            return log_tool_error(logger, f"Error checking ubi_events index: {str(e)}")

        # Truncate name and append date
        truncated_name = _truncate_name_for_date(name)
        final_name = _append_date_suffix(truncated_name)

        # Build request body
        body = {
            "name": final_name,
            "type": "UBI_JUDGMENT",
            "clickModel": click_model,
            "maxRank": max_rank,
        }

        # Add optional date filters if provided
        if start_date:
            body["startDate"] = start_date
        if end_date:
            body["endDate"] = end_date

        response = sr_client.put_judgments(body=body)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error creating UBI judgment: {str(e)}")


@monitored_tool(
    name="GenerateLLMJudgmentsTool",
    description="Generates relevance judgments for query-document pairs using AWS Bedrock Claude. Retrieves documents from the index and uses AI to judge relevance on a 0-3 scale. Processes pairs in batches for improved performance.",
)
async def generate_llm_judgments(
    query_doc_pairs: str,
    index: str,
    judgment_list_name: str,
    system_prompt: str = "",
    fields: list[str | None] = None,
) -> str:
    """
    Generates LLM-based judgments for query-document pairs with batched processing.

    Args:
        query_doc_pairs: JSON string of query-doc pairs, e.g., '[{"query": "laptop", "doc_id": "doc123"}, ...]'
        index: The index to retrieve documents from
        judgment_list_name: Name for the judgment list
        system_prompt: System prompt for the LLM (default: empty string, will use basic relevance judging prompt)
        fields: Optional list of document fields to include in the evaluation (default: None, includes all fields)

    Returns:
        str: Result of creating the judgment list with LLM-generated ratings
    """
    try:
        # Truncate name and append date
        truncated_name = _truncate_name_for_date(judgment_list_name)
        final_name = _append_date_suffix(truncated_name)

        log_info_event(
            logger,
            "[GenerateLLMJudgmentsTool] Starting judgment generation.",
            "tools.judgment_list.generate_start",
            final_name=final_name,
        )
        if fields:
            log_info_event(
                logger,
                "[GenerateLLMJudgmentsTool] Using fields for judging.",
                "tools.judgment_list.generate_fields",
                fields=fields,
            )

        # Check AWS credentials
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            return json.dumps(
                {
                    "error": "AWS credentials not found in environment",
                    "message": "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file",
                },
                indent=2,
            )

        # Create boto3 session and Bedrock runtime client
        bedrock_session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
        bedrock_client = bedrock_session.client("bedrock-runtime")

        # Parse query-doc pairs
        try:
            pairs = json.loads(query_doc_pairs)
            if not isinstance(pairs, list):
                return json.dumps(
                    {
                        "error": "Invalid format",
                        "message": "query_doc_pairs must be a JSON array of objects with 'query' and 'doc_id' fields",
                    },
                    indent=2,
                )
        except json.JSONDecodeError as e:
            return json.dumps(
                {"error": f"Error parsing query_doc_pairs JSON: {str(e)}"}, indent=2
            )

        total_pairs = len(pairs)
        log_info_event(
            logger,
            "[GenerateLLMJudgmentsTool] Processing query-document pairs.",
            "tools.judgment_list.generate_pairs",
            total_pairs=total_pairs,
        )

        # Get OpenSearch client
        client_manager = get_client_manager()
        os_client = client_manager.get_client()
        sr_client = client_manager.get_search_relevance_client()

        # Check if index exists
        try:
            index_exists = os_client.indices.exists(index=index)
            if not index_exists:
                return json.dumps(
                    {
                        "error": f"Index '{index}' does not exist",
                        "message": f"Please ensure the index '{index}' exists before generating judgments",
                    },
                    indent=2,
                )
        except Exception as e:
            return log_tool_error(logger, f"Error checking index: {str(e)}")

        # Helper function to process a single pair
        async def process_pair(
            pair: dict[str, Any], pair_index: int
        ) -> dict[str, Any] | None:
            """Judge relevance of one query-doc pair via LLM; returns dict with query, docId, rating or None on error."""
            try:
                if "query" not in pair or "doc_id" not in pair:
                    log_error_event(
                        logger,
                        "[GenerateLLMJudgmentsTool] ✗ Invalid pair - missing query or doc_id.",
                        "tools.judgment_list.generate_invalid_pair",
                        pair_index=pair_index,
                        exc_info=False,
                    )
                    return None

                query = pair["query"]
                doc_id = pair["doc_id"]

                # Retrieve document from index
                try:
                    # If specific fields are provided, only get those fields
                    if fields:
                        doc_response = os_client.get(
                            index=index, id=doc_id, _source_includes=fields
                        )
                    else:
                        doc_response = os_client.get(index=index, id=doc_id)
                    doc_source = doc_response["_source"]
                except Exception as e:
                    log_error_event(
                        logger,
                        "[GenerateLLMJudgmentsTool] ✗ Error retrieving doc.",
                        "tools.judgment_list.generate_doc_retrieve_error",
                        error=e,
                        doc_id=doc_id,
                        exc_info=False,
                    )
                    return None

                # Format document for LLM
                doc_text = json.dumps(doc_source, indent=2)

                # Add message about fields if specific fields were requested
                fields_info = (
                    f"\nNote: Only the following fields were included for evaluation: {', '.join(fields)}"
                    if fields
                    else ""
                )

                # Create user prompt for relevance judgment
                user_prompt = f"""Query: {query}

Document:
{doc_text}{fields_info}

Rate the relevance of this document to the query on a scale of 0-3:
- 0: Not relevant at all
- 1: Slightly relevant
- 2: Moderately relevant
- 3: Highly relevant

Respond with only the number (0, 1, 2, or 3)."""

                # Call AWS Bedrock Claude for judgment
                try:
                    # Prepare messages for Bedrock Converse API
                    messages = [{"role": "user", "content": [{"text": user_prompt}]}]

                    # Use asyncio.to_thread to run synchronous bedrock call in thread pool
                    response = await asyncio.to_thread(
                        bedrock_client.converse,
                        modelId=INFERENCE_PROFILE_ARN,
                        messages=messages,
                        system=[
                            {
                                "text": system_prompt
                                if system_prompt
                                else "You are a search relevance expert judging document relevance."
                            }
                        ],
                        inferenceConfig={"temperature": 0, "maxTokens": 10},
                    )

                    rating_str = response["output"]["message"]["content"][0][
                        "text"
                    ].strip()

                    # Validate rating
                    try:
                        rating_int = int(rating_str)
                        if rating_int < 0 or rating_int > 3:
                            rating_str = "0"  # Default to 0 if invalid
                    except ValueError:
                        rating_str = "0"  # Default to 0 if not a number

                except Exception as e:
                    log_error_event(
                        logger,
                        "[GenerateLLMJudgmentsTool] ✗ Error calling Bedrock API.",
                        "tools.judgment_list.generate_bedrock_error",
                        error=e,
                        query=query,
                        doc_id=doc_id,
                        exc_info=False,
                    )
                    return None

                return {"query": query, "docId": doc_id, "rating": rating_str}

            except Exception as e:
                log_error_event(
                    logger,
                    "[GenerateLLMJudgmentsTool] ✗ Unexpected error processing pair.",
                    "tools.judgment_list.generate_pair_error",
                    error=e,
                    pair_index=pair_index,
                    exc_info=False,
                )
                return None

        # Process pairs in batches
        BATCH_SIZE = 5  # Process 5 pairs concurrently
        judgment_ratings = {}  # Group by query
        successful_count = 0
        failed_count = 0

        for batch_start in range(0, total_pairs, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_pairs)
            batch = pairs[batch_start:batch_end]

            log_info_event(
                logger,
                "[GenerateLLMJudgmentsTool] Processing batch.",
                "tools.judgment_list.generate_batch",
                batch_num=batch_start // BATCH_SIZE + 1,
                batch_start=batch_start + 1,
                batch_end=batch_end,
                total_pairs=total_pairs,
            )

            # Process batch concurrently
            tasks = [
                process_pair(pair, batch_start + i) for i, pair in enumerate(batch)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful results
            for result in results:
                if result and not isinstance(result, Exception):
                    query = result["query"]
                    if query not in judgment_ratings:
                        judgment_ratings[query] = []
                    judgment_ratings[query].append(
                        {"docId": result["docId"], "rating": result["rating"]}
                    )
                    successful_count += 1
                else:
                    failed_count += 1

            # Add small delay between batches to avoid rate limiting
            if batch_end < total_pairs:
                await asyncio.sleep(0.5)

        log_info_event(
            logger,
            "[GenerateLLMJudgmentsTool] Completed.",
            "tools.judgment_list.generate_completed",
            successful_count=successful_count,
            failed_count=failed_count,
        )

        if successful_count == 0:
            return json.dumps(
                {
                    "error": "All judgment generation attempts failed",
                    "message": "Check logs for details about specific failures",
                },
                indent=2,
            )

        # Build judgment body in OpenSearch format
        body = {
            "name": final_name,
            "type": "IMPORT_JUDGMENT",
            "judgmentRatings": [
                {"query": query, "ratings": ratings}
                for query, ratings in judgment_ratings.items()
            ],
        }

        # Create judgment list in OpenSearch
        response = sr_client.put_judgments(body=body)

        # Add statistics to response
        result = {
            "judgment_list": response,
            "statistics": {
                "total_pairs": total_pairs,
                "successful": successful_count,
                "failed": failed_count,
                "success_rate": round(successful_count / total_pairs * 100, 2)
                if total_pairs > 0
                else 0,
                "fields_used": fields if fields else "all",
            },
        }

        log_info_event(
            logger,
            "[GenerateLLMJudgmentsTool] ✓ Successfully created judgment list",
            "tools.judgment_list.generate_created",
            final_name=final_name,
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        log_error_event(
            logger,
            "[GenerateLLMJudgmentsTool] ✗ ERROR.",
            "tools.judgment_list.generate_error",
            error=e,
        )
        return json.dumps(
            {"error": f"Error generating LLM judgments: {str(e)}"}, indent=2
        )


@monitored_tool(
    name="ExtractPairsFromPairwiseExperimentTool",
    description="Extracts query-document pairs from a pairwise experiment for LLM judgment generation. Returns pairs in the format required by GenerateLLMJudgmentsTool.",
)
def extract_pairs_from_pairwise_experiment(
    experiment_id: str,
    max_docs_per_query: int = 10,
    include_snapshot_index: int | None = None,
) -> str:
    """
    Extracts query-document pairs from a pairwise experiment.

    Args:
        experiment_id: ID of the pairwise experiment
        max_docs_per_query: Maximum number of documents to include per query (default: 10)
        include_snapshot_index: If specified, only include docs from this snapshot (0 or 1). If None, includes docs from both snapshots.

    Returns:
        str: JSON string of query-doc pairs formatted for generate_llm_judgments
    """
    try:
        log_info_event(
            logger,
            "[ExtractPairsFromPairwiseExperimentTool] Extracting pairs from experiment.",
            "tools.judgment_list.extract_start",
            experiment_id=experiment_id,
        )

        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        # Get the experiment
        response = sr_client.get_experiments(experiment_id=experiment_id)

        exp_hits = response.get("hits", {}).get("total", {}).get("value", 0)
        if exp_hits == 0:
            log_warning_event(
                logger,
                "[ExtractPairsFromPairwiseExperimentTool] ✗ Experiment not found.",
                "tools.judgment_list.extract_not_found",
                experiment_id=experiment_id,
            )
            return json.dumps(
                {"error": "Experiment not found", "experiment_id": experiment_id},
                indent=2,
            )

        exp_data = response.get("hits", {}).get("hits", [])[0].get("_source", {})
        exp_type = exp_data.get("type", "")

        # Validate it's a pairwise experiment
        if exp_type != "PAIRWISE_COMPARISON":
            log_error_event(
                logger,
                "[ExtractPairsFromPairwiseExperimentTool] ✗ Wrong experiment type.",
                "tools.judgment_list.extract_wrong_type",
                exp_type=exp_type,
                exc_info=False,
            )
            return json.dumps(
                {
                    "error": f"This tool only works with PAIRWISE_COMPARISON experiments. Found: {exp_type}",
                    "experiment_id": experiment_id,
                    "experiment_type": exp_type,
                },
                indent=2,
            )

        # Extract results
        results = exp_data.get("results", [])
        if not results:
            log_warning_event(
                logger,
                "[ExtractPairsFromPairwiseExperimentTool] ✗ No results in experiment.",
                "tools.judgment_list.extract_no_results",
            )
            return json.dumps(
                {
                    "error": "No results found in experiment",
                    "experiment_id": experiment_id,
                },
                indent=2,
            )

        log_info_event(
            logger,
            "[ExtractPairsFromPairwiseExperimentTool] Found queries in experiment.",
            "tools.judgment_list.extract_queries",
            query_count=len(results),
        )

        # Extract query-doc pairs
        pairs = []
        query_doc_set = set()  # Track unique pairs to avoid duplicates

        for result in results:
            query_text = result.get("query_text", "")
            snapshots = result.get("snapshots", [])

            for snapshot_idx, snapshot in enumerate(snapshots):
                # Skip if we're filtering by snapshot index
                if (
                    include_snapshot_index is not None
                    and snapshot_idx != include_snapshot_index
                ):
                    continue

                doc_ids = snapshot.get("docIds", [])

                # Limit docs per query
                for doc_id in doc_ids[:max_docs_per_query]:
                    # Create unique key to avoid duplicates
                    pair_key = (query_text, doc_id)
                    if pair_key not in query_doc_set:
                        query_doc_set.add(pair_key)
                        pairs.append({"query": query_text, "doc_id": doc_id})

        log_info_event(
            logger,
            "[ExtractPairsFromPairwiseExperimentTool] Extracted unique query-doc pairs.",
            "tools.judgment_list.extract_pairs",
            pair_count=len(pairs),
        )

        result = {
            "experiment_id": experiment_id,
            "experiment_type": exp_type,
            "total_queries": len(results),
            "total_pairs": len(pairs),
            "max_docs_per_query": max_docs_per_query,
            "pairs": pairs,
        }

        log_info_event(
            logger,
            "[ExtractPairsFromPairwiseExperimentTool] ✓ Successfully extracted pairs",
            "tools.judgment_list.extract_done",
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        log_error_event(
            logger,
            "[ExtractPairsFromPairwiseExperimentTool] ERROR.",
            "tools.judgment_list.extract_error",
            error=e,
        )
        return json.dumps(
            {"error": f"Error extracting pairs from pairwise experiment: {str(e)}"},
            indent=2,
        )


@monitored_tool(
    name="DeleteJudgmentTool", description="Deletes a judgment by ID from OpenSearch"
)
def delete_judgment(judgment_id: str) -> str:
    """
    Deletes a judgment by ID.

    Args:
        judgment_id: ID of the judgment to delete

    Returns:
        str: Result of the deletion operation
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.delete_judgments(judgment_id=judgment_id)

        return json.dumps(response, indent=2)
    except Exception as e:
        return log_tool_error(logger, f"Error deleting judgment: {str(e)}")
