"""
Interleaved A/B Testing Tools for Online Search Evaluation

This module provides tools for creating and running interleaved A/B tests with simulation.
Interleaved tests allow comparing two search configurations by showing users a combined
result list and tracking which configuration's results get more engagement.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from datetime import datetime, timezone

from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_info_event,
    log_warning_event,
)
from utils.monitored_tool import monitored_tool
from utils.opensearch_client import OpenSearchClientManager
from utils.tool_utils import format_tool_error

logger = get_logger(__name__)

# Initialize OpenSearch client
client_manager = OpenSearchClientManager()
os_client = client_manager.get_client()


def team_draft_interleave(
    results_a: list[dict],
    results_b: list[dict],
    k: int = 10,
    random_seed: int | None = None,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """
    Team Draft interleaving algorithm.

    Randomly assigns each result list to a team, then alternates picking documents
    from each team. Skips duplicates and tracks source attribution.

    Args:
        results_a: Search results from configuration A (list of dicts with 'id' field)
        results_b: Search results from configuration B (list of dicts with 'id' field)
        k: Maximum number of documents to return
        random_seed: Optional seed for reproducibility

    Returns:
        Tuple of:
        - List of (doc_id, source_config) tuples in interleaved order
        - Dict mapping doc_id -> source_config for attribution
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Randomly assign teams
    if random.random() < 0.5:
        team_a, team_b = results_a, results_b
        team_a_name, team_b_name = "config_a", "config_b"
    else:
        team_a, team_b = results_b, results_a
        team_a_name, team_b_name = "config_b", "config_a"

    interleaved = []
    attribution = {}
    seen_docs = set()

    idx_a = 0
    idx_b = 0
    turn = 0  # 0 = team A's turn, 1 = team B's turn

    while len(interleaved) < k and (idx_a < len(team_a) or idx_b < len(team_b)):
        # Determine which team picks
        if turn == 0:
            current_team = team_a
            current_idx = idx_a
            current_name = team_a_name
        else:
            current_team = team_b
            current_idx = idx_b
            current_name = team_b_name

        # Find next unseen document from current team
        while current_idx < len(current_team):
            doc_id = current_team[current_idx].get("id") or current_team[
                current_idx
            ].get("_id")
            if doc_id and doc_id not in seen_docs:
                interleaved.append((doc_id, current_name))
                attribution[doc_id] = current_name
                seen_docs.add(doc_id)
                current_idx += 1
                break
            current_idx += 1

        # Update index
        if turn == 0:
            idx_a = current_idx
        else:
            idx_b = current_idx

        # Switch turns
        turn = 1 - turn

        # If neither team has documents left, break
        if idx_a >= len(team_a) and idx_b >= len(team_b):
            break

    return interleaved, attribution


def position_based_click_probability(
    position: int,
    base_ctr: float = 0.3,
) -> float:
    """
    Position-based click probability model.

    P(click) = base_ctr / log2(position + 1)

    Args:
        position: Document position (1-indexed)
        base_ctr: Base click-through rate for position 1

    Returns:
        Click probability for the given position
    """
    return base_ctr / math.log2(position + 1)


def cascade_click_probability(
    position: int,
    relevance_prob: float = 0.5,
) -> float:
    """
    Cascade click model.

    Users scan top-down and stop at first relevant result.
    P(examine) = (1 - relevance_prob)^(position-1)
    P(click) = P(examine) * relevance_prob

    Args:
        position: Document position (1-indexed)
        relevance_prob: Probability a document is relevant when examined

    Returns:
        Click probability for the given position
    """
    examination_prob = (1 - relevance_prob) ** (position - 1)
    return examination_prob * relevance_prob


def generate_impression_event(
    query_id: str,
    session_id: str,
    client_id: str,
    query_text: str,
    doc_id: str,
    position: int,
    search_config: str,
    test_id: str | None = None,
) -> dict:
    """Generate UBI impression event."""
    metadata = {
        "search_config": search_config,
        "interleaved": True,
    }
    if test_id:
        metadata["test_id"] = test_id

    return {
        "application": "search_relevance_test",
        "action_name": "impression",
        "query_id": query_id,
        "session_id": session_id,
        "client_id": client_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message_type": "INFO",
        "user_query": query_text,
        "event_attributes": {
            "object": {
                "object_id": doc_id,
                "object_id_field": "id",
            },
            "position": {
                "ordinal": position,
            },
            "metadata": metadata,
        },
    }


def generate_click_event(
    query_id: str,
    session_id: str,
    client_id: str,
    query_text: str,
    doc_id: str,
    position: int,
    search_config: str,
    test_id: str | None = None,
) -> dict:
    """Generate UBI click event."""
    metadata = {
        "search_config": search_config,
        "interleaved": True,
        "dwell_time_ms": random.randint(2000, 30000),
    }
    if test_id:
        metadata["test_id"] = test_id

    return {
        "application": "search_relevance_test",
        "action_name": "click",
        "query_id": query_id,
        "session_id": session_id,
        "client_id": client_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message_type": "INFO",
        "user_query": query_text,
        "event_attributes": {
            "object": {
                "object_id": doc_id,
                "object_id_field": "id",
            },
            "position": {
                "ordinal": position,
            },
            "metadata": metadata,
        },
    }


@monitored_tool(
    name="CreateInterleavedTestTool",
    description="Creates an interleaved A/B test between two search configurations for online evaluation",
)
def create_interleaved_test(
    test_name: str,
    search_configuration_a_id: str,
    search_configuration_b_id: str,
    query_set_id: str,
    algorithm: str = "team_draft",
    description: str = "",
) -> str:
    """
    Registers an interleaved test for online evaluation.

    Args:
        test_name: Name for the test
        search_configuration_a_id: First search configuration (baseline)
        search_configuration_b_id: Second search configuration (variant)
        query_set_id: Query set to use for testing
        algorithm: Interleaving algorithm ("team_draft" or "balanced")
        description: Optional description

    Returns:
        JSON with test_id and test details
    """
    try:
        # Generate test ID
        test_id = f"interleaved_test_{uuid.uuid4().hex[:8]}"

        # Create test document
        test_doc = {
            "test_id": test_id,
            "test_name": test_name,
            "search_configuration_a_id": search_configuration_a_id,
            "search_configuration_b_id": search_configuration_b_id,
            "query_set_id": query_set_id,
            "algorithm": algorithm,
            "status": "active",
            "description": description,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stopped_at": None,
            "metadata": {
                "created_by": "online_testing_agent",
            },
        }

        # Store in OpenSearch
        index_name = "search-relevance-interleaved-test"

        # Create index if it doesn't exist
        if not os_client.indices.exists(index=index_name):
            os_client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "test_id": {
                                "type": "keyword",
                            },
                            "test_name": {
                                "type": "text",
                            },
                            "search_configuration_a_id": {
                                "type": "keyword",
                            },
                            "search_configuration_b_id": {
                                "type": "keyword",
                            },
                            "query_set_id": {
                                "type": "keyword",
                            },
                            "algorithm": {
                                "type": "keyword",
                            },
                            "status": {
                                "type": "keyword",
                            },
                            "description": {
                                "type": "text",
                            },
                            "created_at": {
                                "type": "date",
                            },
                            "stopped_at": {
                                "type": "date",
                            },
                        },
                    },
                },
            )
            log_info_event(
                logger,
                "[CreateInterleavedTestTool] ✓ Created index.",
                "tools.interleaved.create_index",
                index_name=index_name,
            )

        # Index the test document
        os_client.index(
            index=index_name,
            id=test_id,
            body=test_doc,
            refresh=True,
        )

        log_info_event(
            logger,
            "[CreateInterleavedTestTool] ✓ Created test",
            "tools.interleaved.create_done",
            test_id=test_id,
        )

        return json.dumps(
            {
                "success": True,
                "test_id": test_id,
                "test_name": test_name,
                "search_configuration_a_id": search_configuration_a_id,
                "search_configuration_b_id": search_configuration_b_id,
                "query_set_id": query_set_id,
                "algorithm": algorithm,
                "status": "active",
                "created_at": test_doc["created_at"],
            },
            indent=2,
        )

    except Exception as e:
        error_msg = f"Error creating interleaved test: {str(e)}"
        log_error_event(
            logger,
            "[CreateInterleavedTestTool] ✗ Error.",
            "tools.interleaved.create_error",
            error=e,
        )
        return format_tool_error(error_msg)


@monitored_tool(
    name="GetInterleavedTestTool",
    description="Retrieves details of an interleaved test",
)
def get_interleaved_test(test_id: str) -> str:
    """
    Retrieves interleaved test details.

    Args:
        test_id: The interleaved test ID

    Returns:
        JSON with test details
    """
    try:
        response = os_client.get(
            index="search-relevance-interleaved-test",
            id=test_id,
        )

        if not response.get("found"):
            return format_tool_error(f"Test not found: {test_id}")

        test_data = response["_source"]
        log_info_event(
            logger,
            "[GetInterleavedTestTool] ✓ Retrieved test.",
            "tools.interleaved.get_done",
            test_id=test_id,
        )

        return json.dumps(test_data, indent=2)

    except Exception as e:
        error_msg = f"Error retrieving test: {str(e)}"
        log_error_event(
            logger,
            "[GetInterleavedTestTool] ✗ Error.",
            "tools.interleaved.get_error",
            error=e,
        )
        return format_tool_error(error_msg)


@monitored_tool(
    name="StopInterleavedTestTool", description="Stops an active interleaved test"
)
def stop_interleaved_test(test_id: str) -> str:
    """
    Stops an active interleaved test.

    Args:
        test_id: The interleaved test ID

    Returns:
        JSON with updated test status
    """
    try:
        # Update test status
        os_client.update(
            index="search-relevance-interleaved-test",
            id=test_id,
            body={
                "doc": {
                    "status": "stopped",
                    "stopped_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            },
            refresh=True,
        )

        log_info_event(
            logger,
            "[StopInterleavedTestTool] ✓ Stopped test.",
            "tools.interleaved.stop_done",
            test_id=test_id,
        )

        return json.dumps(
            {
                "success": True,
                "test_id": test_id,
                "status": "stopped",
                "message": "Test stopped successfully",
            },
            indent=2,
        )

    except Exception as e:
        error_msg = f"Error stopping test: {str(e)}"
        log_error_event(
            logger,
            "[StopInterleavedTestTool] ✗ Error.",
            "tools.interleaved.stop_error",
            error=e,
        )
        return format_tool_error(error_msg)


@monitored_tool(
    name="DeleteInterleavedTestTool",
    description="Deletes an interleaved test and optionally its associated simulated UBI events",
)
def delete_interleaved_test(
    test_id: str,
    delete_events: bool = False,
) -> str:
    """
    Deletes an interleaved test and optionally its associated events.

    Args:
        test_id: The interleaved test ID
        delete_events: If True, also delete associated UBI events (default: False)

    Returns:
        JSON with deletion results
    """
    try:
        # Check if test exists
        test_response = os_client.get(
            index="search-relevance-interleaved-test", id=test_id
        )
        if not test_response.get("found"):
            return format_tool_error(f"Test not found: {test_id}")

        test_data = test_response["_source"]

        # Delete the test document
        os_client.delete(
            index="search-relevance-interleaved-test", id=test_id, refresh=True
        )

        log_info_event(
            logger,
            "[DeleteInterleavedTestTool] ✓ Deleted test",
            "tools.interleaved.delete_done",
            test_id=test_id,
        )

        result = {
            "success": True,
            "test_id": test_id,
            "test_name": test_data.get("test_name"),
            "test_deleted": True,
            "events_deleted": 0,
        }

        # Optionally delete associated events
        if delete_events:
            log_info_event(
                logger,
                "[DeleteInterleavedTestTool] Deleting events for test.",
                "tools.interleaved.delete_events",
                test_id=test_id,
            )

            # Query to find all events with this test_id
            delete_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"event_attributes.metadata.interleaved": True}},
                            {
                                "term": {
                                    "event_attributes.metadata.test_id.keyword": test_id,
                                },
                            },
                        ],
                    },
                },
            }

            # Delete by query
            delete_response = os_client.delete_by_query(
                index="ubi_events", body=delete_query, refresh=True
            )

            events_deleted = delete_response.get("deleted", 0)
            result["events_deleted"] = events_deleted

            log_info_event(
                logger,
                "[DeleteInterleavedTestTool] ✓ Deleted events.",
                "tools.interleaved.delete_events_done",
                events_deleted=events_deleted,
            )

        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"Error deleting test: {str(e)}"
        log_error_event(
            logger,
            "[DeleteInterleavedTestTool] ✗ Error.",
            "tools.interleaved.delete_error",
            error=e,
        )
        return format_tool_error(error_msg)


@monitored_tool(
    name="SimulateInterleavedTestTool",
    description="Simulates user behavior for an interleaved test by generating UBI events (impressions and clicks)",
)
async def simulate_interleaved_test(
    test_id: str,
    num_sessions: int = 100,
    click_model: str = "position_based",
    impressions_per_query: int = 10,
    seed: int | None = None,
) -> str:
    """
    Simulates user sessions for an interleaved test.

    Args:
        test_id: The interleaved test ID
        num_sessions: Number of user sessions to simulate (default 100)
        click_model: Click probability model - "position_based", "cascade", or "random" (default "position_based")
        impressions_per_query: Max documents shown per query (default 10)
        seed: Random seed for reproducibility (optional)

    Returns:
        JSON with simulation statistics
    """
    try:
        if seed is not None:
            random.seed(seed)

        log_info_event(
            logger,
            "[SimulateInterleavedTestTool] Starting simulation.",
            "tools.interleaved.simulate_start",
            test_id=test_id,
        )
        log_info_event(
            logger,
            "[SimulateInterleavedTestTool] Sessions and click model.",
            "tools.interleaved.simulate_config",
            num_sessions=num_sessions,
            click_model=click_model,
        )

        # Get test details
        test_response = os_client.get(
            index="search-relevance-interleaved-test", id=test_id
        )
        if not test_response.get("found"):
            return format_tool_error(f"Test not found: {test_id}")

        test_data = test_response["_source"]

        # Check if test is active
        test_status = test_data.get("status", "active")
        if test_status != "active":
            return json.dumps(
                {
                    "error": f"Cannot simulate stopped test. Test status: {test_status}",
                    "test_id": test_id,
                    "status": test_status,
                }
            )

        config_a_id = test_data["search_configuration_a_id"]
        config_b_id = test_data["search_configuration_b_id"]
        query_set_id = test_data["query_set_id"]

        # Get search configurations
        config_a_response = os_client.get(
            index="search-relevance-search-config", id=config_a_id
        )
        config_b_response = os_client.get(
            index="search-relevance-search-config", id=config_b_id
        )

        config_a = config_a_response["_source"]["query"]
        config_b = config_b_response["_source"]["query"]
        search_index = config_a_response["_source"]["index"]

        # Get query set
        query_set_response = os_client.get(
            index="search-relevance-queryset", id=query_set_id
        )
        query_set_queries = query_set_response["_source"]["querySetQueries"]

        # Extract query texts from the querySetQueries structure
        queries = [q["queryText"] for q in query_set_queries]

        log_info_event(
            logger,
            "[SimulateInterleavedTestTool] Loaded queries from query set.",
            "tools.interleaved.simulate_queries_loaded",
            query_count=len(queries),
        )

        # Statistics
        stats = {
            "total_sessions": 0,
            "total_queries": 0,
            "total_impressions": 0,
            "total_clicks": 0,
            "clicks_by_config": {"config_a": 0, "config_b": 0},
            "impressions_by_config": {"config_a": 0, "config_b": 0},
        }

        events_to_index = []

        # Simulate sessions
        for session_num in range(num_sessions):
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            client_id = f"client_{uuid.uuid4().hex[:8]}"

            # Select random query
            query_text = random.choice(queries)
            query_id = f"query_{uuid.uuid4().hex[:8]}"

            # Execute both search configurations
            query_a = json.dumps(config_a).replace("%SearchText%", query_text)
            query_b = json.dumps(config_b).replace("%SearchText%", query_text)

            results_a = os_client.search(
                index=search_index, body=json.loads(query_a), size=20
            )

            results_b = os_client.search(
                index=search_index, body=json.loads(query_b), size=20
            )

            hits_a = results_a.get("hits", {}).get("hits", [])
            hits_b = results_b.get("hits", {}).get("hits", [])

            if not hits_a or not hits_b:
                continue

            # Apply Team Draft interleaving
            # Note: Do NOT pass seed here - we want team assignment to be random per query
            # The seed only controls query selection and click simulation for reproducibility
            interleaved_results, attribution = team_draft_interleave(
                hits_a, hits_b, k=impressions_per_query, random_seed=None
            )

            # Generate impression events
            for position, (doc_id, source_config) in enumerate(
                interleaved_results, start=1
            ):
                impression_event = generate_impression_event(
                    query_id=query_id,
                    session_id=session_id,
                    client_id=client_id,
                    query_text=query_text,
                    doc_id=doc_id,
                    position=position,
                    search_config=source_config,
                    test_id=test_id,
                )
                events_to_index.append(impression_event)
                stats["total_impressions"] += 1
                stats["impressions_by_config"][source_config] += 1

            # Simulate clicks based on click model
            for position, (doc_id, source_config) in enumerate(
                interleaved_results, start=1
            ):
                # Calculate click probability
                if click_model == "position_based":
                    click_prob = position_based_click_probability(position)
                elif click_model == "cascade":
                    click_prob = cascade_click_probability(position)
                elif click_model == "random":
                    click_prob = 0.1  # 10% uniform probability
                else:
                    click_prob = position_based_click_probability(position)

                # Simulate click
                if random.random() < click_prob:
                    click_event = generate_click_event(
                        query_id=query_id,
                        session_id=session_id,
                        client_id=client_id,
                        query_text=query_text,
                        doc_id=doc_id,
                        position=position,
                        search_config=source_config,
                        test_id=test_id,
                    )
                    events_to_index.append(click_event)
                    stats["total_clicks"] += 1
                    stats["clicks_by_config"][source_config] += 1

                    # In cascade model, stop after first click
                    if click_model == "cascade":
                        break

            stats["total_sessions"] += 1
            stats["total_queries"] += 1

            # Log progress every 10 sessions
            if (session_num + 1) % 10 == 0:
                log_info_event(
                    logger,
                    "[SimulateInterleavedTestTool] Progress.",
                    "tools.interleaved.simulate_progress",
                    session_num=session_num + 1,
                    num_sessions=num_sessions,
                )

        # Bulk index all events
        if events_to_index:
            log_info_event(
                logger,
                "[SimulateInterleavedTestTool] Indexing events to ubi_events.",
                "tools.interleaved.simulate_index",
                event_count=len(events_to_index),
            )

            # Log a sample event for debugging
            log_info_event(
                logger,
                "[SimulateInterleavedTestTool] Sample impression event.",
                "tools.interleaved.simulate_sample_impression",
                sample=json.dumps(events_to_index[0], indent=2),
            )

            # Find first click event if any
            click_events = [
                e for e in events_to_index if e.get("action_name") == "click"
            ]
            if click_events:
                log_info_event(
                    logger,
                    "[SimulateInterleavedTestTool] Sample click event.",
                    "tools.interleaved.simulate_sample_click",
                    sample=json.dumps(click_events[0], indent=2),
                )

            bulk_body = []
            for event in events_to_index:
                bulk_body.append(
                    {
                        "index": {
                            "_index": "ubi_events",
                        },
                    },
                )
                bulk_body.append(event)

            bulk_response = os_client.bulk(
                body=bulk_body,
                refresh=True,
            )

            if bulk_response.get("errors"):
                log_warning_event(
                    logger,
                    "[SimulateInterleavedTestTool] Some events failed to index.",
                    "tools.interleaved.simulate_index_errors",
                )

                # Log detailed error information
                failed_count = 0
                for idx, item in enumerate(bulk_response.get("items", [])):
                    if "index" in item and item["index"].get("error"):
                        failed_count += 1
                        error_info = item["index"]["error"]
                        event_idx = idx

                        # Log first few errors with details
                        if failed_count <= 3:
                            log_error_event(
                                logger,
                                "[SimulateInterleavedTestTool] Event failed.",
                                "tools.interleaved.simulate_event_failed",
                                error=error_info.get("reason", str(error_info)),
                                event_idx=event_idx,
                                error_type=error_info.get("type"),
                                exc_info=False,
                            )

                log_warning_event(
                    logger,
                    "[SimulateInterleavedTestTool] Total failed events.",
                    "tools.interleaved.simulate_failed_count",
                    failed_count=failed_count,
                    total=len(events_to_index),
                )

        # Calculate overall CTR
        stats["overall_ctr"] = (
            stats["total_clicks"] / stats["total_impressions"]
            if stats["total_impressions"] > 0
            else 0
        )
        stats["ctr_by_config"] = {
            "config_a": stats["clicks_by_config"]["config_a"]
            / stats["impressions_by_config"]["config_a"]
            if stats["impressions_by_config"]["config_a"] > 0
            else 0,
            "config_b": stats["clicks_by_config"]["config_b"]
            / stats["impressions_by_config"]["config_b"]
            if stats["impressions_by_config"]["config_b"] > 0
            else 0,
        }

        log_info_event(
            logger,
            "[SimulateInterleavedTestTool] ✓ Simulation complete",
            "tools.interleaved.simulate_complete",
        )
        log_info_event(
            logger,
            "[SimulateInterleavedTestTool] Total clicks and CTR.",
            "tools.interleaved.simulate_stats",
            total_clicks=stats["total_clicks"],
            overall_ctr=stats["overall_ctr"],
        )

        result = {
            "success": True,
            "test_id": test_id,
            "simulation_config": {
                "num_sessions": num_sessions,
                "click_model": click_model,
                "impressions_per_query": impressions_per_query,
                "seed": seed,
            },
            "statistics": stats,
            "events_created": {
                "impressions": stats["total_impressions"],
                "clicks": stats["total_clicks"],
            },
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"Error simulating interleaved test: {str(e)}"
        log_error_event(
            logger,
            "[SimulateInterleavedTestTool] ✗ Error.",
            "tools.interleaved.simulate_error",
            error=e,
        )
        return format_tool_error(error_msg)


@monitored_tool(
    name="CalculateInterleavedWinnerTool",
    description="Analyzes interleaved test results using click attribution and determines winning configuration with statistical significance",
)
def calculate_interleaved_winner(
    test_id: str,
    significance_level: float = 0.05,
) -> str:
    """
    Calculates which search configuration won the interleaved test.

    Uses click attribution to count clicks for each configuration and
    applies binomial test for statistical significance.

    Args:
        test_id: The interleaved test ID
        significance_level: P-value threshold for significance (default 0.05)

    Returns:
        JSON with winner, click counts, CTR, and statistical metrics
    """
    try:
        from scipy import stats as scipy_stats

        log_info_event(
            logger,
            "[CalculateInterleavedWinnerTool] Analyzing test.",
            "tools.interleaved.calculate_start",
            test_id=test_id,
        )

        # Get test details
        test_response = os_client.get(
            index="search-relevance-interleaved-test", id=test_id
        )
        if not test_response.get("found"):
            return format_tool_error(f"Test not found: {test_id}")

        test_data = test_response["_source"]
        _config_a_id = test_data["search_configuration_a_id"]
        _config_b_id = test_data["search_configuration_b_id"]

        # Query UBI events for this test
        # We need to find events with metadata.interleaved = true
        # and count clicks by search_config

        query = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"event_attributes.metadata.interleaved": True}},
                        {
                            "term": {
                                "event_attributes.metadata.test_id.keyword": test_id
                            }
                        },
                    ],
                },
            },
            "aggs": {
                "by_action": {
                    "terms": {
                        "field": "action_name",
                    },
                    "aggs": {
                        "by_config": {
                            "terms": {
                                "field": "event_attributes.metadata.search_config.keyword",
                            },
                        },
                    },
                },
            },
        }

        response = os_client.search(
            index="ubi_events",
            body=query,
        )

        # Parse aggregations
        clicks_a = 0
        clicks_b = 0
        impressions_a = 0
        impressions_b = 0

        for action_bucket in response["aggregations"]["by_action"]["buckets"]:
            action_name = action_bucket["key"]

            for config_bucket in action_bucket["by_config"]["buckets"]:
                config = config_bucket["key"]
                count = config_bucket["doc_count"]

                if action_name == "click":
                    if config == "config_a":
                        clicks_a = count
                    elif config == "config_b":
                        clicks_b = count
                elif action_name == "impression":
                    if config == "config_a":
                        impressions_a = count
                    elif config == "config_b":
                        impressions_b = count

        total_clicks = clicks_a + clicks_b
        total_impressions = impressions_a + impressions_b

        if total_clicks == 0:
            return json.dumps(
                {"error": "No clicks found for this test. Cannot determine winner."}
            )

        # Calculate CTR
        ctr_a = clicks_a / impressions_a if impressions_a > 0 else 0
        ctr_b = clicks_b / impressions_b if impressions_b > 0 else 0

        # Binomial test for statistical significance
        # Null hypothesis: both configs are equally likely to receive clicks
        # We use a two-tailed binomial test
        # Note: binomtest (no underscore) is the current scipy function, binom_test was deprecated
        p_value = scipy_stats.binomtest(
            clicks_b, total_clicks, p=0.5, alternative="two-sided"
        ).pvalue

        # Determine winner
        if p_value < significance_level:
            if clicks_b > clicks_a:
                winner = "config_b"
                confidence = "high" if p_value < 0.01 else "medium"
            else:
                winner = "config_a"
                confidence = "high" if p_value < 0.01 else "medium"
        else:
            winner = "no_significant_difference"
            confidence = "low"

        # Calculate relative improvement
        if clicks_a > 0:
            relative_improvement = (clicks_b - clicks_a) / clicks_a
        else:
            relative_improvement = 0

        log_info_event(
            logger,
            "[CalculateInterleavedWinnerTool] Winner and p-value.",
            "tools.interleaved.calculate_winner",
            winner=winner,
            p_value=p_value,
        )

        result = {
            "success": True,
            "test_id": test_id,
            "winner": winner,
            "click_counts": {
                "config_a": clicks_a,
                "config_b": clicks_b,
                "total": total_clicks,
            },
            "impression_counts": {
                "config_a": impressions_a,
                "config_b": impressions_b,
                "total": total_impressions,
            },
            "ctr": {"config_a": round(ctr_a, 4), "config_b": round(ctr_b, 4)},
            "relative_improvement": round(relative_improvement, 3),
            "statistical_test": {
                "method": "binomial_test",
                "p_value": round(float(p_value), 4),
                "significance_level": significance_level,
                "is_significant": bool(p_value < significance_level),
            },
            "confidence": confidence,
            "interpretation": _generate_interpretation(
                winner,
                clicks_a,
                clicks_b,
                ctr_a,
                ctr_b,
                relative_improvement,
                p_value,
                significance_level,
            ),
        }

        return json.dumps(result, indent=2)

    except ImportError:
        return json.dumps(
            {
                "error": "scipy is required for statistical analysis. Install with: uv install scipy"
            }
        )
    except Exception as e:
        error_msg = f"Error calculating winner: {str(e)}"
        log_error_event(
            logger,
            "[CalculateInterleavedWinnerTool] Error.",
            "tools.interleaved.calculate_error",
            error=e,
        )
        return format_tool_error(error_msg)


def _generate_interpretation(
    winner: str,
    clicks_a: int,
    clicks_b: int,
    ctr_a: float,
    ctr_b: float,
    relative_improvement: float,
    p_value: float,
    significance_level: float,
) -> str:
    """Generate human-readable interpretation of results."""
    if winner == "no_significant_difference":
        return (
            f"No statistically significant difference detected between configurations "
            f"(p={p_value:.3f}). Both configurations perform similarly. "
            f"Consider running a longer test or using a larger sample size."
        )

    winning_config = winner
    _losing_config = "config_a" if winner == "config_b" else "config_b"
    _winning_clicks = clicks_b if winner == "config_b" else clicks_a
    _losing_clicks = clicks_a if winner == "config_b" else clicks_b
    winning_ctr = ctr_b if winner == "config_b" else ctr_a
    losing_ctr = ctr_a if winner == "config_b" else ctr_b

    improvement_pct = abs(relative_improvement) * 100

    return (
        f"{winning_config.upper()} shows statistically significant improvement "
        f"({improvement_pct:.1f}% more clicks, CTR: {winning_ctr:.3f} vs {losing_ctr:.3f}, "
        f"p={p_value:.3f}). The difference is unlikely due to chance."
    )
