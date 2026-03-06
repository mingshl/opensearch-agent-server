"""Route handlers for run-related AG-UI endpoints.

This module contains route handlers for creating runs, getting run details,
canceling runs, and retrieving run events.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import Request
from fastapi.responses import StreamingResponse

from utils.activity_monitor import AGUIActivityMonitor
from utils.logging_helpers import get_logger, log_info_event, log_warning_event
from server.ag_ui_event_processor import AGUIEventProcessor, generate_events
from server.authorization import require_ownership
from server.config import get_config
from server.constants import DEFAULT_EVENT_LIMIT
from server.error_recovery import (
    create_fallback_events_response,
    create_fallback_run_response,
    handle_read_operation_with_fallback,
)
from server.exceptions import ConflictError, NotFoundError
from server.route_helpers import (
    create_encoder,
    ensure_thread_has_title,
    save_initial_messages,
)
from server.run_manager import get_run_manager
from server.run_route_helpers import (
    consume_event_generator_with_cancellation,
    create_event_queue,
    yield_events_from_queue,
)
from server.strands_agent import StrandsAgent
from server.types import (
    CancelRunResponse,
    PersistenceProtocol,
    RunEventsResponse,
    RunResponse,
)
from server.utils import (
    get_user_id_from_request,
    require_authenticated_if_auth_enabled,
    safe_persistence_operation,
)
from server.validators import ValidatedRunAgentInput

logger = get_logger(__name__)


def create_run_route(
    strands_agent: StrandsAgent,
    persistence: PersistenceProtocol | None,
    input_data: ValidatedRunAgentInput,
    request: Request,
) -> StreamingResponse:
    """Start a new agent run and stream AG-UI events via SSE.

    Follows the official AG-UI Strands integration pattern.

    Files should be included in the RunAgentInput messages as BinaryInputContent
    with base64-encoded data in the JSON payload, not as multipart/form-data.

    Args:
        strands_agent: StrandsAgent instance
        persistence: Optional AGUIPersistence instance
        input_data: ValidatedRunAgentInput with thread_id, run_id, and messages
                   (files should be in messages as BinaryInputContent with base64 data)
        request: FastAPI request object (for Accept header)

    Returns:
        SSE stream of AG-UI events

    Raises:
        UnauthorizedError: If authentication is required (auth enabled) and the request is not authenticated.
        ConflictError: If persistence is enabled and a run with this run_id is already in progress (409).

    """
    require_authenticated_if_auth_enabled(request)
    # Get Accept header for EventEncoder
    accept_header = request.headers.get("accept", "text/event-stream")
    encoder = create_encoder(accept_header)

    # Extract validated fields (validation already performed by Pydantic)
    thread_id = input_data.thread_id
    run_id = input_data.run_id

    # Convert to RunAgentInput for compatibility with downstream functions
    run_agent_input = input_data.to_run_agent_input()

    # Extract user ID for OTel tracing
    user_id = get_user_id_from_request(request)

    # Prevent duplicate concurrent runs: if persistence is enabled, reject when
    # a run with this run_id already exists and is still running.
    if persistence:
        existing_run = persistence.get_run(run_id)
        if existing_run and existing_run.get("status") == "running":
            log_warning_event(
                logger,
                f"Rejected duplicate run: run_id={run_id} already has an active run",
                "ag_ui.duplicate_run_rejected",
                run_id=run_id,
                thread_id=thread_id,
            )
            raise ConflictError(
                f"A run with run_id {run_id} is already in progress. "
                "Use a unique run_id or wait for the current run to finish.",
                context={"runId": run_id, "threadId": thread_id},
            )

    # Save thread and run start to persistence if enabled
    if persistence:
        safe_persistence_operation(
            "save_thread", persistence.save_thread, thread_id=thread_id, user_id=user_id
        )
        safe_persistence_operation(
            "save_run_start",
            persistence.save_run_start,
            run_id=run_id,
            thread_id=thread_id,
        )

        # Save initial user messages
        save_initial_messages(persistence, run_agent_input, thread_id, run_id)

        # Generate thread title from first user message if thread doesn't have one
        ensure_thread_has_title(persistence, thread_id, run_agent_input)

    # Log run start
    message_count = len(input_data.messages)
    log_info_event(
        logger,
        f"Starting AG-UI run: run_id={run_id}, thread_id={thread_id}, "
        f"user_id={user_id}, message_count={message_count}",
        "ag_ui.run_starting",
        run_id=run_id,
        thread_id=thread_id,
        user_id=user_id,
        message_count=message_count,
    )
    start_time = datetime.now()

    # Config from app.state when set by create_app (allows tests to inject); else get_config()
    config = getattr(request.app.state, "config", None) or get_config()

    # Initialize activity monitor for this run
    activity_monitor = AGUIActivityMonitor(run_id=run_id, thread_id=thread_id)

    # Create event processor
    event_processor = AGUIEventProcessor(
        encoder=encoder,
        persistence=persistence,
        activity_monitor=activity_monitor,
    )

    # Create a wrapper async generator that checks for cancellation
    async def cancellable_event_stream() -> AsyncIterator[str]:
        """Event stream wrapper that checks for cancellation."""
        run_manager = get_run_manager()
        # Create bounded event queue to prevent memory exhaustion
        event_queue = create_event_queue()
        generator_done = False
        generator_error: Exception | None = None

        # Create event generator
        event_generator = generate_events(
            strands_agent=strands_agent,
            input_data=run_agent_input,
            event_processor=event_processor,
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            start_time=start_time,
            config=config,
        )

        # Create a coroutine that consumes the event generator and puts events in queue
        async def consume_event_generator() -> None:
            """Consume the event generator and put events in queue."""
            nonlocal generator_done, generator_error
            generator_error = await consume_event_generator_with_cancellation(
                event_generator,
                run_id,
                thread_id,
                encoder,
                event_queue,
            )
            generator_done = True

        # Create task and register it for cancellation tracking
        task = asyncio.create_task(consume_event_generator())
        await run_manager.register_run(run_id, task)

        # Yield events from the queue
        # Use task.done() instead of generator_done flag to avoid closure issue
        # The generator_done flag is set inside consume_event_generator() but yield_events_from_queue()
        # receives it by value, so it never sees the updated value. Checking task.done() directly
        # ensures we see the actual completion status.
        try:
            async for event in yield_events_from_queue(
                event_queue, task, generator_error, run_id, thread_id
            ):
                yield event
        finally:
            # Ensure cleanup - cancel task if still running and unregister run
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await run_manager.unregister_run(run_id)

    return StreamingResponse(
        cancellable_event_stream(),
        media_type=encoder.get_content_type(),
    )


@require_ownership("run", "run_id")
def get_run_route(
    persistence: PersistenceProtocol | None,
    run_id: str,
    request: Request | None = None,
    _cached_run: dict | None = None,
) -> RunResponse:
    """Get run details including status and metadata.

    Uses fallback response if persistence fails, allowing the API to
    continue functioning with degraded capabilities.

    **Access Control:**
    Access control is enforced when:
    - `request` is provided AND
    - User is authenticated (`is_authenticated(request)` returns `True`) AND
    - Persistence is enabled (needed to fetch thread for ownership check)

    When persistence is disabled or request is None, access control is skipped
    for backward compatibility.

    **Order of Operations:**
    The @require_ownership decorator checks existence before ownership:
    - Non-existent runs → 404 NotFoundError (checked first)
    - Existent but unauthorized runs → 403 ForbiddenError (checked second)

    Args:
        persistence: Optional AGUIPersistence instance
        run_id: Run identifier
        request: Optional FastAPI request object for access control
        _cached_run: Optional cached run from ownership decorator (avoids duplicate query)

    Returns:
        Run information including status, timestamps, and metadata.
        Returns fallback response if persistence fails.

    Raises:
        NotFoundError: If the run is not found (valid response, not a failure)
        ForbiddenError: If user doesn't have access to this run's thread (only when authenticated and persistence enabled)
        UnauthorizedError: If authentication is required (auth enabled) and the request is not authenticated.

    """
    require_authenticated_if_auth_enabled(request)
    if not persistence:
        # Return fallback response when persistence is not enabled
        return create_fallback_run_response(run_id)

    # Use cached run from decorator if available to avoid duplicate query
    # The decorator has already verified existence and ownership
    if _cached_run is not None:
        return _cached_run

    def fallback() -> RunResponse:
        return create_fallback_run_response(run_id)

    # Fallback path: fetch run if not cached (e.g., when auth is disabled)
    # Note: When auth is enabled, the decorator ensures the run exists and is owned
    run = handle_read_operation_with_fallback(
        operation_name="run",
        operation_func=persistence.get_run,
        fallback_func=fallback,
        error_event_name="ag_ui.run_retrieval_error",
        error_context={"run_id": run_id, "runId": run_id},
        run_id=run_id,
    )

    if run:
        return run
    # Not found is a valid response - raise exception
    raise NotFoundError("Run", run_id, context={"runId": run_id, "status": "unknown"})


@require_ownership("run", "run_id")
def get_run_events_route(
    persistence: PersistenceProtocol | None,
    run_id: str,
    event_type: str | None = None,
    limit: int = DEFAULT_EVENT_LIMIT,
    offset: int = 0,
    request: Request | None = None,
    _cached_run: dict | None = None,
) -> RunEventsResponse:
    """Get events for a run, optionally filtered by event_type.

    Uses fallback response if persistence fails, allowing the API to
    continue functioning with degraded capabilities.

    **Access Control:**
    When `request` is provided, the user is authenticated, and persistence
    is enabled, verifies the user owns the run's thread. Otherwise access
    control is skipped for backward compatibility.

    **Order of Operations:**
    The @require_ownership decorator checks existence before ownership:
    - Non-existent runs → 404 NotFoundError (checked first)
    - Existent but unauthorized runs → 403 ForbiddenError (checked second)

    **Not found:** Returns 404 when the run_id does not exist. When the run
    exists but has no events yet, returns 200 with ``events: []`` and
    ``count: 0`` (well-defined empty structure).

    Args:
        persistence: Optional AGUIPersistence instance
        run_id: Run identifier
        event_type: Optional event type to filter by (e.g., TEXT_MESSAGE_START)
        limit: Maximum number of events to return (default: DEFAULT_EVENT_LIMIT)
        offset: Offset for pagination (default: 0)
        request: Optional FastAPI request object for access control
        _cached_run: Optional cached run from ownership decorator (avoids duplicate query)

    Returns:
        RunEventsResponse with runId, eventType, events list, and count.
        Empty events when the run exists but has no events, or when
        persistence fails (degraded).

    Raises:
        NotFoundError: If the run does not exist (404).
        ForbiddenError: If the user does not own the run's thread (when
            authenticated and persistence enabled).
        UnauthorizedError: If authentication is required (auth enabled) and the request is not authenticated.

    """
    require_authenticated_if_auth_enabled(request)
    if not persistence:
        # Return fallback response when persistence is not enabled
        return create_fallback_events_response(run_id, event_type)

    # Verify run exists before fetching events: unknown run_id → 404
    # Use cached run from decorator if available to avoid duplicate query
    # The decorator has already verified existence and ownership when auth is enabled
    run = _cached_run
    if run is None:
        # Fallback path: check existence if not cached (e.g., when auth is disabled)
        try:
            run = persistence.get_run(run_id)
        except Exception as e:
            log_warning_event(
                logger,
                f"Could not check run existence for events, using fallback: {e}",
                "ag_ui.run_events_run_check_error",
                run_id=run_id,
                runId=run_id,
            )
            return create_fallback_events_response(run_id, event_type)
    if run is None:
        raise NotFoundError("Run", run_id, context={"runId": run_id})

    # Access control: verify user owns the run's thread when request and auth present
    # Note: Ownership check is performed by @require_ownership decorator before this function runs

    def fallback() -> RunEventsResponse:
        return create_fallback_events_response(run_id, event_type)

    result = handle_read_operation_with_fallback(
        operation_name="events for run",
        operation_func=persistence.get_events,
        fallback_func=fallback,
        error_event_name="ag_ui.run_events_retrieval_error",
        error_context={
            "run_id": run_id,
            "runId": run_id,
            "event_type": event_type,
            "limit": limit,
            "offset": offset,
            "events": [],
        },
        run_id=run_id,
        event_type=event_type,
        limit=limit,
        offset=offset,
    )
    # If fallback was used, result is already a dict with events and count
    # If operation succeeded, result is a list of events
    if isinstance(result, dict):
        return result
    return {
        "runId": run_id,
        "eventType": event_type,
        "events": result,
        "count": len(result),
    }


@require_ownership("run", "run_id")
async def cancel_run_route(
    persistence: PersistenceProtocol | None,
    run_id: str,
    request: Request | None = None,
) -> CancelRunResponse:
    """Cancel a running agent.

    **Access Control:**
    When `request` is provided, the user is authenticated, and persistence
    is enabled, verifies the user owns the run's thread before canceling.
    Otherwise access control is skipped for backward compatibility.

    Args:
        persistence: Optional AGUIPersistence instance (for access control)
        run_id: Run identifier
        request: Optional FastAPI request object for access control

    Returns:
        Cancellation confirmation

    Raises:
        ForbiddenError: If the user does not own the run's thread (when
            authenticated and persistence enabled).
        UnauthorizedError: If authentication is required (auth enabled) and the request is not authenticated.

    """
    require_authenticated_if_auth_enabled(request)

    run_manager = get_run_manager()

    # Check if run is active
    is_active = await run_manager.is_run_active(run_id)

    if not is_active:
        # Check if it was already canceled
        was_canceled = await run_manager.is_run_canceled(run_id)
        if was_canceled:
            log_info_event(
                logger,
                f"Run was already canceled: run_id={run_id}",
                "ag_ui.run_already_canceled",
                run_id=run_id,
            )
            return {
                "runId": run_id,
                "canceled": True,
                "message": "Run was already canceled",
            }
        else:
            log_warning_event(
                logger,
                f"Run not found or already completed: run_id={run_id}",
                "ag_ui.run_not_found_for_cancel",
                run_id=run_id,
            )
            return {
                "runId": run_id,
                "canceled": False,
                "message": "Run not found or already completed",
            }

    # Cancel the run
    canceled = await run_manager.cancel_run(
        run_id, reason="User requested cancellation"
    )

    if canceled:
        log_info_event(
            logger,
            f"Successfully canceled run: run_id={run_id}",
            "ag_ui.run_cancellation_success",
            run_id=run_id,
        )
        return {
            "runId": run_id,
            "canceled": True,
            "message": "Run cancellation requested successfully",
        }
    else:
        # Run may have completed between check and cancel
        log_warning_event(
            logger,
            f"Run cancellation failed (may have completed): run_id={run_id}",
            "ag_ui.run_cancellation_failed",
            run_id=run_id,
        )
        return {
            "runId": run_id,
            "canceled": False,
            "message": "Run may have already completed",
        }
