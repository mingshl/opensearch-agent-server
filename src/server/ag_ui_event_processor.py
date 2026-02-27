"""Event Processor for AG-UI Protocol.

Handles event generation, persistence, and activity monitoring for AG-UI events.

**Key Components:**
- `AGUIEventProcessor` - Main event processor class
- `generate_events()` - High-level event generation function
- `_process_event_stream()` - Internal event stream processing
- `_handle_run_error()` - Error handling for run failures
- `_complete_run()` - Run completion and cleanup

**Responsibilities:**
- Event encoding (to SSE format)
- Event persistence (optional)
- Activity monitoring (optional)
- Message state tracking
- Error handling and fallbacks

**Usage Example:**
```python
from server.ag_ui_event_processor import AGUIEventProcessor, generate_events
from server.route_helpers import create_encoder

encoder = create_encoder("text/event-stream")
processor = AGUIEventProcessor(
    encoder=encoder,
    persistence=persistence,  # Optional
    activity_monitor=activity_monitor,  # Optional
)

# Generate events with processing
async for encoded_event in generate_events(
    strands_agent=agent,
    input_data=input_data,
    event_processor=processor,
    run_id="run-123",
    thread_id="thread-456",
    user_id="user-789",
    start_time=datetime.now(),
):
    # Yield encoded SSE events
    yield encoded_event
```
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime

from ag_ui.core import RunAgentInput

from utils.logging_helpers import (
    get_logger,
    log_debug_event,
    log_error_event,
    log_info_event,
    log_warning_event,
)
from server.ag_ui_event_strategy import (
    AGUIEventContext,
    create_agui_event_handler_chain,
)
from server.config import ServerConfig, get_config
from server.constants import (
    DEFAULT_EVENT_QUEUE_TIMEOUT,
    DEFAULT_EVENT_STREAM_CHECK_TIMEOUT,
)
from server.strands_agent import StrandsAgent
from server.types import (
    ActivityMonitorProtocol,
    AGUIEvent,
    EventEncoderProtocol,
    PersistenceProtocol,
)
from server.utils import (
    create_error_event,
    get_event_type_from_object,
    get_event_type_name,
    safe_persistence_operation,
)

logger = get_logger(__name__)


class AGUIEventProcessor:
    """Processes AG-UI events for persistence, activity monitoring, and encoding."""

    def __init__(
        self,
        encoder: EventEncoderProtocol,
        persistence: PersistenceProtocol | None = None,
        activity_monitor: ActivityMonitorProtocol | None = None,
    ) -> None:
        """Initialize event processor.

        Args:
            encoder: Encoder implementing EventEncoderProtocol (required)
            persistence: Optional AGUIPersistence instance for saving events
            activity_monitor: Optional AGUIActivityMonitor instance for tracking activity

        """
        self.persistence = persistence
        self.activity_monitor = activity_monitor
        self.encoder = encoder
        # Initialize strategy pattern handler chain
        self._handler_chain = create_agui_event_handler_chain(
            persistence=persistence,
            activity_monitor=activity_monitor,
        )

    def process_event(
        self,
        event: AGUIEvent,
        run_id: str,
        thread_id: str,
        current_message_id: str | None,
        current_message_content: list,
    ) -> tuple[str | None, list, str]:
        """Process a single event for persistence and activity monitoring.

        Args:
            event: AG-UI event object
            run_id: Run identifier
            thread_id: Thread identifier
            current_message_id: Current message ID being tracked
            current_message_content: Current message content chunks

        Returns:
            Tuple of (updated message_id, updated message_content, encoded_event_string)

        """
        # Save event to persistence if enabled
        if self.persistence:
            self._save_event_to_persistence(event, run_id, thread_id)

        # Use strategy pattern to handle message tracking and activity monitoring
        context = AGUIEventContext(
            event=event,
            run_id=run_id,
            thread_id=thread_id,
            current_message_id=current_message_id,
            current_message_content=current_message_content,
            persistence=self.persistence,
            activity_monitor=self.activity_monitor,
        )
        updated_message_id, updated_message_content = self._handler_chain.process_event(
            context
        )

        # Encode event
        try:
            encoded_event = self.encoder.encode(event)
        except Exception as e:
            log_error_event(
                logger,
                "Failed to encode event.",
                "ag_ui.encoding_error",
                error=str(e),
                exc_info=True,
                run_id=run_id,
                thread_id=thread_id,
            )
            # Try to encode error event, with fallback if that also fails
            try:
                error_event = create_error_event(
                    message=f"Encoding error: {str(e)}",
                    code="ENCODING_ERROR",
                )
                encoded_event = self.encoder.encode(error_event)
            except Exception as fallback_error:
                # Last resort: return plain text error if encoding error event also fails
                log_error_event(
                    logger,
                    "Failed to encode error event.",
                    "ag_ui.encoding_error_fallback_failed",
                    error=str(fallback_error),
                    exc_info=True,
                    run_id=run_id,
                    thread_id=thread_id,
                    original_error=str(e),
                )
                # Return plain text SSE format error message (use json.dumps to escape)
                encoded_event = f"data: {json.dumps({'error': f'Encoding failed: {str(e)}', 'code': 'ENCODING_ERROR'})}\n\n"

        return updated_message_id, updated_message_content, encoded_event

    def _save_event_to_persistence(
        self, event: AGUIEvent, run_id: str, thread_id: str
    ) -> None:
        """Save event to persistence if enabled.

        Uses safe_persistence_operation() for consistent error handling.

        Args:
            event: AG-UI event object
            run_id: Run identifier
            thread_id: Thread identifier

        """
        # Prepare event data for persistence
        event_id = str(uuid.uuid4())
        event_type = get_event_type_from_object(event)
        event_type_str = (
            get_event_type_name(event_type) if event_type is not None else "UNKNOWN"
        )

        # Convert event to dict for storage
        if hasattr(event, "model_dump"):
            event_data = event.model_dump(exclude_none=True)
        elif hasattr(event, "dict"):
            event_data = event.dict(exclude_none=True)
        else:
            event_data = {"type": event_type_str}

        # Use safe_persistence_operation for consistent error handling
        safe_persistence_operation(
            "save_event",
            self.persistence.save_event,
            event_id=event_id,
            run_id=run_id,
            event_type=event_type_str,
            event_data=event_data,
        )


async def _wait_for_strands_event(
    strands_task: asyncio.Task | None,
    get_next_event: Callable[[], Awaitable[AGUIEvent | None]],
    timeout: float,
) -> tuple[asyncio.Task | None, AGUIEvent | None, bool]:
    """Wait for a Strands event with timeout, managing task lifecycle.

    This helper function manages the Strands event task lifecycle:
    - Creates a new task if needed
    - Waits for task completion with timeout
    - Returns the updated task, event result, and completion status

    State Transition Diagram:
        Task State: None -> Created -> Pending -> Completed/Done/Error
        Return States:
        - (None, None, True):  Stream exhausted, no more events
        - (None, Event, False): Event received, task completed (need new task next time)
        - Raises Exception: Task error occurred, exception propagates to caller for error handling
        - (Task, None, False): Timeout, task still pending (reuse on next call)

    Args:
        strands_task: Current Strands event task (None if not started or completed)
        get_next_event: Async function to get next event from iterator
        timeout: Timeout in seconds for waiting

    Returns:
        Tuple of (updated_task, event_result, is_done):
        - updated_task: Task instance (None if stream is done, new task if needed)
        - event_result: Event if available, None if stream done or timeout
        - is_done: True if stream is exhausted, False otherwise
    """

    # STATE TRANSITION: Task Creation
    # Transition: None or Done -> Created
    # If no task exists or previous task completed, create new task
    # This ensures we always have a pending task to wait on
    if strands_task is None or strands_task.done():
        strands_task = asyncio.create_task(get_next_event())

    # STATE TRANSITION: Task Waiting
    # Transition: Created -> Pending -> (Completed | Timeout | Error)
    # Wait for task completion with timeout. This allows us to periodically
    # check for Python tool events even while waiting for Strands events.
    # Use asyncio.wait() instead of wait_for() to avoid canceling the task on timeout
    done, pending = await asyncio.wait(
        [strands_task],
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # STATE TRANSITION: Task Completion Handling
    # Transition: Pending -> Completed
    if done:
        try:
            result = await strands_task
            # STATE TRANSITION: Stream Exhaustion Check
            # Transition: Completed -> (Stream Done | Event Received)
            if result is None:
                # Stream exhausted - no more events will come
                # Return: (None, None, True) - signals stream is done
                return None, None, True
            else:
                # Event received successfully
                # Return: (None, Event, False) - task completed, event available
                # Note: Return None for task so caller creates new task next iteration
                return None, result, False
        except Exception as e:
            # STATE TRANSITION: Task Error Handling
            # Transition: Completed -> Error
            # Task failed - log error and re-raise so it propagates to generate_events
            # This allows generate_events to handle the error and emit an error event
            log_error_event(
                logger,
                "Failed to process Strands event task.",
                "ag_ui.strands_task_error",
                error=str(e),
                exc_info=True,
            )
            # Re-raise the exception so it propagates to generate_events for proper error handling
            raise
    else:
        # STATE TRANSITION: Timeout Handling
        # Transition: Pending -> Timeout (still pending)
        # Timeout occurred - task still running, return it for reuse
        # Return: (Task, None, False) - task still pending, reuse on next call
        # This allows us to continue waiting on the same task without creating duplicates
        return strands_task, None, False


async def _process_event_stream(
    strands_agent: StrandsAgent,
    input_data: RunAgentInput,
    event_processor: AGUIEventProcessor,
    run_id: str,
    thread_id: str,
    config: ServerConfig | None = None,
) -> AsyncIterator[str]:
    """Process the event stream from StrandsAgent.

    Args:
        strands_agent: StrandsAgent instance
        input_data: RunAgentInput with thread_id, run_id, and messages
        event_processor: AGUIEventProcessor instance
        run_id: Run identifier
        thread_id: Thread identifier
        config: Optional ServerConfig for max_event_queue_size. When None, uses get_config().

    Yields:
        Encoded SSE event strings

    """
    from utils.tool_event_emitter import (
        AGUIToolEventEmitter,
        set_ag_ui_emitter,
    )

    # Initialize tracking variables
    current_message_id = None
    current_message_content = []

    # Create bounded event queue for Python-level tool events (stores encoded events ready to yield)
    cfg = config if config is not None else get_config()
    max_queue_size = cfg.max_event_queue_size
    python_tool_event_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)

    # Create a callback to process and queue encoded events immediately
    # This mimics Chainlit's immediate UI update pattern
    async def process_and_queue_python_event(event: AGUIEvent) -> None:
        """Process Python tool event and queue the encoded result immediately.

        Note on closure behavior: This callback accesses `current_message_id` and
        `current_message_content` from the enclosing scope. Python closures capture
        variables by reference, not by value, so the callback will always see the
        current values of these variables as they are updated in the main event loop.
        This ensures Python tool events are correctly associated with the current
        message being processed, even if the message_id changes during the stream.
        """
        # Process Python-level event through event processor
        # Note: current_message_id and current_message_content are captured by closure
        # and will reflect the latest values from the main event processing loop
        _, _, encoded_python_event = event_processor.process_event(
            event,
            run_id,
            thread_id,
            current_message_id,  # Accessed from closure - sees updated value
            current_message_content,  # Accessed from closure - sees updated value
        )
        # Queue the encoded event (ready to yield) with backpressure handling
        # If queue is full, this will block until space is available
        # If timeout exceeded, log error - event is lost but processing continues
        # Note: This is a callback from tool execution, so we can't break/stop here,
        # but we log the error for monitoring
        try:
            await asyncio.wait_for(
                python_tool_event_queue.put(encoded_python_event),
                timeout=DEFAULT_EVENT_QUEUE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            # Queue full and timeout exceeded - log error (event is lost)
            queue_size = python_tool_event_queue.qsize()
            log_error_event(
                logger,
                f"Python tool event queue full, timeout exceeded: run_id={run_id}, "
                f"queue_size={queue_size}. Event was dropped.",
                "ag_ui.python_tool_event_queue_timeout",
                run_id=run_id,
                thread_id=thread_id,
                queue_size=queue_size,
            )
            # Note: We can't break/stop here since this is a callback from tool execution.
            # The event is lost, but tool execution continues. This is logged for monitoring.

    # Create AG-UI tool event emitter with temporary message_id (will be updated from first event)
    # We create it early so tool wrappers can access it immediately when tools are called
    # Pass the callback so events are processed and queued immediately when tools are called
    # Note: event_queue parameter is not used when callback is provided, but we still create
    # a bounded queue for consistency and in case the emitter falls back to using it
    tool_emitter = AGUIToolEventEmitter(
        event_queue=asyncio.Queue(
            maxsize=max_queue_size
        ),  # Not used when callback is provided, but bounded for safety
        message_id="temp",  # Will be updated from first Strands event
        yield_callback=process_and_queue_python_event,
    )
    set_ag_ui_emitter(tool_emitter)
    log_debug_event(
        logger,
        f"Created AG-UI tool event emitter: run_id={run_id}",
        "ag_ui.tool_emitter_created",
        run_id=run_id,
        thread_id=thread_id,
    )

    # Create Strands event iterator
    strands_event_iterator = strands_agent.run(input_data)

    # Process events from both Strands stream and Python tool event queue concurrently
    strands_done = False

    async def get_next_strands_event() -> AGUIEvent | None:
        """Get next Strands event from iterator, handling stream exhaustion.

        This helper function wraps the async iterator's __anext__() method to
        catch StopAsyncIteration exceptions and return None instead of raising.
        This simplifies the calling code by converting the exception into a
        sentinel value.

        Returns:
            Next AGUIEvent from the Strands iterator, or None if stream is exhausted
        """
        try:
            return await strands_event_iterator.__anext__()
        except StopAsyncIteration:
            return None

    # Use a small timeout to periodically check Python event queue
    # This ensures Python tool events are yielded immediately even while waiting for Strands events
    CHECK_TIMEOUT = DEFAULT_EVENT_STREAM_CHECK_TIMEOUT

    # STATE: Initialize task tracking
    # strands_task transitions: None -> Created -> Pending -> Completed -> None (cycle)
    # strands_done transitions: False -> True (one-way, never goes back)
    strands_task: asyncio.Task | None = None

    # MAIN LOOP: Process events from both Strands stream and Python tool queue
    # Loop continues while: Strands stream active OR Python events pending
    # Exit condition: Both streams done AND queue empty
    while not strands_done or not python_tool_event_queue.empty():
        # STATE TRANSITION: Yield Python Events (Priority)
        # Process Python events first (non-blocking) to ensure real-time display
        # These events are already encoded and ready to yield
        # Use get_nowait() directly - QueueEmpty exception signals queue is empty
        # This avoids race condition between empty() check and get_nowait() call
        while True:
            try:
                encoded_python_event = python_tool_event_queue.get_nowait()
                yield encoded_python_event
            except asyncio.QueueEmpty:
                # Queue is empty - break to continue with Strands event processing
                break

        # STATE TRANSITION: Check Exit Condition
        # Exit if both streams are done and no events remain
        # Transition: Active Loop -> Exit
        if strands_done and python_tool_event_queue.empty():
            break

        # STATE TRANSITION: Wait for Strands Event
        # Only wait if Strands stream is still active
        if not strands_done:
            # Call helper to manage task lifecycle and wait for event
            # Helper returns: (updated_task, event_result, stream_done_flag)
            strands_task, strands_event, stream_done = await _wait_for_strands_event(
                strands_task,
                get_next_strands_event,
                CHECK_TIMEOUT,
            )

            # STATE TRANSITION: Handle Stream Completion
            # Transition: Stream Active -> Stream Exhausted
            if stream_done:
                # Strands stream exhausted - no more events will come
                # Update state: strands_done = True, strands_task = None
                strands_done = True
                strands_task = None
            # STATE TRANSITION: Handle Event Received
            # Transition: Waiting -> Event Available
            elif strands_event is not None:
                # Strands event received - process it
                # Process event (persistence, activity tracking, encoding)
                current_message_id, current_message_content, encoded_event = (
                    event_processor.process_event(
                        strands_event,
                        run_id,
                        thread_id,
                        current_message_id,
                        current_message_content,
                    )
                )

                # STATE TRANSITION: Update Emitter Message ID
                # Transition: temp -> actual_message_id (first event only)
                # Update emitter message_id from first event (or if it changed)
                # This ensures Python tool events are associated with correct message
                if current_message_id and tool_emitter.message_id != current_message_id:
                    old_message_id = tool_emitter.message_id
                    tool_emitter.message_id = current_message_id
                    if old_message_id == "temp":
                        log_debug_event(
                            logger,
                            f"Updated AG-UI tool event emitter message_id: {old_message_id} -> {current_message_id}",
                            "ag_ui.tool_emitter_message_id_updated",
                            old_message_id=old_message_id,
                            new_message_id=current_message_id,
                            run_id=run_id,
                            thread_id=thread_id,
                        )

                yield encoded_event
            # STATE TRANSITION: Handle Timeout
            # Transition: Waiting -> Timeout (no event yet)
            # If strands_event is None and stream_done is False, timeout occurred
            # Task is still pending and will be reused on next iteration
            # Loop continues to check Python queue again
        else:
            # STATE TRANSITION: Strands Stream Done, Process Remaining Python Events
            # Transition: Strands Active -> Strands Done, Python Events Pending
            # Strands stream is done, but Python events might still be coming
            if not python_tool_event_queue.empty():
                # Yield any Python events immediately and continue loop
                # This ensures all Python tool events are processed even after Strands ends
                continue
            else:
                # STATE TRANSITION: Both Streams Done
                # Transition: Active Loop -> Exit
                # Both streams are done and no events remain - exit loop
                break

    # Yield any remaining Python-level tool events before cleanup
    # This ensures all tool calls are displayed even if they complete after the last Strands event
    # These are already encoded and ready to yield
    # Use get_nowait() directly - QueueEmpty exception signals queue is empty
    while True:
        try:
            encoded_python_event = python_tool_event_queue.get_nowait()
            yield encoded_python_event
        except asyncio.QueueEmpty:
            # Queue is empty - all remaining events have been yielded
            break

    # Clean up: clear emitter when stream ends
    if tool_emitter:
        set_ag_ui_emitter(None)
        log_debug_event(
            logger,
            f"Cleared AG-UI tool event emitter: run_id={run_id}",
            "ag_ui.tool_emitter_cleared",
            run_id=run_id,
            thread_id=thread_id,
        )


def _handle_run_error(
    event_processor: AGUIEventProcessor,
    run_id: str,
    thread_id: str,
    user_id: str,
    error: Exception,
) -> str:
    """Handle run error by emitting error event.

    Args:
        event_processor: AGUIEventProcessor instance
        run_id: Run identifier
        thread_id: Thread identifier
        user_id: User identifier
        error: The exception that occurred

    Returns:
        Encoded error event string (or plain text fallback if encoding fails)

    """
    log_error_event(
        logger,
        "Run error.",
        "ag_ui.run_error",
        error=str(error),
        exc_info=True,
        run_id=run_id,
        thread_id=thread_id,
        user_id=user_id,
    )
    # Try to encode error event, with fallback if encoding fails
    try:
        error_event = create_error_event(
            message=str(error),
            code="RUN_ERROR",
        )
        return event_processor.encoder.encode(error_event)
    except Exception as encoding_error:
        # Last resort: return plain text error if encoding fails
        log_error_event(
            logger,
            "Failed to encode run error event.",
            "ag_ui.run_error_encoding_failed",
            error=str(encoding_error),
            exc_info=True,
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            original_error=str(error),
        )
        # Return plain text SSE format error message (use json.dumps to escape)
        return f"data: {json.dumps({'error': str(error), 'code': 'RUN_ERROR'})}\n\n"


def _complete_run(
    event_processor: AGUIEventProcessor,
    run_id: str,
    thread_id: str,
    user_id: str,
    event_count: int,
    start_time: datetime,
) -> None:
    """Complete run by handling cleanup, persistence, and logging.

    Args:
        event_processor: AGUIEventProcessor instance
        run_id: Run identifier
        thread_id: Thread identifier
        user_id: User identifier
        event_count: Number of events processed
        start_time: Start time for duration calculation

    """
    # Complete any remaining active tool calls (edge case handling)
    if event_processor.activity_monitor:
        remaining_tool_calls = (
            event_processor.activity_monitor.get_remaining_tool_calls()
        )
        if remaining_tool_calls:
            log_warning_event(
                logger,
                f"Completing remaining active tool calls at run end: "
                f"run_id={run_id}, thread_id={thread_id}, count={len(remaining_tool_calls)}",
                "ag_ui.completing_remaining_tool_calls",
                run_id=run_id,
                thread_id=thread_id,
                count=len(remaining_tool_calls),
            )
            event_processor.activity_monitor.complete_remaining_tool_calls(
                error="Run completed before tool call finished"
            )

    # Save run finish to persistence if enabled
    if event_processor.persistence:
        safe_persistence_operation(
            "save_run_finish",
            event_processor.persistence.save_run_finish,
            run_id=run_id,
            status="completed",
        )

    # Log run completion with activity summary
    duration = (datetime.now() - start_time).total_seconds()
    log_info_event(
        logger,
        f"Completed AG-UI run: run_id={run_id}, thread_id={thread_id}, "
        f"user_id={user_id}, events={event_count}, duration={duration:.2f}s",
        "ag_ui.run_completed",
        run_id=run_id,
        thread_id=thread_id,
        user_id=user_id,
        event_count=event_count,
        duration_seconds=duration,
    )

    # Log activity summary
    if event_processor.activity_monitor:
        event_processor.activity_monitor.log_summary()


async def generate_events(
    strands_agent: StrandsAgent,
    input_data: RunAgentInput,
    event_processor: AGUIEventProcessor,
    run_id: str,
    thread_id: str,
    user_id: str,
    start_time: datetime,
    config: ServerConfig | None = None,
) -> AsyncIterator[str]:
    """Generate SSE events from StrandsAgent with processing.

    Args:
        strands_agent: StrandsAgent instance
        input_data: RunAgentInput with thread_id, run_id, and messages
        event_processor: AGUIEventProcessor instance
        run_id: Run identifier
        thread_id: Thread identifier
        user_id: User identifier
        start_time: Start time for duration calculation
        config: Optional ServerConfig for event queue size etc. When None, uses get_config().

    Yields:
        Encoded SSE event strings

    """
    from contextlib import nullcontext

    # Initialize counter
    event_count = 0

    # Phoenix tracing removed for opensearch-agent-server; use nullcontext as no-op
    with nullcontext():
        try:
            # Process event stream - `async for` iterates over async generator
            async for encoded_event in _process_event_stream(
                strands_agent,
                input_data,
                event_processor,
                run_id,
                thread_id,
                config=config,
            ):
                event_count += 1
                # Yield each encoded event to caller
                yield encoded_event
        except Exception as e:
            # Catch any exception during processing and emit run error event
            error_event_str = _handle_run_error(
                event_processor, run_id, thread_id, user_id, e
            )
            yield error_event_str
        finally:
            # Ensure cleanup happens regardless of success/failure
            # Always complete run cleanup
            _complete_run(
                event_processor, run_id, thread_id, user_id, event_count, start_time
            )
