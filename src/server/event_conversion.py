"""Convert Strands events to AG-UI protocol events and process handler results.

Handles conversion of Strands events to AG-UI protocol events and processing
of handler results. This module contains the core event transformation logic
that converts between different event formats.

**Key Components:**
- `format_error_message()` - Format error messages with markdown
- `handle_initialization_error()` - Convert initialization errors to AG-UI events
- `extract_user_message()` - Extract latest user message from input data
- `process_handler_result()` - Process handler results and yield events

**Features:**
- Error event generation following AG-UI protocol
- Handler result processing with state updates
- User message extraction from multimodal content
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    TextMessageEndEvent,
    ToolCallEndEvent,
)

from utils.logging_helpers import (
    get_logger,
    log_debug_event,
    log_error_event,
    log_info_event,
    log_warning_event,
)
from server.constants import DEFAULT_USER_MESSAGE, ROLE_USER
from server.strands_message_processing import (
    extract_user_message_from_multimodal_content,
)
from server.types import (
    ActiveToolCallInfo,
    CompletionSignal,
    EventResult,
    HandlerResult,
    ToolCallInfo,
    ToolCallStateUpdate,
    ToolResultStateUpdate,
)
from server.utils import create_error_event, is_event_type

logger = get_logger(__name__)


def format_error_message(
    error_title: str, error_message: str, additional_info: str = ""
) -> str:
    """Format error message with markdown for better readability in AG-UI.

    Args:
        error_title: Title of the error (e.g., "Initialization Error")
        error_message: The actual error message
        additional_info: Optional additional information to append

    Returns:
        Formatted error message with markdown

    """
    formatted = f"❌ **{error_title}**\n\n```\n{error_message}\n```"
    if additional_info:
        formatted += f"\n\n{additional_info}"
    return formatted


async def handle_initialization_error(
    thread_id: str, run_id: str, error: Exception
) -> AsyncIterator[RunErrorEvent | RunFinishedEvent]:
    """Handle agent initialization errors by emitting error events.

    This function ensures that initialization failures are properly communicated
    via error events in the event stream, following the AG-UI protocol pattern.
    It emits both a RunErrorEvent (to indicate the error) and a RunFinishedEvent
    (to signal that the run has completed, albeit with an error).

    Args:
        thread_id: Thread identifier
        run_id: Run identifier
        error: The initialization error

    Yields:
        RunErrorEvent: Error event describing the initialization failure
        RunFinishedEvent: Completion event signaling the run has ended

    Note:
        This function does not raise exceptions. Errors are communicated via
        events in the stream, allowing the caller to handle them gracefully.

    Example Usage:
        async for event in handle_initialization_error(thread_id, run_id, error):
            # Process each yielded event
            process(event)

    """
    # Convert exception to string for logging
    error_message = str(error)
    log_error_event(
        logger,
        "Agent initialization failed.",
        "ag_ui.agent_initialization_failed",
        error=error_message,
        exc_info=True,  # Include full exception traceback in logs
        thread_id=thread_id,
        run_id=run_id,
    )
    # `yield` makes this an async generator - each yield produces one value
    # The caller uses `async for event in ...` to receive these values one at a time
    yield create_error_event(
        message=format_error_message(
            "Initialization Error",
            error_message,
            "Please check your OpenSearch connection and API keys.",
        ),
        code="INITIALIZATION_ERROR",
    )
    yield RunFinishedEvent(
        type=EventType.RUN_FINISHED,
        thread_id=thread_id,
        run_id=run_id,
    )


def extract_user_message(input_data: RunAgentInput) -> str:
    """Extract the latest user message from input_data.

    Args:
        input_data: RunAgentInput with messages

    Returns:
        Extracted user message text

    """
    user_message = DEFAULT_USER_MESSAGE
    if input_data.messages:
        for msg in reversed(input_data.messages):
            role = (
                msg.role
                if hasattr(msg, "role")
                else msg.get("role")
                if isinstance(msg, dict)
                else ROLE_USER
            )
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if role == ROLE_USER and content:
                user_message = extract_user_message_from_multimodal_content(content)
                break
    return user_message


async def process_handler_result(
    handler_result: HandlerResult,
    event: dict[str, Any],
    message_id: str,
    message_started: bool,
    tool_calls_seen: dict[str, ToolCallInfo],
    active_tool_calls: dict[str, ActiveToolCallInfo],
) -> AsyncIterator[
    tuple[Any, bool, dict[str, ToolCallInfo], dict[str, ActiveToolCallInfo], bool]
]:
    """Process a handler result and yield events, updating state accordingly.

    Handles different types of HandlerResult objects, yields events, and returns
    updated state. This function processes handler results from the event handler
    chain and converts them into AG-UI events with appropriate state updates.

    Args:
        handler_result: HandlerResult object from event handler chain
        event: Original Strands event dictionary
        message_id: Current message ID
        message_started: Whether message has been started
        tool_calls_seen: Dictionary of seen tool calls
        active_tool_calls: Dictionary of active tool calls

    Yields:
        Tuples of (event_to_yield, updated_message_started, updated_tool_calls_seen,
                  updated_active_tool_calls, should_stop)
        - event_to_yield: Event to yield (may be None)
        - updated_message_started: Updated message_started flag
        - updated_tool_calls_seen: Updated tool_calls_seen dict
        - updated_active_tool_calls: Updated active_tool_calls dict
        - should_stop: Whether processing should stop

    Example:
        ```python
        async for (
            event_to_yield,
            updated_msg_started,
            updated_tool_calls,
            updated_active_calls,
            should_stop,
        ) in process_handler_result(
            handler_result=EventResult(event=some_event, should_stop=False),
            event=original_event,
            message_id="msg-123",
            message_started=True,
            tool_calls_seen={},
            active_tool_calls={},
        ):
            if event_to_yield:
                yield event_to_yield
            if should_stop:
                break
            # Update state from returned values
            message_started = updated_msg_started
            tool_calls_seen = updated_tool_calls
            active_tool_calls = updated_active_calls
        ```

    """
    updated_message_started = message_started
    updated_tool_calls_seen = tool_calls_seen
    updated_active_tool_calls = active_tool_calls
    should_stop = False

    if isinstance(handler_result, CompletionSignal):
        # Handler signaled completion (e.g., ResultEventHandler)
        # Complete any active tool calls
        if "result" in event:
            # `list(dict.keys())` creates a list copy of keys
            # We do this because we're modifying the dict during iteration
            # Iterating over `active_tool_calls.keys()` directly would raise RuntimeError
            for tool_use_id in list(updated_active_tool_calls.keys()):
                yield (
                    ToolCallEndEvent(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=tool_use_id,
                    ),
                    updated_message_started,
                    updated_tool_calls_seen,
                    updated_active_tool_calls,
                    should_stop,
                )
            # `dict.clear()` removes all items from dictionary
            updated_active_tool_calls.clear()
            yield (
                None,
                updated_message_started,
                updated_tool_calls_seen,
                updated_active_tool_calls,
                should_stop,
            )
    elif isinstance(handler_result, ToolResultStateUpdate):
        # Tool result handler returned state update with optional stop flag
        updated_active_tool_calls = handler_result.active_tool_calls
        event_to_yield = None
        if handler_result.should_stop:
            # Properly close text message if started before breaking
            if updated_message_started:
                event_to_yield = TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )
                updated_message_started = False
            log_debug_event(
                logger,
                "Stopping text streaming after tool result",
                "ag_ui.stopping_text_streaming_after_tool_result",
            )
            should_stop = True
        yield (
            event_to_yield,
            updated_message_started,
            updated_tool_calls_seen,
            updated_active_tool_calls,
            should_stop,
        )
    elif isinstance(handler_result, ToolCallStateUpdate):
        # Tool call handler returned state update
        updated_tool_calls_seen = handler_result.tool_calls_seen
        updated_active_tool_calls = handler_result.active_tool_calls
        event_to_yield = None
        if handler_result.should_stop:
            # Properly close text message if started before breaking
            if updated_message_started:
                event_to_yield = TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )
                updated_message_started = False
            should_stop = True
        yield (
            event_to_yield,
            updated_message_started,
            updated_tool_calls_seen,
            updated_active_tool_calls,
            should_stop,
        )
    elif isinstance(handler_result, EventResult):
        # Regular AG-UI event
        event_to_yield = handler_result.event
        # Update message_started flag if this is a TEXT_MESSAGE_START event
        # This ensures the flag is set even if context sync fails.
        # When should_stop and message not yet started, we yield the event but
        # do not set updated_message_started so the test/caller sees consistent state.
        if event_to_yield:
            event_type_check = is_event_type(
                event_to_yield, EventType.TEXT_MESSAGE_START
            )
            if event_type_check and (not handler_result.should_stop or message_started):
                updated_message_started = True
                # Log for debugging
                log_info_event(
                    logger,
                    "Detected TEXT_MESSAGE_START in process_handler_result - setting updated_message_started=True",
                    "ag_ui.text_message_start_detected_in_handler_result",
                    message_id=message_id,
                    event_type=str(getattr(event_to_yield, "type", None)),
                    previous_message_started=message_started,
                )
        if handler_result.should_stop:
            # Only emit TextMessageEndEvent to close the message if it was already
            # started before this handler (message_started param). If we just set
            # updated_message_started from this TEXT_MESSAGE_START, yield the
            # original event and stop without emitting a spurious TEXT_MESSAGE_END.
            if message_started:
                event_to_yield = TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )
                updated_message_started = False
            should_stop = True
        yield (
            event_to_yield,
            updated_message_started,
            updated_tool_calls_seen,
            updated_active_tool_calls,
            should_stop,
        )
    else:
        # Fallback for unexpected types (shouldn't happen with proper typing)
        log_warning_event(
            logger,
            f"Unexpected handler result type: {type(handler_result)}",
            "ag_ui.unexpected_handler_result_type",
            result_type=str(type(handler_result)),
        )
        yield (
            None,
            updated_message_started,
            updated_tool_calls_seen,
            updated_active_tool_calls,
            should_stop,
        )
