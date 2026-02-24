"""Strategy Pattern Event Handlers for StrandsAgent.

Implements the strategy pattern for handling different event types from the Strands agent,
replacing complex nested conditionals with a clean handler-based approach.

**Architecture Note:**
This module is part of Stage 1 of the event processing pipeline:
- Stage 1 (this module): Converts raw Strands SDK events → AG-UI events
- Stage 2 (ag_ui_event_strategy.py): Processes AG-UI events for persistence/activity monitoring

This module operates on raw Strands SDK event dictionaries (with keys like "data",
"current_tool_use", "message") and converts them into standardized AG-UI protocol
events (Pydantic models). It uses async generators to yield AG-UI events, and follows
a first-match handler pattern where only the first matching handler processes each event.

**Key Components:**
- `StrandsEventHandler` - Base class for event handlers
- `TextStreamingHandler` - Handles text streaming events
- `ToolCallHandler` - Handles tool call events
- `ToolResultHandler` - Handles tool result events
- `MultiAgentEventHandler` - Handles multi-agent events
- `ResultEventHandler` - Handles completion events
- `StrandsEventHandlerChain` - Chains handlers together

**Usage Example:**
```python
from server.strands_event_strategy import (
    StrandsEventHandler,
    StrandsEventContext,
    create_strands_event_handler_chain,
)

class MyCustomHandler(StrandsEventHandler):
    def can_handle(self, event: StrandsEvent) -> bool:
        return "my_field" in event

    async def handle(self, context: StrandsEventContext):
        # Process event
        yield EventResult(event=some_event, should_stop=False)
```
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ag_ui.core import EventType, RunAgentInput, ToolCallStartEvent

from utils.logging_helpers import get_logger, log_debug_event, log_warning_event
from server.constants import ROLE_ASSISTANT, ROLE_USER
from server.strands_event_handlers import StrandsEventHandlers
from server.types import (
    ActiveToolCallInfo,
    CompletionSignal,
    EventResult,
    HandlerResult,
    StrandsEvent,
    ToolCallInfoInternal,
    ToolCallStateUpdate,
    ToolResultStateUpdate,
)
from server.utils import get_tool_call_id, parse_json_with_fallback

logger = get_logger(__name__)


class StrandsEventContext:
    """Context object passed to event handlers containing all necessary state.

    This encapsulates the state needed by handlers, making the handler interface
    cleaner and more testable.
    """

    def __init__(
        self,
        event: StrandsEvent,
        message_id: str,
        message_started: bool,
        tool_calls_seen: dict[str, ToolCallInfoInternal],
        active_tool_calls: dict[str, ActiveToolCallInfo],
        input_data: RunAgentInput,
        event_handlers: StrandsEventHandlers,
        text_buffer: list[str] | None = None,
        frontend_tool_names: set[str] | None = None,
        has_pending_tool_result: bool = False,
    ) -> None:
        """Initialize event context.

        Args:
            event: Strands event dictionary
            message_id: Current message ID
            message_started: Whether message has been started
            tool_calls_seen: Dictionary of seen tool calls
            active_tool_calls: Dictionary of active tool calls
            input_data: RunAgentInput instance
            event_handlers: StrandsEventHandlers instance
            text_buffer: Optional list to accumulate text chunks (for non-streaming mode)
            frontend_tool_names: Set of frontend tool names (Phase 1: Frontend Tool Detection)
            has_pending_tool_result: Whether there is a pending tool result (Phase 2: Pending Tool Result Detection)

        """
        self.event = event
        self.message_id = message_id
        self.message_started = message_started
        self.tool_calls_seen = tool_calls_seen
        self.active_tool_calls = active_tool_calls
        self.input_data = input_data
        self.event_handlers = event_handlers
        self.text_buffer = text_buffer if text_buffer is not None else []
        self.frontend_tool_names = (
            frontend_tool_names if frontend_tool_names is not None else set()
        )
        self.has_pending_tool_result = has_pending_tool_result


class StrandsEventHandler(ABC):
    """Base class for Strands event handlers using the strategy pattern.

    Each handler implements can_handle() to determine if it can process an event,
    and handle() to process the event and yield AG-UI events.
    """

    @abstractmethod
    def can_handle(self, event: StrandsEvent) -> bool:
        """Check if this handler can process the given event.

        Args:
            event: Strands event dictionary

        Returns:
            True if this handler can process the event, False otherwise

        """
        raise NotImplementedError

    @abstractmethod
    async def handle(
        self, context: StrandsEventContext
    ) -> AsyncIterator[HandlerResult]:
        """Handle the event and yield handler results.

        Args:
            context: StrandsEventContext containing event and state

        Yields:
            HandlerResult objects that clearly indicate the type of result:
            - EventResult: AG-UI event to yield
            - ToolCallStateUpdate: State update for tool calls
            - ToolResultStateUpdate: State update for tool results with optional stop flag
            - CompletionSignal: Signal that processing is complete

        """
        raise NotImplementedError


class TextStreamingHandler(StrandsEventHandler):
    """Handler for text streaming events (data field)."""

    # Streaming enabled - text chunks are emitted as they arrive
    DISABLE_STREAMING = False

    def can_handle(self, event: StrandsEvent) -> bool:
        """Check if event contains text streaming data."""
        return "data" in event and bool(event.get("data"))

    async def handle(
        self, context: StrandsEventContext
    ) -> AsyncIterator[HandlerResult]:
        """Handle text streaming and yield text message events."""
        message_started, start_event, content_event = (
            context.event_handlers.handle_text_streaming(
                context.event, context.message_id, context.message_started
            )
        )
        if start_event:
            yield EventResult(event=start_event, should_stop=False)

        # TEMPORARY: If streaming disabled, accumulate chunks instead of emitting
        if self.DISABLE_STREAMING:
            if content_event and content_event.delta:
                # Accumulate text chunk in buffer
                context.text_buffer.append(content_event.delta)
                log_warning_event(
                    logger,
                    "[STREAMING_DISABLED] Accumulated text chunk.",
                    "ag_ui.streaming_disabled_accumulate",
                    delta_length=len(content_event.delta),
                    buffer_length=len("".join(context.text_buffer)),
                )
            # Don't emit content events during streaming
        else:
            # Normal streaming behavior
            if content_event:
                yield EventResult(event=content_event, should_stop=False)

        # Update context state
        context.message_started = message_started


class AssistantToolCallsHandler(StrandsEventHandler):
    """Handler for assistant messages with tool_calls field.

    Extracts tool calls from assistant messages and emits TOOL_CALL_START events
    for internal tool calls that don't generate current_tool_use events.
    """

    def can_handle(self, event: StrandsEvent) -> bool:
        """Check if event contains an assistant message with tool_calls."""
        if "message" not in event or not isinstance(event.get("message"), dict):
            return False
        message = event.get("message", {})
        if message.get("role") != ROLE_ASSISTANT:
            return False
        tool_calls = message.get("tool_calls", [])
        return bool(tool_calls and isinstance(tool_calls, list))

    async def handle(
        self, context: StrandsEventContext
    ) -> AsyncIterator[HandlerResult]:
        """Handle assistant messages with tool_calls and emit TOOL_CALL_START events."""
        message = context.event.get("message", {})
        tool_calls = message.get("tool_calls", [])

        log_debug_event(
            logger,
            f"Assistant message with tool_calls: count={len(tool_calls)}",
            "ag_ui.assistant_tool_calls_detected",
            tool_calls_count=len(tool_calls),
        )

        # Extract tool calls from assistant message
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            # Extract tool call information
            tool_call_id = get_tool_call_id(tool_call, generate_default=False)
            if not tool_call_id:
                log_debug_event(
                    logger,
                    f"Skipping tool call without ID: {tool_call}",
                    "ag_ui.tool_call_no_id",
                )
                continue

            # Get tool name from function.name or name field
            function = tool_call.get("function", {})
            tool_name = function.get("name") if isinstance(function, dict) else None
            if not tool_name:
                tool_name = tool_call.get("name")
            if not tool_name:
                log_debug_event(
                    logger,
                    f"Skipping tool call without name: tool_call_id={tool_call_id}",
                    "ag_ui.tool_call_no_name",
                )
                continue

            # Check if we've already seen this tool call
            if tool_call_id in context.tool_calls_seen:
                log_debug_event(
                    logger,
                    f"Skipping already seen tool call: tool_name={tool_name}, tool_id={tool_call_id}",
                    "ag_ui.tool_call_already_seen",
                    tool_name=tool_name,
                    tool_id=tool_call_id,
                )
                continue

            # Extract tool arguments
            tool_args_str = (
                function.get("arguments", "{}") if isinstance(function, dict) else "{}"
            )
            if isinstance(tool_args_str, str):
                # Use centralized JSON parsing utility with fallback
                tool_input = parse_json_with_fallback(tool_args_str, fallback_value={})
            else:
                tool_input = tool_args_str

            # Register tool call
            context.tool_calls_seen[tool_call_id] = {
                "name": tool_name,
                "args": tool_args_str
                if isinstance(tool_args_str, str)
                else json.dumps(tool_args_str),
                "input": tool_input,
            }
            context.active_tool_calls[tool_call_id] = {
                "name": tool_name,
                "input": tool_input,
            }

            log_debug_event(
                logger,
                f"Emitting TOOL_CALL_START from assistant message: tool_name={tool_name}, tool_id={tool_call_id}",
                "ag_ui.tool_call_start_from_assistant",
                tool_name=tool_name,
                tool_id=tool_call_id,
            )

            # Emit TOOL_CALL_START event
            yield EventResult(
                event=ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=tool_call_id,
                    tool_call_name=tool_name,
                    parent_message_id=context.message_id,
                ),
                should_stop=False,
            )

            # Emit state update (context is modified in-place, but we also yield update for consistency)
            yield ToolCallStateUpdate(
                tool_calls_seen=context.tool_calls_seen.copy(),
                active_tool_calls=context.active_tool_calls.copy(),
            )


class ToolResultHandler(StrandsEventHandler):
    """Handler for tool result events from message events."""

    def can_handle(self, event: StrandsEvent) -> bool:
        """Check if event contains a message with tool results."""
        return (
            "message" in event
            and isinstance(event.get("message"), dict)
            and event.get("message", {}).get("role") == ROLE_USER
        )

    async def handle(
        self, context: StrandsEventContext
    ) -> AsyncIterator[HandlerResult]:
        """Handle tool results and yield related events."""
        async for result in context.event_handlers.handle_tool_results(
            context.event,
            context.input_data,
            context.tool_calls_seen,
            context.active_tool_calls,
            context.message_id,
        ):
            if isinstance(result, ToolResultStateUpdate):
                # Update context state
                context.active_tool_calls = result.active_tool_calls
            yield result


class ToolCallHandler(StrandsEventHandler):
    """Handler for tool call events (current_tool_use field)."""

    def can_handle(self, event: StrandsEvent) -> bool:
        """Check if event contains a tool call."""
        return "current_tool_use" in event and bool(event.get("current_tool_use"))

    async def handle(
        self, context: StrandsEventContext
    ) -> AsyncIterator[HandlerResult]:
        """Handle tool calls and yield tool call events."""
        async for result in context.event_handlers.handle_tool_calls(
            context.event,
            context.input_data,
            context.tool_calls_seen,
            context.active_tool_calls,
            context.message_id,
            context.frontend_tool_names,
            context.has_pending_tool_result,
        ):
            if isinstance(result, ToolCallStateUpdate):
                # Update context state
                context.tool_calls_seen = result.tool_calls_seen
                context.active_tool_calls = result.active_tool_calls
            yield result


class MultiAgentEventHandler(StrandsEventHandler):
    """Handler for multi-agent events (node start, stop, handoff)."""

    def can_handle(self, event: StrandsEvent) -> bool:
        """Check if event contains multi-agent information."""
        return (
            "multiagent_node_start" in event
            or "multiagent_node_stop" in event
            or "multiagent_handoff" in event
        )

    async def handle(
        self, context: StrandsEventContext
    ) -> AsyncIterator[HandlerResult]:
        """Handle multi-agent events and yield related events."""
        async for result in context.event_handlers.handle_multi_agent_events(
            context.event, context.message_id
        ):
            yield result


class ResultEventHandler(StrandsEventHandler):
    """Handler for final result events."""

    def can_handle(self, event: StrandsEvent) -> bool:
        """Check if event contains a final result."""
        return "result" in event

    async def handle(
        self, context: StrandsEventContext
    ) -> AsyncIterator[HandlerResult]:
        """Handle final result event.

        Note: This handler doesn't yield events directly, but signals that
        active tool calls should be completed. The caller handles this.
        """
        # This handler signals completion - actual handling is done by caller
        yield CompletionSignal()


class StrandsEventHandlerChain:
    """Chain of event handlers that processes events using the strategy pattern.

    Handlers are checked in order, and the first matching handler processes the event.
    """

    def __init__(self, handlers: list[StrandsEventHandler]) -> None:
        """Initialize handler chain.

        Args:
            handlers: List of event handlers in priority order

        """
        self.handlers = handlers

    async def process_event(
        self, context: StrandsEventContext
    ) -> AsyncIterator[HandlerResult]:
        """Process an event using the handler chain.

        Args:
            context: StrandsEventContext containing event and state

        Yields:
            HandlerResult objects with clear semantics. When no handler
            matches the event, the generator yields no items.

        """
        # Iterate through handlers in priority order
        for handler in self.handlers:
            # Check if this handler can process the event
            if handler.can_handle(context.event):
                # Process event through handler
                async for result in handler.handle(context):
                    yield result
                # Only one handler processes each event (first-match pattern)
                return

        # No handler matched - log debug info
        log_debug_event(
            logger,
            f"No handler matched event: event_keys={list(context.event.keys())}",
            "ag_ui.no_handler_matched",
            event_keys=list(context.event.keys()),
        )


def create_strands_event_handler_chain(
    event_handlers: StrandsEventHandlers,
) -> StrandsEventHandlerChain:
    """Create a handler chain with default handlers.

    Args:
        event_handlers: StrandsEventHandlers instance

    Returns:
        StrandsEventHandlerChain configured with default handlers

    """
    handlers = [
        TextStreamingHandler(),
        AssistantToolCallsHandler(),  # Process assistant messages with tool_calls before tool results
        ToolResultHandler(),
        ToolCallHandler(),
        MultiAgentEventHandler(),
        ResultEventHandler(),
    ]
    return StrandsEventHandlerChain(handlers)
