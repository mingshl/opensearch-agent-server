"""Event Handlers for StrandsAgent.

Handles various event types from the Strands agent and converts them to AG-UI events.

This module provides a facade class that delegates to specialized handler modules
organized by event type for improved maintainability.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ag_ui.core import (
    RunAgentInput,
    TextMessageContentEvent,
    TextMessageStartEvent,
)

from server.strands_agent_config import (
    StatePayload,
    StrandsAgentConfig,
)
from server.strands_multi_agent_event_handler import MultiAgentEventHandler
from server.strands_text_event_handler import TextEventHandler
from server.strands_tool_call_event_handler import ToolCallEventHandler
from server.strands_tool_result_event_handler import (
    HAS_JSONPATCH,
    ToolResultEventHandler,
)
from server.types import (
    ActiveToolCallInfo,
    HandlerResult,
    StrandsEvent,
    ToolCallInfoInternal,
)

# Re-export for tests and backward compatibility
__all__ = ["StrandsEventHandlers", "HAS_JSONPATCH"]


class StrandsEventHandlers:
    """Event handlers for converting Strands events to AG-UI events.

    This class provides a unified interface for handling different types of events from
    the Strands agent and converting them to AG-UI protocol events. It delegates to
    specialized handler modules organized by event type.

    **Architecture:**
    - `TextEventHandler`: Handles text streaming events
    - `ToolCallEventHandler`: Handles tool call events
    - `ToolResultEventHandler`: Handles tool result events
    - `MultiAgentEventHandler`: Handles multi-agent events
    """

    def __init__(self, config: StrandsAgentConfig) -> None:
        """Initialize event handlers with configuration.

        Args:
            config: StrandsAgentConfig instance for customizing tool behavior

        """
        self.config = config
        # Track previous state for delta generation (keyed by thread_id)
        # `dict[str, StatePayload]`: Dictionary mapping thread IDs to state dictionaries
        self._previous_state: dict[str, StatePayload] = {}

        # Initialize specialized handlers
        self._text_handler = TextEventHandler()
        self._tool_call_handler = ToolCallEventHandler(config)
        self._tool_result_handler = ToolResultEventHandler(config, self._previous_state)
        self._multi_agent_handler = MultiAgentEventHandler()

    def handle_text_streaming(
        self, event: StrandsEvent, message_id: str, message_started: bool
    ) -> tuple[bool, TextMessageStartEvent | None, TextMessageContentEvent | None]:
        """Handle text streaming events from the agent.

        Args:
            event: Strands event dictionary
            message_id: Current message ID
            message_started: Whether message has been started

        Returns:
            Tuple of (updated message_started flag, start_event or None, content_event or None)

        """
        return self._text_handler.handle_text_streaming(
            event, message_id, message_started
        )

    def parse_tool_result_data(self, result_content: list) -> str | dict | None:
        """Parse tool result data from content list.

        Args:
            result_content: List of content items from tool result

        Returns:
            Parsed result data (dict, str, or None)

        """
        return self._tool_result_handler.parse_tool_result_data(result_content)

    async def handle_tool_calls(
        self,
        event: StrandsEvent,
        input_data: RunAgentInput,
        tool_calls_seen: dict[str, ToolCallInfoInternal],
        active_tool_calls: dict[str, ActiveToolCallInfo],
        message_id: str,
        frontend_tool_names: set[str],
        has_pending_tool_result: bool,
    ) -> AsyncIterator[HandlerResult]:
        """Handle tool call events from the agent.

        Args:
            event: Strands event dictionary
            input_data: RunAgentInput
            tool_calls_seen: Dictionary of seen tool calls
            active_tool_calls: Dictionary of active tool calls
            message_id: Current message ID
            frontend_tool_names: Set of frontend tool names (Phase 1: Frontend Tool Detection)
            has_pending_tool_result: Whether there is a pending tool result (Phase 2: Pending Tool Result Detection)

        Yields:
            HandlerResult objects:
            - EventResult: Tool call related AG-UI events
            - ToolCallStateUpdate: State update when tool call is processed

        """
        async for result in self._tool_call_handler.handle_tool_calls(
            event,
            input_data,
            tool_calls_seen,
            active_tool_calls,
            message_id,
            frontend_tool_names,
            has_pending_tool_result,
        ):
            yield result

    async def handle_tool_results(
        self,
        event: StrandsEvent,
        input_data: RunAgentInput,
        tool_calls_seen: dict[str, ToolCallInfoInternal],
        active_tool_calls: dict[str, ActiveToolCallInfo],
        message_id: str,
    ) -> AsyncIterator[HandlerResult]:
        """Handle tool result events from message events.

        Args:
            event: Strands event dictionary
            input_data: RunAgentInput
            tool_calls_seen: Dictionary of seen tool calls
            active_tool_calls: Dictionary of active tool calls
            message_id: Current message ID

        Yields:
            Various AG-UI events related to tool results, including error events for malformed results

        """
        async for result in self._tool_result_handler.handle_tool_results(
            event, input_data, tool_calls_seen, active_tool_calls, message_id
        ):
            yield result

    async def handle_multi_agent_events(
        self, event: StrandsEvent, message_id: str
    ) -> AsyncIterator[HandlerResult]:
        """Handle multi-agent events (node start, node stop, handoff).

        Args:
            event: Strands event dictionary
            message_id: Current message ID

        Yields:
            Multi-agent related AG-UI events

        """
        async for result in self._multi_agent_handler.handle_multi_agent_events(
            event, message_id
        ):
            yield result
