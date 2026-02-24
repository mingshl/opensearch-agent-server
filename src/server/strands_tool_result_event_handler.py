"""Tool Result Event Handler for StrandsAgent.

Handles tool result events from the Strands agent and converts them to AG-UI events.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

try:
    import jsonpatch  # type: ignore[import-untyped]

    HAS_JSONPATCH = True
except ImportError:
    HAS_JSONPATCH = False
    jsonpatch = None  # type: ignore[assignment]

from ag_ui.core import (
    AssistantMessage,
    CustomEvent,
    EventType,
    MessagesSnapshotEvent,
    RunAgentInput,
    StateDeltaEvent,
    StateSnapshotEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    ToolMessage,
)

from utils.logging_helpers import (
    get_logger,
    log_debug_event,
    log_error_event,
    log_info_event,
    log_warning_event,
)
from server.constants import ROLE_USER
from server.sanitization import sanitize_for_tool_result
from server.strands_agent_config import (
    StatePayload,
    StrandsAgentConfig,
    ToolResultContext,
    maybe_await,
)
from server.types import (
    ActiveToolCallInfo,
    EventResult,
    HandlerResult,
    StrandsEvent,
    ToolCallInfoInternal,
    ToolCallStateUpdate,
    ToolResultStateUpdate,
)
from server.utils import (
    create_error_event,
    format_result_content,
    get_tool_call_id,
    handle_operation_error,
    log_hook_error_if_not_silent,
    parse_json_with_fallback,
)

logger = get_logger(__name__)


class ToolResultEventHandler:
    """Handler for tool result events from the Strands agent.

    Converts Strands tool result events into AG-UI events, handling
    messages snapshots, state hooks, and tool call completion.
    """

    def __init__(
        self, config: StrandsAgentConfig, previous_state: dict[str, StatePayload]
    ) -> None:
        """Initialize tool result event handler with configuration.

        Args:
            config: StrandsAgentConfig instance for customizing tool behavior
            previous_state: Dictionary mapping thread IDs to state dictionaries for delta tracking
        """
        self.config = config
        self._previous_state = previous_state

    def parse_tool_result_data(self, result_content: list[Any]) -> dict[str, Any] | str:
        """Parse tool result data from content list.

        Args:
            result_content: List of content items from tool result

        Returns:
            Parsed result data (dict, str, or None)

        """
        result_data = None
        # `result_content and ...` checks if list is not empty (short-circuit evaluation)
        # `isinstance()` checks type at runtime
        if result_content and isinstance(result_content, list):
            # Collect all text content items
            text_parts = []
            # Iterate over list items
            for content_item in result_content:
                # Check if item is dict AND has "text" key
                if isinstance(content_item, dict) and "text" in content_item:
                    text_content = content_item["text"]
                    # Only process non-empty text
                    if text_content:
                        text_parts.append(text_content)

            # Combine all text parts if multiple items
            if text_parts:
                # Ensure all parts are strings for join (content may be str or other)
                str_parts = [str(p).strip() for p in text_parts if p is not None]
                str_parts = [p for p in str_parts if p]
                if not str_parts:
                    return result_data
                combined_text = (
                    "\n".join(str_parts) if len(str_parts) > 1 else str_parts[0]
                )
                # Only attempt JSON parse when content looks like JSON; otherwise
                # treat as plain text (e.g. markdown) to avoid log noise and errors.
                stripped = combined_text.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    result_data = parse_json_with_fallback(
                        combined_text, fallback_value=combined_text
                    )
                else:
                    result_data = combined_text
        return result_data

    async def _emit_messages_snapshot_for_tool_result(
        self,
        input_data: RunAgentInput,
        result_tool_id: str,
        tool_name: str,
        call_info: ToolCallInfoInternal,
        result_data: Any,
    ) -> MessagesSnapshotEvent | None:
        """Emit MessagesSnapshotEvent for a tool result.

        Args:
            input_data: RunAgentInput
            result_tool_id: Tool call ID
            tool_name: Name of the tool
            call_info: Tool call information
            result_data: Parsed tool result data

        Returns:
            MessagesSnapshotEvent or None if emission failed

        """
        try:
            assistant_msg = AssistantMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=result_tool_id,
                        type="function",
                        function={
                            "name": tool_name,
                            "arguments": call_info.get("args", "{}"),
                        },
                    )
                ],
            )
            content_str = format_result_content(result_data)
            tool_msg = ToolMessage(
                id=str(uuid.uuid4()),
                role="tool",
                content=content_str,
                tool_call_id=result_tool_id,
            )
            all_messages = list(input_data.messages) + [assistant_msg, tool_msg]
            return MessagesSnapshotEvent(
                type=EventType.MESSAGES_SNAPSHOT,
                messages=all_messages,
            )
        except Exception as e:
            handle_operation_error(
                operation_name="emit MessagesSnapshotEvent for tool result",
                error_event_name="ag_ui.messages_snapshot_emit_failed",
                error=e,
                tool_call_id=result_tool_id,
                tool_name=tool_name,
            )
            return None

    def _generate_state_delta(
        self,
        previous_state: StatePayload,
        new_state: StatePayload,
        tool_name: str,
    ) -> StateDeltaEvent | None:
        """Generate a StateDeltaEvent from state changes using JSON patch.

        Args:
            previous_state: Previous state dictionary
            new_state: New state dictionary
            tool_name: Name of the tool (for logging)

        Returns:
            StateDeltaEvent if jsonpatch is available and delta can be generated, None otherwise

        """
        if not HAS_JSONPATCH:
            log_warning_event(
                logger,
                f"jsonpatch not available, cannot generate state delta: tool_name={tool_name}. "
                "Install jsonpatch package to enable state delta support.",
                "ag_ui.jsonpatch_unavailable",
                tool_name=tool_name,
            )
            return None

        try:
            # Generate JSON patch from previous state to new state
            patch = jsonpatch.make_patch(previous_state, new_state)

            # Only emit delta if there are actual changes
            if patch.patch:
                return StateDeltaEvent(type=EventType.STATE_DELTA, delta=patch.patch)
            return None
        except Exception as e:
            handle_operation_error(
                operation_name="generate state delta",
                error_event_name="ag_ui.state_delta_generation_error",
                error=e,
                tool_name=tool_name,
            )
            return None

    def _make_result_context(
        self,
        input_data: RunAgentInput,
        tool_name: str,
        result_tool_id: str,
        call_info: ToolCallInfoInternal,
        result_data: Any,
        message_id: str,
    ) -> ToolResultContext:
        """Build a ToolResultContext for result hooks.

        Args:
            input_data: RunAgentInput for the current run
            tool_name: Name of the tool
            result_tool_id: Tool call ID
            call_info: Tool call information (input, args, etc.)
            result_data: Parsed tool result data
            message_id: Current message ID

        Returns:
            ToolResultContext instance for use by state/custom result hooks
        """
        return ToolResultContext(
            input_data=input_data,
            tool_name=tool_name,
            tool_use_id=result_tool_id,
            tool_input=call_info.get("input", {}),
            args_str=call_info.get("args", "{}"),
            result_data=result_data,
            message_id=message_id,
        )

    async def _handle_state_from_result_hook(
        self,
        input_data: RunAgentInput,
        tool_name: str,
        result_tool_id: str,
        call_info: ToolCallInfoInternal,
        result_data: Any,
        message_id: str,
        tool_behavior: Any,
    ) -> AsyncIterator[StateSnapshotEvent]:
        """Handle state_from_result hook for tool result.

        Args:
            input_data: RunAgentInput
            tool_name: Name of the tool
            result_tool_id: Tool call ID
            call_info: Tool call information
            result_data: Parsed tool result data
            message_id: Current message ID
            tool_behavior: Tool behavior configuration

        Yields:
            StateSnapshotEvent if state payload is generated

        """
        # Early return pattern: exit if hook not configured
        if not (tool_behavior and tool_behavior.state_from_result):
            return

        try:
            result_context = self._make_result_context(
                input_data,
                tool_name,
                result_tool_id,
                call_info,
                result_data,
                message_id,
            )
            # `await maybe_await(...)` handles both sync and async hook functions
            # `maybe_await()` returns value if sync, awaits if async
            state_payload = await maybe_await(
                tool_behavior.state_from_result(result_context)
            )
            # `if state_payload:` checks if value is truthy (not None, not empty)
            if state_payload:
                # `yield` produces event to caller (makes this an async generator)
                yield StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT, snapshot=state_payload
                )
                log_debug_event(
                    logger,
                    f"Emitted state snapshot from tool result: tool_name={tool_name}",
                    "ag_ui.emitted_state_snapshot_from_result",
                    tool_name=tool_name,
                )
        except Exception as e:
            log_hook_error_if_not_silent(
                silent=self.config.silent_hook_errors,
                hook_name="state_from_result",
                error_event_name="ag_ui.state_from_result_hook_error",
                error=e,
                tool_name=tool_name,
                tool_call_id=result_tool_id,
            )

    async def _handle_state_delta_from_result_hook(
        self,
        input_data: RunAgentInput,
        tool_name: str,
        result_tool_id: str,
        call_info: ToolCallInfoInternal,
        result_data: Any,
        message_id: str,
        tool_behavior: Any,
        previous_state: StatePayload,
    ) -> AsyncIterator[StateDeltaEvent | StateSnapshotEvent]:
        """Handle state_delta_from_result hook for tool result.

        This hook allows tools to emit incremental state updates via JSON patches.
        The hook receives the previous state and returns the new state, and a delta
        is automatically generated.

        Args:
            input_data: RunAgentInput
            tool_name: Name of the tool
            result_tool_id: Tool call ID
            call_info: Tool call information
            result_data: Parsed tool result data
            message_id: Current message ID
            tool_behavior: Tool behavior configuration
            previous_state: Previous state dictionary to generate delta from

        Yields:
            StateDeltaEvent if delta can be generated, otherwise StateSnapshotEvent

        """
        # Early return pattern: exit if hook not configured
        if not (tool_behavior and tool_behavior.state_delta_from_result):
            return  # Exit early - no hook to execute

        try:
            result_context = self._make_result_context(
                input_data,
                tool_name,
                result_tool_id,
                call_info,
                result_data,
                message_id,
            )
            # `await maybe_await(...)` handles both sync and async hook functions
            new_state = await maybe_await(
                tool_behavior.state_delta_from_result(result_context, previous_state)
            )

            if new_state:
                # Try to generate delta first
                delta_event = self._generate_state_delta(
                    previous_state, new_state, tool_name
                )
                if delta_event:
                    yield delta_event
                    log_debug_event(
                        logger,
                        f"Emitted state delta from tool result: tool_name={tool_name}",
                        "ag_ui.emitted_state_delta_from_result",
                        tool_name=tool_name,
                    )
                else:
                    # Fallback to snapshot if delta generation fails
                    yield StateSnapshotEvent(
                        type=EventType.STATE_SNAPSHOT, snapshot=new_state
                    )
                    log_debug_event(
                        logger,
                        f"Emitted state snapshot (delta fallback) from tool result: tool_name={tool_name}",
                        "ag_ui.emitted_state_snapshot_delta_fallback",
                        tool_name=tool_name,
                    )
        except Exception as e:
            log_hook_error_if_not_silent(
                silent=self.config.silent_hook_errors,
                hook_name="state_delta_from_result",
                error_event_name="ag_ui.state_delta_from_result_hook_error",
                error=e,
                tool_name=tool_name,
                tool_call_id=result_tool_id,
            )

    async def _handle_custom_result_handler_hook(
        self,
        input_data: RunAgentInput,
        tool_name: str,
        result_tool_id: str,
        call_info: ToolCallInfoInternal,
        result_data: Any,
        message_id: str,
        tool_behavior: Any,
    ) -> AsyncIterator[CustomEvent]:
        """Handle custom_result_handler hook for tool result.

        Args:
            input_data: RunAgentInput
            tool_name: Name of the tool
            result_tool_id: Tool call ID
            call_info: Tool call information
            result_data: Parsed tool result data
            message_id: Current message ID
            tool_behavior: Tool behavior configuration

        Yields:
            Custom events from the handler

        """
        if not (tool_behavior and tool_behavior.custom_result_handler):
            return

        try:
            result_context = self._make_result_context(
                input_data,
                tool_name,
                result_tool_id,
                call_info,
                result_data,
                message_id,
            )
            async for custom_event in tool_behavior.custom_result_handler(
                result_context
            ):
                if custom_event is not None:
                    yield custom_event
        except Exception as e:
            log_hook_error_if_not_silent(
                silent=self.config.silent_hook_errors,
                hook_name="custom_result_handler",
                error_event_name="ag_ui.custom_result_handler_error",
                error=e,
                tool_name=tool_name,
                tool_call_id=result_tool_id,
            )

    async def _process_single_tool_result(
        self,
        input_data: RunAgentInput,
        result_tool_id: str,
        result_data: Any,
        tool_calls_seen: dict[str, ToolCallInfoInternal],
        active_tool_calls: dict[str, ActiveToolCallInfo],
        message_id: str,
        tool_result_dict: dict[str, Any] | None = None,
    ) -> AsyncIterator[HandlerResult]:
        """Process a single tool result and yield related events.

        Extracted from handle_tool_results to improve readability and testability.
        Handles all aspects of tool result processing including event emission,
        hook execution, and tool call completion.

        Args:
            input_data: RunAgentInput
            result_tool_id: Tool call ID for the result (may be strands_tool_id for frontend tools)
            result_data: Parsed tool result data
            tool_calls_seen: Dictionary of seen tool calls
            active_tool_calls: Dictionary of active tool calls (modified in place)
            message_id: Current message ID
            tool_result_dict: Optional raw tool result dictionary (for extracting tool name)

        Yields:
            HandlerResult objects:
            - EventResult: AG-UI events (MessagesSnapshotEvent, StateSnapshotEvent, CustomEvent, ToolCallEndEvent)
            - ToolResultStateUpdate: State update with optional stop flag
        """
        # Phase 1: Map strands_tool_id to tool_use_id for frontend tools
        # Frontend tools use unique UUIDs, but results come with strands_tool_id
        actual_tool_id = result_tool_id
        if result_tool_id not in tool_calls_seen:
            # Try to find by strands_tool_id mapping (for frontend tools)
            for tool_use_id, call_info in tool_calls_seen.items():
                strands_tool_id = call_info.get("strands_tool_id")
                if strands_tool_id == result_tool_id:
                    actual_tool_id = tool_use_id
                    log_debug_event(
                        logger,
                        f"Mapped strands_tool_id to tool_use_id for frontend tool: "
                        f"strands_tool_id={result_tool_id}, tool_use_id={tool_use_id}",
                        "ag_ui.frontend_tool_id_mapped",
                        strands_tool_id=result_tool_id,
                        tool_use_id=tool_use_id,
                    )
                    break

        call_info = tool_calls_seen.get(actual_tool_id, {})
        tool_name = call_info.get("name", "unknown")

        # Handle tool result for unknown tool call - emit retroactive TOOL_CALL_START event
        if tool_name == "unknown":
            # Try to extract tool name from tool_result_dict
            inferred_tool_name = None
            if tool_result_dict:
                inferred_tool_name = tool_result_dict.get("name")

            # Use inferred name or fallback to "unknown_tool"
            tool_name = inferred_tool_name or "unknown_tool"

            # Register the tool call retroactively (use actual_tool_id, not result_tool_id)
            tool_calls_seen[actual_tool_id] = {
                "name": tool_name,
                "args": "{}",
                "input": {},
            }
            active_tool_calls[actual_tool_id] = {
                "name": tool_name,
                "input": {},
            }

            log_info_event(
                logger,
                f"Emitting retroactive TOOL_CALL_START for unknown tool call: "
                f"tool_name={tool_name}, tool_id={actual_tool_id}",
                "ag_ui.retroactive_tool_call_start",
                tool_name=tool_name,
                tool_id=actual_tool_id,
                inferred_from_result=inferred_tool_name is not None,
            )

            # Emit retroactive TOOL_CALL_START event
            yield EventResult(
                event=ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=actual_tool_id,
                    tool_call_name=tool_name,
                    parent_message_id=message_id,
                ),
                should_stop=False,
            )

            # Emit state update
            yield ToolCallStateUpdate(
                tool_calls_seen=tool_calls_seen.copy(),
                active_tool_calls=active_tool_calls.copy(),
            )

            # Log the issue; we still emit an error event below so the frontend is notified
            log_error_event(
                logger,
                f"Tool result received for unknown tool call ID: {actual_tool_id}. "
                "This may indicate out-of-order events (tool result before tool call) or a missing tool call event.",
                "strands.tool_result_before_call",
                exc_info=False,
                tool_call_id=actual_tool_id,
                message_id=message_id,
            )
            # Emit error event to notify frontend of the issue
            yield EventResult(
                event=create_error_event(
                    message=(
                        f"Tool result received for unknown tool call ID: {actual_tool_id}. "
                        "This may indicate out-of-order events or a missing tool call event."
                    ),
                    code="TOOL_RESULT_UNKNOWN_CALL",
                ),
                should_stop=False,
            )
            # Continue processing to mark the tool call as complete if it exists in active_tool_calls
            # This prevents hanging tool calls

        tool_behavior = self.config.tool_behaviors.get(tool_name)

        # Emit MessagesSnapshotEvent
        if not (tool_behavior and tool_behavior.skip_messages_snapshot):
            snapshot_event = await self._emit_messages_snapshot_for_tool_result(
                input_data, actual_tool_id, tool_name, call_info, result_data
            )
            if snapshot_event:
                yield EventResult(event=snapshot_event, should_stop=False)

        # Get thread_id once for state tracking
        thread_id = getattr(input_data, "thread_id", "default")

        # Handle state_from_result hook (for snapshots)
        async for state_event in self._handle_state_from_result_hook(
            input_data,
            tool_name,
            actual_tool_id,
            call_info,
            result_data,
            message_id,
            tool_behavior,
        ):
            # Update tracked state if snapshot is emitted
            if isinstance(state_event, StateSnapshotEvent) and hasattr(
                state_event, "snapshot"
            ):
                self._previous_state[thread_id] = state_event.snapshot
            yield EventResult(event=state_event, should_stop=False)

        # Handle state_delta_from_result hook (for incremental updates)
        previous_state = self._previous_state.get(thread_id, {})
        async for state_event in self._handle_state_delta_from_result_hook(
            input_data,
            tool_name,
            actual_tool_id,
            call_info,
            result_data,
            message_id,
            tool_behavior,
            previous_state,
        ):
            # Update tracked state after delta/snapshot is emitted
            if isinstance(state_event, StateDeltaEvent):
                # Apply delta to previous state to get new state
                if HAS_JSONPATCH and hasattr(state_event, "delta"):
                    try:
                        new_state = jsonpatch.apply_patch(
                            previous_state, state_event.delta
                        )
                        self._previous_state[thread_id] = new_state
                    except Exception as e:
                        handle_operation_error(
                            operation_name="apply state delta",
                            error_event_name="ag_ui.state_delta_apply_failed",
                            error=e,
                            tool_name=tool_name,
                        )
            elif isinstance(state_event, StateSnapshotEvent) and hasattr(
                state_event, "snapshot"
            ):
                # Update tracked state with snapshot
                self._previous_state[thread_id] = state_event.snapshot
            yield EventResult(event=state_event, should_stop=False)

        # Handle custom_result_handler hook
        async for custom_event in self._handle_custom_result_handler_hook(
            input_data,
            tool_name,
            actual_tool_id,
            call_info,
            result_data,
            message_id,
            tool_behavior,
        ):
            yield EventResult(event=custom_event, should_stop=False)

        # Mark tool call as complete
        if actual_tool_id in active_tool_calls:
            # Emit ToolCallResultEvent if we have result data (AG-UI protocol)
            if result_data is not None:
                # Convert result_data to JSON string for ToolCallResultEvent.content
                result_content = format_result_content(result_data)
                log_debug_event(
                    logger,
                    f"Emitting TOOL_CALL_RESULT: tool_id={actual_tool_id}, "
                    f"result_type={type(result_data).__name__}",
                    "ag_ui.tool_call_result_emitted",
                    tool_call_id=actual_tool_id,
                    result_type=type(result_data).__name__,
                )
                yield EventResult(
                    event=ToolCallResultEvent(
                        type=EventType.TOOL_CALL_RESULT,
                        tool_call_id=actual_tool_id,
                        message_id=message_id,
                        content=result_content,
                        # role is intentionally omitted - without role="tool",
                        # the frontend won't add this to conversation history
                    ),
                    should_stop=False,
                )

            # Emit TOOL_CALL_END to mark tool call as complete (AG-UI protocol)
            yield EventResult(
                event=ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=actual_tool_id,
                ),
                should_stop=False,
            )
            del active_tool_calls[actual_tool_id]

            # Check if we should stop streaming after result
            if tool_behavior and tool_behavior.stop_streaming_after_result:
                yield ToolResultStateUpdate(
                    active_tool_calls=active_tool_calls, should_stop=True
                )
                return

        # Yield state update after processing tool result (even if not in active_tool_calls)
        # This ensures consistent behavior and allows caller to know processing completed
        yield ToolResultStateUpdate(
            active_tool_calls=active_tool_calls, should_stop=False
        )

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
        message_obj = event["message"]
        if message_obj.get("role") == ROLE_USER:
            message_content = message_obj.get("content", [])
            if isinstance(message_content, list):
                for item in message_content:
                    if isinstance(item, dict) and "toolResult" in item:
                        tool_result = item["toolResult"]
                        result_tool_id = get_tool_call_id(
                            tool_result, generate_default=False
                        )
                        result_content = tool_result.get("content", [])

                        # Debug: Log full tool result structure to understand what fields are available
                        # Use INFO level so it shows up in logs
                        log_info_event(
                            logger,
                            f"Tool result received: tool_result_keys={list(tool_result.keys())}, "
                            f"tool_call_id={result_tool_id}, "
                            f"has_name={'name' in tool_result}, "
                            f"name_value={tool_result.get('name') if 'name' in tool_result else None}, "
                            f"tool_result_preview={str(tool_result)[:200]}",
                            "ag_ui.tool_result_structure",
                            tool_result_keys=list(tool_result.keys()),
                            tool_call_id=result_tool_id,
                            has_name="name" in tool_result,
                            name_value=tool_result.get("name")
                            if "name" in tool_result
                            else None,
                        )

                        # Validate tool result before processing
                        if not result_tool_id:
                            # Missing tool call ID - emit error event
                            log_error_event(
                                logger,
                                f"Malformed tool result: missing tool_call_id. "
                                f"message_id={message_id}",
                                "strands.malformed_tool_result_missing_id",
                                exc_info=False,
                                message_id=message_id,
                                tool_result=str(tool_result)[
                                    :200
                                ],  # Truncate for logging
                            )
                            yield EventResult(
                                event=create_error_event(
                                    message=(
                                        "Received tool result with missing tool call ID. "
                                        "This may indicate a malformed event from the agent."
                                    ),
                                    code="MALFORMED_TOOL_RESULT",
                                ),
                                should_stop=False,
                            )
                            continue  # Skip this tool result

                        result_data = self.parse_tool_result_data(result_content)

                        if result_data is None:
                            # Failed to parse result data - emit error event
                            log_error_event(
                                logger,
                                f"Malformed tool result: failed to parse result data. "
                                f"tool_call_id={result_tool_id}, message_id={message_id}",
                                "strands.malformed_tool_result_parse_failed",
                                exc_info=False,
                                tool_call_id=result_tool_id,
                                message_id=message_id,
                                result_content=str(result_content)[
                                    :200
                                ],  # Truncate for logging
                            )
                            yield EventResult(
                                event=create_error_event(
                                    message=(
                                        f"Failed to parse tool result data for tool call {result_tool_id}. "
                                        "The result content may be malformed."
                                    ),
                                    code="TOOL_RESULT_PARSE_ERROR",
                                ),
                                should_stop=False,
                            )
                            # Still process the tool result with None data to mark it as complete
                            # This prevents the tool call from hanging indefinitely
                            result_data = {"error": "Failed to parse result data"}

                        # Sanitize tool result before emitting to events/hooks (control chars, depth/size limits)
                        result_data = sanitize_for_tool_result(result_data)

                        # Process valid tool result
                        should_stop = False
                        async for result in self._process_single_tool_result(
                            input_data=input_data,
                            result_tool_id=result_tool_id,
                            result_data=result_data,
                            tool_calls_seen=tool_calls_seen,
                            active_tool_calls=active_tool_calls,
                            message_id=message_id,
                            tool_result_dict=tool_result,
                        ):
                            # Check if this is a state update signaling to stop
                            if (
                                isinstance(result, ToolResultStateUpdate)
                                and result.should_stop
                            ):
                                should_stop = True
                            yield result
                        # Stop processing if tool result signaled to stop streaming
                        if should_stop:
                            return
