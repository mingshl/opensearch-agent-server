"""Tool Call Event Handler for StrandsAgent.

Handles tool call events from the Strands agent and converts them to AG-UI tool call events.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ag_ui.core import (
    CustomEvent,
    EventType,
    RunAgentInput,
    StateSnapshotEvent,
    ToolCallArgsEvent,
    ToolCallStartEvent,
)

from utils.logging_helpers import (
    get_logger,
    log_debug_event,
)
from server.strands_agent_config import (
    StrandsAgentConfig,
    ToolCallContext,
    maybe_await,
    normalize_predict_state,
)
from server.types import (
    ActiveToolCallInfo,
    EventResult,
    HandlerResult,
    StrandsEvent,
    ToolCallInfoInternal,
    ToolCallStateUpdate,
)
from server.utils import (
    get_tool_call_id,
    log_hook_error_if_not_silent,
)

logger = get_logger(__name__)


class ToolCallEventHandler:
    """Handler for tool call events from the Strands agent.

    Converts Strands tool call events into AG-UI tool call events, handling
    args streaming, state hooks, and PredictState mappings.
    """

    def __init__(self, config: StrandsAgentConfig) -> None:
        """Initialize tool call event handler with configuration.

        Args:
            config: StrandsAgentConfig instance for customizing tool behavior
        """
        self.config = config

    async def _handle_args_streamer(
        self,
        input_data: RunAgentInput,
        tool_name: str,
        strands_tool_id: str,
        tool_input: dict,
        args_str: str,
        tool_behavior: Any,
    ) -> AsyncIterator[ToolCallArgsEvent]:
        """Handle args_streamer hook for tool call.

        Args:
            input_data: RunAgentInput
            tool_name: Name of the tool
            strands_tool_id: Tool call ID
            tool_input: Tool input dictionary
            args_str: Serialized tool arguments
            tool_behavior: Tool behavior configuration

        Yields:
            ToolCallArgsEvent instances

        """
        if tool_behavior and tool_behavior.args_streamer:
            try:
                call_context = ToolCallContext(
                    input_data=input_data,
                    tool_name=tool_name,
                    tool_use_id=strands_tool_id,
                    tool_input=tool_input,
                    args_str=args_str,
                )
                async for args_chunk in tool_behavior.args_streamer(call_context):
                    yield ToolCallArgsEvent(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=strands_tool_id,
                        delta=args_chunk,
                    )
            except Exception as e:
                log_hook_error_if_not_silent(
                    silent=self.config.silent_hook_errors,
                    hook_name="args_streamer",
                    error_event_name="ag_ui.args_streamer_error",
                    error=e,
                    tool_name=tool_name,
                    tool_call_id=strands_tool_id,
                )
                # Fallback to single args event
                yield ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=strands_tool_id,
                    delta=args_str,
                )
        else:
            yield ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=strands_tool_id,
                delta=args_str,
            )

    async def _handle_state_from_args_hook(
        self,
        input_data: RunAgentInput,
        tool_name: str,
        strands_tool_id: str,
        tool_input: dict,
        args_str: str,
        tool_behavior: Any,
    ) -> AsyncIterator[StateSnapshotEvent]:
        """Handle state_from_args hook for tool call.

        Args:
            input_data: RunAgentInput
            tool_name: Name of the tool
            strands_tool_id: Tool call ID
            tool_input: Tool input dictionary
            args_str: Serialized tool arguments
            tool_behavior: Tool behavior configuration

        Yields:
            StateSnapshotEvent if state payload is generated

        """
        if not (tool_behavior and tool_behavior.state_from_args):
            return

        try:
            call_context = ToolCallContext(
                input_data=input_data,
                tool_name=tool_name,
                tool_use_id=strands_tool_id,
                tool_input=tool_input,
                args_str=args_str,
            )
            state_payload = await maybe_await(
                tool_behavior.state_from_args(call_context)
            )
            if state_payload:
                yield StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT, snapshot=state_payload
                )
                log_debug_event(
                    logger,
                    f"Emitted state snapshot from tool args: {tool_name}",
                    "ag_ui.emitted_state_snapshot_from_args",
                    tool_name=tool_name,
                )
        except Exception as e:
            log_hook_error_if_not_silent(
                silent=self.config.silent_hook_errors,
                hook_name="state_from_args",
                error_event_name="ag_ui.state_from_args_hook_error",
                error=e,
                tool_name=tool_name,
                tool_call_id=strands_tool_id,
            )

    async def _handle_predict_state(
        self,
        tool_name: str,
        strands_tool_id: str,
        tool_behavior: Any,
    ) -> AsyncIterator[CustomEvent]:
        """Handle PredictState mappings for tool call.

        Args:
            tool_name: Name of the tool
            strands_tool_id: Tool call ID
            tool_behavior: Tool behavior configuration

        Yields:
            CustomEvent with PredictState if mappings exist

        """
        if not tool_behavior:
            return

        try:
            predict_state_payload = [
                mapping.to_payload()
                for mapping in normalize_predict_state(tool_behavior.predict_state)
            ]
            if predict_state_payload:
                yield CustomEvent(
                    type=EventType.CUSTOM,
                    name="PredictState",
                    value=predict_state_payload,
                )
                log_debug_event(
                    logger,
                    f"Emitted PredictState event: tool_name={tool_name}, "
                    f"mapping_count={len(predict_state_payload)}",
                    "ag_ui.emitted_predict_state_event",
                    tool_name=tool_name,
                    mapping_count=len(predict_state_payload),
                )
        except Exception as e:
            log_hook_error_if_not_silent(
                silent=self.config.silent_hook_errors,
                hook_name="PredictState",
                error_event_name="ag_ui.predict_state_emit_error",
                error=e,
                tool_name=tool_name,
                tool_call_id=strands_tool_id,
            )

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
        tool_use = event["current_tool_use"]
        tool_name = tool_use.get("name")
        strands_tool_id = get_tool_call_id(tool_use, generate_default=True)
        tool_input = tool_use.get("input", {})
        args_str = (
            json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
        )

        # Phase 2: Skip tool call events if pending tool result exists
        if has_pending_tool_result:
            log_debug_event(
                logger,
                f"Skipping tool call START event due to pending tool result: "
                f"tool_name={tool_name}, tool_call_id={strands_tool_id}",
                "ag_ui.tool_call_skipped_pending_result",
                tool_name=tool_name,
                tool_call_id=strands_tool_id,
            )
            # Still track the tool call in tool_calls_seen for state management
            # but don't emit events
            if tool_name and strands_tool_id:
                tool_calls_seen[strands_tool_id] = {
                    "name": tool_name,
                    "args": args_str,
                    "input": tool_input,
                }
                active_tool_calls[strands_tool_id] = {
                    "name": tool_name,
                    "input": tool_input,
                }
            return  # Don't emit tool call events

        # Debug logging for tool calls
        log_debug_event(
            logger,
            f"Processing tool call: tool_name={tool_name}, tool_id={strands_tool_id}, "
            f"already_seen={strands_tool_id in tool_calls_seen}, "
            f"tool_calls_seen_count={len(tool_calls_seen)}",
            "ag_ui.tool_call_processing",
            tool_name=tool_name,
            tool_id=strands_tool_id,
            already_seen=strands_tool_id in tool_calls_seen,
            tool_calls_seen_count=len(tool_calls_seen),
            tool_input_keys=list(tool_input.keys())
            if isinstance(tool_input, dict)
            else [],
        )

        # Only process if this is a new tool call
        if not (tool_name and strands_tool_id not in tool_calls_seen):
            log_debug_event(
                logger,
                f"Skipping tool call: tool_name={tool_name}, tool_id={strands_tool_id}, "
                f"reason={'no_name' if not tool_name else 'already_seen'}",
                "ag_ui.tool_call_skipped",
                tool_name=tool_name,
                tool_id=strands_tool_id,
                reason="no_name" if not tool_name else "already_seen",
            )
            return  # No tool call to process, don't yield anything

        # Phase 1: Check if this is a frontend tool and generate unique ID
        is_frontend_tool = tool_name in frontend_tool_names if tool_name else False

        # Generate unique ID for frontend tools, use Strands ID for backend tools
        if is_frontend_tool:
            # Generate new UUID for frontend tools to avoid ID conflicts
            tool_use_id = str(uuid.uuid4())
            log_debug_event(
                logger,
                f"Generated unique ID for frontend tool: tool_name={tool_name}, "
                f"strands_tool_id={strands_tool_id}, tool_use_id={tool_use_id}",
                "ag_ui.frontend_tool_id_generated",
                tool_name=tool_name,
                strands_tool_id=strands_tool_id,
                tool_use_id=tool_use_id,
            )
            # Store mapping: tool_use_id -> strands_tool_id for result lookup
            # We'll store both IDs in tool_calls_seen using tool_use_id as key
            # but also track the mapping
            tool_calls_seen[tool_use_id] = {
                "name": tool_name,
                "args": args_str,
                "input": tool_input,
                "strands_tool_id": strands_tool_id,  # Store mapping for result lookup
            }
        else:
            # Use Strands' ID for backend tools (so result lookup works)
            tool_use_id = strands_tool_id or str(uuid.uuid4())
            tool_calls_seen[tool_use_id] = {
                "name": tool_name,
                "args": args_str,
                "input": tool_input,
            }

        active_tool_calls[tool_use_id] = {
            "name": tool_name,
            "input": tool_input,
        }

        tool_behavior = self.config.tool_behaviors.get(tool_name)

        # Emit tool call events
        log_debug_event(
            logger,
            f"Emitting TOOL_CALL_START: tool_name={tool_name}, tool_id={tool_use_id}",
            "ag_ui.tool_call_start_emitted",
            tool_name=tool_name,
            tool_id=tool_use_id,
            parent_message_id=message_id,
        )
        yield EventResult(
            event=ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=tool_use_id,
                tool_call_name=tool_name,
                parent_message_id=message_id,
            ),
            should_stop=False,
        )

        # Handle args_streamer if configured
        async for args_event in self._handle_args_streamer(
            input_data, tool_name, tool_use_id, tool_input, args_str, tool_behavior
        ):
            yield EventResult(event=args_event, should_stop=False)

        # Handle state_from_args hook
        async for state_event in self._handle_state_from_args_hook(
            input_data, tool_name, tool_use_id, tool_input, args_str, tool_behavior
        ):
            yield EventResult(event=state_event, should_stop=False)

        # Handle PredictState mappings
        async for predict_event in self._handle_predict_state(
            tool_name, tool_use_id, tool_behavior
        ):
            yield EventResult(event=predict_event, should_stop=False)

        # Phase 1: Halt stream after frontend tool call (unless continue_after_frontend_call is set)
        should_stop_after_tool_call = False
        if is_frontend_tool:
            if not (tool_behavior and tool_behavior.continue_after_frontend_call):
                # Halt event stream after frontend tool call
                log_debug_event(
                    logger,
                    f"Halting event stream after frontend tool call: tool_name={tool_name}, "
                    f"tool_call_id={tool_use_id}",
                    "ag_ui.halt_after_frontend_tool",
                    tool_name=tool_name,
                    tool_call_id=tool_use_id,
                )
                should_stop_after_tool_call = True

        # Yield state update only when a tool call was processed
        yield ToolCallStateUpdate(
            tool_calls_seen=tool_calls_seen,
            active_tool_calls=active_tool_calls,
            should_stop=should_stop_after_tool_call,
        )
