"""Event processing logic for Strands agent event streams.

This module contains the core event processing logic that processes
Strands agent events and converts them to AG-UI protocol events.

**Key Functions:**
- `process_strands_event_stream` - Main event processing loop
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunFinishedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from strands import Agent as StrandsAgentCore

from utils.logging_helpers import get_logger, log_debug_event, log_warning_event
from server.event_conversion import extract_user_message, process_handler_result
from server.state_management import apply_state_context, handle_state_snapshot
from server.strands_agent_config import StrandsAgentConfig
from server.strands_event_handlers import StrandsEventHandlers
from server.strands_event_strategy import (
    StrandsEventContext,
    create_strands_event_handler_chain,
)
from server.types import (
    ActiveToolCallInfo,
    AGUIEvent,
    ToolCallInfoInternal,
)
from server.utils import is_event_type

logger = get_logger(__name__)


async def process_strands_event_stream(
    agent: StrandsAgentCore,
    input_data: RunAgentInput,
    thread_id: str,
    run_id: str,
    config: StrandsAgentConfig,
    event_handlers: StrandsEventHandlers,
) -> AsyncIterator[AGUIEvent]:
    """Process Strands agent event stream and yield AG-UI events.

    This function handles the core event processing loop, converting
    Strands agent events into AG-UI protocol events. It uses a strategy
    pattern with event handlers to process different event types and
    maintains state throughout the event stream.

    Args:
        agent: StrandsAgentCore instance that generates event stream
        input_data: RunAgentInput with thread_id, run_id, messages, etc.
        thread_id: Thread identifier for the conversation
        run_id: Run identifier for this execution
        config: StrandsAgentConfig for state management and tool behavior
        event_handlers: StrandsEventHandlers instance for event processing

    Yields:
        AG-UI Event objects (TextMessageStartEvent, ToolCallStartEvent, etc.)

    Example:
        ```python
        async for event in process_strands_event_stream(
            agent=agent,
            input_data=input_data,
            thread_id="thread-123",
            run_id="run-456",
            config=config,
            event_handlers=event_handlers,
        ):
            # Process each AG-UI event
            if event.type == EventType.TEXT_MESSAGE_START:
                print("Message started")
            elif event.type == EventType.TOOL_CALL_START:
                print(f"Tool call: {event.tool_name}")
        ```

    """
    # Handle state snapshot if provided
    is_resume, snapshot_event = handle_state_snapshot(input_data, thread_id)
    if snapshot_event:
        yield snapshot_event

    # Extract and prepare user message
    user_message = extract_user_message(input_data)
    user_message = apply_state_context(
        config, input_data, user_message, thread_id, is_resume
    )

    # Extract frontend tool names from input_data.tools (Phase 1: Frontend Tool Detection)
    frontend_tool_names: set[str] = set()
    if hasattr(input_data, "tools") and input_data.tools:
        for tool_def in input_data.tools:
            tool_name = (
                tool_def.get("name")
                if isinstance(tool_def, dict)
                else getattr(tool_def, "name", None)
            )
            if tool_name:
                frontend_tool_names.add(tool_name)
    log_debug_event(
        logger,
        f"Extracted frontend tool names: {frontend_tool_names}, "
        f"thread_id={thread_id}, run_id={run_id}",
        "ag_ui.frontend_tool_names_extracted",
        frontend_tool_names=list(frontend_tool_names),
        thread_id=thread_id,
        run_id=run_id,
    )

    # Phase 2: Detect pending tool result (Phase 2: Pending Tool Result Detection)
    has_pending_tool_result = False
    if hasattr(input_data, "messages") and input_data.messages:
        last_msg = input_data.messages[-1]
        if (hasattr(last_msg, "role") and last_msg.role == "tool") or (
            isinstance(last_msg, dict) and last_msg.get("role") == "tool"
        ):
            has_pending_tool_result = True
            tool_call_id = (
                getattr(last_msg, "tool_call_id", "unknown")
                if not isinstance(last_msg, dict)
                else last_msg.get("tool_call_id", "unknown")
            )
            log_debug_event(
                logger,
                f"Has pending tool result detected: tool_call_id={tool_call_id}, "
                f"thread_id={thread_id}",
                "ag_ui.pending_tool_result_detected",
                tool_call_id=tool_call_id,
                thread_id=thread_id,
            )

    # Phase 3: Track expected tool call IDs from messages (Phase 3: Orphaned Tool Message Handling)
    expected_tool_call_ids: set[str] = set()
    if hasattr(input_data, "messages") and input_data.messages:
        for msg in input_data.messages:
            # Handle assistant messages with tool_calls
            if (
                hasattr(msg, "role")
                and msg.role == "assistant"
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                expected_tool_call_ids.clear()  # Reset for this assistant message
                for tc in msg.tool_calls:
                    tool_call_id = (
                        tc.id if hasattr(tc, "id") else getattr(tc, "id", None)
                    )
                    if tool_call_id:
                        expected_tool_call_ids.add(tool_call_id)

            # Handle tool messages
            elif hasattr(msg, "role") and msg.role == "tool":
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id and tool_call_id in expected_tool_call_ids:
                    expected_tool_call_ids.remove(tool_call_id)
                elif tool_call_id:
                    # Orphaned tool message - log but don't fail
                    log_debug_event(
                        logger,
                        f"Skipping orphaned tool message: tool_call_id={tool_call_id}, "
                        f"thread_id={thread_id}",
                        "ag_ui.orphaned_tool_message",
                        tool_call_id=tool_call_id,
                        thread_id=thread_id,
                    )

    # Initialize tracking variables
    message_id = str(uuid.uuid4())
    message_started = False
    text_message_start_emitted = False  # Track if TEXT_MESSAGE_START was emitted
    tool_calls_seen: dict[str, ToolCallInfoInternal] = {}
    active_tool_calls: dict[str, ActiveToolCallInfo] = {}
    text_buffer: list[str] = []  # TEMPORARY: Buffer for non-streaming mode
    halt_event_stream = (
        False  # Flag to silently consume events after frontend tool halt
    )

    message_count = len(input_data.messages) if hasattr(input_data, "messages") else 0
    log_debug_event(
        logger,
        f"Starting agent run: thread_id={thread_id}, run_id={run_id}, message_count={message_count}",
        "ag_ui.agent_run_starting",
        thread_id=thread_id,
        run_id=run_id,
        message_count=message_count,
    )

    # Initialize handler chain
    handler_chain = create_strands_event_handler_chain(event_handlers)

    # Stream from Strands agent
    # `agent.stream_async()` returns an async generator
    agent_stream = agent.stream_async(user_message)

    # `async for` iterates over async generator values
    # Each iteration waits for the next value to be produced
    # This is non-blocking - other async tasks can run while waiting
    context: StrandsEventContext | None = None
    try:
        async for event in agent_stream:
            # If we've halted, consume remaining events silently to allow proper cleanup
            # This matches the official implementation pattern to avoid async generator cleanup issues
            if halt_event_stream:
                # Continue consuming events silently until generator ends naturally
                # or we receive a complete/force_stop event
                if event.get("complete") or event.get("force_stop"):
                    log_debug_event(
                        logger,
                        f"Stream complete or force_stop after halt: complete={event.get('complete')}, "
                        f"force_stop={event.get('force_stop')}",
                        "ag_ui.stream_complete_after_halt",
                        complete=event.get("complete"),
                        force_stop=event.get("force_stop"),
                    )
                    break
                continue

            # Enhanced logging for tool call events
            has_tool_use = "current_tool_use" in event and bool(
                event.get("current_tool_use")
            )
            tool_name = None
            if has_tool_use:
                tool_use = event.get("current_tool_use", {})
                tool_name = tool_use.get("name")

            # Check for assistant messages with tool_calls
            has_message = "message" in event and isinstance(event.get("message"), dict)
            message_role = None
            has_tool_calls = False
            tool_calls_count = 0
            if has_message:
                message = event.get("message", {})
                message_role = message.get("role")
                if message_role == "assistant":
                    tool_calls = message.get("tool_calls", [])
                    has_tool_calls = bool(tool_calls)
                    tool_calls_count = (
                        len(tool_calls) if isinstance(tool_calls, list) else 0
                    )

            log_debug_event(
                logger,
                f"Received Strands event: event_keys={list(event.keys())}, "
                f"has_current_tool_use={has_tool_use}, tool_name={tool_name}, "
                f"has_message={has_message}, message_role={message_role}, "
                f"has_tool_calls={has_tool_calls}, tool_calls_count={tool_calls_count}",
                "ag_ui.received_strands_event",
                event_keys=list(event.keys()),
                has_current_tool_use=has_tool_use,
                tool_name=tool_name,
                has_message=has_message,
                message_role=message_role,
                has_tool_calls=has_tool_calls,
                tool_calls_count=tool_calls_count,
            )

            # Skip lifecycle events
            if event.get("init_event_loop") or event.get("start_event_loop"):
                continue
            if event.get("complete") or event.get("force_stop"):
                log_debug_event(
                    logger,
                    f"Stream complete or force_stop: complete={event.get('complete')}, "
                    f"force_stop={event.get('force_stop')}",
                    "ag_ui.stream_complete_or_force_stop",
                    complete=event.get("complete"),
                    force_stop=event.get("force_stop"),
                )
                break

            # Use strategy pattern to handle events
            # Create context object with all state needed by handlers
            context = StrandsEventContext(
                event=event,
                message_id=message_id,
                message_started=message_started,
                tool_calls_seen=tool_calls_seen,
                active_tool_calls=active_tool_calls,
                input_data=input_data,
                event_handlers=event_handlers,
                text_buffer=text_buffer,
                frontend_tool_names=frontend_tool_names,
                has_pending_tool_result=has_pending_tool_result,
            )

            # Process event through handler chain
            # `async for` iterates over async generator results from handler chain
            should_stop_processing = False
            async for handler_result in handler_chain.process_event(context):
                # Update state from context (handlers may have modified it)
                # Python dictionaries are mutable - handlers can modify them in-place
                # Read updated state from context AFTER handler has processed it
                message_started = context.message_started
                tool_calls_seen = context.tool_calls_seen
                active_tool_calls = context.active_tool_calls
                text_buffer = context.text_buffer  # Sync text buffer from context

                # Process handler result and yield events/update state
                async for (
                    event_to_yield,
                    updated_msg_started,
                    updated_tool_calls,
                    updated_active_calls,
                    should_stop,
                ) in process_handler_result(
                    handler_result,
                    event,
                    message_id,
                    message_started,
                    tool_calls_seen,
                    active_tool_calls,
                ):
                    # Update state from handler result processing
                    # Log if message_started is being set to True
                    if not message_started and updated_msg_started:
                        log_debug_event(
                            logger,
                            "message_started changed from False to True via process_handler_result",
                            "ag_ui.message_started_set_via_handler_result",
                            thread_id=thread_id,
                            run_id=run_id,
                            message_id=message_id,
                            previous_value=message_started,
                            new_value=updated_msg_started,
                        )
                    message_started = updated_msg_started
                    tool_calls_seen = updated_tool_calls
                    active_tool_calls = updated_active_calls

                    # Yield event if present
                    if event_to_yield is not None:
                        # Track if TEXT_MESSAGE_START was emitted
                        if is_event_type(event_to_yield, EventType.TEXT_MESSAGE_START):
                            text_message_start_emitted = True
                            log_debug_event(
                                logger,
                                "TEXT_MESSAGE_START event emitted - marking for TEXT_MESSAGE_END",
                                "ag_ui.text_message_start_emitted",
                                thread_id=thread_id,
                                run_id=run_id,
                                message_id=message_id,
                            )
                        try:
                            yield event_to_yield
                        except GeneratorExit:
                            # Catch GeneratorExit during yield - this happens when the consumer
                            # closes the generator. We should stop processing and return
                            # to trigger the finally block for cleanup.
                            log_debug_event(
                                logger,
                                f"GeneratorExit caught during yield: run_id={run_id}",
                                "ag_ui.generator_exit_during_yield",
                                run_id=run_id,
                                thread_id=thread_id,
                            )
                            return

                    # Stop processing if signaled (set halt flag to consume remaining events silently)
                    if should_stop:
                        halt_event_stream = True
                        log_debug_event(
                            logger,
                            f"Halting event stream: will consume remaining events silently, "
                            f"thread_id={thread_id}, run_id={run_id}",
                            "ag_ui.halt_event_stream_set",
                            thread_id=thread_id,
                            run_id=run_id,
                        )
                        # Don't break here - continue to consume events silently
                        # The halt_event_stream check at the top of the loop will handle it
                        should_stop_processing = True
                        break

                # Break outer loop if should_stop was set (but we'll continue consuming silently)
                if should_stop_processing:
                    break
    except GeneratorExit:
        # Catch GeneratorExit during iteration - this happens when the consumer
        # closes the generator. We should stop processing and return
        # to trigger the finally block for cleanup.
        log_debug_event(
            logger,
            f"GeneratorExit caught during iteration: run_id={run_id}",
            "ag_ui.generator_exit_during_iteration",
            run_id=run_id,
            thread_id=thread_id,
        )
        return
    except (asyncio.CancelledError, Exception):
        # Re-raise other exceptions to be handled by the caller
        raise
    finally:
        # Properly close the async generator to avoid context detachment errors
        # This matches the official implementation pattern
        # The generator should complete naturally when we consume all events,
        # but we still try to close it explicitly to be safe
        try:
            # Check if generator is already closed/exhausted
            if hasattr(agent_stream, "ag_running") and not agent_stream.ag_running:
                # Generator is already closed, nothing to do
                pass
            elif hasattr(agent_stream, "aclose"):
                # Try to close gracefully, but suppress context-related errors
                await agent_stream.aclose()
        except (
            GeneratorExit,
            ValueError,
            RuntimeError,
            StopAsyncIteration,
        ):
            # Suppress context detachment errors - they occur when the generator
            # is closed in a different context, but don't affect functionality
            # These errors are logged by Strands internally, we just prevent them from propagating
            pass
        except AttributeError:
            # Generator doesn't have ag_running attribute (older Python versions)
            # Just try to close it
            if hasattr(agent_stream, "aclose"):
                try:
                    await agent_stream.aclose()
                except (
                    GeneratorExit,
                    ValueError,
                    RuntimeError,
                    StopAsyncIteration,
                ):
                    pass
        except Exception as e:
            # Log other unexpected errors but don't fail
            log_debug_event(
                logger,
                f"Error closing agent stream: {e}, thread_id={thread_id}, run_id={run_id}",
                "ag_ui.agent_stream_close_error",
                error=str(e),
                thread_id=thread_id,
                run_id=run_id,
            )

    # End message if started OR if we have buffered text (streaming disabled case)
    if text_buffer:
        accumulated_text = "".join(text_buffer)
        if not message_started:
            log_warning_event(
                logger,
                "[STREAMING_DISABLED] Sending TEXT_MESSAGE_START for buffered text.",
                "ag_ui.streaming_disabled_send_start",
                message_id=message_id,
                text_length=len(accumulated_text),
            )
            yield TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message_id,
                role="assistant",
            )
            message_started = True

        log_warning_event(
            logger,
            "[STREAMING_DISABLED] Sending accumulated TEXT_MESSAGE_CONTENT.",
            "ag_ui.streaming_disabled_send_content",
            message_id=message_id,
            text_length=len(accumulated_text),
        )
        yield TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=message_id,
            delta=accumulated_text,
        )
        text_buffer.clear()  # Clear buffer after sending

    # Final state synchronization: sync from context one last time after loop ends
    # This ensures we capture any state updates that handlers made during the last
    # iteration that might not have been synced.
    # Handlers update context.message_started (see strands_event_strategy.py:194),
    # which we sync. However, if handlers update it after we've already
    # synced, or if the loop breaks early, we need this final sync to ensure
    # TEXT_MESSAGE_END is emitted correctly.
    #
    # IMPORTANT: We use the existing message_started value (from process_handler_result)
    # as the source of truth, since process_handler_result correctly sets it when
    # TEXT_MESSAGE_START events are processed. Only sync from context if message_started
    # is still False (meaning it wasn't set by process_handler_result).
    try:
        # Sync final state from context if it exists (events were processed)
        # context is only defined if the event loop executed at least once
        # Only update if message_started is False (context might have the correct value)
        if context and hasattr(context, "message_started"):
            # Use context.message_started if our current value is False
            # This handles cases where handlers set context.message_started but
            # process_handler_result didn't catch it (edge cases)
            if not message_started and context.message_started:
                log_debug_event(
                    logger,
                    "Syncing message_started from context (was False, context has True).",
                    "ag_ui.sync_message_started_from_context",
                    thread_id=thread_id,
                    run_id=run_id,
                    context_message_started=context.message_started,
                )
                message_started = context.message_started
    except (NameError, AttributeError):
        # context not defined (no events processed) - use existing message_started value
        pass

    # Emit TEXT_MESSAGE_END if message was started
    # This ensures every TEXT_MESSAGE_START has a matching TEXT_MESSAGE_END
    #
    # FALLBACK: If message_started is False but we have text_buffer content,
    # that means a message was started but the flag wasn't set correctly.
    # In this case, we should still emit TEXT_MESSAGE_END.
    #
    # ADDITIONAL FALLBACK: Check context.message_started as final fallback
    # This handles cases where the flag wasn't properly synced
    has_text_content = bool(text_buffer) if "text_buffer" in locals() else False
    context_has_message_started = (
        context.message_started
        if "context" in locals() and context and hasattr(context, "message_started")
        else False
    )

    # Emit TEXT_MESSAGE_END if message was started OR if we have text content OR if context says message started
    # OR if we emitted a TEXT_MESSAGE_START event (most reliable indicator)
    # This handles edge cases where message_started flag wasn't set correctly
    should_emit_end = (
        message_started
        or has_text_content
        or context_has_message_started
        or text_message_start_emitted
    )

    if should_emit_end:
        yield TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=message_id,
        )
    else:
        log_warning_event(
            logger,
            "Not emitting TEXT_MESSAGE_END - message_started is False and no text buffer.",
            "ag_ui.text_message_end_not_emitted",
            thread_id=thread_id,
            run_id=run_id,
            message_started=message_started,
            has_text_buffer=has_text_content,
        )

    # Always finish the run
    yield RunFinishedEvent(
        type=EventType.RUN_FINISHED,
        thread_id=thread_id,
        run_id=run_id,
    )
