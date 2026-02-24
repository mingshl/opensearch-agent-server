"""
AG-UI Tool Event Emitter

Emits AG-UI events when tools are called at the Python level.
This allows internal tool calls (that don't generate Strands events) to be
displayed in the AG-UI frontend.

Similar to Chainlit's AgentActivityMonitor, but emits AG-UI protocol events
instead of creating Chainlit steps.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from ag_ui.core import (
    EventType,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from utils.logging_helpers import get_logger, log_debug_event, log_warning_event
from server.types import AGUIEvent
from server.utils import format_result_content

logger = get_logger(__name__)

# Thread-local storage for event emitter
# This allows tool wrappers to access the emitter without passing it explicitly
#
# Note: We use thread-local storage (dictionary with thread IDs) instead of contextvars
# because:
# 1. The current implementation works correctly for our use case where each request
#    runs in its own thread/async context
# 2. Thread-local storage is simpler and sufficient for the current architecture
# 3. Multiple async tasks running in the same thread would be rare in our use case
#
# Future consideration: If we need better async support with multiple concurrent
# tasks in the same thread, we could migrate to contextvars. However, this would
# require careful testing to ensure it doesn't break existing functionality.
_thread_local_storage: dict[str, AGUIToolEventEmitter | None] = {}


class AGUIToolEventEmitter:
    """Emits AG-UI events when tools are called at Python level.

    This class provides a mechanism for Python-level tool wrappers to emit
    AG-UI protocol events, allowing internal tool calls to be displayed
    in the frontend even though they don't generate Strands events.
    """

    def __init__(
        self,
        event_queue: asyncio.Queue[AGUIEvent],
        message_id: str,
        yield_callback: Callable[[AGUIEvent], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize AG-UI tool event emitter.

        Args:
            event_queue: Queue to put AG-UI events into (will be merged with Strands events)
            message_id: Current message ID for associating tool calls with messages
            yield_callback: Optional async callback to process and queue events immediately
        """
        self.event_queue = event_queue
        self.message_id = message_id
        self.yield_callback = yield_callback
        self.tool_call_counter = 0
        self.active_tool_calls: dict[str, str] = {}  # Maps tool_call_id -> tool_name
        self.tool_call_results: dict[str, Any] = {}  # Maps tool_call_id -> result

    async def emit_tool_call_start(
        self, tool_name: str, arguments: dict[str, Any | None] = None
    ) -> str:
        """Emit TOOL_CALL_START event.

        Args:
            tool_name: Name of the tool being called
            arguments: Optional dictionary with tool call arguments

        Returns:
            tool_call_id: Unique identifier for this tool call
        """
        self.tool_call_counter += 1
        tool_call_id = f"python_tool_{self.tool_call_counter}_{uuid.uuid4().hex[:8]}"

        self.active_tool_calls[tool_call_id] = tool_name

        event = ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=tool_call_id,
            tool_call_name=tool_name,
            parent_message_id=self.message_id,
        )

        log_debug_event(
            logger,
            f"Emitting Python-level TOOL_CALL_START: tool_name={tool_name}, tool_id={tool_call_id}",
            "ag_ui.python_tool_call_start",
            tool_name=tool_name,
            tool_id=tool_call_id,
            message_id=self.message_id,
        )

        # If callback is available, yield directly (like Chainlit does)
        # Otherwise, queue for later processing
        if self.yield_callback:
            try:
                await self.yield_callback(event)
            except Exception as e:
                # Fallback to queue if callback fails
                log_warning_event(
                    logger,
                    "Callback failed, falling back to queue.",
                    "ag_ui.emitter_callback_fallback",
                    error=str(e),
                )
                await self.event_queue.put(event)
        else:
            await self.event_queue.put(event)
        return tool_call_id

    async def emit_tool_call_end(
        self,
        tool_call_id: str,
        success: bool = True,
        error: str | None = None,
        result: Any | None = None,
    ) -> None:
        """Emit TOOL_CALL_END event following AG-UI protocol.

        If result is provided, emits TOOL_CALL_RESULT first, then TOOL_CALL_END.

        Args:
            tool_call_id: Unique identifier for the tool call (returned from emit_tool_call_start)
            success: Whether the tool call succeeded
            error: Optional error message if failed
            result: Optional tool call result to include (emitted as TOOL_CALL_RESULT per AG-UI protocol)
        """
        if tool_call_id not in self.active_tool_calls:
            log_warning_event(
                logger,
                "Tool call end emitted for unknown tool_call_id.",
                "ag_ui.tool_call_end_unknown_id",
                tool_call_id=tool_call_id,
            )
            return

        tool_name = self.active_tool_calls.pop(tool_call_id)

        # Emit ToolCallResultEvent if we have a result (AG-UI protocol)
        if success and not error and result is not None:
            # Convert result to JSON string for ToolCallResultEvent.content
            result_content = format_result_content(result)

            result_event = ToolCallResultEvent(
                type=EventType.TOOL_CALL_RESULT,
                tool_call_id=tool_call_id,
                message_id=self.message_id,
                content=result_content,
                # role is intentionally omitted - without role="tool",
                # the frontend won't add this to conversation history
            )

            log_debug_event(
                logger,
                f"Emitting Python-level TOOL_CALL_RESULT: tool_name={tool_name}, tool_id={tool_call_id}",
                "ag_ui.python_tool_call_result",
                tool_name=tool_name,
                tool_id=tool_call_id,
            )

            # Emit result event
            if self.yield_callback:
                try:
                    await self.yield_callback(result_event)
                except Exception as e:
                    log_warning_event(
                        logger,
                        "Callback failed, falling back to queue.",
                        "ag_ui.emitter_callback_fallback",
                        error=str(e),
                    )
                    await self.event_queue.put(result_event)
            else:
                await self.event_queue.put(result_event)

        # Emit TOOL_CALL_END to mark completion (AG-UI protocol)
        event = ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=tool_call_id,
        )

        log_debug_event(
            logger,
            f"Emitting Python-level TOOL_CALL_END: tool_name={tool_name}, tool_id={tool_call_id}, success={success}",
            "ag_ui.python_tool_call_end",
            tool_name=tool_name,
            tool_id=tool_call_id,
            success=success,
            error=error,
        )

        # If callback is available, yield directly (like Chainlit does)
        # Otherwise, queue for later processing
        if self.yield_callback:
            try:
                await self.yield_callback(event)
            except Exception as e:
                # Fallback to queue if callback fails
                log_warning_event(
                    logger,
                    "Callback failed, falling back to queue.",
                    "ag_ui.emitter_callback_fallback",
                    error=str(e),
                )
                await self.event_queue.put(event)
        else:
            await self.event_queue.put(event)

    async def set_tool_call_result(self, tool_call_id: str, result: Any) -> None:
        """Set the result for a tool call (emitted as TOOL_CALL_RESULT event per AG-UI protocol).

        Args:
            tool_call_id: Tool call ID
            result: Tool call result (will be emitted via ToolCallResultEvent)
        """
        self.tool_call_results[tool_call_id] = result

    @asynccontextmanager
    async def tool_call(self, tool_name: str, **kwargs: Any) -> Any:
        """Context manager to track tool calls and emit events.

        Usage:
            async with emitter.tool_call("SearchIndexTool", query="laptop") as tool_call_id:
                result = await actual_tool_call()
                await emitter.set_tool_call_result(tool_call_id, result)
                return result

        Args:
            tool_name: Name of the tool being called
            **kwargs: Tool arguments (for logging/debugging)
        """
        tool_call_id = await self.emit_tool_call_start(tool_name, kwargs)
        try:
            yield tool_call_id
            # Get result if it was set via set_tool_call_result
            result = self.tool_call_results.pop(tool_call_id, None)
            await self.emit_tool_call_end(tool_call_id, success=True, result=result)
        except Exception as e:
            # Clean up result if exception occurred
            self.tool_call_results.pop(tool_call_id, None)
            await self.emit_tool_call_end(tool_call_id, success=False, error=str(e))
            raise


def set_ag_ui_emitter(emitter: AGUIToolEventEmitter | None) -> None:
    """Set the AG-UI event emitter for the current thread/context.

    This allows tool wrappers to access the emitter without explicit passing.
    Similar to Chainlit's user_session pattern.

    Args:
        emitter: AGUIToolEventEmitter instance or None to clear
    """
    thread_id = threading.current_thread().ident
    if thread_id is None:
        # Fallback to process-level storage if thread ID unavailable
        _thread_local_storage["emitter"] = emitter
    else:
        _thread_local_storage[f"emitter_{thread_id}"] = emitter


def get_ag_ui_emitter() -> AGUIToolEventEmitter | None:
    """Get the AG-UI event emitter for the current thread/context.

    Returns:
        AGUIToolEventEmitter instance if set, None otherwise
    """
    thread_id = threading.current_thread().ident
    if thread_id is None:
        return _thread_local_storage.get("emitter")
    return _thread_local_storage.get(f"emitter_{thread_id}")
