"""
Monitored Tool Decorator

Wraps Strands @tool decorator to add real-time UI monitoring via AgentActivityMonitor.
Use this instead of @tool directly to get automatic Chainlit step creation.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from strands.tools.decorator import tool

# Chainlit monitor not used in opensearch-agent-server
MONITOR_AVAILABLE = False
get_monitor = None

# Exceptions that may indicate monitor/emitter unavailable at runtime (e.g. wrong
# context). We catch these so tools still run; we do not catch BaseException
# (KeyboardInterrupt, SystemExit) or other critical errors.
_MONITOR_UNAVAILABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    LookupError,
)
try:
    from chainlit.context import (
        ChainlitContextException,
    )  # "context not found" when not in Chainlit app (e.g. pytest)

    _MONITOR_UNAVAILABLE_EXCEPTIONS = (
        *_MONITOR_UNAVAILABLE_EXCEPTIONS,
        ChainlitContextException,
    )
except ImportError:
    pass  # Chainlit not installed; base tuple is sufficient

# Try to import AG-UI emitter - will work in AG-UI context
try:
    from utils.tool_event_emitter import get_ag_ui_emitter

    GET_AG_UI_EMITTER_AVAILABLE = True
except ImportError:
    GET_AG_UI_EMITTER_AVAILABLE = False
    get_ag_ui_emitter = None


def _result_preview(result: Any, max_len: int = 200) -> str:
    """Safe preview string for tool result; never raises."""
    try:
        s = str(result)
        return s[:max_len] + "..." if len(s) > max_len else s
    except Exception:
        return "[output omitted]"


def monitored_tool(
    name: str | None = None,
    description: str | None = None,
    inputSchema: dict | None = None,
) -> Callable[..., Any]:
    """
    Decorator that wraps @tool and adds real-time monitoring for Chainlit UI.

    This decorator:
    1. Adds activity monitoring (creates Chainlit steps showing tool execution)
    2. Applies the standard Strands @tool decorator
    3. Gracefully degrades if monitor is not available (non-Chainlit contexts)

    Usage:
        @monitored_tool(
            name="MyTool",
            description="Does something useful"
        )
        def my_tool(param1: str) -> str:
            return f"Result: {param1}"

    Args:
        name: Tool name (will be shown in UI)
        description: Tool description (for LLM)
        inputSchema: JSON schema for tool inputs

    Returns:
        Decorated function with monitoring and Strands tool capabilities
    """

    def decorator(func: Callable) -> Callable:
        # Check if function is async or sync
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Async wrapper that adds monitoring and AG-UI event emission to tool calls.

                This wrapper:
                1. Attempts to get Chainlit monitor (if available)
                2. Attempts to get AG-UI event emitter (if available)
                3. Executes the tool function with appropriate monitoring
                4. Gracefully degrades if monitoring is not available

                Args:
                    *args: Positional arguments passed to the tool function
                    **kwargs: Keyword arguments passed to the tool function

                Returns:
                    Result from the wrapped tool function
                """
                # Try to get monitor
                monitor = None
                if MONITOR_AVAILABLE and get_monitor:
                    try:
                        monitor = get_monitor()
                    except _MONITOR_UNAVAILABLE_EXCEPTIONS:
                        pass  # Monitor not available in this context

                # Try to get AG-UI emitter
                ag_ui_emitter = None
                if GET_AG_UI_EMITTER_AVAILABLE and get_ag_ui_emitter:
                    try:
                        ag_ui_emitter = get_ag_ui_emitter()
                    except _MONITOR_UNAVAILABLE_EXCEPTIONS:
                        pass  # Emitter not available in this context

                tool_name = name or func.__name__

                # Use monitoring if available
                if monitor and ag_ui_emitter:
                    # Both Chainlit monitor and AG-UI emitter available
                    async with monitor.tool_call(tool_name, **kwargs) as step:
                        async with ag_ui_emitter.tool_call(
                            tool_name, **kwargs
                        ) as tool_call_id:
                            result = await func(*args, **kwargs)
                            # Set result for AG-UI emitter
                            await ag_ui_emitter.set_tool_call_result(
                                tool_call_id, result
                            )
                            # Show result preview in step (safe if str(result) fails)
                            step.output = f"✓ Result: {_result_preview(result)}"
                            return result
                elif monitor:
                    # Only Chainlit monitor available
                    async with monitor.tool_call(tool_name, **kwargs) as step:
                        result = await func(*args, **kwargs)
                        # Show result preview in step (safe if str(result) fails)
                        step.output = f"✓ Result: {_result_preview(result)}"
                        return result
                elif ag_ui_emitter:
                    # Only AG-UI emitter available
                    async with ag_ui_emitter.tool_call(
                        tool_name, **kwargs
                    ) as tool_call_id:
                        result = await func(*args, **kwargs)
                        await ag_ui_emitter.set_tool_call_result(tool_call_id, result)
                        return result
                else:
                    # Execute without monitoring
                    return await func(*args, **kwargs)

            # Apply @tool decorator (only pass inputSchema when provided)
            tool_kwargs: dict[str, Any] = {"name": name, "description": description}
            if inputSchema is not None:
                tool_kwargs["inputSchema"] = inputSchema
            return tool(**tool_kwargs)(async_wrapper)

        else:
            # Convert sync function to async and add monitoring
            @functools.wraps(func)
            async def async_sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Async wrapper for sync functions that adds monitoring and AG-UI event emission.

                This wrapper:
                1. Converts sync function to async execution
                2. Attempts to get Chainlit monitor (if available)
                3. Attempts to get AG-UI event emitter (if available)
                4. Executes the sync tool function with appropriate monitoring
                5. Gracefully degrades if monitoring is not available

                Args:
                    *args: Positional arguments passed to the tool function
                    **kwargs: Keyword arguments passed to the tool function

                Returns:
                    Result from the wrapped tool function
                """
                # Try to get monitor
                monitor = None
                if MONITOR_AVAILABLE and get_monitor:
                    try:
                        monitor = get_monitor()
                    except _MONITOR_UNAVAILABLE_EXCEPTIONS:
                        pass  # Monitor not available in this context

                # Try to get AG-UI emitter
                ag_ui_emitter = None
                if GET_AG_UI_EMITTER_AVAILABLE and get_ag_ui_emitter:
                    try:
                        ag_ui_emitter = get_ag_ui_emitter()
                    except _MONITOR_UNAVAILABLE_EXCEPTIONS:
                        pass  # Emitter not available in this context

                tool_name = name or func.__name__

                # Use monitoring if available
                if monitor and ag_ui_emitter:
                    # Both Chainlit monitor and AG-UI emitter available
                    async with monitor.tool_call(tool_name, **kwargs) as step:
                        async with ag_ui_emitter.tool_call(
                            tool_name, **kwargs
                        ) as tool_call_id:
                            # Call sync function in async context
                            result = func(*args, **kwargs)
                            # Set result for AG-UI emitter
                            await ag_ui_emitter.set_tool_call_result(
                                tool_call_id, result
                            )
                            # Show result preview in step (safe if str(result) fails)
                            step.output = f"✓ Result: {_result_preview(result)}"
                            return result
                elif monitor:
                    # Only Chainlit monitor available
                    async with monitor.tool_call(tool_name, **kwargs) as step:
                        # Call sync function in async context
                        result = func(*args, **kwargs)
                        # Show result preview in step (safe if str(result) fails)
                        step.output = f"✓ Result: {_result_preview(result)}"
                        return result
                elif ag_ui_emitter:
                    # Only AG-UI emitter available
                    async with ag_ui_emitter.tool_call(
                        tool_name, **kwargs
                    ) as tool_call_id:
                        # Call sync function in async context
                        result = func(*args, **kwargs)
                        await ag_ui_emitter.set_tool_call_result(tool_call_id, result)
                        return result
                else:
                    # Execute without monitoring
                    return func(*args, **kwargs)

            # Apply @tool decorator (only pass inputSchema when provided)
            tool_kwargs = {"name": name, "description": description}
            if inputSchema is not None:
                tool_kwargs["inputSchema"] = inputSchema
            return tool(**tool_kwargs)(async_sync_wrapper)

    return decorator
