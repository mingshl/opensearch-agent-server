"""Configuration primitives for customizing Strands agent behavior.

This module provides configuration classes for customizing tool behavior,
state management, and thread resume functionality.

**Key Classes:**
- `StrandsAgentConfig` - Top-level configuration for agent behavior
- `ToolBehavior` - Per-tool configuration and hooks
- `ToolCallContext` - Context passed to tool call hooks
- `ToolResultContext` - Context passed to tool result hooks
- `PredictStateMapping` - Declarative state prediction configuration

**Usage Example:**
```python
from server.strands_agent_config import (
    StrandsAgentConfig,
    ToolBehavior,
    PredictStateMapping,
)

config = StrandsAgentConfig(
    tool_behaviors={
        "my_tool": ToolBehavior(
            state_from_result=lambda ctx: {"result": ctx.result_data},
            predict_state=[
                PredictStateMapping(
                    state_key="query",
                    tool="my_tool",
                    tool_argument="query"
                )
            ],
        )
    }
)

agent = StrandsAgent(config=config)
```
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import (
    Any,
    TypeVar,
    cast,
)

from ag_ui.core import RunAgentInput

# TypeVar for generic awaitable values
T = TypeVar("T")

StatePayload = dict[str, Any]


@dataclass
class ToolCallContext:
    """Context passed to tool call hooks.

    Attributes:
        input_data: RunAgentInput instance containing thread and message data
        tool_name: Name of the tool being called
        tool_use_id: Unique identifier for this tool call
        tool_input: Tool input dictionary with arguments
        args_str: Serialized JSON string representation of tool arguments

    """

    input_data: RunAgentInput
    tool_name: str
    tool_use_id: str
    tool_input: Any
    args_str: str


@dataclass
class ToolResultContext(ToolCallContext):
    """Context passed to tool result hooks.

    Inherits all fields from ToolCallContext and adds result-specific fields.

    Attributes:
        result_data: Parsed tool result data (dict, str, or other type)
        message_id: Current message ID associated with this tool result

    """

    result_data: Any
    message_id: str


# Type aliases for function signatures
# `Callable[[ArgTypes...], ReturnType]` describes a function type
# `AsyncIterator[Type]` means async generator yielding Type values
# `Awaitable[Type]` means value that can be awaited (async function result)
# `|` is union operator (Python 3.10+) meaning "or" (alternative to Union)
ArgsStreamer = Callable[[ToolCallContext], AsyncIterator[str]]
StateFromArgs = Callable[
    [ToolCallContext], Awaitable[StatePayload | None] | StatePayload | None
]
StateFromResult = Callable[
    [ToolResultContext], Awaitable[StatePayload | None] | StatePayload | None
]
StateDeltaFromResult = Callable[
    [ToolResultContext, StatePayload],
    Awaitable[StatePayload | None] | StatePayload | None,
]
CustomResultHandler = Callable[[ToolResultContext], AsyncIterator[Any]]
StateContextBuilder = Callable[[RunAgentInput, str], str]


@dataclass
class PredictStateMapping:
    """Declarative mapping telling the UI how to predict state from tool args.

    This mapping allows the frontend to optimistically update UI state based on
    tool call arguments before the tool result is received.

    Attributes:
        state_key: Key in the state object to update
        tool: Name of the tool that will update this state
        tool_argument: Name of the tool argument to use as the state value

    """

    state_key: str
    tool: str
    tool_argument: str

    def to_payload(self) -> dict[str, str]:
        """Convert to dictionary payload for PredictState events.

        Returns:
            Dictionary with state_key, tool, and tool_argument fields

        """
        # Dictionary literal syntax: `{"key": value}` creates dictionary
        return {
            "state_key": self.state_key,
            "tool": self.tool,
            "tool_argument": self.tool_argument,
        }


@dataclass
class ToolBehavior:
    """Declarative configuration for tool-specific handling.

    Attributes:
        skip_messages_snapshot: If True, skip emitting MessagesSnapshotEvent for this tool
        continue_after_frontend_call: If True, continue streaming after frontend tool call
        stop_streaming_after_result: If True, stop text streaming after tool result
        predict_state: Optional PredictState mappings for optimistic UI updates
        args_streamer: Optional async generator for streaming large tool arguments
        state_from_args: Optional hook to emit state snapshot from tool arguments
        state_from_result: Optional hook to emit state snapshot from tool result
        state_delta_from_result: Optional hook to emit state delta from tool result.
                                  Takes (context, previous_state) and returns new state.
                                  A JSON patch delta will be generated automatically.
        custom_result_handler: Optional async generator for custom events after tool result

    """

    skip_messages_snapshot: bool = False
    continue_after_frontend_call: bool = False
    stop_streaming_after_result: bool = False
    predict_state: Iterable[PredictStateMapping | None] = None
    args_streamer: ArgsStreamer | None = None
    state_from_args: StateFromArgs | None = None
    state_from_result: StateFromResult | None = None
    state_delta_from_result: StateDeltaFromResult | None = None
    custom_result_handler: CustomResultHandler | None = None


@dataclass
class StrandsAgentConfig:
    """Top-level configuration for the Strands agent adapter.

    Attributes:
        tool_behaviors: Dictionary mapping tool names to ToolBehavior configurations
        state_context_builder: Optional callable to inject state into user prompts
        silent_hook_errors: If True, silently swallow hook errors (official pattern).
                           If False, log warnings for debugging (default).

    """

    # `field(default_factory=dict)` creates new empty dict for each instance
    # `default_factory` is a function called to create default value
    # This prevents all instances from sharing the same dict (mutable default issue)
    tool_behaviors: dict[str, ToolBehavior] = field(default_factory=dict)
    state_context_builder: StateContextBuilder | None = None
    silent_hook_errors: bool = False


async def maybe_await(value: T | Awaitable[T]) -> T:
    """Await coroutine-like values produced by hook callables.

    Helper function to handle both sync and async hook implementations.
    If the value is awaitable (coroutine), awaits it; otherwise returns it as-is.

    Args:
        value: Value that may be awaitable (coroutine) or a regular value

    Returns:
        Awaited result if awaitable, otherwise the value itself

    """
    # `inspect.isawaitable()` checks if value is a coroutine/awaitable
    if inspect.isawaitable(value):
        # `await` pauses execution until coroutine completes
        result = await value
        return cast(T, result)
    # If not awaitable, return as-is (sync function result)
    return cast(T, value)


def normalize_predict_state(
    value: PredictStateMapping | Iterable[PredictStateMapping | None],
) -> list[PredictStateMapping]:
    """Normalize predict state config into a concrete list.

    Converts various input formats (None, single mapping, iterable) into
    a consistent list format for processing.

    Args:
        value: Optional iterable of PredictStateMapping objects, single mapping, or None

    Returns:
        List of PredictStateMapping objects (empty list if value is None)

    Raises:
        TypeError: If value is a str (str is iterable but not a valid container here)

    """
    if value is None:
        return []
    if isinstance(value, PredictStateMapping):
        return [value]
    # Reject str: it is iterable so list("x") would yield wrong type and break .to_payload()
    if isinstance(value, str):
        raise TypeError(
            "predict_state must be None, a PredictStateMapping, or an iterable of "
            "PredictStateMapping; got str"
        )
    return list(value)
