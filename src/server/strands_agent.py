"""Main StrandsAgent Class for AG-UI Protocol.

Wraps Strands agents to emit AG-UI protocol events following the official
AG-UI Strands integration pattern. Uses a pluggable agent factory to support
different sub-agents routed by the orchestrator.

**Key Components:**
- `StrandsAgent` - Main agent wrapper class

**Features:**
- Pluggable agent factory (orchestrator injects the right sub-agent per request)
- Thread-based agent caching (LRU eviction)
- State snapshot support for resuming conversations
- Error handling via event emission (AG-UI protocol compliance)
- Configurable tool behavior via StrandsAgentConfig
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Awaitable

from ag_ui.core import EventType, RunAgentInput, RunStartedEvent
from strands import Agent as StrandsAgentCore

from utils.logging_helpers import get_logger, log_debug_event, log_error_event
from server.agent_cache import LRUAgentCache
from server.config import get_config
from server.event_conversion import format_error_message, handle_initialization_error
from server.strands_agent_config import StrandsAgentConfig
from server.strands_event_handlers import StrandsEventHandlers
from server.strands_event_processing import process_strands_event_stream
from server.types import AGUIEvent
from server.utils import create_error_event

logger = get_logger(__name__)


def _extract_app_id_from_context(context: list) -> str | None:
    """Extract appId from the AG-UI context array.

    OpenSearch Dashboards sends page context as a Context entry with a JSON
    value containing ``appId`` (e.g. "discover", "explore", "home").
    This function finds the first entry whose value contains an appId.

    Args:
        context: List of AG-UI Context objects (description + value).

    Returns:
        The appId string, or None if not found.
    """
    for ctx in context:
        try:
            value = ctx.value if isinstance(ctx.value, dict) else json.loads(ctx.value)
            if isinstance(value, dict) and "appId" in value:
                app_id = value["appId"]
                log_debug_event(
                    logger,
                    f"Extracted appId='{app_id}' from AG-UI context",
                    "strands_agent.context_app_id",
                    app_id=app_id,
                )
                return app_id
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return None

# Type for the agent factory: takes page_context (optional), returns a Strands Agent
AgentFactory = Callable[[str | None], Awaitable[StrandsAgentCore]]


class StrandsAgent:
    """Strands Agent wrapper for AG-UI integration.

    Uses a pluggable agent_factory to create the appropriate sub-agent
    based on page_context from the incoming request. The orchestrator
    provides the factory function that handles routing.
    """

    def __init__(
        self,
        agent_factory: AgentFactory | None = None,
        agent: StrandsAgentCore | None = None,
        name: str = "opensearch-agent-server",
        description: str = "Multi-agent orchestrator for OpenSearch Dashboards",
        enable_thread_caching: bool = True,
        config: StrandsAgentConfig | None = None,
        cache_max_size: int | None = None,
    ) -> None:
        """Initialize StrandsAgent wrapper.

        Args:
            agent_factory: Async callable that takes page_context and returns a Strands Agent.
                          This is the primary way to create agents — provided by the orchestrator.
            agent: Optional pre-initialized Strands Agent (bypasses factory).
            name: Agent name
            description: Agent description
            enable_thread_caching: If True, caches agents per thread_id for reuse (default: True)
            config: Optional StrandsAgentConfig for customizing tool behavior
            cache_max_size: Max cached agents (default from config or 100).
        """
        self.strands_agent = agent
        self.agent_factory = agent_factory

        if cache_max_size is None:
            server_config = get_config()
            cache_max_size = server_config.agent_cache_size

        self.name = name
        self.description = description
        self._enable_thread_caching = enable_thread_caching
        self.config = config or StrandsAgentConfig()

        self._thread_agents = LRUAgentCache(max_size=cache_max_size)
        self._event_handlers = StrandsEventHandlers(self.config)

    async def _create_agent(self, page_context: str | None = None) -> StrandsAgentCore:
        """Create a new agent instance via the factory.

        Args:
            page_context: Page context from the request (used for routing).

        Returns:
            StrandsAgentCore instance

        Raises:
            RuntimeError: If no agent factory is configured.
            Exception: If agent creation fails.
        """
        if self.agent_factory is None:
            raise RuntimeError(
                "No agent_factory configured. Provide an agent_factory or a pre-initialized agent."
            )

        try:
            agent = await self.agent_factory(page_context)
            return agent
        except Exception as e:
            error_msg = str(e)
            log_error_event(
                logger,
                "Failed to create agent via factory.",
                "ag_ui.agent_factory_failed",
                error=error_msg,
                exc_info=True,
                page_context=page_context,
            )
            context_message = (
                f"Failed to create agent: {error_msg}\n\n"
                "Please check:\n"
                "- OpenSearch connection\n"
                "- LLM credentials\n"
                "- MCP connection configuration"
            )
            try:
                raise type(e)(context_message) from e
            except TypeError:
                raise RuntimeError(context_message) from e

    async def _get_or_create_agent(
        self, thread_id: str, page_context: str | None = None
    ) -> StrandsAgentCore:
        """Get or create agent for a thread, using caching if enabled.

        Args:
            thread_id: Thread identifier for caching
            page_context: Page context for routing to the right sub-agent

        Returns:
            StrandsAgentCore instance

        Raises:
            Exception: If agent creation fails
        """
        # If a pre-initialized agent is provided, use it for all threads
        if self.strands_agent is not None:
            return self.strands_agent

        # If thread caching is disabled, create a new agent each time
        if not self._enable_thread_caching:
            return await self._create_agent(page_context)

        # Use thread-based LRU caching (cache methods are internally thread-safe)
        cached_agent = await self._thread_agents.get(thread_id)
        if cached_agent is not None:
            cache_size = await self._thread_agents.size()
            log_debug_event(
                logger,
                f"Reusing cached agent: thread_id={thread_id}, cache_size={cache_size}",
                "ag_ui.reusing_cached_agent",
                thread_id=thread_id,
                cache_size=cache_size,
            )
            return cached_agent

        # Initialize new agent for this thread
        cache_size = await self._thread_agents.size()
        log_debug_event(
            logger,
            f"Creating new agent: thread_id={thread_id}, page_context={page_context}, cache_size={cache_size}",
            "ag_ui.creating_new_agent",
            thread_id=thread_id,
            page_context=page_context,
            cache_size=cache_size,
        )
        agent = await self._create_agent(page_context)
        evicted_thread_id = await self._thread_agents.put(thread_id, agent)

        if evicted_thread_id:
            cache_size = await self._thread_agents.size()
            log_debug_event(
                logger,
                f"Evicted LRU agent from cache: evicted_thread_id={evicted_thread_id}, "
                f"new_thread_id={thread_id}, cache_size={cache_size}",
                "ag_ui.evicted_lru_agent",
                evicted_thread_id=evicted_thread_id,
                new_thread_id=thread_id,
                cache_size=cache_size,
            )

        return agent

    async def cleanup_thread(self, thread_id: str) -> None:
        """Clean up cached agent for a specific thread.

        Args:
            thread_id: Thread identifier to clean up

        """
        if await self._thread_agents.remove(thread_id):
            cache_size = await self._thread_agents.size()
            log_debug_event(
                logger,
                f"Cleaned up cached agent: thread_id={thread_id}, cache_size={cache_size}",
                "ag_ui.cleaned_up_cached_agent",
                thread_id=thread_id,
                cache_size=cache_size,
            )

    async def cleanup_all_threads(self) -> int:
        """Clean up all cached thread agents.

        Returns:
            Number of agents that were cleaned up
        """
        thread_count = await self._thread_agents.clear()
        log_debug_event(
            logger,
            f"Cleaned up cached thread agents: count={thread_count}",
            "ag_ui.cleaned_up_all_cached_agents",
            count=thread_count,
        )
        return thread_count

    async def run(self, input_data: RunAgentInput) -> AsyncIterator[AGUIEvent]:
        """Run the Strands agent and yield AG-UI events.

        This method follows the AG-UI protocol pattern where errors are communicated
        via error events in the stream rather than raising exceptions. This allows
        the event stream to close gracefully after emitting error events, enabling
        proper cleanup and error handling on the client side.

        **Error Handling Behavior:**
        - Exceptions are caught and converted to RunErrorEvent objects
        - Error events are yielded in the event stream
        - Exceptions are **not** re-raised (by design, following AG-UI protocol)
        - All error paths emit appropriate error events:
          - Initialization errors: RunErrorEvent + RunFinishedEvent
          - Runtime errors: RunErrorEvent (stream closes after error)
        - The caller can detect failures by checking for RunErrorEvent in the stream

        **Event Sequence:**
        - On success: RunStartedEvent, then various AG-UI events, then RunFinishedEvent
        - On initialization error: RunErrorEvent, then RunFinishedEvent (no RunStartedEvent)
        - On runtime error: RunStartedEvent, then events until failure, then RunErrorEvent (stream closes)

        **Frontend Tool Handling:**
        - Frontend tools (identified via `input_data.tools`) generate unique UUIDs
        - After frontend tool calls, the stream halts (unless `continue_after_frontend_call` is set)
        - Remaining Strands events are consumed silently to allow proper generator cleanup
        - This matches the official AG-UI Strands integration pattern

        Args:
            input_data: RunAgentInput with thread_id, run_id, messages, etc.
                Must include thread_id and run_id; the AG-UI RunAgentInput type
                (ag_ui.core) requires both. API callers use ValidatedRunAgentInput
                which enforces them and returns 422 when missing.

        Yields:
            AG-UI Event objects (RunStartedEvent, various events, RunFinishedEvent or RunErrorEvent)

        Note:
            This method never raises exceptions. All errors are communicated via
            RunErrorEvent objects in the event stream. Callers should check for
            RunErrorEvent to detect failures.

        Example Usage:
            async for event in strands_agent.run(input_data):
                # Process each event as it's yielded
                process_event(event)

        """
        # Extract thread/run IDs (AG-UI RunAgentInput requires both)
        thread_id = input_data.thread_id
        run_id = input_data.run_id

        # Extract page_context for routing.
        # Strategy:
        #   1. Check forwardedProps.page_context (direct override, useful for curl testing)
        #   2. Check AG-UI context array for page context with appId (sent by Dashboard)
        page_context = None
        if hasattr(input_data, "forwarded_props") and input_data.forwarded_props:
            page_context = input_data.forwarded_props.get("page_context")

        if not page_context and hasattr(input_data, "context") and input_data.context:
            page_context = _extract_app_id_from_context(input_data.context)

        # Get or create agent (with thread-based caching + page_context routing)
        try:
            agent = await self._get_or_create_agent(thread_id, page_context=page_context)
        except Exception as e:
            # If initialization fails, yield error events instead of raising exception
            async for event in handle_initialization_error(thread_id, run_id, e):
                yield event
            return

        # Start run
        yield RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=thread_id,
            run_id=run_id,
        )

        try:
            # Process event stream using extracted event processing logic
            async for event in process_strands_event_stream(
                agent=agent,
                input_data=input_data,
                thread_id=thread_id,
                run_id=run_id,
                config=self.config,
                event_handlers=self._event_handlers,
            ):
                yield event

        except Exception as e:
            error_message = str(e)
            log_error_event(
                logger,
                "StrandsAgent.run error.",
                "ag_ui.strands_agent_run_error",
                error=error_message,
                exc_info=True,
                thread_id=thread_id,
                run_id=run_id,
            )
            # Emit error event to communicate failure to the client
            # Following AG-UI protocol: errors are communicated via events, not exceptions
            yield create_error_event(
                message=format_error_message("Error processing request", error_message),
                code="STRANDS_ERROR",
            )
