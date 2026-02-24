"""Multi-Agent Event Handler for StrandsAgent.

Handles multi-agent events from the Strands agent and converts them to AG-UI events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ag_ui.core import (
    EventType,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

from utils.logging_helpers import (
    get_logger,
    log_debug_event,
)
from server.types import EventResult, HandlerResult, StrandsEvent

logger = get_logger(__name__)


class MultiAgentEventHandler:
    """Handler for multi-agent events from the Strands agent.

    Converts Strands multi-agent events (node start, node stop, handoff) into AG-UI tool call events.
    """

    async def handle_multi_agent_events(
        self, event: StrandsEvent, message_id: str
    ) -> AsyncIterator[HandlerResult]:
        """Handle multi-agent events (node start, node stop, handoff).

        Args:
            event: Strands event dictionary
            message_id: Current message ID

        Yields:
            EventResult wrapping AG-UI tool call events (TOOL_CALL_START, TOOL_CALL_END).

        """
        if "multiagent_node_start" in event:
            node_info = event["multiagent_node_start"]
            node_id = node_info.get("node_id", "unknown")
            node_type = node_info.get("node_type", "agent")
            log_debug_event(
                logger,
                f"Multi-agent node start: node_id={node_id}, node_type={node_type}",
                "ag_ui.multi_agent_node_start",
                node_id=node_id,
                node_type=node_type,
            )

            yield EventResult(
                event=ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=f"agent_{node_id}",
                    tool_call_name=f"{node_type}_{node_id}",
                    parent_message_id=message_id,
                ),
                should_stop=False,
            )

        elif "multiagent_node_stop" in event:
            node_info = event["multiagent_node_stop"]
            node_id = node_info.get("node_id", "unknown")

            yield EventResult(
                event=ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=f"agent_{node_id}",
                ),
                should_stop=False,
            )

        elif "multiagent_handoff" in event:
            handoff = event["multiagent_handoff"]
            to_nodes = handoff.get("to_node_ids", [])

            for to_node in to_nodes:
                yield EventResult(
                    event=ToolCallStartEvent(
                        type=EventType.TOOL_CALL_START,
                        tool_call_id=f"handoff_{to_node}",
                        tool_call_name="agent_handoff",
                        parent_message_id=message_id,
                    ),
                    should_stop=False,
                )
