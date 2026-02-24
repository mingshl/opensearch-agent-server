"""Fallback Agent — General OpenSearch Assistant.

A simple Strands agent with all OpenSearch MCP Server tools.
Handles general queries when no specialized sub-agent matches the page context.
"""

from __future__ import annotations

from strands import Agent

from utils.logging_helpers import get_logger, log_info_event
from utils.mcp_connection import MCPConnectionManager

logger = get_logger(__name__)

FALLBACK_SYSTEM_PROMPT = """You are a helpful OpenSearch assistant. You help users understand
and manage their OpenSearch clusters.

You have access to OpenSearch tools via the MCP Server. Use them to answer questions about:
- Cluster health and status
- Index management (list, create, delete, mappings)
- Searching and querying indices
- Cluster settings and configuration
- Node and shard information

When answering:
- Use the available tools to fetch real data from OpenSearch
- Present results clearly and concisely
- If a tool call fails, explain what went wrong and suggest alternatives
- If you don't have the right tool for a request, explain what's available
"""


async def create_fallback_agent(opensearch_url: str) -> Agent:
    """Create the fallback agent with all OpenSearch MCP tools.

    Args:
        opensearch_url: OpenSearch cluster URL.

    Returns:
        A Strands Agent configured with OpenSearch MCP tools.
    """
    mcp_manager = MCPConnectionManager(opensearch_url=opensearch_url)
    opensearch_tools = await mcp_manager.initialize()

    log_info_event(
        logger,
        f"Fallback agent initialized with {len(opensearch_tools)} MCP tools.",
        "fallback_agent.initialized",
        tool_count=len(opensearch_tools),
        opensearch_url=opensearch_url,
    )

    agent = Agent(
        system_prompt=FALLBACK_SYSTEM_PROMPT,
        tools=opensearch_tools,
    )

    return agent
