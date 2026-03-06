"""ART Agent — Search Relevance Testing Sub-Agent.

Orchestrates 3 specialized agents for search relevance tuning:
  - hypothesis_agent: Generate and test search improvement hypotheses
  - evaluation_agent: Offline relevance evaluation (NDCG, MAP, Precision)
  - user_behavior_analysis_agent: Analyze UBI click data
"""

from __future__ import annotations

import os

import boto3
from botocore.config import Config as BotocoreConfig
from strands import Agent
from strands.models.bedrock import BedrockModel

from agents.specialized_agents import (
    evaluation_agent,
    hypothesis_agent,
    set_opensearch_tools,
    user_behavior_analysis_agent,
)
from utils.logging_helpers import get_logger, log_info_event
from utils.mcp_connection import MCPConnectionManager

logger = get_logger(__name__)

# Default Bedrock model — same as Strands SDK default.
# Used when BEDROCK_INFERENCE_PROFILE_ARN is not explicitly set.
_DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

ORCHESTRATOR_SYSTEM_PROMPT = """You are a search relevance tuning orchestrator agent.

Your role is to help users solve search quality issues in OpenSearch by coordinating with specialized agents:

1. **user_behavior_analysis_agent**: Use this agent when users need to understand actual user engagement
   and behavior patterns. This agent analyzes UBI data including CTR metrics, click patterns, zero-click
   rates, and can identify poorly performing queries or high-engagement content.

2. **hypothesis_agent**: Use this agent when users report search relevance issues or need help
   understanding why search results are poor. This agent will analyze queries, examine results,
   check user behavior data, and generate hypotheses about how to resolve the search relevance issues
   and improve search result quality. Use this agent to run pairwise comparison experiments that
   show the quantitative difference between search configurations through change based metrics.
   A quantitative change on its own does not automatically mean improved quality.

3. **evaluation_agent**: Use this agent when users need to evaluate search relevance offline,
   compare search configurations, calculate metrics to measure search relevance
   or summarize past experiments. No other metrics than NDCG, MAP, Precision are available. Stick to these.
   Use this agent to run pointwise experiments that show the qualitative difference between search configurations
   through search quality metrics.


Your process:
1. Understand the user's request
2. Determine which specialized agent(s) to use
3. Delegate tasks to the appropriate agent(s)
4. Synthesize and present the results to the user

Apply the search relevance tuning process where possible:
1. Identify the relevance issue
2. Generate hypotheses including a first sanity check and smoke test using pairwise experiments.
3. Test the hypothesis using:
   a) Offline evaluation with pointwise experiments (judgment-based metrics)
   Choose based on what the user needs.
   Prefer offline if no offline evaluation has been done yet for the current process.
4. Go back to generating hypotheses if validation fails.

Always ask the user for confirmation before executing a step, unless the user gives you the permission
to proceed without their confirmation.

Be helpful, clear, and ensure the user gets complete answers by leveraging the right specialists.
"""


def _get_aws_session() -> boto3.Session:
    """Create a boto3 session from environment credentials."""
    return boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def _create_orchestrator_model(inference_profile_arn: str) -> BedrockModel:
    """Create a BedrockModel for the orchestrator with a 30-minute read timeout."""
    boto_config = BotocoreConfig(
        connect_timeout=60,
        read_timeout=1800,  # 30 minutes — orchestrator coordinates multiple sub-agents
        retries={"max_attempts": 0},
    )
    return BedrockModel(
        model_id=inference_profile_arn,
        boto_session=_get_aws_session(),
        boto_client_config=boto_config,
        streaming=True,
    )


async def create_art_agent(opensearch_url: str) -> Agent:
    """Create the ART orchestrator agent.

    Initializes the MCP connection to OpenSearch, configures the specialized
    sub-agents with the resulting tools, and returns the orchestrator Agent.

    Args:
        opensearch_url: OpenSearch cluster URL.

    Returns:
        A Strands Agent configured as the ART orchestrator.
    """
    # Default BEDROCK_INFERENCE_PROFILE_ARN if not set so specialized agents
    # can create their own BedrockModel instances without error.
    if not os.getenv("BEDROCK_INFERENCE_PROFILE_ARN"):
        os.environ["BEDROCK_INFERENCE_PROFILE_ARN"] = _DEFAULT_BEDROCK_MODEL_ID
        log_info_event(
            logger,
            f"BEDROCK_INFERENCE_PROFILE_ARN not set, defaulting to {_DEFAULT_BEDROCK_MODEL_ID}",
            "art_agent.default_model",
            model_id=_DEFAULT_BEDROCK_MODEL_ID,
        )

    if not os.getenv("BEDROCK_HAIKU_INFERENCE_PROFILE_ARN"):
        os.environ["BEDROCK_HAIKU_INFERENCE_PROFILE_ARN"] = _DEFAULT_BEDROCK_MODEL_ID

    inference_profile_arn = os.environ["BEDROCK_INFERENCE_PROFILE_ARN"]

    log_info_event(
        logger,
        f"Initializing ART agent with OpenSearch at {opensearch_url}",
        "art_agent.initializing",
        opensearch_url=opensearch_url,
    )

    # Initialize MCP connection and retrieve OpenSearch tools
    mcp_manager = MCPConnectionManager(opensearch_url=opensearch_url)
    opensearch_tools = await mcp_manager.initialize()

    log_info_event(
        logger,
        f"MCP initialized with {len(opensearch_tools)} OpenSearch tools",
        "art_agent.mcp_initialized",
        tool_count=len(opensearch_tools),
    )

    # Share the MCP tools with all specialized sub-agents
    set_opensearch_tools(opensearch_tools)

    # Build the orchestrator agent
    orchestrator = Agent(
        model=_create_orchestrator_model(inference_profile_arn),
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=[
            user_behavior_analysis_agent,
            hypothesis_agent,
            evaluation_agent,
        ],
    )

    log_info_event(
        logger,
        "ART agent initialized successfully",
        "art_agent.initialized",
    )

    return orchestrator
