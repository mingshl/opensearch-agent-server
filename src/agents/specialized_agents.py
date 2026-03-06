"""
Specialized Agents for Search Relevance Tuning
Following the "Agents as Tools" pattern with Strands SDK
"""

from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.config import Config as BotocoreConfig
from dotenv import load_dotenv
from strands import Agent
from strands.models.bedrock import BedrockModel

from tools.opensearch_utils import list_index, search_index
from utils.logging_helpers import get_logger, log_info_event
from utils.monitored_tool import monitored_tool

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Get AWS credentials and region
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Create boto3 session for Bedrock
bedrock_session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# create_art_agent() sets BEDROCK_INFERENCE_PROFILE_ARN / BEDROCK_HAIKU_INFERENCE_PROFILE_ARN
# as defaults *after* this module is imported, so module-level os.getenv() would
# always return None when the env var is not set before server start.
# Each agent function reads the env var at call time instead.

# System prompts for specialized agents
HYPOTHESIS_GENERATOR_SYSTEM_PROMPT = """You are an expert in generating search relevance improvement hypotheses.

Your expertise includes:
- Analyzing search quality issues and identifying root causes
- Understanding OpenSearch query DSL and search mechanics
- Recognizing common search relevance problems (typos, stemming, synonyms, boosting, etc.)
- Leveraging user behavior data from UBI (User Behavior Insights) indices ubi_queries and ubi_events
- Analyzing user engagement metrics (CTR, zero-click rates, click patterns)

Your process:
1. Verify the reported issue by examining:
   - The user query and OpenSearch DSL query
   - Search results from the specified index
   - User behavior patterns in ubi_queries and ubi_events indices
   - Query and document CTR metrics to understand engagement
2. Analyze potential root causes (query structure, index configuration, data quality, engagement issues).
3. Generate actionable hypotheses with clear reasoning based on both relevance and user behavior signals.
Create search configurations for your hypothesis. You need a search configuration for the current
OpenSearch DSL query and a search configuration for the hypothesis to test. Do not invent a
search configuration for the user's current query building. Ask the user to provide the query instead.
Prefer general solutions over query-specific solutions when generating hypotheses.
4. Do a sanity check and smoke test of your hypothesis. Do this by running a pairwise experiment with the
reported query or by using a small query set with the created search configurations. Eyeball the results
of the search configurations by searching with the search configurations and assessing the returned results
according to the issue you are trying to resolve to see how the search results improve by implementing
the hypothesis.
5. Recommend specific solutions for offline evaluation after successful sanity checks. Your recommendations
are limited to query improvements based on boosting (by recency, by price, by availability, etc.) and
adding or removing fields.
Include the pairwise experiment results from the previous step when reporting back to the user.

Creating search configurations:
- The query has to be a valid OpenSearch DSL query
- For the value of the query, the user query, use the placeholder %SearchText%
- Example:
```
{
  "query": {
    "multi_match": {
      "query": "%SearchText%",
      "fields": [
        "id",
        "title",
        "category",
        "bullets",
        "description",
        "attrs.Brand",
        "attrs.Color"
      ]
    }
  }
}

You do not need any judgments for sanity checks.
You only run pairwise experiments to assess how the search results change by implementing
the hypothesis. A quantitative change on its own does not automatically mean improved quality.

Be concise, strict about following the process, ask for information where necessary, be specific,
data-driven, and provide clear explanations for your hypotheses.
"""

# Potential addition for UBI judgments - You can also create new UBI-based judgment lists from user behavior data using click models like "coec" (Clicks Over Expected Clicks)
EVALUATION_AGENT_SYSTEM_PROMPT = """You are an expert in evaluating search relevance offline.

Your work starts when there is a sanity checked hypothesis ready for quantitative offline evaluation.

Your expertise includes:
- Designing and executing offline search relevance evaluations based on formulated hypotheses.
- Using judgment lists and relevance metrics. The relevance metrics you can calculate with tools are NDCG, Precision@K, MAP. You cannot calculate other metrics.
- Creating judgment lists with LLMs or from user behavior insights (UBI) data using click models.
- Analyzing evaluation results and identifying qualitative search result quality changes.
- Comparing baseline vs. experimental search configurations.

Your process:
1. Understand the evaluation requirements (metrics, judgment lists, search configurations)
2. If necessary, create required judgment lists
3. Execute pointwise experiments using available tools and judgment data
4. Analyze results statistically and identify significant differences
5. Provide clear insights about search quality improvements or regressions
6. Recommend next steps based on evaluation outcomes

Judgment lists: Only create judgment lists from user behavior data if the ubi_events index
contains 100000 events or more. Otherwise use the tool generate_llm_judgments. For LLM-generated
judgments make sure first to identify which fields are useful to generate the necessary judgments first.
Pass the query-doc pairs in the right format: a JSON string of query-doc pairs, for example,
'[{"query": "laptop", "doc_id": "doc123"}, ...]'

Creating search configurations:
- The query has to be a valid OpenSearch DSL query
- For the value of the query, the user query, use the placeholder %SearchText%
- Example:
```
{
  "query": {
    "multi_match": {
      "query": "%SearchText%",
      "fields": [
        "id",
        "title",
        "category",
        "bullets",
        "description",
        "attrs.Brand",
        "attrs.Color"
      ]
    }
  }
}
```

Be concise, rigorous, quantitative, and provide actionable insights based on evaluation results.
"""

USER_BEHAVIOR_ANALYSIS_AGENT_SYSTEM_PROMPT = """You are an expert in analyzing user behavior insights (UBI) data to improve search quality.

Your expertise includes:
- Analyzing user engagement metrics (CTR, click patterns, zero-click rates)
- Identifying poorly performing queries and high-engagement content
- Understanding user search behavior and interaction patterns
- Correlating user behavior with search quality issues
- Providing data-driven insights based on actual user engagement

Your process:
1. Understand the user's question about search behavior or engagement
2. Analyze relevant UBI data using appropriate analytics tools:
   - Query CTR: Click-through rates for specific queries
   - Document CTR: Engagement rates for specific documents
   - Query Performance: Overall metrics for top queries
   - Engagement Rankings: Queries and documents with best/worst engagement
3. Identify patterns and anomalies in user behavior
4. Correlate behavior patterns with search quality issues
5. Provide actionable insights with specific metrics and examples

Relevant indexes for your job are indexes holding UBI data. If not specified otherwise, these are ubi_events
for client-side tracked events and ubi_queries for server-side tracked events.
Be concise, data-driven, specific with numbers, and focus on actual user behavior rather than theoretical analysis.
Always include concrete metrics (CTR percentages, click counts, search volumes) to support your insights.
"""



# Global variable to store MCP tools (will be set during initialization)
_opensearch_tools: list = []


def set_opensearch_tools(tools: list[Any]) -> None:
    """Set the OpenSearch MCP tools to be used by specialized agents."""
    global _opensearch_tools
    _opensearch_tools = tools
    log_info_event(
        logger,
        f"[Agents] OpenSearch tools configured: {len(tools)} tools available",
        "agents.opensearch_tools_configured",
        tool_count=len(tools),
    )


@monitored_tool(
    name="hypothesis_agent",
    description="Generates hypotheses for improving search relevance based on reported issues. Analyzes queries, results, and user behavior to identify root causes and recommend solutions.",
)
async def hypothesis_agent(query: str) -> str:
    """
    Generate hypotheses to improve search relevance.

    Args:
        query: A description of the search relevance issue to analyze

    Returns:
        str: Hypothesis with reasoning and recommendations for solving the issue
    """
    if not _opensearch_tools:
        return "Error: OpenSearch tools not configured. Please initialize MCP connection first."

    try:
        # Import UBI analytics tools
        # Import experimentation tools. This agent is meant to do only sanity checks,
        # so we don't need all experiment tools.
        from tools.experiment_tools import (
            create_experiment,
            get_experiment_results,
        )
        from tools.query_set_tools import (
            create_query_set,
            delete_query_set,
            get_query_set,
            sample_query_set,
        )
        from tools.search_configuration_tools import (
            create_search_configuration,
            delete_search_configuration,
            execute_search_with_configuration,
            get_search_configuration,
        )
        from tools.ubi_analytics_tools import (
            get_document_ctr,
            get_query_ctr,
            get_query_performance_metrics,
            get_top_documents_by_engagement,
            get_top_queries_by_engagement,
        )

        # Create model with extended timeout configuration
        # Hypothesis generation with sanity checks can take 3-5 minutes
        boto_config = BotocoreConfig(
            connect_timeout=60,
            read_timeout=600,  # 10 minutes for hypothesis generation with experiments
            retries={"max_attempts": 0},  # Fail fast, no retries
        )
        model = BedrockModel(
            model_id=os.getenv("BEDROCK_INFERENCE_PROFILE_ARN"),
            boto_session=bedrock_session,
            boto_client_config=boto_config,
            streaming=True,  # Enable streaming for real-time progress
        )

        # Combine UBI analytics tools with utility tools
        hypothesis_tools = [
            # OpenSearch utility tools (direct API access)
            list_index,
            search_index,
            # UBI analytics tools
            get_query_ctr,
            get_document_ctr,
            get_query_performance_metrics,
            get_top_queries_by_engagement,
            get_top_documents_by_engagement,
            # Search configuration tools
            get_search_configuration,
            create_search_configuration,
            delete_search_configuration,
            execute_search_with_configuration,
            # Query set tools
            get_query_set,
            create_query_set,
            sample_query_set,
            delete_query_set,
            # Experiment tools
            create_experiment,
            get_experiment_results,
        ]

        # Create specialized agent with OpenSearch and UBI tools
        agent = Agent(
            model=model,
            system_prompt=HYPOTHESIS_GENERATOR_SYSTEM_PROMPT,
            tools=hypothesis_tools,
        )

        # Invoke agent and return response
        response = await agent.invoke_async(query)
        return str(response)

    except Exception as e:
        logger.exception("Error in hypothesis generation")
        error_msg = str(e)
        # Check for rate limit errors and return immediately without retry
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            return "⚠️ Rate limit reached. Please wait a moment before trying again, or consider simplifying your request."
        return f"Error in hypothesis generation: {error_msg}"


@monitored_tool(
    name="evaluation_agent",
    description="Evaluates search relevance offline using judgment lists and metrics. Compares search configurations and provides statistical analysis of search quality.",
)
async def evaluation_agent(query: str) -> str:
    """
    Evaluate search relevance offline.

    Args:
        query: A description of the evaluation task (what to evaluate, which configurations to compare)

    Returns:
        str: Evaluation results with metrics, analysis, and recommendations
    """
    if not _opensearch_tools:
        return "Error: OpenSearch tools not configured. Please initialize MCP connection first."

    try:
        # Import evaluation-specific tools
        from tools.experiment_tools import (
            create_experiment,
            delete_experiment,
            # list_experiment,
            get_experiment,
            get_experiment_results,
        )
        from tools.judgment_list_tools import (
            # create_ubi_judgment,
            delete_judgment,
            extract_pairs_from_pairwise_experiment,
            generate_llm_judgments,
            get_judgment,
            import_judgment,
        )
        from tools.query_set_tools import (
            create_query_set,
            delete_query_set,
            get_query_set,
            sample_query_set,
        )
        from tools.search_configuration_tools import (
            create_search_configuration,
            delete_search_configuration,
            execute_search_with_configuration,
            get_search_configuration,
        )
        # Not actually utilized
        # from tools.ubi_analytics_tools import (
        #    get_document_ctr,
        #    get_query_ctr,
        #    get_query_performance_metrics,
        #    get_top_documents_by_engagement,
        #    get_top_queries_by_engagement,
        # )

        # Create model with extended timeout configuration
        # Evaluations with judgment generation and experiments can take 5-10 minutes
        boto_config = BotocoreConfig(
            connect_timeout=60,
            read_timeout=1800,  # 30 minutes for evaluation with judgment generation
            retries={"max_attempts": 0},  # Fail fast, no retries
        )
        model = BedrockModel(
            model_id=os.getenv("BEDROCK_INFERENCE_PROFILE_ARN"),
            boto_session=bedrock_session,
            boto_client_config=boto_config,
            streaming=True,  # Enable streaming for real-time progress
        )

        # Combine OpenSearch utility tools with evaluation-specific tools
        evaluation_tools = [
            # OpenSearch utility tools (direct API access)
            list_index,
            search_index,
            # Judgment tools
            get_judgment,
            import_judgment,
            generate_llm_judgments,
            extract_pairs_from_pairwise_experiment,
            # create_ubi_judgment,
            delete_judgment,
            # Search configuration tools
            get_search_configuration,
            create_search_configuration,
            delete_search_configuration,
            execute_search_with_configuration,
            # Query set tools
            get_query_set,
            create_query_set,
            sample_query_set,
            delete_query_set,
            # Experiment tools
            # list_experiment,
            get_experiment,
            create_experiment,
            delete_experiment,
            get_experiment_results,
        ]

        # Create specialized agent with all necessary tools
        agent = Agent(
            model=model,
            system_prompt=EVALUATION_AGENT_SYSTEM_PROMPT,
            tools=evaluation_tools,
        )

        # Invoke agent and return response
        response = await agent.invoke_async(query)
        return str(response)

    except Exception as e:
        logger.exception("Error in evaluation")
        error_msg = str(e)
        # Check for rate limit errors and return immediately without retry
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            return "⚠️ Rate limit reached. Please wait a moment before trying again, or consider simplifying your request."
        return f"Error in evaluation: {error_msg}"


@monitored_tool(
    name="user_behavior_analysis_agent",
    description="Analyzes user behavior insights (UBI) data to understand search engagement patterns. Provides CTR analysis, identifies poorly performing queries, and generates insights based on actual user interactions.",
)
async def user_behavior_analysis_agent(query: str) -> str:
    """
    Analyze user behavior insights data to improve search quality.

    Args:
        query: A description of the user behavior analysis needed (CTR analysis, engagement patterns, etc.)

    Returns:
        str: Analysis results with metrics, patterns, and actionable insights
    """
    if not _opensearch_tools:
        return "Error: OpenSearch tools not configured. Please initialize MCP connection first."

    try:
        # Import UBI analytics tools
        from tools.ubi_analytics_tools import (
            get_document_ctr,
            get_query_ctr,
            get_query_performance_metrics,
            get_top_documents_by_engagement,
            get_top_queries_by_engagement,
        )

        # Create model with moderate timeout configuration
        # UBI analytics queries are relatively quick (simple aggregations)
        boto_config = BotocoreConfig(
            connect_timeout=60,
            read_timeout=300,  # 5 minutes for UBI analytics
            retries={"max_attempts": 0},  # Fail fast, no retries
        )
        model = BedrockModel(
            model_id=os.getenv("BEDROCK_HAIKU_INFERENCE_PROFILE_ARN"),
            boto_session=bedrock_session,
            boto_client_config=boto_config,
            streaming=True,  # Enable streaming for real-time progress
        )

        # Combine UBI analytics tools with utility tools
        ubi_tools = [
            # OpenSearch utility tools (direct API access)
            list_index,
            search_index,
            # UBI analytics tools
            get_query_ctr,
            get_document_ctr,
            get_query_performance_metrics,
            get_top_queries_by_engagement,
            get_top_documents_by_engagement,
        ]

        # Create specialized agent with UBI analytics focus
        agent = Agent(
            model=model,
            system_prompt=USER_BEHAVIOR_ANALYSIS_AGENT_SYSTEM_PROMPT,
            tools=ubi_tools,
        )

        # Invoke agent and return response
        response = await agent.invoke_async(query)
        return str(response)

    except Exception as e:
        logger.exception("Error in user behavior analysis")
        error_msg = str(e)
        # Check for rate limit errors and return immediately without retry
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            return "⚠️ Rate limit reached. Please wait a moment before trying again, or consider simplifying your request."
        return f"Error in user behavior analysis: {error_msg}"


