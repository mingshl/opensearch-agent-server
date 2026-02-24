"""AG-UI Server Package.

Provides AG-UI protocol server for the multi-agent system.
"""

# Re-export public API at package level for cleaner imports.
# Allows: from server import StrandsAgent, StrandsAgentConfig, ToolBehavior
from server.strands_agent import StrandsAgent
from server.event_conversion import format_error_message
from server.strands_agent_config import (
    StrandsAgentConfig,
    ToolBehavior,
    ToolCallContext,
    ToolResultContext,
    PredictStateMapping,
    maybe_await,
    normalize_predict_state,
)

__all__ = [
    "StrandsAgent",
    "format_error_message",
    "StrandsAgentConfig",
    "ToolBehavior",
    "ToolCallContext",
    "ToolResultContext",
    "PredictStateMapping",
    "maybe_await",
    "normalize_predict_state",
]
