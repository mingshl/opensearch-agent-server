"""State Management for AG-UI Protocol.

Handles state snapshot restoration and state context injection for resuming
conversations and maintaining context across agent runs.

**Key Components:**
- `handle_state_snapshot()` - Handle state snapshot emission if provided
- `apply_state_context()` - Apply state_context_builder if configured

**Features:**
- State snapshot validation and sanitization
- State context injection into user messages
- Resume operation support
"""

from __future__ import annotations

from ag_ui.core import EventType, RunAgentInput, StateSnapshotEvent

from utils.logging_helpers import (
    get_logger,
    log_debug_event,
    log_error_event,
    log_warning_event,
)
from server.constants import DEFAULT_USER_MESSAGE
from server.strands_agent_config import StrandsAgentConfig
from server.utils import validate_state_snapshot

logger = get_logger(__name__)


def handle_state_snapshot(
    input_data: RunAgentInput, thread_id: str
) -> tuple[bool, StateSnapshotEvent | None]:
    """Handle state snapshot emission if provided in input_data.

    Args:
        input_data: RunAgentInput with potential state
        thread_id: Thread identifier for logging

    Returns:
        Tuple of (is_resume: bool, StateSnapshotEvent or None)

    """
    is_resume = False
    snapshot_event = None

    if hasattr(input_data, "state") and input_data.state is not None:
        if isinstance(input_data.state, dict):
            # Filter out messages from state to avoid "Unknown message role" errors
            raw_state = {k: v for k, v in input_data.state.items() if k != "messages"}
            if raw_state:
                # Validate and sanitize state snapshot
                try:
                    state_snapshot, warnings = validate_state_snapshot(raw_state)

                    # Log warnings for any filtered values
                    if warnings:
                        for warning in warnings:
                            log_warning_event(
                                logger,
                                warning,
                                "ag_ui.state_snapshot_validation_warning",
                                thread_id=thread_id,
                            )

                    if state_snapshot:
                        snapshot_event = StateSnapshotEvent(
                            type=EventType.STATE_SNAPSHOT, snapshot=state_snapshot
                        )
                        is_resume = True
                        log_debug_event(
                            logger,
                            f"Restored state snapshot: thread_id={thread_id}, keys={list(state_snapshot.keys())}",
                            "ag_ui.restored_state_snapshot",
                            thread_id=thread_id,
                            keys=list(state_snapshot.keys()),
                        )
                    else:
                        log_warning_event(
                            logger,
                            f"State snapshot validation filtered out all values: thread_id={thread_id}",
                            "ag_ui.state_snapshot_all_filtered",
                            thread_id=thread_id,
                            original_keys=list(raw_state.keys()),
                        )
                except Exception as e:
                    log_error_event(
                        logger,
                        f"Error validating state snapshot: thread_id={thread_id}, error={e}. "
                        "Skipping state snapshot restoration.",
                        "ag_ui.state_snapshot_validation_error",
                        error=str(e),
                        exc_info=True,
                        thread_id=thread_id,
                    )
        else:
            log_warning_event(
                logger,
                f"Expected state to be dict, got {type(input_data.state)}: thread_id={thread_id}. "
                "Skipping state snapshot.",
                "ag_ui.invalid_state_type",
                thread_id=thread_id,
                state_type=str(type(input_data.state)),
            )

    return is_resume, snapshot_event


def apply_state_context(
    config: StrandsAgentConfig,
    input_data: RunAgentInput,
    user_message: str,
    thread_id: str,
    is_resume: bool,
) -> str:
    """Apply state_context_builder if configured.

    Args:
        config: StrandsAgentConfig with optional state_context_builder
        input_data: RunAgentInput
        user_message: Current user message
        thread_id: Thread identifier for logging
        is_resume: Whether this is a resume operation

    Returns:
        Modified user message with state context injected

    """
    if config.state_context_builder:
        try:
            user_message = config.state_context_builder(input_data, user_message)
            if not isinstance(user_message, str):
                log_warning_event(
                    logger,
                    f"state_context_builder returned {type(user_message)}, expected str: thread_id={thread_id}. "
                    "Converting to string.",
                    "ag_ui.invalid_state_context_builder_return_type",
                    thread_id=thread_id,
                    return_type=str(type(user_message)),
                )
                user_message = (
                    str(user_message) if user_message else DEFAULT_USER_MESSAGE
                )
            if is_resume:
                log_debug_event(
                    logger,
                    f"Injected state context into prompt: thread_id={thread_id}",
                    "ag_ui.injected_state_context",
                    thread_id=thread_id,
                )
        except Exception as e:
            if not config.silent_hook_errors:
                log_error_event(
                    logger,
                    f"Error in state_context_builder: thread_id={thread_id}, error={e}. Using original message.",
                    "ag_ui.state_context_builder_error",
                    error=str(e),
                    exc_info=True,
                    thread_id=thread_id,
                )
    return user_message
