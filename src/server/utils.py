"""Utility functions for AG-UI server.

This module contains helper functions to reduce code duplication and improve
maintainability across the server codebase.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast

from ag_ui.core import EventType, RunErrorEvent
from fastapi import Request

from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_info_event,
    log_warning_event,
)
from server.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_INITIAL_DELAY,
    DEFAULT_RETRY_MAX_DELAY_SHORT,
)
from server.types import AGUIEvent

logger = get_logger(__name__)

# TypeVar for generic operation results
T = TypeVar("T")


class EventLike(Protocol):
    """Protocol for objects that have an event type attribute.

    This protocol is used for structural typing - any object that has
    a 'type' or 'eventType' attribute (even if None) will match.
    """

    type: EventType | str | None
    eventType: EventType | str | None  # noqa: N815  # camelCase to match AG-UI protocol


def get_event_type_from_object(
    event: EventLike | dict[str, Any] | AGUIEvent | object,
) -> EventType | None:
    """Extract event type from event object, normalizing to EventType enum.

    Handles cases where event objects may use either 'type' or 'eventType' as the attribute name,
    and normalizes string values to EventType enum. This ensures internal code always works with
    EventType enum values.

    Args:
        event: Event object (can be Pydantic model, dict, or any object with type/eventType attribute)

    Returns:
        EventType enum value, or None if neither attribute exists or value cannot be normalized

    """
    # Use explicit None check to avoid falling back when first attribute exists but is falsy
    event_type = getattr(event, "type", None)
    if event_type is None:
        event_type = getattr(event, "eventType", None)

    # If already an EventType enum, return as-is
    if isinstance(event_type, EventType):
        return event_type

    # If None, return None
    if event_type is None:
        return None

    # If string, try to convert to EventType enum
    if isinstance(event_type, str):
        # Handle cases like "EventType.TOOL_CALL_START" -> "TOOL_CALL_START"
        type_name = event_type.split(".")[-1] if "." in event_type else event_type
        try:
            return EventType[type_name]
        except (KeyError, AttributeError):
            # Invalid event type string - log warning and return None
            log_warning_event(
                logger,
                f"Invalid event type string: {event_type}",
                "ag_ui.invalid_event_type_string",
                event_type=event_type,
            )
            return None

    # Unknown type - return None
    return None


def create_error_event(message: str, code: str) -> RunErrorEvent:
    """Create a RunErrorEvent with consistent structure.

    Helper function to create error events with standardized format,
    reducing code duplication across the codebase.

    Args:
        message: Error message (can be plain text or formatted markdown)
        code: Error code (e.g., "ENCODING_ERROR", "RUN_ERROR", "INITIALIZATION_ERROR")

    Returns:
        RunErrorEvent instance with the specified message and code

    """
    return RunErrorEvent(
        type=EventType.RUN_ERROR,
        message=message,
        code=code,
    )


def get_tool_call_id(tool_data: dict, generate_default: bool = False) -> str | None:
    """Extract tool call ID from dictionary, handling both naming conventions.

    Handles cases where tool data dictionaries may use either 'toolUseId' or 'tool_use_id'
    as the key name, providing a consistent way to extract the tool call ID regardless
    of naming convention.

    Args:
        tool_data: Dictionary containing tool call data
        generate_default: If True and no ID is found, generate a UUID. If False, return None.

    Returns:
        The tool call ID as a string, or None if not found and generate_default is False,
        or a generated UUID string if not found and generate_default is True

    Example:
        ```python
        # With camelCase key
        tool_data = {"toolUseId": "call-123"}
        tool_id = get_tool_call_id(tool_data)  # Returns "call-123"

        # With snake_case key
        tool_data = {"tool_use_id": "call-456"}
        tool_id = get_tool_call_id(tool_data)  # Returns "call-456"

        # Missing ID, generate default
        tool_data = {}
        tool_id = get_tool_call_id(tool_data, generate_default=True)  # Returns UUID string
        ```

    """
    # Use explicit None check to avoid falling back when first key exists but is falsy (e.g., empty string)
    tool_id = tool_data.get("toolUseId")
    if tool_id is None:
        tool_id = tool_data.get("tool_use_id")
    if tool_id:
        # Cast to str since we've verified it's truthy (non-empty string)
        return str(tool_id)
    if generate_default:
        return str(uuid.uuid4())
    return None


def validate_state_snapshot(state: dict) -> tuple[dict, list[str]]:
    """Validate and sanitize state snapshot dictionary.

    Ensures state snapshot values are JSON-serializable and filters out
    problematic values that could cause issues downstream. Applies
    security-oriented sanitization (control chars, depth/size limits, key
    validation) before JSON-serializability checks. Logs warnings for
    any filtered values.

    Args:
        state: State snapshot dictionary to validate

    Returns:
        Tuple of (sanitized_state: dict, warnings: list[str])
        - sanitized_state: State dict with only JSON-serializable, sanitized values
        - warnings: List of warning messages for filtered values

    Example:
        ```python
        state = {
            "key1": "value1",
            "key2": 123,
            "key3": {"nested": "data"},
            "key4": lambda x: x,  # Non-serializable
        }
        sanitized, warnings = validate_state_snapshot(state)
        # sanitized = {"key1": "value1", "key2": 123, "key3": {"nested": "data"}}
        # warnings = ["Non-serializable value for key 'key4' (type: function) filtered..."]
        ```

    """
    from server.sanitization import sanitize_for_state_snapshot, sanitize_key

    # Security sanitization: control chars, depth/size limits, key validation
    sanitized_by_security = sanitize_for_state_snapshot(state)

    sanitized_state = {}
    warnings = []

    for key in state:
        if key not in sanitized_by_security:
            # Key was dropped by sanitization: bad key or non-serializable value
            if not isinstance(key, str):
                warnings.append(
                    f"Non-string key '{key}' (type: {type(key).__name__}) filtered from state snapshot"
                )
            elif sanitize_key(key) is None:
                warnings.append(
                    f"Invalid key '{key}' (invalid characters or length) filtered from state snapshot"
                )
            else:
                try:
                    json.dumps(state[key])
                except (TypeError, ValueError) as e:
                    warnings.append(
                        f"Non-serializable value for key '{key}' (type: {type(state[key]).__name__}) filtered from state snapshot: {str(e)}"
                    )
            continue

        value = sanitized_by_security[key]
        # Ensure key is a string (sanity; keys from sanitization are always str)
        if not isinstance(key, str):
            warnings.append(
                f"Non-string key '{key}' (type: {type(key).__name__}) filtered from state snapshot"
            )
            continue

        # Check if value is JSON-serializable
        try:
            json.dumps(value)  # Test if value can be serialized
            sanitized_state[key] = value
        except (TypeError, ValueError) as e:
            warnings.append(
                f"Non-serializable value for key '{key}' (type: {type(value).__name__}) filtered from state snapshot: {str(e)}"
            )

    return sanitized_state, warnings


def get_event_type_name(event_type: EventType) -> str:
    """Extract event type name from EventType enum.

    Converts EventType enum to its string name. This is used at API boundaries
    (e.g., persistence, logging) where string representation is needed.

    Args:
        event_type: The EventType enum value

    Returns:
        The event type name as a string (e.g., "TOOL_CALL_START")

    """
    return event_type.name


def is_event_type(event: AGUIEvent | dict[str, Any], target_type: EventType) -> bool:
    """Check if an event object matches a target event type.

    Convenience function that extracts event type from an event object and compares
    it to a target EventType enum. This is the most common pattern when checking
    event types in handlers. The function normalizes string types to EventType enum
    internally, ensuring consistent enum-based comparison.

    Args:
        event: Event object (can be Pydantic model, dict, or any object with type/eventType attribute)
        target_type: The target EventType enum to compare against

    Returns:
        True if the event's type matches the target_type, False otherwise

    Example:
        if is_event_type(event, EventType.TEXT_MESSAGE_START):
            # Handle text message start

    """
    event_type = get_event_type_from_object(event)
    if event_type is None:
        return False
    return event_type == target_type


def get_user_id_from_request(request: Request) -> str:
    """Extract user ID from a FastAPI request.

    Attempts to get user ID from:
    1. Request state (set by authentication middleware if authenticated)
    2. X-User-Id header
    3. Authorization header (hash of the header as a partitioning identifier when
       no proper user ID is available; see Note below)
    4. Request client host (as fallback)

    Args:
        request: FastAPI Request object

    Returns:
        User ID string extracted using the fallback chain described above.
        Always returns a non-empty string.

    Note:
        The hashed Authorization-header fallback (step 3) is a
        **non-cryptographic, partitioning-only identifier**. It is used solely
        to partition data (e.g., per-request or per-token scoping) when a
        real identity is not available. It is **not** a security identity and
        must not be relied upon for authentication or authorization decisions.

        The X-User-Id header (step 2) is **trusted** when the header auth
        strategy is used. That is only safe when the server is behind a
        trusted proxy or frontend (e.g. OpenSearch Dashboards) that validates
        and sets X-User-Id. When not behind such a frontend, anyone can
        spoof X-User-Id. See auth middleware and AG-UI auth documentation.

    """
    # Try request state first (set by authentication middleware)
    if hasattr(request.state, "user_id") and request.state.user_id:
        return request.state.user_id

    # Try X-User-Id header
    user_id = request.headers.get("X-User-Id")
    if user_id:
        return user_id  # Early return - exit function immediately

    # Try to extract from Authorization header (partitioning identifier when
    # no proper user ID is available; see get_user_id_from_request docstring)
    auth_header = request.headers.get("Authorization")
    if auth_header:
        # `[:8]` slices string to get first 8 characters
        return hashlib.sha256(auth_header.encode()).hexdigest()[:8]

    # Fallback to client host
    # Ternary operator: `value if condition else other_value`
    # Equivalent to: `if request.client: client_host = getattr(...) else: client_host = "unknown"`
    client_host = (
        getattr(request.client, "host", "unknown") if request.client else "unknown"
    )
    return f"client_{client_host}"


def is_authenticated(request: Request) -> bool:
    """Check if request is authenticated.

    Args:
        request: FastAPI Request object

    Returns:
        True if request is authenticated, False otherwise

    """
    return hasattr(request.state, "authenticated") and request.state.authenticated


def log_security_event(
    logger: logging.Logger,
    event_type: str,  # "auth_failed", "access_denied", "auth_success"
    request: Request | None = None,
    user_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    reason: str | None = None,
    **kwargs,
) -> None:
    """Log security events for audit trail.

    Logs security-related events (authentication failures, access denials, etc.)
    with structured context for security monitoring, incident investigation,
    and compliance auditing.

    Args:
        logger: Logger instance
        event_type: Type of security event ("auth_failed", "access_denied", "auth_success")
        request: Optional FastAPI Request object (for IP address and path extraction)
        user_id: Optional user ID associated with the event
        resource_type: Optional type of resource (e.g., "thread", "run")
        resource_id: Optional resource identifier
        reason: Optional reason for the security event
        **kwargs: Additional context fields to include in the log

    Example:
        ```python
        # Log authentication failure
        log_security_event(
            logger,
            "auth_failed",
            request=request,
            reason="Missing or invalid credentials"
        )

        # Log access denial
        log_security_event(
            logger,
            "access_denied",
            request=request,
            user_id=user_id,
            resource_type="thread",
            resource_id=thread_id,
            owner_id=thread.get("user_id"),
            reason="User does not own resource"
        )
        ```
    """
    context = {
        "event_type": event_type,
        "user_id": user_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "reason": reason,
        "ip_address": request.client.host if request and request.client else None,
        "path": request.url.path if request else None,
        **kwargs,
    }

    if event_type in ("auth_failed", "access_denied"):
        log_warning_event(
            logger,
            f"Security event: {event_type}",
            f"ag_ui.security.{event_type}",
            **context,
        )
    else:
        log_info_event(
            logger,
            f"Security event: {event_type}",
            f"ag_ui.security.{event_type}",
            **context,
        )


def require_authenticated_if_auth_enabled(request: Request | None) -> None:
    """Require authentication when auth is enabled; raise if not authenticated.

    When authentication is enabled (config.auth_enabled), unauthenticated
    requests must be rejected to prevent bypassing auth. This helper runs
    that check at the route level as a defense in depth (in addition to
    auth middleware).

    - If ``request`` is None: no-op (e.g. internal or test call without request).
    - If auth is disabled: no-op.
    - If authenticated: no-op.
    - If auth enabled and not authenticated: raises UnauthorizedError (401).

    Args:
        request: FastAPI Request or None. When None, the check is skipped.

    Raises:
        UnauthorizedError: When auth is enabled and the request is not authenticated.

    """
    if request is None:
        return
    from server.config import get_config
    from server.exceptions import UnauthorizedError

    # Prefer app.state.config (set by create_app) so tests can inject without reset_config
    config = getattr(request.app.state, "config", None) or get_config()
    if not config.auth_enabled:
        return
    if is_authenticated(request):
        return

    # Log security event before raising exception
    log_security_event(
        logger, "auth_failed", request=request, reason="Missing or invalid credentials"
    )
    raise UnauthorizedError("Authentication required")


async def safe_persistence_operation_async(
    operation_name: str, operation_func: Callable[..., T], *args: Any, **kwargs: Any
) -> T | None:
    """Safely execute a persistence write operation with retry logic for transient errors.

    This is the async version that includes retry logic. For sync operations,
    use safe_persistence_operation().

    **INTENTIONAL SILENT FAILURE PATTERN:**
    This function implements intentional silent failure for write operations with
    automatic retry for transient errors. Failures are logged as warnings but do
    not raise exceptions, allowing the application to continue operating even when
    persistence is unavailable or failing.

    **Retry Logic:**
    - Automatically retries transient errors (network, timeout, connection errors)
    - Uses exponential backoff with jitter
    - Maximum 3 retries by default
    - Permanent errors are not retried

    **Rationale:**
    - Write operations (save operations) are non-critical for core functionality
    - The application can continue serving requests even if persistence fails
    - Retry logic improves success rate for transient failures
    - Failures are logged with full context for monitoring and debugging
    - This pattern prevents persistence issues from cascading to API failures

    **Monitoring:**
    Structured logs use event ``ag_ui.persistence_operation_failed`` with
    ``operation_name``, ``run_id``, ``thread_id``, ``error``, and ``attempts``.
    Operators should monitor and alert on this event (e.g. when failure rate or
    count exceeds a threshold); see AG_UI_LOGGING.md and AG_UI_PERSISTENCE.md.

    Args:
        operation_name: Name of the operation for logging (e.g., "save_run_start")
        operation_func: The persistence function to call (can be sync or async)
        *args: Positional arguments to pass to the operation function
        **kwargs: Keyword arguments to pass to the operation function

    Returns:
        Result of the operation function, or None if an error occurred after retries.

    """
    from server.error_classification import is_transient_error
    from server.retry import retry_with_backoff

    # Extract context for logging
    context = {
        "run_id": kwargs.get("run_id"),
        "thread_id": kwargs.get("thread_id"),
    }

    # Wrap operation for retry logic
    async def operation_with_retry() -> T:
        if asyncio.iscoroutinefunction(operation_func):
            return cast(T, await operation_func(*args, **kwargs))
        else:
            return cast(T, operation_func(*args, **kwargs))

    # Attempt operation with retry logic
    retry_result = await retry_with_backoff(
        operation=operation_with_retry,
        max_retries=DEFAULT_MAX_RETRIES,
        initial_delay=DEFAULT_RETRY_INITIAL_DELAY,
        max_delay=DEFAULT_RETRY_MAX_DELAY_SHORT,
        retry_on=is_transient_error,
        operation_name=operation_name,
        context=context,
    )

    if retry_result.success:
        return retry_result.result
    else:
        # All retries exhausted - log final failure
        context_parts = []
        if "run_id" in kwargs:
            context_parts.append(f"run_id={kwargs['run_id']}")
        if "thread_id" in kwargs:
            context_parts.append(f"thread_id={kwargs['thread_id']}")

        context_str = ", ".join(context_parts)
        if context_str:
            error_msg = f"Failed to {operation_name} after {retry_result.attempts} attempts: {context_str}"
        else:
            error_msg = (
                f"Failed to {operation_name} after {retry_result.attempts} attempts"
            )

        log_warning_event(
            logger,
            error_msg,
            "ag_ui.persistence_operation_failed",
            exc_info=True,
            operation_name=operation_name,
            run_id=kwargs.get("run_id"),
            thread_id=kwargs.get("thread_id"),
            error=str(retry_result.errors[-1])
            if retry_result.errors
            else "Unknown error",
            attempts=retry_result.attempts,
        )
        return None


def safe_persistence_operation(
    operation_name: str, operation_func: Callable[..., T], *args: Any, **kwargs: Any
) -> T | None:
    """Safely execute a persistence write operation with error handling.

    **INTENTIONAL SILENT FAILURE PATTERN:**
    This function implements intentional silent failure for write operations. Failures
    are logged as warnings but do not raise exceptions, allowing the application to
    continue operating even when persistence is unavailable or failing.

    **Note:** For async operations or when retry logic is needed, use
    `safe_persistence_operation_async()` instead.

    **Rationale:**
    - Write operations (save operations) are non-critical for core functionality
    - The application can continue serving requests even if persistence fails
    - Failures are logged with full context for monitoring and debugging
    - This pattern prevents persistence issues from cascading to API failures

    **Behavior:**
    - Logs warnings on failure (WARNING level, event: "ag_ui.persistence_operation_failed")
    - Returns None if operation fails (does not raise exceptions)
    - Includes full context (run_id, thread_id, error details) in logs
    - Uses exc_info=True for stack traces in logs

    **When to Use:**
    Use this wrapper for all persistence write operations:
    - save_thread()
    - save_run_start()
    - save_run_finish()
    - save_message()
    - save_event()

    **Monitoring:**
    Structured logs use event ``ag_ui.persistence_operation_failed`` with
    ``operation_name``, ``run_id``, ``thread_id``, and ``error``. Operators
    should monitor and alert on this event (e.g. when failure rate or count
    exceeds a threshold); see AG_UI_LOGGING.md and AG_UI_PERSISTENCE.md.

    Args:
        operation_name: Name of the operation for logging (e.g., "save_run_start")
        operation_func: The persistence function to call
        *args: Positional arguments to pass to the operation function
        **kwargs: Keyword arguments to pass to the operation function

    Returns:
        Result of the operation function, or None if an error occurred.
        Callers should not rely on the return value for write operations.

    """
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        # Extract context from kwargs for better error messages
        # `**kwargs` is a dict of keyword arguments passed to the function
        context_parts = []
        # `"key" in dict` checks if key exists
        if "run_id" in kwargs:
            context_parts.append(f"run_id={kwargs['run_id']}")
        if "thread_id" in kwargs:
            context_parts.append(f"thread_id={kwargs['thread_id']}")

        context_str = ", ".join(context_parts)
        if context_str:
            error_msg = f"Failed to {operation_name}: {context_str}, error={e}"
        else:
            error_msg = f"Failed to {operation_name}: error={e}"

        log_warning_event(
            logger,
            error_msg,
            "ag_ui.persistence_operation_failed",
            exc_info=True,
            operation_name=operation_name,
            run_id=kwargs.get("run_id"),
            thread_id=kwargs.get("thread_id"),
            error=str(e),
        )
        return None


def handle_persistence_read_operation(
    operation_name: str,
    operation_func: Callable[..., T],
    error_event_name: str,
    error_context: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a persistence read operation with standardized error handling.

    **EXPLICIT ERROR RAISING PATTERN:**
    This function implements explicit error raising for read operations. Failures
    are logged as errors and exceptions are raised, ensuring API consumers receive
    proper error responses.

    **Rationale:**
    - Read operations are part of the API contract - clients expect data or errors
    - Failures must be communicated to API consumers via HTTP error responses
    - This pattern ensures consistent error handling across all read endpoints
    - Errors are logged with full context before being raised

    **Behavior:**
    - Logs errors on failure (ERROR level, custom event name)
    - Re-raises expected exceptions (PersistenceNotEnabledError, NotFoundError)
    - Converts unexpected exceptions to InternalServerError
    - Includes full context in error logs and exception context
    - Uses exc_info=True for stack traces in logs

    **When to Use:**
    Use this wrapper for all persistence read operations:
    - get_thread()
    - get_threads()
    - get_run()
    - get_runs()
    - get_messages()
    - get_events()

    **Error Handling:**
    Expected exceptions (PersistenceNotEnabledError, NotFoundError) are re-raised
    as-is. Unexpected exceptions are wrapped in InternalServerError with context.

    Args:
        operation_name: Name of the operation for logging (e.g., "get_run")
        operation_func: The persistence function to call
        error_event_name: Event name for logging (e.g., "ag_ui.run_retrieval_error")
        error_context: Dictionary with context for error logging (e.g., {"run_id": "123"})
        *args: Positional arguments to pass to the operation function
        **kwargs: Keyword arguments to pass to the operation function

    Returns:
        Result of the operation function

    Raises:
        PersistenceNotEnabledError: If persistence is required but not enabled (re-raised)
        NotFoundError: If resource is not found (re-raised)
        InternalServerError: If an unexpected error occurs (wraps original exception)

    """
    from server.exceptions import (
        InternalServerError,
        NotFoundError,
        PersistenceNotEnabledError,
    )

    try:
        return operation_func(*args, **kwargs)
    except (PersistenceNotEnabledError, NotFoundError):
        # Re-raise expected exceptions
        raise
    except Exception as e:
        # Log unexpected errors with standardized format
        context_str = ", ".join(f"{k}={v}" for k, v in error_context.items())
        log_error_event(
            logger,
            f"Error retrieving {operation_name}: {context_str}, error={e}",
            error_event_name,
            error=str(e),
            exc_info=True,
            operation_name=operation_name,
            **error_context,
        )
        # Raise InternalServerError with context
        raise InternalServerError(
            f"Failed to retrieve {operation_name}: {str(e)}", context=error_context
        ) from e


def parse_json_with_fallback(
    text_content: str, fallback_value: Any | None = None
) -> Any | None:
    """Parse JSON string with fallback attempts for common formatting issues.

    Attempts to parse JSON with multiple strategies:
    1. Direct JSON parsing
    2. Replace single quotes with double quotes (common mistake)
    3. Return fallback value or raw text if all parsing fails

    This function reduces code duplication for JSON parsing patterns that appear
    throughout the codebase, particularly in tool result parsing.

    Args:
        text_content: String content to parse as JSON
        fallback_value: Value to return if all parsing attempts fail (default: None).
                       If None, returns the raw text_content.

    Returns:
        Parsed JSON data (dict/list), fallback_value, or raw text_content if parsing fails

    Example:
        result = parse_json_with_fallback('{"key": "value"}')  # Returns dict
        result = parse_json_with_fallback("{'key': 'value'}")  # Returns dict (after fix)
        result = parse_json_with_fallback("invalid", fallback="default")  # Returns "default"

    """
    if not text_content:
        return fallback_value

    try:
        # First attempt: direct JSON parsing
        parsed = json.loads(text_content)
        # Return parsed value (can be dict, list, str, int, float, bool, None)
        return parsed
    except json.JSONDecodeError:
        try:
            # Second attempt: replace single quotes with double quotes
            json_text = text_content.replace("'", '"')
            parsed = json.loads(json_text)
            # Return parsed value (can be dict, list, str, int, float, bool, None)
            return parsed
        except Exception as e:
            # All parsing attempts failed
            log_warning_event(
                logger,
                f"Failed to parse JSON after fallback attempts: {str(e)}",
                "ag_ui.json_parse_failed",
                error=str(e),
                exc_info=True,
            )
            return fallback_value if fallback_value is not None else text_content


def format_result_content(result: Any) -> str:
    """Format tool result content for ToolCallResultEvent.

    Converts tool result data to a JSON string for use in ToolCallResultEvent.content.
    This function centralizes the formatting logic to reduce code duplication.

    Args:
        result: Tool result data (can be dict, list, str, or any other type)

    Returns:
        JSON string if result is dict or list, otherwise string representation

    Example:
        format_result_content({"status": "success"})  # Returns: '{"status": "success"}'
        format_result_content([1, 2, 3])  # Returns: '[1, 2, 3]'
        format_result_content("plain text")  # Returns: 'plain text'
        format_result_content(42)  # Returns: '42'

    """
    if isinstance(result, (dict, list)):
        return json.dumps(result)
    return str(result)


def format_error_context(**kwargs: Any) -> str:
    """Format error context dictionary into a standardized string.

    Creates a consistent format for error context strings used in logging.
    Filters out None values and formats key-value pairs.

    Args:
        **kwargs: Key-value pairs to include in the context string

    Returns:
        Formatted context string (e.g., "tool_name=test, tool_call_id=123, error=ValueError")

    Example:
        context = format_error_context(
            tool_name="test_tool",
            tool_call_id="123",
            error="ValueError"
        )
        # Returns: "tool_name=test_tool, tool_call_id=123, error=ValueError"

    """
    # Filter out None values and format as key=value pairs
    context_parts = [f"{k}={v}" for k, v in kwargs.items() if v is not None]
    return ", ".join(context_parts)


def log_hook_error_if_not_silent(
    silent: bool,
    hook_name: str,
    error_event_name: str,
    error: Exception,
    **error_context: Any,
) -> None:
    """Log hook error if not in silent mode.

    Provides a standardized way to handle errors in hook functions, respecting
    the `silent_hook_errors` configuration. Errors are logged as warnings unless
    silent mode is enabled.

    **Usage Pattern:**
    ```python
    try:
        result = await hook_function(context)
    except Exception as e:
        log_hook_error_if_not_silent(
            silent=config.silent_hook_errors,
            hook_name="state_from_result",
            error_event_name="ag_ui.state_from_result_hook_error",
            error=e,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )
    ```

    Args:
        silent: If True, suppress error logging. If False, log warnings.
        hook_name: Name of the hook for error messages (e.g., "state_from_result")
        error_event_name: Event name for logging (e.g., "ag_ui.state_from_result_hook_error")
        error: The exception that occurred
        **error_context: Additional context fields for error logging

    Example:
        try:
            async for chunk in streamer(context):
                yield chunk
        except Exception as e:
            log_hook_error_if_not_silent(
                silent=config.silent_hook_errors,
                hook_name="args_streamer",
                error_event_name="ag_ui.args_streamer_error",
                error=e,
                tool_name="test_tool",
                tool_call_id="123",
            )

    """
    if not silent:
        context_str = format_error_context(**error_context, error=str(error))
        log_warning_event(
            logger,
            f"Error in {hook_name} hook: {context_str}",
            error_event_name,
            exc_info=True,
            hook_name=hook_name,
            **error_context,
            error=str(error),
        )


def handle_operation_error(
    operation_name: str,
    error_event_name: str,
    error: Exception,
    **error_context: Any,
) -> None:
    """Log an operation error with standardized formatting.

    Centralizes error logging for operations that don't use decorators or
    context managers. Ensures consistent error message formatting across
    the codebase.

    Args:
        operation_name: Name of the operation (e.g., "emit MessagesSnapshotEvent")
        error_event_name: Event name for logging (e.g., "ag_ui.messages_snapshot_emit_failed")
        error: The exception that occurred
        **error_context: Additional context fields for error logging

    Example:
        try:
            result = some_operation()
        except Exception as e:
            handle_operation_error(
                operation_name="emit MessagesSnapshotEvent",
                error_event_name="ag_ui.messages_snapshot_emit_failed",
                error=e,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            return None

    """
    context_str = format_error_context(**error_context, error=str(error))
    log_warning_event(
        logger,
        f"Failed to {operation_name}: {context_str}",
        error_event_name,
        exc_info=True,
        operation_name=operation_name,
        **error_context,
        error=str(error),
    )
