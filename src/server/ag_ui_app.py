"""AG-UI Protocol FastAPI Server.

Exposes the multi-agent system via AG-UI protocol for frontend integration.
create_app() builds the FastAPI app; the lifespan context manager handles
startup (config validation, persistence, telemetry, rate limiting, StrandsAgent)
and shutdown (agent thread cleanup).
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from utils.logging_helpers import (
    get_logger,
    log_debug_event,
    log_error_event,
    log_info_event,
    log_warning_event,
)
from server.constants import (
    DEFAULT_EVENT_LIMIT,
)
from server.exceptions import APIError
from server.types import (
    CancelRunResponse,
    RunEventsResponse,
    RunResponse,
)
from server.validators import ValidatedRunAgentInput

# Configure logging (will be configured by ag_ui_server.py or can be configured here)
# If not already configured, use default human-readable format
if not logging.root.handlers:
    from server.logging_config import (  # noqa: E402  # conditional import for config
        configure_logging,
        get_logging_config,
    )

    use_json, log_level = get_logging_config()
    configure_logging(use_json=use_json, log_level=log_level)

logger = get_logger(__name__)

# Imports below run after logging config so that:
# - OTel and route modules see configured logging if we configured it here.
# - No need to reorder if new modules depend on config or logging.
from utils.otel_init import initialize_telemetry  # noqa: E402
from server.metrics import capture_span  # noqa: E402
from server.auth_middleware import (  # noqa: E402
    AuthenticationMiddleware,
    create_auth_middleware,
)
from server.config import ServerConfig, get_config  # noqa: E402
from server.rate_limiting import (  # noqa: E402
    create_rate_limiter,
    get_rate_limit_decorator,
    setup_rate_limiting,
)
from server.request_id_middleware import RequestIdMiddleware  # noqa: E402
from server.run_routes import (  # noqa: E402
    cancel_run_route,
    create_run_route,
    get_run_events_route,
    get_run_route,
)
from server.strands_agent import StrandsAgent  # noqa: E402

# Set by lifespan at startup; used by routes at request time.
persistence: Any | None = None
strands_agent: StrandsAgent | None = None


def _suppress_mcp_cancel_scope_error(
    loop: asyncio.AbstractEventLoop, context: dict[str, Any]
) -> None:
    """Suppress RuntimeError about cancel scopes from MCP connection cleanup.

    This error occurs when MCP stdio_client async generators are garbage collected
    in a different task than they were created in. Since we manually close streams
    in cleanup(), this error is harmless and can be safely suppressed.

    Args:
        loop: The asyncio event loop
        context: Exception context dict with 'exception' and 'message' keys
    """
    exception = context.get("exception")
    if isinstance(exception, RuntimeError):
        error_msg = str(exception).lower()
        if "cancel scope" in error_msg and "different task" in error_msg:
            # Suppress this specific error - it's harmless since streams are already closed
            log_debug_event(
                logger,
                "Suppressed MCP cancel scope error during cleanup (expected during GC).",
                "ag_ui.mcp_cancel_scope_error_suppressed",
                error=str(exception),
            )
            return

    # For all other exceptions, log them using the standard event helper
    # This mimics asyncio's default exception handler behavior.
    # Use exc_info tuple so the logged traceback matches the exception from context
    # (sys.exc_info() is not reliable in asyncio exception handler callbacks).
    message = context.get("message", "Unhandled exception in event loop")
    if exception:
        exc_info_tuple = (
            type(exception),
            exception,
            getattr(exception, "__traceback__", None),
        )
        log_error_event(
            logger,
            f"✗ {message}: {exception}",
            "ag_ui.asyncio_exception",
            error=exception,
            exception_type=type(exception).__name__,
            exc_info=exc_info_tuple,
        )
    else:
        log_error_event(
            logger,
            f"✗ {message}",
            "ag_ui.asyncio_exception",
            context=context,
            exc_info=False,
        )


def _register_mcp_cancel_scope_exception_handler(
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Register the MCP cancel-scope exception handler on the event loop.

    Called by the create_app() lifespan at startup. Suppresses harmless
    RuntimeErrors from MCP stdio_client cleanup; other exceptions are logged.

    Args:
        loop: The asyncio event loop to register the handler on.
    """
    loop.set_exception_handler(_suppress_mcp_cancel_scope_error)


def _noop_rate_limit(f: Any) -> Any:
    """Placeholder no-op so route handlers can use @rate_limit before the app exists.

    create_app() overwrites the module-level rate_limit with the real decorator.
    """
    return f


rate_limit: Any = _noop_rate_limit


class _MaxBodySizeMiddleware:
    """ASGI middleware that returns 413 when Content-Length exceeds max_bytes.

    When max_bytes is 0, the check is disabled. Only checks when Content-Length
    is present; chunked or missing Content-Length bypass the check.
    """

    def __init__(self, app: Any, max_bytes: int = 0) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if self.max_bytes > 0 and scope.get("type") == "http":
            content_length = None
            for key, value in scope.get("headers") or []:
                if key.lower() == b"content-length":
                    try:
                        content_length = int(value.decode())
                    except (ValueError, TypeError, UnicodeDecodeError):
                        pass
                    break
            if content_length is not None:
                if content_length < 0:
                    body = json.dumps(
                        {"detail": "Invalid Content-Length: must be non-negative."}
                    ).encode()
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 400,
                            "headers": [[b"content-type", b"application/json"]],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                    return
                if content_length > self.max_bytes:
                    body = json.dumps(
                        {
                            "detail": (
                                f"Request body too large. "
                                f"Maximum size is {self.max_bytes} bytes."
                            )
                        }
                    ).encode()
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 413,
                            "headers": [[b"content-type", b"application/json"]],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                    return
        await self.app(scope, receive, send)


def create_app(config_override: ServerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Config is resolved at call time (not import time), so tests can pass an
    alternative config without using reset_config. When config_override is None,
    get_config() is used.

    Args:
        config_override: Optional ServerConfig. When None, uses get_config().

    Returns:
        Configured FastAPI application instance.
    """
    config_resolved = config_override if config_override is not None else get_config()
    rate_limiter = create_rate_limiter(config_resolved)
    rate_limit_deco = get_rate_limit_decorator(rate_limiter, config=config_resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        """FastAPI lifespan: startup then yield, then shutdown.

        Startup: register MCP exception handler, validate config, init
        persistence (if enabled), telemetry, rate limiting, and StrandsAgent.
        Sets module-level persistence and strands_agent for use by routes.
        Shutdown: cleanup cached agent threads and log result.
        """
        loop = asyncio.get_running_loop()
        _register_mcp_cancel_scope_exception_handler(loop)

        # Validate configuration before proceeding with initialization
        from server.config import validate_config_on_startup

        validate_config_on_startup(config_resolved)

        global persistence, strands_agent
        persistence = None
        if config_resolved.enable_persistence:
            try:
                from utils.persistence import AGUIPersistence

                persistence = AGUIPersistence(db_path=config_resolved.db_path)
                log_info_event(
                    logger,
                    "✓ AG-UI data persistence enabled",
                    "ag_ui.persistence_enabled",
                    enabled=True,
                )
            except Exception as e:
                log_warning_event(
                    logger,
                    f"✗ Failed to initialize AG-UI persistence: {e}. "
                    "Continuing without persistence.",
                    "ag_ui.persistence_initialization_failed",
                    exc_info=True,
                    error=str(e),
                    enabled=False,
                )
                persistence = None
        else:
            log_info_event(
                logger,
                "AG-UI data persistence disabled (set AG_UI_ENABLE_PERSISTENCE=true to enable)",
                "ag_ui.persistence_disabled",
                enabled=False,
            )

        otel_endpoint = config_resolved.otel_exporter_endpoint
        otel_service_name = config_resolved.otel_service_name
        telemetry_initialized = initialize_telemetry(
            otel_endpoint=otel_endpoint,
            service_name=otel_service_name,
        )
        if telemetry_initialized:
            log_info_event(
                logger,
                f"OpenTelemetry tracing initialized: endpoint={otel_endpoint}, "
                f"service={otel_service_name}",
                "ag_ui.otel_initialized",
                otel_endpoint=otel_endpoint,
                service_name=otel_service_name,
            )
        else:
            log_warning_event(
                logger,
                f"OpenTelemetry tracing initialization failed or unavailable: "
                f"endpoint={otel_endpoint}. "
                "Server will continue without tracing.",
                "ag_ui.otel_initialization_failed",
                otel_endpoint=otel_endpoint,
                service_name=otel_service_name,
            )

        setup_rate_limiting(app, rate_limiter)

        # --- Orchestrator setup ---
        from orchestrator.registry import AgentRegistration, AgentRegistry
        from orchestrator.router import PageContextRouter
        from agents.fallback_agent import create_fallback_agent
        from agents.art_agent import create_art_agent

        registry = AgentRegistry()
        opensearch_url = config_resolved.opensearch_url

        # Register ART agent (search relevance page)
        registry.register(AgentRegistration(
            name="art",
            description="Search Relevance Testing agent (ART) — hypothesis generation, "
            "evaluation, UBI analysis, and online A/B testing",
            factory=create_art_agent,
            page_contexts=["search-relevance", "searchRelevance"],
            is_fallback=False,
        ))

        # Register fallback agent (handles all unmatched page contexts)
        registry.register(AgentRegistration(
            name="fallback",
            description="General OpenSearch assistant with MCP tools",
            factory=create_fallback_agent,
            page_contexts=[],
            is_fallback=True,
        ))

        log_info_event(
            logger,
            f"Registered {len(registry.list_agents())} agent(s): "
            + ", ".join(a.name for a in registry.list_agents()),
            "ag_ui.agents_registered",
            agent_count=len(registry.list_agents()),
        )

        # Store registry on app for the /agents endpoint
        app.state.registry = registry

        router = PageContextRouter(registry)

        # Agent cache: keyed by agent name (one instance per sub-agent type)
        _agent_cache: dict[str, object] = {}

        async def create_agent_for_request(page_context: str | None = None) -> object:
            """Orchestrator factory: route by page_context, create/cache agent."""
            registration = router.route(page_context)
            if registration.name not in _agent_cache:
                _agent_cache[registration.name] = await registration.factory(opensearch_url)
                log_info_event(
                    logger,
                    f"Created agent '{registration.name}' for page_context='{page_context}'",
                    "ag_ui.agent_created",
                    agent_name=registration.name,
                    page_context=page_context,
                )
            return _agent_cache[registration.name]

        strands_agent = StrandsAgent(
            agent_factory=create_agent_for_request,
            name="opensearch-agent-server",
            description="Multi-agent orchestrator for OpenSearch Dashboards",
            cache_max_size=config_resolved.agent_cache_size,
        )

        yield
        try:
            thread_count = await strands_agent.cleanup_all_threads()
            log_info_event(
                logger,
                f"Server shutdown: cleaned up {thread_count} cached agent(s)",
                "ag_ui.server_shutdown_cleanup",
                thread_count=thread_count,
            )
        except Exception as e:
            log_error_event(
                logger,
                f"✗ Error during server shutdown cleanup: {e}",
                "ag_ui.server_shutdown_cleanup_error",
                exc_info=True,
                error=str(e),
            )

    app = FastAPI(
        title="OpenSearch Agent Server for OpenSearch Dashboards",
        description=(
            "Multi-agent orchestrator for OpenSearch Dashboards. Routes requests "
            "by page context to specialized sub-agents via the AG-UI protocol (SSE)."
        ),
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "runs", "description": "Agent run management endpoints."},
            {"name": "agents", "description": "Agent discovery and registry."},
            {
                "name": "health",
                "description": "Health check and system status endpoints.",
            },
        ],
    )
    app.state.config = config_resolved

    cors_origins = config_resolved.get_cors_origins_list()
    if cors_origins == ["*"]:
        log_warning_event(
            logger,
            "CORS configured to allow all origins (*). This is insecure and should only be used for development. "
            "For production, specify exact origins in AG_UI_CORS_ORIGINS environment variable.",
            "ag_ui.cors_wildcard_warning",
            cors_origins_env=config_resolved.cors_origins or "",
        )
    elif not cors_origins:
        log_info_event(
            logger,
            "CORS disabled (no AG_UI_CORS_ORIGINS set). To enable CORS, set AG_UI_CORS_ORIGINS environment variable.",
            "ag_ui.cors_disabled",
        )
    cors_methods = config_resolved.get_cors_methods_list()
    cors_headers = config_resolved.get_cors_headers_list()
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=cors_methods,
            allow_headers=cors_headers,
        )
        log_info_event(
            logger,
            f"CORS enabled: origins={cors_origins}, methods={cors_methods}, headers={cors_headers}",
            "ag_ui.cors_enabled",
            origins=cors_origins,
            methods=cors_methods,
            headers=cors_headers,
        )

    auth_middleware_config = create_auth_middleware(app, config_resolved)
    if auth_middleware_config:
        app.add_middleware(
            AuthenticationMiddleware,
            enabled=auth_middleware_config["enabled"],
            mode=auth_middleware_config["mode"],
            strategies=auth_middleware_config["strategies"],
            config=auth_middleware_config["config"],
        )

    app.add_middleware(RequestIdMiddleware)

    global rate_limit
    rate_limit = rate_limit_deco

    app.add_middleware(
        _MaxBodySizeMiddleware, max_bytes=config_resolved.max_request_body_bytes
    )

    return app


app = create_app()


def get_strands_agent() -> StrandsAgent:
    """Dependency function to provide StrandsAgent instance.

    This enables dependency injection for better testability and flexibility.
    The function returns the module-level strands_agent instance, which can be
    overridden in tests by patching this dependency function.

    Returns:
        StrandsAgent instance configured for the application

    Raises:
        RuntimeError: If called before app lifespan has run (strands_agent is None).

    """
    if strands_agent is None:
        raise RuntimeError(
            "StrandsAgent not initialized. Ensure app lifespan has run (e.g. use "
            "TestClient with lifespan context or start the server)."
        )
    return strands_agent


# Exception handlers for consistent error responses
# `@app.exception_handler(APIError)` decorator registers exception handler
# FastAPI calls this function when APIError exception is raised
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle API errors with consistent response format.

    Args:
        request: FastAPI request object
        exc: APIError exception instance

    Returns:
        JSONResponse with error details

    """
    log_error_event(
        logger,
        f"✗ API error: code={exc.code}, message={exc.message}, path={request.url.path}",
        "ag_ui.api_error",
        exc_info=True,
        error_code=exc.code,
        error_message=exc.message,
        status_code=exc.status_code,
        path=request.url.path,
    )
    # Build response with error, code, and detail (stable keys for frontend handling).
    # "detail" is included for conventional 4xx/5xx JSON; "error" is the message.
    response_content = {
        "error": exc.message,
        "code": exc.code,
        "detail": exc.message,
    }
    if exc.context:
        response_content.update(exc.context)

    return JSONResponse(status_code=exc.status_code, content=response_content)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors (422) with a sanitized detail.

    Returns 422 with a detail that includes only loc, msg, and type for each
    error. Omits ctx, input, and other keys that might contain stack traces,
    internal paths, or other sensitive implementation details.

    Args:
        request: FastAPI request object
        exc: RequestValidationError from Pydantic/FastAPI validation

    Returns:
        JSONResponse with status 422 and sanitized detail list
    """
    # Keep only loc, msg, type per error to avoid leaking stack traces or paths
    sanitized = [
        {k: v for k, v in e.items() if k in ("loc", "msg", "type")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": sanitized})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPException with proper status code.

    This handler ensures HTTPException (like 401 Authentication required)
    are returned with their proper status codes instead of being caught
    by the general exception handler and converted to 500 errors.

    Args:
        request: FastAPI request object
        exc: HTTPException instance

    Returns:
        JSONResponse with the exception's status code and detail
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with consistent error response.

    Note: HTTPException is handled by http_exception_handler above. FastAPI
    should match HTTPException to the more specific handler first, but this
    check ensures HTTPException is properly handled if the handler order
    is incorrect.

    Args:
        request: FastAPI request object
        exc: Exception instance

    Returns:
        JSONResponse with error details (401 for HTTPException, 500 for others)

    """
    # Handle HTTPException - delegate to the specific handler logic
    # This ensures HTTPException gets proper status codes even if handler
    # registration order is wrong
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    log_error_event(
        logger,
        f"✗ Unexpected error: {exc}, path={request.url.path}",
        "ag_ui.unexpected_error",
        error=str(exc),
        exc_info=True,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "An internal server error occurred",
            "code": "INTERNAL_SERVER_ERROR",
        },
    )


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Status object indicating server health
    """
    return {"status": "ok"}


@app.get("/agents", tags=["agents"])
async def list_agents(request: Request) -> dict:
    """List registered agents and their page contexts.

    Returns:
        Dictionary with list of registered agents.
    """
    registry = request.app.state.registry
    return {
        "agents": [
            {
                "name": reg.name,
                "description": reg.description,
                "page_contexts": reg.page_contexts,
                "is_fallback": reg.is_fallback,
            }
            for reg in registry.list_agents()
        ]
    }


# `@app.post("/runs")` decorator registers HTTP POST endpoint
# FastAPI automatically parses and validates request body as ValidatedRunAgentInput (Pydantic model)
@app.post("/runs", tags=["runs"])
@rate_limit
@capture_span
async def create_run(
    input_data: ValidatedRunAgentInput,
    request: Request,
    strands_agent: StrandsAgent = Depends(get_strands_agent),
) -> StreamingResponse:
    """Start a new agent run and stream AG-UI events via SSE.

    Follows the official AG-UI Strands integration pattern.

    Files should be included in the RunAgentInput messages as BinaryInputContent
    with base64-encoded data in the JSON payload, not as multipart/form-data.

    Args:
        input_data: ValidatedRunAgentInput with thread_id, run_id, and messages
                   (files should be in messages as BinaryInputContent with base64 data)
        request: FastAPI request object (for Accept header)
        strands_agent: StrandsAgent instance injected via dependency injection

    Returns:
        SSE stream of AG-UI events

    """
    return create_run_route(
        strands_agent=strands_agent,
        persistence=persistence,
        input_data=input_data,
        request=request,
    )


@app.get("/runs/{run_id}", tags=["runs"])
async def get_run(run_id: str, request: Request) -> RunResponse:
    """Get run details including status and metadata.

    Args:
        run_id: Run identifier
        request: FastAPI request object (for access control)

    Returns:
        Run information including status, timestamps, and metadata

    """
    return get_run_route(persistence=persistence, run_id=run_id, request=request)


@app.get("/runs/{run_id}/events", tags=["runs"])
async def get_run_events(
    run_id: str,
    request: Request,
    event_type: str | None = None,
    limit: int = DEFAULT_EVENT_LIMIT,
    offset: int = 0,
) -> RunEventsResponse:
    """Get events for a run, optionally filtered by event_type.

    Args:
        run_id: Run identifier
        request: FastAPI request object (for access control)
        event_type: Optional event type to filter by (e.g., TEXT_MESSAGE_START)
        limit: Maximum number of events to return (default: DEFAULT_EVENT_LIMIT)
        offset: Offset for pagination (default: 0)

    Returns:
        List of event dictionaries

    """
    return get_run_events_route(
        persistence=persistence,
        run_id=run_id,
        event_type=event_type,
        limit=limit,
        offset=offset,
        request=request,
    )


@app.post("/runs/{run_id}/cancel", tags=["runs"])
async def cancel_run(run_id: str, request: Request) -> CancelRunResponse:
    """Cancel a running agent.

    Args:
        run_id: Run identifier
        request: FastAPI request object (for access control)

    Returns:
        Cancellation confirmation

    """
    return await cancel_run_route(
        persistence=persistence, run_id=run_id, request=request
    )


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(app, host=config.server_host, port=config.server_port)
