from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from functools import wraps
from typing import Any, TypeVar

from fastapi import Request
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from utils.logging_helpers import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Initialize tracer
tracer = trace.get_tracer(__name__)

def capture_span(func: F) -> F:
    """Decorator to capture an OpenTelemetry span for a FastAPI route.

    Handles both regular responses and StreamingResponses.
    For StreamingResponses, the span covers until the stream is closed.

    Args:
        func: The route handler function

    Returns:
        The decorated function
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Find request object in arguments
        request: Request | None = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if not request:
            for v in kwargs.values():
                if isinstance(v, Request):
                    request = v
                    break

        attributes = {}
        if request:
            attributes["http.method"] = request.method
            attributes["http.target"] = request.url.path

        span_name = f"{request.method} {request.url.path}" if request else func.__name__

        span = tracer.start_span(span_name, attributes=attributes)
        trace_id = format(span.get_span_context().trace_id, "032x")

        with trace.use_span(span, end_on_exit=False):
            try:
                response = await func(*args, **kwargs)

                if hasattr(response, "status_code"):
                    span.set_attribute("http.status_code", response.status_code)
                    if response.status_code >= 400:
                        span.set_status(Status(StatusCode.ERROR))

                if hasattr(response, "headers"):
                    # Return trace ID.
                    response.headers["X-Trace-Id"] = trace_id

                if isinstance(response, StreamingResponse):
                    original_body_iterator = response.body_iterator

                    async def wrapped_body_iterator() -> AsyncIterator[Any]:
                        # Re-enter the span context for the iterator and end it on exit
                        with trace.use_span(span, end_on_exit=True):
                            try:
                                async for chunk in original_body_iterator:
                                    yield chunk
                            except Exception as e:
                                span.record_exception(e)
                                span.set_status(Status(StatusCode.ERROR))
                                raise

                    response.body_iterator = wrapped_body_iterator()
                    return response
                else:
                    span.end()
                    return response

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("http.status_code", 500)
                span.end()
                raise

    return wrapper  # type: ignore[return-value]