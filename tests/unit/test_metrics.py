"""
Unit tests for the capture_span OTel decorator (server/metrics.py).

Tests span lifecycle for:
- Regular (non-streaming) responses
- StreamingResponse (span stays open until stream is exhausted)
- Exceptions raised by the route handler
- Request object discovery (positional arg vs keyword arg)
- Missing request object fallback
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry.trace import StatusCode

from server.metrics import capture_span

pytestmark = pytest.mark.unit

_TRACER_PATCH = "server.metrics.tracer"


def _make_span(trace_id: int = 0x1234567890ABCDEF1234567890ABCDEF) -> MagicMock:
    """Return a mock span with a deterministic trace_id."""
    span = MagicMock()
    span.get_span_context.return_value.trace_id = trace_id
    return span


def _make_request(method: str = "GET", path: str = "/test") -> Request:
    """Build a real Starlette Request so isinstance(req, Request) passes."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [],
    }
    return Request(scope)


class TestCaptureSpanRegularResponse:
    """Span behaviour for synchronous (non-streaming) route responses."""

    async def test_span_started_and_ended(self):
        """Span is started and ended for a regular response."""
        mock_tracer = MagicMock()
        mock_span = _make_span()
        mock_tracer.start_span.return_value = mock_span

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                return JSONResponse({"ok": True}, status_code=200)

            await route(_make_request())

        mock_tracer.start_span.assert_called_once()
        mock_span.end.assert_called_once()

    async def test_http_attributes_set(self):
        """http.method, http.target, and http.status_code are recorded on the span."""
        mock_tracer = MagicMock()
        mock_span = _make_span()
        mock_tracer.start_span.return_value = mock_span

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                return JSONResponse({}, status_code=201)

            await route(_make_request("POST", "/runs"))

        mock_tracer.start_span.assert_called_once_with(
            "POST /runs", attributes={"http.method": "POST", "http.target": "/runs"}
        )
        mock_span.set_attribute.assert_any_call("http.status_code", 201)

    async def test_trace_id_header_injected(self):
        """X-Trace-Id header is added to the response."""
        mock_tracer = MagicMock()
        mock_span = _make_span(0x1234567890ABCDEF1234567890ABCDEF)
        mock_tracer.start_span.return_value = mock_span

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                return JSONResponse({}, status_code=200)

            response = await route(_make_request())

        assert response.headers["X-Trace-Id"] == "1234567890abcdef1234567890abcdef"

    async def test_error_status_set_for_4xx(self):
        """ERROR status is set on the span when the response code is >= 400."""
        mock_tracer = MagicMock()
        mock_span = _make_span()
        mock_tracer.start_span.return_value = mock_span

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                return JSONResponse({"error": "bad"}, status_code=400)

            await route(_make_request())

        args, _ = mock_span.set_status.call_args
        assert args[0].status_code == StatusCode.ERROR

    async def test_request_found_as_keyword_arg(self):
        """Decorator finds the Request object when passed as a keyword argument."""
        mock_tracer = MagicMock()
        mock_span = _make_span()
        mock_tracer.start_span.return_value = mock_span

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(*, request: Request):
                return JSONResponse({}, status_code=200)

            await route(request=_make_request("DELETE", "/threads/1"))

        mock_tracer.start_span.assert_called_once_with(
            "DELETE /threads/1",
            attributes={"http.method": "DELETE", "http.target": "/threads/1"},
        )

    async def test_no_request_object_uses_function_name(self):
        """Falls back to func.__name__ as span name when no Request is found."""
        mock_tracer = MagicMock()
        mock_span = _make_span()
        mock_tracer.start_span.return_value = mock_span

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def my_handler():
                return JSONResponse({}, status_code=200)

            await my_handler()

        call_args = mock_tracer.start_span.call_args
        assert call_args[0][0] == "my_handler"


class TestCaptureSpanStreamingResponse:
    """Span behaviour for StreamingResponse routes."""

    async def test_span_not_ended_before_stream_exhausted(self):
        """Span is open while the stream is still being consumed."""
        mock_tracer = MagicMock()
        mock_span = _make_span()
        mock_tracer.start_span.return_value = mock_span

        async def gen():
            yield b"a"
            yield b"b"

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                return StreamingResponse(gen(), status_code=200)

            response = await route(_make_request("POST", "/runs"))

        assert isinstance(response, StreamingResponse)
        mock_span.end.assert_not_called()

    async def test_span_ended_after_stream_exhausted(self):
        """Span is closed once the body iterator is fully consumed."""
        mock_tracer = MagicMock()
        mock_span = _make_span()
        mock_tracer.start_span.return_value = mock_span

        async def gen():
            yield b"chunk"

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                return StreamingResponse(gen(), status_code=200)

            response = await route(_make_request())
            async for _ in response.body_iterator:
                pass

        mock_span.end.assert_called_once()

    async def test_stream_chunks_pass_through_unchanged(self):
        """The wrapped iterator yields exactly the same chunks as the original."""
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = _make_span()

        async def gen():
            for chunk in [b"x", b"y", b"z"]:
                yield chunk

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                return StreamingResponse(gen())

            response = await route(_make_request())
            chunks = [c async for c in response.body_iterator]

        assert chunks == [b"x", b"y", b"z"]

    async def test_trace_id_header_on_streaming_response(self):
        """X-Trace-Id header is injected on StreamingResponse too."""
        mock_tracer = MagicMock()
        mock_span = _make_span(0xDEADBEEFDEADBEEFDEADBEEFDEADBEEF)
        mock_tracer.start_span.return_value = mock_span

        async def gen():
            yield b""

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                return StreamingResponse(gen())

            response = await route(_make_request())

        assert response.headers["X-Trace-Id"] == "deadbeefdeadbeefdeadbeefdeadbeef"

    async def test_stream_exception_records_error_on_span(self):
        """An exception mid-stream records the exception and sets ERROR status."""
        mock_tracer = MagicMock()
        mock_span = _make_span()
        mock_tracer.start_span.return_value = mock_span

        async def failing_gen():
            yield b"partial"
            raise RuntimeError("stream failure")

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                return StreamingResponse(failing_gen())

            response = await route(_make_request())

            with pytest.raises(RuntimeError, match="stream failure"):
                async for _ in response.body_iterator:
                    pass

        mock_span.record_exception.assert_called_once()
        args, _ = mock_span.set_status.call_args
        assert args[0].status_code == StatusCode.ERROR


class TestCaptureSpanException:
    """Span behaviour when the route handler itself raises."""

    async def test_exception_records_on_span_and_reraises(self):
        """Exception is recorded on the span, status set to ERROR, and re-raised."""
        mock_tracer = MagicMock()
        mock_span = _make_span()
        mock_tracer.start_span.return_value = mock_span

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                raise ValueError("boom")

            with pytest.raises(ValueError, match="boom"):
                await route(_make_request())

        mock_span.record_exception.assert_called_once()
        args, _ = mock_span.set_status.call_args
        assert args[0].status_code == StatusCode.ERROR
        mock_span.set_attribute.assert_any_call("http.status_code", 500)
        mock_span.end.assert_called_once()

    async def test_span_always_ended_on_exception(self):
        """Span.end() is called even when the handler raises."""
        mock_tracer = MagicMock()
        mock_span = _make_span()
        mock_tracer.start_span.return_value = mock_span

        with patch(_TRACER_PATCH, mock_tracer):
            @capture_span
            async def route(request: Request):
                raise RuntimeError("oops")

            with pytest.raises(RuntimeError):
                await route(_make_request())

        mock_span.end.assert_called_once()
