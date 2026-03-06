"""OpenTelemetry Observability Initialization

Shared module for initializing OpenTelemetry tracing with OpenInference.
Sends traces via OTLP to any compatible collector endpoint (e.g., Jaeger,
Grafana Tempo, any OpenTelemetry Collector).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager

from utils.logging_helpers import (
    get_logger,
    log_info_event,
    log_warning_event,
)

logger = get_logger(__name__)

try:
    from openinference.instrumentation import using_user
    from openinference.instrumentation.bedrock import BedrockInstrumentor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Create no-op context manager if OpenTelemetry/OpenInference not available

    @contextmanager
    def using_user(user_id: str) -> Generator[None, None, None]:
        """No-op context manager when OpenTelemetry/OpenInference is not available."""
        yield


def initialize_telemetry(
    otel_endpoint: str | None = None,
    service_name: str | None = None,
) -> bool:
    """Initialize OpenTelemetry tracing with OTLP export.

    Configures a TracerProvider with an OTLP HTTP exporter that sends traces
    to any OpenInference-compatible endpoint. Also instruments Bedrock for
    detailed LLM call traces.

    Args:
        otel_endpoint: OTLP collector endpoint (defaults to OTEL_EXPORTER_OTLP_ENDPOINT
            env var or http://localhost:4318). This is the base URL; /v1/traces is
            appended automatically.
        service_name: Service name for trace attribution (defaults to OTEL_SERVICE_NAME
            env var or "opensearch-agent-server").

    Returns:
        True if telemetry was successfully initialized, False otherwise
    """
    if not OTEL_AVAILABLE:
        log_warning_event(
            logger,
            "OpenTelemetry/OpenInference packages not installed, tracing disabled",
            "otel.not_available",
        )
        return False

    if otel_endpoint is None:
        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    if service_name is None:
        service_name = os.getenv("OTEL_SERVICE_NAME", "opensearch-agent-server")

    try:
        resource = Resource.create({
            "service.name": service_name,
            "openinference.project.name": service_name,
        })

        # --- Tracing ---
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{otel_endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        BedrockInstrumentor().instrument()

        # Suppress noisy "Failed to detach context" errors that occur when
        # OTel spans cross async await boundaries. This is a known upstream
        # issue (opentelemetry-python#2606) and is harmless — spans are still
        # exported correctly.
        logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

        log_info_event(
            logger,
            f"OpenTelemetry tracing initialized: endpoint={otel_endpoint}, "
            f"service={service_name}",
            "otel.initialized",
            otel_endpoint=otel_endpoint,
            service_name=service_name,
        )
        return True
    except Exception as e:
        log_warning_event(
            logger,
            f"Failed to initialize OpenTelemetry tracing: {e}",
            "otel.init_failed",
            error=str(e),
        )
        return False