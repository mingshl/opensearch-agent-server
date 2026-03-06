"""
Unit tests for OpenTelemetry initialization.

Tests critical paths for OpenTelemetry/OpenInference observability initialization including:
- OTel availability detection
- Initialization with and without OTel packages available
- Endpoint and service name resolution (args, env vars, defaults)
- Error handling
"""

import os
from unittest.mock import Mock, patch

import pytest

from utils.otel_init import initialize_telemetry

pytestmark = pytest.mark.unit

_MODULE = "utils.otel_init"


class TestInitializeTelemetry:
    """Tests for initialize_telemetry function."""

    def test_returns_false_when_otel_not_available(self):
        """Returns False immediately when OTel packages are not installed."""
        with patch(f"{_MODULE}.OTEL_AVAILABLE", False):
            assert initialize_telemetry() is False

    def test_success_wires_up_provider_and_instruments_bedrock(self):
        """Successful init configures TracerProvider, OTLP exporter, and Bedrock."""
        with (
            patch(f"{_MODULE}.OTEL_AVAILABLE", True),
            patch(f"{_MODULE}.TracerProvider", create=True) as mock_provider_cls,
            patch(f"{_MODULE}.OTLPSpanExporter", create=True) as mock_exporter_cls,
            patch(f"{_MODULE}.BatchSpanProcessor", create=True) as mock_processor_cls,
            patch(f"{_MODULE}.Resource", create=True) as mock_resource_cls,
            patch(f"{_MODULE}.trace", create=True) as mock_trace,
            patch(f"{_MODULE}.BedrockInstrumentor", create=True) as mock_bedrock,
        ):
            mock_provider = Mock()
            mock_provider_cls.return_value = mock_provider
            mock_exporter = Mock()
            mock_exporter_cls.return_value = mock_exporter
            mock_processor = Mock()
            mock_processor_cls.return_value = mock_processor
            mock_resource = Mock()
            mock_resource_cls.create.return_value = mock_resource
            mock_bedrock.return_value = Mock()

            result = initialize_telemetry(
                otel_endpoint="http://collector:4318",
                service_name="opensearch-agent-server",
            )

            assert result is True
            mock_resource_cls.create.assert_called_once_with({
                "service.name": "opensearch-agent-server",
                "openinference.project.name": "opensearch-agent-server",
            })
            mock_provider_cls.assert_called_once_with(resource=mock_resource)
            mock_exporter_cls.assert_called_once_with(
                endpoint="http://collector:4318/v1/traces"
            )
            mock_processor_cls.assert_called_once_with(mock_exporter)
            mock_provider.add_span_processor.assert_called_once_with(mock_processor)
            mock_trace.set_tracer_provider.assert_called_once_with(mock_provider)
            mock_bedrock.return_value.instrument.assert_called_once()

    def test_appends_v1_traces_to_endpoint(self):
        """The /v1/traces suffix is always appended to the provided endpoint."""
        with (
            patch(f"{_MODULE}.OTEL_AVAILABLE", True),
            patch(f"{_MODULE}.TracerProvider", create=True),
            patch(f"{_MODULE}.OTLPSpanExporter", create=True) as mock_exporter_cls,
            patch(f"{_MODULE}.BatchSpanProcessor", create=True),
            patch(f"{_MODULE}.Resource", create=True),
            patch(f"{_MODULE}.trace", create=True),
            patch(f"{_MODULE}.BedrockInstrumentor", create=True) as mock_bedrock,
        ):
            mock_bedrock.return_value = Mock()

            initialize_telemetry(otel_endpoint="http://tempo:4318", service_name="svc")

            mock_exporter_cls.assert_called_once_with(
                endpoint="http://tempo:4318/v1/traces"
            )

    def test_custom_service_name_propagated_to_resource(self):
        """Custom service_name is set on both resource attributes."""
        with (
            patch(f"{_MODULE}.OTEL_AVAILABLE", True),
            patch(f"{_MODULE}.TracerProvider", create=True),
            patch(f"{_MODULE}.OTLPSpanExporter", create=True),
            patch(f"{_MODULE}.BatchSpanProcessor", create=True),
            patch(f"{_MODULE}.Resource", create=True) as mock_resource_cls,
            patch(f"{_MODULE}.trace", create=True),
            patch(f"{_MODULE}.BedrockInstrumentor", create=True) as mock_bedrock,
        ):
            mock_bedrock.return_value = Mock()

            initialize_telemetry(
                otel_endpoint="http://collector:4318",
                service_name="my-custom-service",
            )

            mock_resource_cls.create.assert_called_once_with({
                "service.name": "my-custom-service",
                "openinference.project.name": "my-custom-service",
            })

    def test_reads_endpoint_from_env_var(self):
        """OTEL_EXPORTER_OTLP_ENDPOINT env var is used when endpoint arg is omitted."""
        with (
            patch(f"{_MODULE}.OTEL_AVAILABLE", True),
            patch(f"{_MODULE}.TracerProvider", create=True),
            patch(f"{_MODULE}.OTLPSpanExporter", create=True) as mock_exporter_cls,
            patch(f"{_MODULE}.BatchSpanProcessor", create=True),
            patch(f"{_MODULE}.Resource", create=True),
            patch(f"{_MODULE}.trace", create=True),
            patch(f"{_MODULE}.BedrockInstrumentor", create=True) as mock_bedrock,
            patch.dict(
                os.environ,
                {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://env-collector:4318"},
            ),
        ):
            mock_bedrock.return_value = Mock()

            result = initialize_telemetry()

            assert result is True
            mock_exporter_cls.assert_called_once_with(
                endpoint="http://env-collector:4318/v1/traces"
            )

    def test_reads_service_name_from_env_var(self):
        """OTEL_SERVICE_NAME env var is used when service_name arg is omitted."""
        with (
            patch(f"{_MODULE}.OTEL_AVAILABLE", True),
            patch(f"{_MODULE}.TracerProvider", create=True),
            patch(f"{_MODULE}.OTLPSpanExporter", create=True),
            patch(f"{_MODULE}.BatchSpanProcessor", create=True),
            patch(f"{_MODULE}.Resource", create=True) as mock_resource_cls,
            patch(f"{_MODULE}.trace", create=True),
            patch(f"{_MODULE}.BedrockInstrumentor", create=True) as mock_bedrock,
            patch.dict(os.environ, {"OTEL_SERVICE_NAME": "env-service"}),
        ):
            mock_bedrock.return_value = Mock()

            result = initialize_telemetry(otel_endpoint="http://collector:4318")

            assert result is True
            mock_resource_cls.create.assert_called_once_with({
                "service.name": "env-service",
                "openinference.project.name": "env-service",
            })

    def test_default_endpoint_when_no_arg_or_env(self):
        """Falls back to http://localhost:4318 when neither arg nor env var is set."""
        with (
            patch(f"{_MODULE}.OTEL_AVAILABLE", True),
            patch(f"{_MODULE}.TracerProvider", create=True),
            patch(f"{_MODULE}.OTLPSpanExporter", create=True) as mock_exporter_cls,
            patch(f"{_MODULE}.BatchSpanProcessor", create=True),
            patch(f"{_MODULE}.Resource", create=True),
            patch(f"{_MODULE}.trace", create=True),
            patch(f"{_MODULE}.BedrockInstrumentor", create=True) as mock_bedrock,
            patch.dict(os.environ, {}, clear=True),
        ):
            mock_bedrock.return_value = Mock()

            initialize_telemetry()

            mock_exporter_cls.assert_called_once_with(
                endpoint="http://localhost:4318/v1/traces"
            )

    def test_default_service_name_when_no_arg_or_env(self):
        """Falls back to 'opensearch-agent-server' when neither arg nor env var is set."""
        with (
            patch(f"{_MODULE}.OTEL_AVAILABLE", True),
            patch(f"{_MODULE}.TracerProvider", create=True),
            patch(f"{_MODULE}.OTLPSpanExporter", create=True),
            patch(f"{_MODULE}.BatchSpanProcessor", create=True),
            patch(f"{_MODULE}.Resource", create=True) as mock_resource_cls,
            patch(f"{_MODULE}.trace", create=True),
            patch(f"{_MODULE}.BedrockInstrumentor", create=True) as mock_bedrock,
            patch.dict(os.environ, {}, clear=True),
        ):
            mock_bedrock.return_value = Mock()

            initialize_telemetry()

            mock_resource_cls.create.assert_called_once_with({
                "service.name": "opensearch-agent-server",
                "openinference.project.name": "opensearch-agent-server",
            })

    def test_returns_false_on_exception(self):
        """Returns False (does not raise) when any initialization step throws."""
        with (
            patch(f"{_MODULE}.OTEL_AVAILABLE", True),
            patch(f"{_MODULE}.TracerProvider", create=True),
            patch(f"{_MODULE}.OTLPSpanExporter", create=True) as mock_exporter_cls,
            patch(f"{_MODULE}.BatchSpanProcessor", create=True),
            patch(f"{_MODULE}.Resource", create=True),
            patch(f"{_MODULE}.trace", create=True),
        ):
            mock_exporter_cls.side_effect = Exception("connection refused")

            result = initialize_telemetry(otel_endpoint="http://collector:4318")

            assert result is False
