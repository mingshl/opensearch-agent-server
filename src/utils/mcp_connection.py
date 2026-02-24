"""
MCP Connection Utility for OpenSearch
Provides shared access to OpenSearch MCP tools across multiple agents.
"""

from __future__ import annotations

import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from strands.tools.decorator import tool

from utils.logging_helpers import (
    get_logger,
    log_debug_event,  # noqa: F401 (used in MCP client methods)
    log_error_event,
    log_info_event,
    log_warning_event,
)

logger = get_logger(__name__)

# Import monitor - will be available if in Chainlit context
# Chainlit monitor removed — not used in remote-agent-server
MONITOR_AVAILABLE = False
get_monitor = None

# Import AG-UI emitter - will be available if in AG-UI context
try:
    from utils.tool_event_emitter import get_ag_ui_emitter

    GET_AG_UI_EMITTER_AVAILABLE = True
except ImportError:
    GET_AG_UI_EMITTER_AVAILABLE = False
    get_ag_ui_emitter = None


class MCPConnectionManager:
    """Manages MCP connection to OpenSearch and provides wrapped tools."""

    def __init__(
        self,
        opensearch_url: str | None = None,
        username: str = "admin",
        password: str = "admin",
    ) -> None:
        """Initialize MCP connection manager.

        Args:
            opensearch_url: OpenSearch server URL (defaults to OPENSEARCH_URL env var or localhost:9200)
            username: OpenSearch username (defaults to "admin")
            password: OpenSearch password (defaults to "admin")
        """
        self.opensearch_url = opensearch_url or os.getenv(
            "OPENSEARCH_URL",
            "http://localhost:9200",
        )
        self.username = username
        self.password = password
        self.read_stream = None
        self.write_stream = None
        self.session = None
        self._stdio_ctx = None
        self._session_ctx = None
        self.mcp_tools = []

    async def initialize(self) -> list:
        """Initialize the MCP session and wrap OpenSearch tools."""
        # Get absolute path to config file in the same directory as this script
        config_path = os.path.join(os.path.dirname(__file__), "os_mcp_config.yml")
        log_info_event(
            logger,
            f"Initializing MCP with config: {config_path}",
            "mcp.initializing",
            config_path=config_path,
        )

        # Use sys.executable to ensure the MCP server runs in the same
        # Python environment (venv) as this process.
        import sys

        mcp_command = os.getenv("MCP_SERVER_COMMAND", sys.executable)

        # Inherit current environment so subprocess has PATH, VIRTUAL_ENV, etc.
        # Then overlay OpenSearch connection settings.
        mcp_env = {**os.environ}
        mcp_env.update({
            "OPENSEARCH_URL": self.opensearch_url,
            "OPENSEARCH_USERNAME": self.username,
            "OPENSEARCH_PASSWORD": self.password,
        })

        server_params = StdioServerParameters(
            command=mcp_command,
            args=["-m", "mcp_server_opensearch", "--config", config_path],
            env=mcp_env,
        )

        # Enter stdio_client context
        self._stdio_ctx = stdio_client(server_params)
        self.read_stream, self.write_stream = await self._stdio_ctx.__aenter__()
        log_info_event(logger, "[MCP] Stdio client connected", "mcp.stdio_connected")

        # Enter ClientSession context
        self._session_ctx = ClientSession(self.read_stream, self.write_stream)
        self.session = await self._session_ctx.__aenter__()
        log_info_event(logger, "[MCP] ClientSession created", "mcp.session_created")

        # Initialize the session
        await self.session.initialize()
        log_info_event(logger, "[MCP] Session initialized", "mcp.session_initialized")

        # Get raw tool definitions from the active MCP session
        raw_mcp_tools = await self.session.list_tools()
        log_info_event(
            logger,
            f"[MCP] Discovered {len(raw_mcp_tools.tools)} OpenSearch tools",
            "mcp.tools_discovered",
            tool_count=len(raw_mcp_tools.tools),
        )

        # Create tool wrappers using @tool decorator
        mcp_tools = []

        for tool_def in raw_mcp_tools.tools:
            wrapped_tool = self._create_tool_wrapper(tool_def)
            if wrapped_tool:
                mcp_tools.append(wrapped_tool)
                log_info_event(
                    logger,
                    f"[MCP] Wrapped tool: {tool_def.name}",
                    "mcp.tool_wrapped",
                    tool_name=tool_def.name,
                )

        self.mcp_tools = mcp_tools
        return mcp_tools

    def _create_tool_wrapper(self, tool_def: Any) -> Any:
        """Create a Strands-compatible tool wrapper for an MCP tool.

        This method dynamically generates a wrapper function that:
        1. Extracts parameters from the MCP tool's input schema
        2. Wraps the tool call with Chainlit monitoring (if available)
        3. Wraps the tool call with AG-UI event emission (if available)
        4. Calls the actual MCP tool via the active session
        5. Returns the tool result

        The generated wrapper function is decorated with @tool to make it
        available to Strands agents.

        Args:
            tool_def: MCP tool definition object with name, description, and inputSchema

        Returns:
            Decorated tool function ready for use by Strands agents, or None if creation fails
        """
        try:
            # Build parameter list from the input schema
            params = []
            defaults = []
            schema = getattr(tool_def, "inputSchema", None) or {}
            schema_props = schema.get("properties", {})

            for param_name, param_info in schema_props.items():
                params.append(param_name)
                if "default" in param_info:
                    defaults.append(param_info["default"])

            # Build parameter string with defaults
            if defaults:
                non_default_count = len(params) - len(defaults)
                param_strs = params[:non_default_count]
                param_strs += [f"{p}=None" for p in params[non_default_count:]]
                param_signature = ", ".join(param_strs)
            else:
                param_signature = ", ".join(params) if params else ""

            # Create function code dynamically
            # Note: The generated wrapper function doesn't have a docstring because
            # it's dynamically created. The function signature and behavior are
            # documented in the _create_tool_wrapper method docstring above.
            func_code = f"""
async def {tool_def.name}_wrapper({param_signature}):
    \"\"\"Wrapper for MCP tool '{tool_def.name}': {tool_def.description or "No description"}.\"\"\"
    kwargs = {{}}
"""
            for param in params:
                func_code += f"    if {param} is not None:\n"
                func_code += f"        kwargs['{param}'] = {param}\n"

            func_code += f"""
    # Get current session from manager (not the captured one)
    current_session = manager.session
    if not current_session:
        raise RuntimeError("MCP session is not initialized or has been closed")

    # Get activity monitor from module-level import (if available)
    monitor = None
    if MONITOR_AVAILABLE and get_monitor:
        try:
            monitor = get_monitor()
            log_debug_event(
                logger,
                "[MCP Tool Wrapper] Monitor retrieved.",
                "mcp.monitor_retrieved",
                tool_name="{tool_def.name}",
            )
        except Exception as e:
            log_debug_event(
                logger,
                "[MCP Tool Wrapper] Failed to get monitor.",
                "mcp.monitor_get_failed",
                tool_name="{tool_def.name}",
                error=str(e),
            )
            pass  # Monitor not available in this context
    else:
        log_debug_event(
            logger,
            "[MCP Tool Wrapper] Monitor not available.",
            "mcp.monitor_not_available",
            monitor_available=MONITOR_AVAILABLE,
            get_monitor=get_monitor,
        )

    # Get AG-UI event emitter (if available)
    ag_ui_emitter = None
    if GET_AG_UI_EMITTER_AVAILABLE and get_ag_ui_emitter:
        try:
            ag_ui_emitter = get_ag_ui_emitter()
        except Exception as e:
            log_debug_event(
                logger,
                "[MCP Tool Wrapper] Failed to get AG-UI emitter.",
                "mcp.ag_ui_emitter_get_failed",
                tool_name="{tool_def.name}",
                error=str(e),
            )
            pass  # Emitter not available in this context

    # Use monitor and/or AG-UI emitter if available
    if monitor and ag_ui_emitter:
        # Both Chainlit monitor and AG-UI emitter available
        async with monitor.tool_call('{tool_def.name}', **kwargs) as step:
            async with ag_ui_emitter.tool_call('{tool_def.name}', **kwargs) as tool_call_id:
                try:
                    result = await current_session.call_tool('{tool_def.name}', kwargs)
                except Exception as e:
                    error_msg = str(e)
                    if "ClosedResourceError" in error_msg or "closed" in error_msg.lower():
                        return f"Error: MCP connection closed. Tool '{tool_def.name}' failed. The OpenSearch MCP server may have crashed or the connection was lost. Please refresh the page to reinitialize."
                    raise
                if isinstance(result.content, list) and len(result.content) > 0:
                    content_item = result.content[0]
                    if hasattr(content_item, 'text'):
                        result_str = content_item.text
                    else:
                        result_str = str(content_item)
                else:
                    result_str = str(result.content)

                # Set result for AG-UI emitter
                await ag_ui_emitter.set_tool_call_result(tool_call_id, result_str)
                # Show result preview in step output
                preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                step.output = "✓ Result: " + preview
                return result_str
    elif monitor:
        # Only Chainlit monitor available
        async with monitor.tool_call('{tool_def.name}', **kwargs) as step:
            try:
                result = await current_session.call_tool('{tool_def.name}', kwargs)
            except Exception as e:
                error_msg = str(e)
                if "ClosedResourceError" in error_msg or "closed" in error_msg.lower():
                    return f"Error: MCP connection closed. Tool '{tool_def.name}' failed. The OpenSearch MCP server may have crashed or the connection was lost. Please refresh the page to reinitialize."
                raise
            if isinstance(result.content, list) and len(result.content) > 0:
                content_item = result.content[0]
                if hasattr(content_item, 'text'):
                    result_str = content_item.text
                else:
                    result_str = str(content_item)
            else:
                result_str = str(result.content)

            # Show result preview in step output
            preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
            step.output = "✓ Result: " + preview
            return result_str
    elif ag_ui_emitter:
        # Only AG-UI emitter available
        async with ag_ui_emitter.tool_call('{tool_def.name}', **kwargs) as tool_call_id:
            try:
                result = await current_session.call_tool('{tool_def.name}', kwargs)
            except Exception as e:
                error_msg = str(e)
                if "ClosedResourceError" in error_msg or "closed" in error_msg.lower():
                    return f"Error: MCP connection closed. Tool '{tool_def.name}' failed. The OpenSearch MCP server may have crashed or the connection was lost. Please refresh the page to reinitialize."
                raise
            if isinstance(result.content, list) and len(result.content) > 0:
                content_item = result.content[0]
                if hasattr(content_item, 'text'):
                    result_str = content_item.text
                else:
                    result_str = str(content_item)
            else:
                result_str = str(result.content)
            # Set result for AG-UI emitter
            await ag_ui_emitter.set_tool_call_result(tool_call_id, result_str)
            return result_str
    else:
        # No monitoring available, execute normally
        try:
            result = await current_session.call_tool('{tool_def.name}', kwargs)
            if isinstance(result.content, list) and len(result.content) > 0:
                content_item = result.content[0]
                if hasattr(content_item, 'text'):
                    return content_item.text
                else:
                    return str(content_item)
            else:
                return str(result.content)
        except Exception as e:
            error_msg = str(e)
            if "ClosedResourceError" in error_msg or "closed" in error_msg.lower():
                return f"Error: MCP connection closed. Tool '{tool_def.name}' failed. The OpenSearch MCP server may have crashed or the connection was lost. Please refresh the page to reinitialize."
            raise
"""

            # Execute the function definition (include log_debug_event for wrapper body)
            local_vars = {
                "manager": self,
                "logger": logger,
                "log_debug_event": log_debug_event,
                "MONITOR_AVAILABLE": MONITOR_AVAILABLE,
                "get_monitor": get_monitor,
                "GET_AG_UI_EMITTER_AVAILABLE": GET_AG_UI_EMITTER_AVAILABLE,
                "get_ag_ui_emitter": get_ag_ui_emitter,
            }
            exec(func_code, local_vars)
            wrapper_func = local_vars[f"{tool_def.name}_wrapper"]

            # Apply the @tool decorator (schema may be {} if inputSchema was missing)
            decorated = tool(
                name=tool_def.name,
                description=tool_def.description or f"MCP tool: {tool_def.name}",
                inputSchema=schema,
            )(wrapper_func)

            return decorated
        except Exception as e:
            log_error_event(
                logger,
                f"[MCP] ✗ Failed to wrap tool: {tool_def.name}",
                "mcp.tool_wrap_failed",
                tool_name=tool_def.name,
                error=str(e),
            )
            return None

    async def cleanup(self) -> None:
        """Clean up the MCP session by closing streams and clearing references.

        Note: We don't explicitly call __aexit__ on contexts because they may be
        in different async tasks (anyio cancel scope limitation). Instead, we close
        the underlying streams and let the contexts clean up naturally.

        This method is idempotent - it can be called multiple times safely.
        """
        # Early return if already cleaned up
        if self.read_stream is None and self.write_stream is None:
            log_info_event(
                logger, "[MCP] Connection already cleaned up", "mcp.already_cleaned_up"
            )
            return

        try:
            # Close streams gracefully, checking if already closed
            if self.write_stream:
                try:
                    # Check if stream is already closed using _closed attribute (anyio streams)
                    if (
                        hasattr(self.write_stream, "_closed")
                        and self.write_stream._closed
                    ):
                        log_info_event(
                            logger,
                            "[MCP] Write stream already closed",
                            "mcp.write_stream_already_closed",
                        )
                    else:
                        await self.write_stream.aclose()
                        log_info_event(
                            logger,
                            "[MCP] Write stream closed",
                            "mcp.write_stream_closed",
                        )
                except (RuntimeError, ValueError) as e:
                    # Handle already-closed stream exceptions
                    error_msg = str(e).lower()
                    if "closed" in error_msg or "already closed" in error_msg:
                        log_info_event(
                            logger,
                            "[MCP] Write stream was already closed",
                            "mcp.write_stream_already_closed",
                        )
                    else:
                        log_warning_event(
                            logger,
                            f"[MCP] ✗ Error closing write stream: {e}",
                            "mcp.write_stream_close_error",
                            error=str(e),
                        )
                except Exception as e:
                    # Catch any other exceptions during close
                    log_warning_event(
                        logger,
                        f"[MCP] ✗ Error closing write stream: {e}",
                        "mcp.write_stream_close_error",
                        error=str(e),
                    )

            if self.read_stream:
                try:
                    # Check if stream is already closed using _closed attribute (anyio streams)
                    if (
                        hasattr(self.read_stream, "_closed")
                        and self.read_stream._closed
                    ):
                        log_info_event(
                            logger,
                            "[MCP] Read stream already closed",
                            "mcp.read_stream_already_closed",
                        )
                    else:
                        await self.read_stream.aclose()
                        log_info_event(
                            logger, "[MCP] Read stream closed", "mcp.read_stream_closed"
                        )
                except (RuntimeError, ValueError) as e:
                    # Handle already-closed stream exceptions
                    error_msg = str(e).lower()
                    if "closed" in error_msg or "already closed" in error_msg:
                        log_info_event(
                            logger,
                            "[MCP] Read stream was already closed",
                            "mcp.read_stream_already_closed",
                        )
                    else:
                        log_warning_event(
                            logger,
                            f"[MCP] ✗ Error closing read stream: {e}",
                            "mcp.read_stream_close_error",
                            error=str(e),
                        )
                except Exception as e:
                    # Catch any other exceptions during close
                    log_warning_event(
                        logger,
                        f"[MCP] ✗ Error closing read stream: {e}",
                        "mcp.read_stream_close_error",
                        error=str(e),
                    )

            log_info_event(
                logger,
                "[MCP] Connection cleaned up successfully",
                "mcp.cleanup_success",
            )
        except Exception as e:
            log_error_event(
                logger,
                f"[MCP] ✗ Error during cleanup: {e}",
                "mcp.cleanup_error",
                error=str(e),
            )
        finally:
            # Clear all references to allow garbage collection
            # Note: Setting _stdio_ctx to None may trigger garbage collection of the
            # async generator, which can cause a RuntimeError about cancel scopes.
            # This is expected and harmless since we've already closed the streams.
            # The asyncio exception handler in ag_ui_app.py will suppress this error.
            self.session = None
            self._session_ctx = None
            self._stdio_ctx = None
            self.read_stream = None
            self.write_stream = None
            self.mcp_tools = []

    def __del__(self) -> None:
        """Finalizer to handle cleanup during garbage collection.

        This is a best-effort cleanup method. Suppresses RuntimeError about
        cancel scopes that can occur when the async generator context manager
        is garbage collected in a different task than it was created in.
        This is expected behavior when streams are already closed.

        Note: This method is not guaranteed to be called deterministically.
        The asyncio exception handler in ag_ui_app.py is the primary defense
        against these errors. This method provides an additional safety net
        during garbage collection, but the actual cleanup happens in cleanup().
        """
        # Clear reference to prevent issues during GC
        # The actual cleanup happens in cleanup() which closes streams
        # This just prevents the async generator from being held longer than needed
        try:
            if hasattr(self, "_stdio_ctx"):
                self._stdio_ctx = None
        except Exception as e:
            # Log exception during finalization but don't raise
            # We can't do async cleanup here anyway, and __del__ must not raise
            # Use logging helper function (not async) since this is in __del__
            try:
                log_warning_event(
                    logger,
                    f"✗ Exception during MCPConnectionManager finalization: {e}",
                    "mcp.finalization_error",
                    error=str(e),
                    exc_info=True,
                )
            except Exception:
                # If logging fails too, suppress it (can happen during shutdown)
                pass

    def get_tools(self) -> list[Any]:
        """Get the list of wrapped MCP tools."""
        return self.mcp_tools
