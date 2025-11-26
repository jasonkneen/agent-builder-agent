"""
MCP Server Implementation for Kortix

Provides a complete MCP server that exposes Kortix tools to any MCP client.
Supports stdio transport for CLI usage and can be extended for SSE/WebSocket.
"""
import asyncio
import json
import sys
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field
import traceback

from .protocol import (
    MCPRequest,
    MCPResponse,
    MCPTool,
    MCPToolCallResult,
    MCPToolCallContent,
    MCPServerCapabilities,
    MCPServerInfo,
    MCPError,
    MCPErrorCode,
    MCPMethod,
    openai_schema_to_mcp_tool,
)
from ..config import get_config, get_mcp_config


@dataclass
class ToolRegistry:
    """Registry for MCP tools with their handlers."""
    _tools: Dict[str, MCPTool] = field(default_factory=dict)
    _handlers: Dict[str, Callable] = field(default_factory=dict)
    _instances: Dict[str, Any] = field(default_factory=dict)

    def register_tool(self, tool: MCPTool, handler: Callable, instance: Any = None) -> None:
        """Register a tool with its handler function."""
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler
        if instance:
            self._instances[tool.name] = instance

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get tool definition by name."""
        return self._tools.get(name)

    def get_handler(self, name: str) -> Optional[Callable]:
        """Get tool handler by name."""
        return self._handlers.get(name)

    def get_instance(self, name: str) -> Optional[Any]:
        """Get tool instance by name."""
        return self._instances.get(name)

    def list_tools(self) -> List[MCPTool]:
        """List all registered tools."""
        return list(self._tools.values())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


class MCPServer:
    """
    MCP Server that exposes Kortix tools via the Model Context Protocol.

    Usage:
        server = MCPServer()
        server.register_unit(FilesTool)
        server.register_unit(TerminalTool)
        await server.run_stdio()
    """

    def __init__(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
    ):
        mcp_config = get_mcp_config()
        self.name = name or mcp_config.server_name
        self.version = version or mcp_config.server_version

        self.registry = ToolRegistry()
        self.capabilities = MCPServerCapabilities(
            tools=True,
            resources=False,
            prompts=False,
            logging=True,
        )

        self._initialized = False
        self._running = False
        self._method_handlers: Dict[str, Callable] = {}
        self._setup_method_handlers()

    def _setup_method_handlers(self) -> None:
        """Setup handlers for MCP methods."""
        self._method_handlers = {
            MCPMethod.INITIALIZE.value: self._handle_initialize,
            MCPMethod.SHUTDOWN.value: self._handle_shutdown,
            MCPMethod.TOOLS_LIST.value: self._handle_tools_list,
            MCPMethod.TOOLS_CALL.value: self._handle_tools_call,
            MCPMethod.LOGGING_SET_LEVEL.value: self._handle_set_log_level,
        }

    def register_unit(self, unit_class: Type) -> None:
        """
        Register a Kortix Unit class, converting its schema to MCP tools.

        Args:
            unit_class: A Unit subclass with a schema() method
        """
        # Create instance for tool execution
        try:
            instance = unit_class()
        except Exception:
            # If instantiation fails, we'll create on-demand
            instance = None

        # Get schema and convert to MCP tools
        schemas = unit_class.schema()
        for schema in schemas:
            mcp_tool = openai_schema_to_mcp_tool(schema)

            # Get the handler method from the instance or class
            func_name = schema.get("function", {}).get("name", "")
            if instance and hasattr(instance, func_name):
                handler = getattr(instance, func_name)
            elif hasattr(unit_class, func_name):
                handler = getattr(unit_class, func_name)
            else:
                continue

            self.registry.register_tool(mcp_tool, handler, instance)

    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
    ) -> None:
        """
        Register a standalone function as an MCP tool.

        Args:
            name: Tool name
            description: Tool description
            parameters: JSON schema for parameters
            handler: Function to call when tool is invoked
        """
        from .protocol import MCPToolInputSchema

        input_schema = MCPToolInputSchema(
            type=parameters.get("type", "object"),
            properties=parameters.get("properties", {}),
            required=parameters.get("required", []),
        )

        tool = MCPTool(name=name, description=description, input_schema=input_schema)
        self.registry.register_tool(tool, handler)

    async def _handle_initialize(
        self, request: MCPRequest
    ) -> Dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.capabilities.to_dict(),
            "serverInfo": MCPServerInfo(self.name, self.version).to_dict(),
        }

    async def _handle_shutdown(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle shutdown request."""
        self._running = False
        return {}

    async def _handle_tools_list(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = self.registry.list_tools()
        return {"tools": [t.to_dict() for t in tools]}

    async def _handle_tools_call(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle tools/call request."""
        params = request.params or {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if not self.registry.has_tool(tool_name):
            raise MCPError(
                code=MCPErrorCode.TOOL_NOT_FOUND,
                message=f"Tool not found: {tool_name}",
            )

        handler = self.registry.get_handler(tool_name)
        if not handler:
            raise MCPError(
                code=MCPErrorCode.INTERNAL_ERROR,
                message=f"No handler for tool: {tool_name}",
            )

        try:
            # Execute the tool
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = await asyncio.to_thread(handler, **arguments)

            # Convert result to MCP format
            return self._format_tool_result(result)

        except Exception as e:
            return MCPToolCallResult.text_result(
                f"Error executing tool: {str(e)}\n{traceback.format_exc()}",
                is_error=True
            ).to_dict()

    def _format_tool_result(self, result: Any) -> Dict[str, Any]:
        """Convert tool result to MCP format."""
        # Handle UnitResult type
        if hasattr(result, 'success') and hasattr(result, 'output'):
            return MCPToolCallResult.text_result(
                result.output,
                is_error=not result.success
            ).to_dict()

        # Handle string
        if isinstance(result, str):
            return MCPToolCallResult.text_result(result).to_dict()

        # Handle dict/list - serialize to JSON
        if isinstance(result, (dict, list)):
            return MCPToolCallResult.text_result(
                json.dumps(result, indent=2)
            ).to_dict()

        # Default: convert to string
        return MCPToolCallResult.text_result(str(result)).to_dict()

    async def _handle_set_log_level(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle logging/setLevel request."""
        params = request.params or {}
        level = params.get("level", "info")
        # TODO: Actually update log level
        return {}

    async def handle_request(self, request: MCPRequest) -> Optional[MCPResponse]:
        """
        Process an incoming MCP request and return a response.

        Args:
            request: The MCP request to process

        Returns:
            MCPResponse or None for notifications
        """
        # Notifications don't need responses
        if request.is_notification():
            return None

        method = request.method
        handler = self._method_handlers.get(method)

        if not handler:
            return MCPResponse.failure(
                id=request.id,
                code=MCPErrorCode.METHOD_NOT_FOUND,
                message=f"Unknown method: {method}",
            )

        try:
            result = await handler(request)
            return MCPResponse.success(id=request.id, result=result)
        except MCPError as e:
            return MCPResponse.failure(
                id=request.id,
                code=e.code,
                message=e.message,
                data=e.data,
            )
        except Exception as e:
            return MCPResponse.failure(
                id=request.id,
                code=MCPErrorCode.INTERNAL_ERROR,
                message=str(e),
                data=traceback.format_exc(),
            )

    async def run_stdio(self) -> None:
        """
        Run the MCP server using stdio transport.

        Reads JSON-RPC messages from stdin and writes responses to stdout.
        This is the standard transport for MCP servers.
        """
        self._running = True
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)

        loop = asyncio.get_event_loop()
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        # For stdout, we write directly
        async def write_response(response: MCPResponse) -> None:
            output = response.to_json() + "\n"
            sys.stdout.write(output)
            sys.stdout.flush()

        while self._running:
            try:
                # Read a line from stdin
                line = await reader.readline()
                if not line:
                    break

                line = line.decode('utf-8').strip()
                if not line:
                    continue

                # Parse request
                try:
                    request = MCPRequest.from_json(line)
                except json.JSONDecodeError as e:
                    response = MCPResponse.failure(
                        id=None,
                        code=MCPErrorCode.PARSE_ERROR,
                        message=f"Invalid JSON: {str(e)}",
                    )
                    await write_response(response)
                    continue

                # Handle request
                response = await self.handle_request(request)
                if response:
                    await write_response(response)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue running
                sys.stderr.write(f"Error processing request: {e}\n")
                sys.stderr.flush()

    async def handle_message(self, message: str) -> Optional[str]:
        """
        Handle a single message and return the response.

        Useful for non-stdio transports (HTTP, WebSocket, etc.)

        Args:
            message: JSON-RPC message string

        Returns:
            JSON response string or None for notifications
        """
        try:
            request = MCPRequest.from_json(message)
            response = await self.handle_request(request)
            return response.to_json() if response else None
        except json.JSONDecodeError as e:
            return MCPResponse.failure(
                id=None,
                code=MCPErrorCode.PARSE_ERROR,
                message=f"Invalid JSON: {str(e)}",
            ).to_json()


def create_mcp_server() -> MCPServer:
    """
    Create an MCP server with all Kortix tools registered.

    Returns:
        Configured MCPServer instance
    """
    server = MCPServer()

    # Register available tools
    # Import here to avoid circular imports
    try:
        from ..units.files_tool import FilesTool
        server.register_unit(FilesTool)
    except Exception:
        pass

    try:
        from ..units.terminal_tool import TerminalTool
        server.register_unit(TerminalTool)
    except Exception:
        pass

    return server


async def main() -> None:
    """Main entry point for MCP server."""
    server = create_mcp_server()
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
