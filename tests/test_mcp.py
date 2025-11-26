"""Tests for MCP Server integration."""
import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch

# Set environment variables before importing
import os
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")


class TestMCPProtocol:
    """Tests for MCP protocol types."""

    def test_mcp_request_from_json(self):
        """Test parsing MCP request from JSON."""
        from core.mcp.protocol import MCPRequest

        json_str = '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
        request = MCPRequest.from_json(json_str)

        assert request.method == "tools/list"
        assert request.id == 1
        assert request.jsonrpc == "2.0"

    def test_mcp_request_to_json(self):
        """Test converting MCP request to JSON."""
        from core.mcp.protocol import MCPRequest

        request = MCPRequest(method="tools/call", id=1, params={"name": "test"})
        json_str = request.to_json()
        parsed = json.loads(json_str)

        assert parsed["method"] == "tools/call"
        assert parsed["id"] == 1
        assert parsed["params"]["name"] == "test"

    def test_mcp_request_is_notification(self):
        """Test identifying notification requests."""
        from core.mcp.protocol import MCPRequest

        notification = MCPRequest(method="notifications/initialized")
        request = MCPRequest(method="tools/list", id=1)

        assert notification.is_notification() is True
        assert request.is_notification() is False

    def test_mcp_response_success(self):
        """Test creating success response."""
        from core.mcp.protocol import MCPResponse

        response = MCPResponse.success(id=1, result={"tools": []})
        json_str = response.to_json()
        parsed = json.loads(json_str)

        assert parsed["id"] == 1
        assert parsed["result"] == {"tools": []}
        assert "error" not in parsed

    def test_mcp_response_failure(self):
        """Test creating error response."""
        from core.mcp.protocol import MCPResponse, MCPErrorCode

        response = MCPResponse.failure(
            id=1,
            code=MCPErrorCode.METHOD_NOT_FOUND,
            message="Method not found"
        )
        json_str = response.to_json()
        parsed = json.loads(json_str)

        assert parsed["id"] == 1
        assert parsed["error"]["code"] == MCPErrorCode.METHOD_NOT_FOUND
        assert parsed["error"]["message"] == "Method not found"

    def test_mcp_tool_to_dict(self):
        """Test converting MCP tool to dict."""
        from core.mcp.protocol import MCPTool, MCPToolInputSchema

        schema = MCPToolInputSchema(
            properties={"path": {"type": "string"}},
            required=["path"]
        )
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            input_schema=schema
        )
        tool_dict = tool.to_dict()

        assert tool_dict["name"] == "read_file"
        assert tool_dict["description"] == "Read a file"
        assert tool_dict["inputSchema"]["properties"]["path"]["type"] == "string"

    def test_openai_schema_to_mcp_tool(self):
        """Test converting OpenAI function schema to MCP tool."""
        from core.mcp.protocol import openai_schema_to_mcp_tool

        openai_schema = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }

        mcp_tool = openai_schema_to_mcp_tool(openai_schema)

        assert mcp_tool.name == "get_weather"
        assert mcp_tool.description == "Get weather for a location"
        assert "location" in mcp_tool.input_schema.properties

    def test_mcp_tool_call_result(self):
        """Test MCP tool call result formatting."""
        from core.mcp.protocol import MCPToolCallResult

        result = MCPToolCallResult.text_result("Hello, world!")
        result_dict = result.to_dict()

        assert result_dict["isError"] is False
        assert len(result_dict["content"]) == 1
        assert result_dict["content"][0]["type"] == "text"
        assert result_dict["content"][0]["text"] == "Hello, world!"

    def test_mcp_tool_call_result_error(self):
        """Test MCP tool call result with error."""
        from core.mcp.protocol import MCPToolCallResult

        result = MCPToolCallResult.text_result("Error occurred", is_error=True)
        result_dict = result.to_dict()

        assert result_dict["isError"] is True


class TestToolRegistry:
    """Tests for tool registry."""

    def test_register_and_get_tool(self):
        """Test registering and retrieving a tool."""
        from core.mcp.server import ToolRegistry
        from core.mcp.protocol import MCPTool, MCPToolInputSchema

        registry = ToolRegistry()

        schema = MCPToolInputSchema()
        tool = MCPTool(name="test_tool", description="Test", input_schema=schema)

        def handler(**kwargs):
            return "result"

        registry.register_tool(tool, handler)

        assert registry.has_tool("test_tool")
        assert registry.get_tool("test_tool") == tool
        assert registry.get_handler("test_tool") == handler

    def test_list_tools(self):
        """Test listing all registered tools."""
        from core.mcp.server import ToolRegistry
        from core.mcp.protocol import MCPTool, MCPToolInputSchema

        registry = ToolRegistry()
        schema = MCPToolInputSchema()

        tool1 = MCPTool(name="tool1", description="Tool 1", input_schema=schema)
        tool2 = MCPTool(name="tool2", description="Tool 2", input_schema=schema)

        registry.register_tool(tool1, lambda: None)
        registry.register_tool(tool2, lambda: None)

        tools = registry.list_tools()
        assert len(tools) == 2
        assert any(t.name == "tool1" for t in tools)
        assert any(t.name == "tool2" for t in tools)


class TestMCPServer:
    """Tests for MCP server."""

    @pytest.fixture
    def server(self):
        """Create MCP server instance."""
        from core.mcp.server import MCPServer
        return MCPServer(name="test-server", version="1.0.0")

    @pytest.mark.asyncio
    async def test_handle_initialize(self, server):
        """Test handling initialize request."""
        from core.mcp.protocol import MCPRequest

        request = MCPRequest(method="initialize", id=1)
        response = await server.handle_request(request)

        assert response.error is None
        assert response.result["serverInfo"]["name"] == "test-server"
        assert "capabilities" in response.result

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, server):
        """Test handling tools/list request."""
        from core.mcp.protocol import MCPRequest, MCPTool, MCPToolInputSchema

        # Register a test tool
        schema = MCPToolInputSchema()
        tool = MCPTool(name="test", description="Test tool", input_schema=schema)
        server.registry.register_tool(tool, lambda: "result")

        request = MCPRequest(method="tools/list", id=1)
        response = await server.handle_request(request)

        assert response.error is None
        assert len(response.result["tools"]) == 1
        assert response.result["tools"][0]["name"] == "test"

    @pytest.mark.asyncio
    async def test_handle_tools_call(self, server):
        """Test handling tools/call request."""
        from core.mcp.protocol import MCPRequest, MCPTool, MCPToolInputSchema

        # Register a test tool
        schema = MCPToolInputSchema(
            properties={"value": {"type": "string"}},
            required=["value"]
        )
        tool = MCPTool(name="echo", description="Echo value", input_schema=schema)

        def echo_handler(value: str):
            return f"Echo: {value}"

        server.registry.register_tool(tool, echo_handler)

        request = MCPRequest(
            method="tools/call",
            id=1,
            params={"name": "echo", "arguments": {"value": "hello"}}
        )
        response = await server.handle_request(request)

        assert response.error is None
        assert "Echo: hello" in response.result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_tools_call_not_found(self, server):
        """Test handling tools/call for non-existent tool."""
        from core.mcp.protocol import MCPRequest, MCPErrorCode

        request = MCPRequest(
            method="tools/call",
            id=1,
            params={"name": "nonexistent", "arguments": {}}
        )
        response = await server.handle_request(request)

        assert response.error is not None
        assert response.error.code == MCPErrorCode.TOOL_NOT_FOUND

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, server):
        """Test handling unknown method."""
        from core.mcp.protocol import MCPRequest, MCPErrorCode

        request = MCPRequest(method="unknown/method", id=1)
        response = await server.handle_request(request)

        assert response.error is not None
        assert response.error.code == MCPErrorCode.METHOD_NOT_FOUND

    @pytest.mark.asyncio
    async def test_notification_returns_none(self, server):
        """Test that notifications don't return a response."""
        from core.mcp.protocol import MCPRequest

        # Notification has no id
        request = MCPRequest(method="notifications/initialized")
        response = await server.handle_request(request)

        assert response is None

    def test_register_function(self, server):
        """Test registering a standalone function."""
        def my_function(arg1: str) -> str:
            return f"Result: {arg1}"

        server.register_function(
            name="my_function",
            description="A test function",
            parameters={
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
                "required": ["arg1"]
            },
            handler=my_function
        )

        assert server.registry.has_tool("my_function")

    @pytest.mark.asyncio
    async def test_handle_message(self, server):
        """Test handling raw message string."""
        message = '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
        response_str = await server.handle_message(message)
        response = json.loads(response_str)

        assert response["id"] == 1
        assert "result" in response

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, server):
        """Test handling invalid JSON message."""
        response_str = await server.handle_message("not valid json")
        response = json.loads(response_str)

        assert response["error"]["code"] == -32700  # Parse error
