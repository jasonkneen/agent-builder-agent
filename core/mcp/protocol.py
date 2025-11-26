"""
MCP Protocol Types and Constants

Implements the Model Context Protocol specification for tool communication.
https://modelcontextprotocol.io/
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum, IntEnum
import json


class MCPErrorCode(IntEnum):
    """Standard MCP error codes (JSON-RPC 2.0 compatible)."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # MCP-specific error codes
    TOOL_NOT_FOUND = -32000
    TOOL_EXECUTION_ERROR = -32001
    RESOURCE_NOT_FOUND = -32002
    PERMISSION_DENIED = -32003
    RATE_LIMITED = -32004
    VALIDATION_ERROR = -32005


class MCPMethod(str, Enum):
    """Standard MCP methods."""
    # Lifecycle
    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"
    SHUTDOWN = "shutdown"

    # Tools
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"

    # Resources
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_UNSUBSCRIBE = "resources/unsubscribe"

    # Prompts
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"

    # Logging
    LOGGING_SET_LEVEL = "logging/setLevel"

    # Notifications
    PROGRESS = "notifications/progress"
    RESOURCE_UPDATED = "notifications/resources/updated"
    RESOURCE_LIST_CHANGED = "notifications/resources/list_changed"
    TOOL_LIST_CHANGED = "notifications/tools/list_changed"
    PROMPT_LIST_CHANGED = "notifications/prompts/list_changed"


class MCPError(Exception):
    """MCP error response (also usable as exception)."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass
class MCPToolInputSchema:
    """JSON Schema for tool input parameters."""
    type: str = "object"
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    additional_properties: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "properties": self.properties,
            "required": self.required,
            "additionalProperties": self.additional_properties,
        }


@dataclass
class MCPTool:
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: MCPToolInputSchema

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema.to_dict(),
        }


@dataclass
class MCPResource:
    """MCP Resource definition."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "uri": self.uri,
            "name": self.name,
        }
        if self.description:
            result["description"] = self.description
        if self.mime_type:
            result["mimeType"] = self.mime_type
        return result


@dataclass
class MCPPrompt:
    """MCP Prompt definition."""
    name: str
    description: Optional[str] = None
    arguments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.arguments:
            result["arguments"] = self.arguments
        return result


@dataclass
class MCPServerCapabilities:
    """Server capabilities advertisement."""
    tools: bool = True
    resources: bool = False
    prompts: bool = False
    logging: bool = True

    def to_dict(self) -> Dict[str, Any]:
        caps = {}
        if self.tools:
            caps["tools"] = {}
        if self.resources:
            caps["resources"] = {"subscribe": True}
        if self.prompts:
            caps["prompts"] = {}
        if self.logging:
            caps["logging"] = {}
        return caps


@dataclass
class MCPServerInfo:
    """Server information for initialize response."""
    name: str
    version: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version}


@dataclass
class MCPRequest:
    """MCP JSON-RPC request."""
    method: str
    id: Optional[Union[str, int]] = None
    params: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPRequest":
        return cls(
            method=data.get("method", ""),
            id=data.get("id"),
            params=data.get("params"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MCPRequest":
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        result = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.id is not None:
            result["id"] = self.id
        if self.params is not None:
            result["params"] = self.params
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def is_notification(self) -> bool:
        return self.id is None


@dataclass
class MCPResponse:
    """MCP JSON-RPC response."""
    id: Optional[Union[str, int]]
    result: Optional[Any] = None
    error: Optional[MCPError] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        response = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            response["error"] = self.error.to_dict()
        else:
            response["result"] = self.result
        return response

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def success(cls, id: Optional[Union[str, int]], result: Any) -> "MCPResponse":
        return cls(id=id, result=result)

    @classmethod
    def failure(cls, id: Optional[Union[str, int]], code: int, message: str, data: Any = None) -> "MCPResponse":
        return cls(id=id, error=MCPError(code=code, message=message, data=data))


@dataclass
class MCPToolCallContent:
    """Content types for tool call results."""
    type: str  # "text", "image", "resource"
    text: Optional[str] = None
    data: Optional[str] = None  # Base64 encoded for images
    mime_type: Optional[str] = None
    uri: Optional[str] = None  # For resource type

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.type == "text" and self.text:
            result["text"] = self.text
        elif self.type == "image":
            result["data"] = self.data
            result["mimeType"] = self.mime_type
        elif self.type == "resource":
            result["resource"] = {"uri": self.uri}
        return result


@dataclass
class MCPToolCallResult:
    """Result of a tool call."""
    content: List[MCPToolCallContent]
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": [c.to_dict() for c in self.content],
            "isError": self.is_error,
        }

    @classmethod
    def text_result(cls, text: str, is_error: bool = False) -> "MCPToolCallResult":
        return cls(
            content=[MCPToolCallContent(type="text", text=text)],
            is_error=is_error
        )


def openai_schema_to_mcp_tool(schema: Dict[str, Any]) -> MCPTool:
    """Convert OpenAI function schema to MCP tool format."""
    func = schema.get("function", schema)
    params = func.get("parameters", {})

    input_schema = MCPToolInputSchema(
        type=params.get("type", "object"),
        properties=params.get("properties", {}),
        required=params.get("required", []),
    )

    return MCPTool(
        name=func.get("name", "unknown"),
        description=func.get("description", ""),
        input_schema=input_schema,
    )
