"""
MCP (Model Context Protocol) Server Integration for Kortix

Exposes Kortix tools as MCP-compatible tools that can be used by
Claude Desktop, other MCP clients, and AI applications.
"""
from .server import MCPServer
from .protocol import (
    MCPRequest,
    MCPResponse,
    MCPTool,
    MCPError,
    MCPErrorCode,
)

__all__ = [
    "MCPServer",
    "MCPRequest",
    "MCPResponse",
    "MCPTool",
    "MCPError",
    "MCPErrorCode",
]
