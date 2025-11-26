"""
Centralized Configuration Management for Kortix Agent Framework

Uses Pydantic for validation and environment-based configuration.
All settings can be overridden via environment variables.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from functools import lru_cache
import json


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    logs_db_path: str = field(default_factory=lambda: os.getenv(
        "KORTIX_LOGS_DB",
        str(Path.cwd() / "data" / "logs.db")
    ))
    working_memory_db_path: str = field(default_factory=lambda: os.getenv(
        "KORTIX_WORKING_MEMORY_DB",
        str(Path.cwd() / "data" / "working_memory.db")
    ))
    pool_size: int = field(default_factory=lambda: int(os.getenv("KORTIX_DB_POOL_SIZE", "5")))
    max_overflow: int = field(default_factory=lambda: int(os.getenv("KORTIX_DB_MAX_OVERFLOW", "10")))
    pool_timeout: float = field(default_factory=lambda: float(os.getenv("KORTIX_DB_POOL_TIMEOUT", "30.0")))
    check_same_thread: bool = False


@dataclass
class LLMConfig:
    """LLM API configuration settings."""
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    default_model: str = field(default_factory=lambda: os.getenv("KORTIX_DEFAULT_MODEL", "gpt-4o"))
    default_temperature: float = field(default_factory=lambda: float(os.getenv("KORTIX_DEFAULT_TEMPERATURE", "0")))
    default_max_tokens: Optional[int] = field(default_factory=lambda: (
        int(os.getenv("KORTIX_DEFAULT_MAX_TOKENS")) if os.getenv("KORTIX_DEFAULT_MAX_TOKENS") else None
    ))
    # Retry settings
    max_retries: int = field(default_factory=lambda: int(os.getenv("KORTIX_LLM_MAX_RETRIES", "3")))
    retry_base_delay: float = field(default_factory=lambda: float(os.getenv("KORTIX_LLM_RETRY_DELAY", "1.0")))
    retry_max_delay: float = field(default_factory=lambda: float(os.getenv("KORTIX_LLM_RETRY_MAX_DELAY", "60.0")))
    request_timeout: float = field(default_factory=lambda: float(os.getenv("KORTIX_LLM_TIMEOUT", "120.0")))


@dataclass
class WorkspaceConfig:
    """Workspace and file system configuration."""
    base_path: str = field(default_factory=lambda: os.getenv(
        "KORTIX_WORKSPACE_PATH",
        str(Path.cwd() / "workspace")
    ))
    terminal_logs_dir: str = field(default_factory=lambda: os.getenv(
        "KORTIX_TERMINAL_LOGS_DIR",
        str(Path.cwd() / "workspace" / "terminal_logs")
    ))
    docker_container_name: Optional[str] = field(default_factory=lambda: os.getenv(
        "KORTIX_DOCKER_CONTAINER",
        "workspace_dev-env_1"
    ))
    max_file_size_bytes: int = field(default_factory=lambda: int(os.getenv(
        "KORTIX_MAX_FILE_SIZE",
        str(10 * 1024 * 1024)  # 10MB default
    )))


@dataclass
class MCPConfig:
    """MCP Server configuration."""
    enabled: bool = field(default_factory=lambda: os.getenv("KORTIX_MCP_ENABLED", "true").lower() == "true")
    server_name: str = field(default_factory=lambda: os.getenv("KORTIX_MCP_SERVER_NAME", "kortix-agent"))
    server_version: str = field(default_factory=lambda: os.getenv("KORTIX_MCP_SERVER_VERSION", "1.0.0"))
    transport: str = field(default_factory=lambda: os.getenv("KORTIX_MCP_TRANSPORT", "stdio"))  # stdio, sse, websocket
    host: str = field(default_factory=lambda: os.getenv("KORTIX_MCP_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("KORTIX_MCP_PORT", "3000")))


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("KORTIX_LOG_LEVEL", "INFO"))
    format: str = field(default_factory=lambda: os.getenv(
        "KORTIX_LOG_FORMAT",
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    ))
    rotation: str = field(default_factory=lambda: os.getenv("KORTIX_LOG_ROTATION", "10 MB"))
    retention: str = field(default_factory=lambda: os.getenv("KORTIX_LOG_RETENTION", "7 days"))
    serialize_json: bool = field(default_factory=lambda: os.getenv("KORTIX_LOG_JSON", "false").lower() == "true")


@dataclass
class MiddlewareConfig:
    """Middleware configuration."""
    enabled_middleware: List[str] = field(default_factory=lambda: (
        os.getenv("KORTIX_MIDDLEWARE", "logging,validation,retry").split(",")
    ))
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv("KORTIX_RATE_LIMIT", "100")))
    rate_limit_window_seconds: int = field(default_factory=lambda: int(os.getenv("KORTIX_RATE_LIMIT_WINDOW", "60")))


@dataclass
class Config:
    """Main configuration container."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    middleware: MiddlewareConfig = field(default_factory=MiddlewareConfig)

    # Application metadata
    app_name: str = field(default_factory=lambda: os.getenv("KORTIX_APP_NAME", "Kortix Agent Framework"))
    app_version: str = field(default_factory=lambda: os.getenv("KORTIX_APP_VERSION", "0.2.0"))
    debug: bool = field(default_factory=lambda: os.getenv("KORTIX_DEBUG", "false").lower() == "true")

    def __post_init__(self):
        """Ensure required directories exist."""
        Path(self.database.logs_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.database.working_memory_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.workspace.base_path).mkdir(parents=True, exist_ok=True)
        Path(self.workspace.terminal_logs_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary (hiding sensitive values)."""
        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for field_name in obj.__dataclass_fields__:
                    value = getattr(obj, field_name)
                    # Hide API keys
                    if 'key' in field_name.lower() or 'password' in field_name.lower():
                        result[field_name] = '***' if value else None
                    else:
                        result[field_name] = _sanitize(value)
                return result
            return obj
        return _sanitize(self)

    @classmethod
    def from_env_file(cls, env_file: str = ".env") -> "Config":
        """Load configuration from .env file if it exists."""
        env_path = Path(env_file)
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip().strip('"\''))
        return cls()


# Singleton configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = Config.from_env_file()
    return _config


def reset_config() -> None:
    """Reset configuration (useful for testing)."""
    global _config
    _config = None


# Convenience accessors
def get_db_config() -> DatabaseConfig:
    return get_config().database


def get_llm_config() -> LLMConfig:
    return get_config().llm


def get_workspace_config() -> WorkspaceConfig:
    return get_config().workspace


def get_mcp_config() -> MCPConfig:
    return get_config().mcp


def get_logging_config() -> LoggingConfig:
    return get_config().logging
