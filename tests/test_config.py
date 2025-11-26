"""Tests for the configuration system."""
import os
import pytest
from pathlib import Path
from unittest.mock import patch


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        # Clear any cached config
        import core.config as config_module
        config_module.reset_config()

        from core.config import DatabaseConfig
        config = DatabaseConfig()

        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.pool_timeout == 30.0
        assert config.check_same_thread is False

    def test_env_override(self):
        """Test environment variable overrides."""
        import core.config as config_module
        config_module.reset_config()

        with patch.dict(os.environ, {"KORTIX_DB_POOL_SIZE": "10"}):
            from core.config import DatabaseConfig
            config = DatabaseConfig()
            assert config.pool_size == 10


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_model(self):
        """Test default model setting."""
        import core.config as config_module
        config_module.reset_config()

        from core.config import LLMConfig
        config = LLMConfig()

        assert config.default_model == "gpt-4o"
        assert config.default_temperature == 0.0
        assert config.max_retries == 3

    def test_api_key_from_env(self):
        """Test API key loading from environment."""
        import core.config as config_module
        config_module.reset_config()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
            from core.config import LLMConfig
            config = LLMConfig()
            assert config.openai_api_key == "test-key-123"


class TestWorkspaceConfig:
    """Tests for WorkspaceConfig."""

    def test_default_paths(self):
        """Test default workspace paths."""
        import core.config as config_module
        config_module.reset_config()

        from core.config import WorkspaceConfig
        config = WorkspaceConfig()

        assert "workspace" in config.base_path
        assert "terminal_logs" in config.terminal_logs_dir


class TestMCPConfig:
    """Tests for MCPConfig."""

    def test_default_values(self):
        """Test default MCP configuration."""
        import core.config as config_module
        config_module.reset_config()

        from core.config import MCPConfig
        config = MCPConfig()

        assert config.enabled is True
        assert config.server_name == "kortix-agent"
        assert config.transport == "stdio"
        assert config.port == 3000


class TestConfig:
    """Tests for main Config class."""

    def test_singleton_pattern(self):
        """Test that get_config returns the same instance."""
        import core.config as config_module
        config_module.reset_config()

        from core.config import get_config
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reset_config(self):
        """Test config reset functionality."""
        import core.config as config_module
        config_module.reset_config()

        from core.config import get_config, reset_config
        config1 = get_config()
        reset_config()
        config2 = get_config()

        assert config1 is not config2

    def test_to_dict_hides_secrets(self):
        """Test that to_dict hides sensitive values."""
        import core.config as config_module
        config_module.reset_config()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-key"}):
            from core.config import Config
            config = Config()
            config_dict = config.to_dict()

            # API keys should be masked
            assert config_dict["llm"]["openai_api_key"] == "***"

    def test_creates_directories(self, tmp_path):
        """Test that directories are created on init."""
        import core.config as config_module
        config_module.reset_config()

        logs_db = tmp_path / "logs" / "logs.db"
        wm_db = tmp_path / "wm" / "wm.db"
        workspace = tmp_path / "workspace"

        with patch.dict(os.environ, {
            "KORTIX_LOGS_DB": str(logs_db),
            "KORTIX_WORKING_MEMORY_DB": str(wm_db),
            "KORTIX_WORKSPACE_PATH": str(workspace),
            "KORTIX_TERMINAL_LOGS_DIR": str(workspace / "terminal_logs"),
        }):
            from core.config import Config
            config = Config()

            assert logs_db.parent.exists()
            assert wm_db.parent.exists()
            assert workspace.exists()


class TestConfigFromEnvFile:
    """Tests for loading config from .env file."""

    def test_loads_env_file(self, tmp_path):
        """Test loading configuration from .env file."""
        import core.config as config_module
        config_module.reset_config()

        env_file = tmp_path / ".env"
        env_file.write_text('KORTIX_APP_NAME="Test App"\nKORTIX_DEBUG=true\n')

        from core.config import Config
        config = Config.from_env_file(str(env_file))

        assert config.app_name == "Test App"
        assert config.debug is True

    def test_handles_missing_env_file(self):
        """Test graceful handling of missing .env file."""
        import core.config as config_module
        config_module.reset_config()

        from core.config import Config
        # Should not raise
        config = Config.from_env_file("/nonexistent/.env")
        assert config is not None
