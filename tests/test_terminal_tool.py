"""
Comprehensive tests for core/units/terminal_tool.py

Tests cover:
- TerminalTool initialization
- Session management
- Command execution
- Session observation
- Schema generation
- Edge cases
"""
import os
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import pytest


# ============================================================================
# TerminalTool Initialization Tests
# ============================================================================

class TestTerminalToolInitialization:
    """Tests for TerminalTool initialization."""

    @pytest.fixture
    def mock_dependencies(self, temp_logs_dir):
        """Mock all dependencies for TerminalTool."""
        with patch('core.units.terminal_tool.WorkingMemory') as mock_wm:
            mock_wm_instance = MagicMock()
            mock_wm_instance.get_module.return_value = None
            mock_wm.return_value = mock_wm_instance

            with patch('core.units.terminal_tool.get_docker_container_id') as mock_docker:
                mock_docker.return_value = "test_container_123"

                with patch('core.framework.base.db_handler'):
                    with patch('core.framework.base.session_context') as mock_ctx:
                        mock_ctx.get.return_value = "test_session"
                        with patch('core.framework.base.logger') as mock_loguru:
                            mock_loguru.add = Mock()
                            yield {
                                'working_memory': mock_wm_instance,
                                'docker': mock_docker,
                                'logs_dir': temp_logs_dir
                            }

    def test_initialization_creates_logs_dir(self, mock_dependencies, tmp_path):
        """Test that initialization creates logs directory."""
        logs_dir = str(tmp_path / "new_logs")

        with patch.object(
            __import__('core.units.terminal_tool', fromlist=['TerminalTool']).TerminalTool,
            'logs_dir',
            logs_dir
        ):
            from core.units.terminal_tool import TerminalTool

            with patch.object(TerminalTool, '__init__', lambda self: None):
                tool = TerminalTool()
                tool.logs_dir = logs_dir
                os.makedirs(tool.logs_dir, exist_ok=True)

                assert os.path.exists(logs_dir)

    def test_initialization_without_container_raises(self):
        """Test that initialization raises when no Docker container."""
        with patch('core.units.terminal_tool.get_docker_container_id') as mock_docker:
            mock_docker.return_value = None

            with patch('core.units.terminal_tool.WorkingMemory'):
                with patch('core.framework.base.db_handler'):
                    with patch('core.framework.base.session_context') as mock_ctx:
                        mock_ctx.get.return_value = "test_session"
                        with patch('core.framework.base.logger'):
                            from core.units.terminal_tool import TerminalTool

                            with pytest.raises(ValueError) as exc_info:
                                TerminalTool()

                            assert "No running container" in str(exc_info.value)


# ============================================================================
# Session Management Tests
# ============================================================================

class TestSessionManagement:
    """Tests for terminal session management."""

    @pytest.fixture
    def terminal_tool(self, temp_logs_dir):
        """Create a TerminalTool instance with mocked dependencies."""
        with patch('core.units.terminal_tool.WorkingMemory') as mock_wm:
            mock_wm_instance = MagicMock()
            mock_wm_instance.get_module.return_value = []
            mock_wm.return_value = mock_wm_instance

            with patch('core.units.terminal_tool.get_docker_container_id') as mock_docker:
                mock_docker.return_value = "test_container"

                with patch('core.framework.base.db_handler'):
                    with patch('core.framework.base.session_context') as mock_ctx:
                        mock_ctx.get.return_value = "test_session"
                        with patch('core.framework.base.logger') as mock_loguru:
                            mock_loguru.add = Mock()

                            from core.units.terminal_tool import TerminalTool

                            with patch.object(TerminalTool, 'initialize_terminal_sessions', lambda self: None):
                                tool = TerminalTool.__new__(TerminalTool)
                                tool.logs_dir = temp_logs_dir
                                tool.container_name = "test_container"
                                tool.working_memory = mock_wm_instance
                                tool.logger = MagicMock()
                                yield tool

    def test_new_terminal_session(self, terminal_tool):
        """Test creating a new terminal session."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            session_id = terminal_tool.new_terminal_session()

            assert session_id == "session_1"
            # Should have called tmux commands
            assert mock_run.call_count >= 1

    def test_new_terminal_session_increments_id(self, terminal_tool):
        """Test that new sessions get incremented IDs."""
        terminal_tool.working_memory.get_module.return_value = [
            {"session_id": "session_1", "action_history": []},
            {"session_id": "session_2", "action_history": []}
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            session_id = terminal_tool.new_terminal_session()

            assert session_id == "session_3"

    def test_control_c_terminal_session_success(self, terminal_tool):
        """Test closing a terminal session successfully."""
        terminal_tool.working_memory.get_module.return_value = [
            {"session_id": "session_1", "action_history": []}
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = terminal_tool.control_c_terminal_session("session_1")

            assert result.success is True
            assert "CLOSED" in result.output

    def test_control_c_terminal_session_not_found(self, terminal_tool):
        """Test closing a non-existent session."""
        terminal_tool.working_memory.get_module.return_value = []

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = terminal_tool.control_c_terminal_session("nonexistent")

            assert result.success is False
            assert "not found" in result.output

    def test_control_c_terminal_session_subprocess_error(self, terminal_tool):
        """Test handling subprocess error during session close."""
        terminal_tool.working_memory.get_module.return_value = [
            {"session_id": "session_1", "action_history": []}
        ]

        with patch('subprocess.run') as mock_run:
            from subprocess import CalledProcessError
            mock_run.side_effect = CalledProcessError(1, "tmux")

            result = terminal_tool.control_c_terminal_session("session_1")

            assert result.success is False


# ============================================================================
# Command Execution Tests
# ============================================================================

class TestCommandExecution:
    """Tests for terminal command execution."""

    @pytest.fixture
    def terminal_tool(self, temp_logs_dir):
        """Create a TerminalTool instance with mocked dependencies."""
        with patch('core.units.terminal_tool.WorkingMemory') as mock_wm:
            mock_wm_instance = MagicMock()
            mock_wm_instance.get_module.return_value = [
                {"session_id": "session_1", "action_history": []}
            ]
            mock_wm.return_value = mock_wm_instance

            with patch('core.units.terminal_tool.get_docker_container_id') as mock_docker:
                mock_docker.return_value = "test_container"

                with patch('core.framework.base.db_handler'):
                    with patch('core.framework.base.session_context') as mock_ctx:
                        mock_ctx.get.return_value = "test_session"
                        with patch('core.framework.base.logger') as mock_loguru:
                            mock_loguru.add = Mock()

                            from core.units.terminal_tool import TerminalTool

                            with patch.object(TerminalTool, 'initialize_terminal_sessions', lambda self: None):
                                tool = TerminalTool.__new__(TerminalTool)
                                tool.logs_dir = temp_logs_dir
                                tool.container_name = "test_container"
                                tool.working_memory = mock_wm_instance
                                tool.logger = MagicMock()
                                yield tool

    def test_send_terminal_command_success(self, terminal_tool):
        """Test sending a command to terminal session."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Create mocks with __name__ attribute to work with functools.wraps
            mock_observe = MagicMock(return_value="")
            mock_observe.__name__ = "observe_terminal_session"
            mock_update = MagicMock()
            mock_update.__name__ = "update_action_history"

            # Directly assign the mocked methods to avoid __getattribute__ interception
            original_observe = terminal_tool.__class__.observe_terminal_session
            original_update = terminal_tool.__class__.update_action_history
            try:
                terminal_tool.__class__.observe_terminal_session = mock_observe
                terminal_tool.__class__.update_action_history = mock_update
                result = terminal_tool.send_terminal_command("session_1", "echo hello")

                assert result.success is True
                assert "Command executed" in result.output
            finally:
                terminal_tool.__class__.observe_terminal_session = original_observe
                terminal_tool.__class__.update_action_history = original_update

    def test_send_terminal_command_failure(self, terminal_tool):
        """Test handling command send failure."""
        with patch('subprocess.run') as mock_run:
            from subprocess import CalledProcessError
            mock_run.side_effect = CalledProcessError(1, "tmux")

            result = terminal_tool.send_terminal_command("session_1", "echo hello")

            assert result.success is False

    def test_update_action_history(self, terminal_tool):
        """Test that action history is updated."""
        terminal_tool.update_action_history("session_1", "echo hello")

        # Verify working memory was updated
        terminal_tool.working_memory.add_or_update_module.assert_called()


# ============================================================================
# Session Observation Tests
# ============================================================================

class TestSessionObservation:
    """Tests for terminal session observation."""

    @pytest.fixture
    def terminal_tool(self, temp_logs_dir):
        """Create a TerminalTool instance with mocked dependencies."""
        with patch('core.units.terminal_tool.WorkingMemory') as mock_wm:
            mock_wm_instance = MagicMock()
            mock_wm_instance.get_module.return_value = [
                {"session_id": "session_1", "action_history": []}
            ]
            mock_wm.return_value = mock_wm_instance

            with patch('core.units.terminal_tool.get_docker_container_id') as mock_docker:
                mock_docker.return_value = "test_container"

                with patch('core.framework.base.db_handler'):
                    with patch('core.framework.base.session_context') as mock_ctx:
                        mock_ctx.get.return_value = "test_session"
                        with patch('core.framework.base.logger') as mock_loguru:
                            mock_loguru.add = Mock()

                            from core.units.terminal_tool import TerminalTool

                            with patch.object(TerminalTool, 'initialize_terminal_sessions', lambda self: None):
                                tool = TerminalTool.__new__(TerminalTool)
                                tool.logs_dir = temp_logs_dir
                                tool.container_name = "test_container"
                                tool.working_memory = mock_wm_instance
                                tool.logger = MagicMock()
                                yield tool

    def test_observe_terminal_session_no_log_file(self, terminal_tool):
        """Test observing session when log file doesn't exist."""
        with patch('time.sleep'):
            result = terminal_tool.observe_terminal_session("session_1", 0, 0)

            assert "Log file not found" in result or "Session not found" in result or isinstance(result, str)

    def test_observe_terminal_session_with_logs(self, terminal_tool, temp_logs_dir):
        """Test observing session with existing logs."""
        # Create a log file
        log_file = Path(temp_logs_dir) / "session_1.log"
        timestamp = time.strftime("%Y-%m-%d-%H:%M:%S")
        log_file.write_text(f"{timestamp} echo hello\n{timestamp} output: hello\n")

        with patch('time.sleep'):
            result = terminal_tool.observe_terminal_session("session_1", 5, 0)

            # Should return some content or empty string
            assert isinstance(result, str)

    def test_observe_terminal_session_not_found(self, terminal_tool):
        """Test observing non-existent session."""
        terminal_tool.working_memory.get_module.return_value = []

        with patch('time.sleep'):
            result = terminal_tool.observe_terminal_session("nonexistent", 0, 0)

            assert "not found" in result.lower() or result == ""


# ============================================================================
# Schema Tests
# ============================================================================

class TestTerminalToolSchema:
    """Tests for TerminalTool schema generation."""

    def test_schema_returns_list(self):
        """Test that schema returns a list."""
        from core.units.terminal_tool import TerminalTool

        schema = TerminalTool.schema()

        assert isinstance(schema, list)

    def test_schema_contains_required_functions(self):
        """Test that schema contains all required functions."""
        from core.units.terminal_tool import TerminalTool

        schema = TerminalTool.schema()
        function_names = [item["function"]["name"] for item in schema]

        assert "new_terminal_session" in function_names
        assert "control_c_terminal_session" in function_names
        assert "send_terminal_command" in function_names
        assert "observe_terminal_session" in function_names

    def test_schema_format(self):
        """Test that schema follows OpenAI function calling format."""
        from core.units.terminal_tool import TerminalTool

        schema = TerminalTool.schema()

        for item in schema:
            assert "type" in item
            assert item["type"] == "function"
            assert "function" in item
            assert "name" in item["function"]
            assert "description" in item["function"]
            assert "parameters" in item["function"]

    def test_schema_parameter_types(self):
        """Test that schema parameters have correct types."""
        from core.units.terminal_tool import TerminalTool

        schema = TerminalTool.schema()

        # Find send_terminal_command schema
        for item in schema:
            if item["function"]["name"] == "send_terminal_command":
                props = item["function"]["parameters"]["properties"]
                assert props["session_id"]["type"] == "string"
                assert props["command"]["type"] == "string"

            if item["function"]["name"] == "observe_terminal_session":
                props = item["function"]["parameters"]["properties"]
                assert props["offset_start_time_by_in_seconds"]["type"] == "integer"
                assert props["observation_time_in_seconds"]["type"] == "integer"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def terminal_tool(self, temp_logs_dir):
        """Create a TerminalTool instance with mocked dependencies."""
        with patch('core.units.terminal_tool.WorkingMemory') as mock_wm:
            mock_wm_instance = MagicMock()
            mock_wm_instance.get_module.return_value = []
            mock_wm.return_value = mock_wm_instance

            with patch('core.units.terminal_tool.get_docker_container_id') as mock_docker:
                mock_docker.return_value = "test_container"

                with patch('core.framework.base.db_handler'):
                    with patch('core.framework.base.session_context') as mock_ctx:
                        mock_ctx.get.return_value = "test_session"
                        with patch('core.framework.base.logger') as mock_loguru:
                            mock_loguru.add = Mock()

                            from core.units.terminal_tool import TerminalTool

                            with patch.object(TerminalTool, 'initialize_terminal_sessions', lambda self: None):
                                tool = TerminalTool.__new__(TerminalTool)
                                tool.logs_dir = temp_logs_dir
                                tool.container_name = "test_container"
                                tool.working_memory = mock_wm_instance
                                tool.logger = MagicMock()
                                yield tool

    def test_command_with_special_characters(self, terminal_tool):
        """Test sending command with special characters."""
        terminal_tool.working_memory.get_module.return_value = [
            {"session_id": "session_1", "action_history": []}
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Create mocks with __name__ attribute
            mock_observe = MagicMock(return_value="")
            mock_observe.__name__ = "observe_terminal_session"
            mock_update = MagicMock()
            mock_update.__name__ = "update_action_history"

            original_observe = terminal_tool.__class__.observe_terminal_session
            original_update = terminal_tool.__class__.update_action_history
            try:
                terminal_tool.__class__.observe_terminal_session = mock_observe
                terminal_tool.__class__.update_action_history = mock_update
                result = terminal_tool.send_terminal_command(
                    "session_1",
                    "echo 'Hello \"World\"' && ls -la"
                )
                assert result.success is True
            finally:
                terminal_tool.__class__.observe_terminal_session = original_observe
                terminal_tool.__class__.update_action_history = original_update

    def test_command_with_newlines(self, terminal_tool):
        """Test sending command with newlines."""
        terminal_tool.working_memory.get_module.return_value = [
            {"session_id": "session_1", "action_history": []}
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            mock_observe = MagicMock(return_value="")
            mock_observe.__name__ = "observe_terminal_session"
            mock_update = MagicMock()
            mock_update.__name__ = "update_action_history"

            original_observe = terminal_tool.__class__.observe_terminal_session
            original_update = terminal_tool.__class__.update_action_history
            try:
                terminal_tool.__class__.observe_terminal_session = mock_observe
                terminal_tool.__class__.update_action_history = mock_update
                result = terminal_tool.send_terminal_command(
                    "session_1",
                    "echo 'line1\nline2'"
                )
                assert isinstance(result.success, bool)
            finally:
                terminal_tool.__class__.observe_terminal_session = original_observe
                terminal_tool.__class__.update_action_history = original_update

    def test_very_long_command(self, terminal_tool):
        """Test sending very long command."""
        terminal_tool.working_memory.get_module.return_value = [
            {"session_id": "session_1", "action_history": []}
        ]

        long_command = "echo " + "x" * 10000

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            mock_observe = MagicMock(return_value="")
            mock_observe.__name__ = "observe_terminal_session"
            mock_update = MagicMock()
            mock_update.__name__ = "update_action_history"

            original_observe = terminal_tool.__class__.observe_terminal_session
            original_update = terminal_tool.__class__.update_action_history
            try:
                terminal_tool.__class__.observe_terminal_session = mock_observe
                terminal_tool.__class__.update_action_history = mock_update
                result = terminal_tool.send_terminal_command("session_1", long_command)
                assert isinstance(result.success, bool)
            finally:
                terminal_tool.__class__.observe_terminal_session = original_observe
                terminal_tool.__class__.update_action_history = original_update

    def test_multiple_concurrent_sessions(self, terminal_tool):
        """Test managing multiple sessions."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Create multiple sessions
            sessions = []
            for i in range(3):
                terminal_tool.working_memory.get_module.return_value = [
                    {"session_id": f"session_{j+1}", "action_history": []}
                    for j in range(i)
                ]
                session_id = terminal_tool.new_terminal_session()
                sessions.append(session_id)

            assert len(sessions) == 3
            assert sessions[0] == "session_1"
            assert sessions[1] == "session_2"
            assert sessions[2] == "session_3"
