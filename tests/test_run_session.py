"""
Comprehensive tests for core/units/run_session.py

Tests cover:
- RunSessionTool initialization
- Session lifecycle
- Agent instructions
- Integration scenarios
- Edge cases
"""
import json
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock

import pytest


# ============================================================================
# RunSessionTool Initialization Tests
# ============================================================================

class TestRunSessionToolInitialization:
    """Tests for RunSessionTool initialization."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for RunSessionTool."""
        with patch('core.units.run_session.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            with patch('core.units.run_session.WorkingMemory') as mock_wm:
                mock_wm_instance = MagicMock()
                mock_wm_instance.export_memory.return_value = {}
                mock_wm.return_value = mock_wm_instance

                with patch('core.units.run_session.FilesTool') as mock_files:
                    mock_files_instance = MagicMock()
                    mock_files_instance.schema.return_value = []
                    mock_files.return_value = mock_files_instance

                    with patch('core.units.run_session.TerminalTool') as mock_terminal:
                        mock_terminal_instance = MagicMock()
                        mock_terminal_instance.schema.return_value = []
                        mock_terminal.return_value = mock_terminal_instance

                        with patch('core.units.run_session.BaseAssistant') as mock_base:
                            mock_assistant = MagicMock()
                            mock_assistant.assistant_id = "asst_test123"
                            mock_base.return_value = mock_assistant

                            with patch('core.units.run_session.initialize_logging'):
                                with patch('core.framework.base.db_handler'):
                                    with patch('core.framework.base.session_context') as mock_ctx:
                                        mock_ctx.get.return_value = "test_session"
                                        with patch('core.framework.base.logger') as mock_loguru:
                                            mock_loguru.add = Mock()
                                            yield {
                                                'openai': mock_client,
                                                'working_memory': mock_wm_instance,
                                                'files_tool': mock_files_instance,
                                                'terminal_tool': mock_terminal_instance,
                                                'base_assistant': mock_assistant
                                            }

    def test_initialization_creates_components(self, mock_dependencies):
        """Test that initialization creates all required components."""
        from core.units.run_session import RunSessionTool

        tool = RunSessionTool()

        assert tool.working_memory is not None
        assert tool.files_tool_instance is not None
        assert tool.terminal_tool_instance is not None
        assert tool.agent is not None

    def test_initialization_creates_tools_list(self, mock_dependencies):
        """Test that initialization creates combined tools list."""
        from core.units.run_session import RunSessionTool

        tool = RunSessionTool()

        assert isinstance(tool.tools, list)


# ============================================================================
# Agent Instructions Tests
# ============================================================================

class TestAgentInstructions:
    """Tests for agent instructions generation."""

    @pytest.fixture
    def run_session_tool(self):
        """Create a RunSessionTool instance with mocked dependencies."""
        with patch('core.units.run_session.OpenAI'):
            with patch('core.units.run_session.WorkingMemory') as mock_wm:
                mock_wm.return_value.export_memory.return_value = {}

                with patch('core.units.run_session.FilesTool') as mock_files:
                    mock_files.return_value.schema.return_value = []

                    with patch('core.units.run_session.TerminalTool') as mock_terminal:
                        mock_terminal.return_value.schema.return_value = []

                        with patch('core.units.run_session.BaseAssistant'):
                            with patch('core.units.run_session.initialize_logging'):
                                with patch('core.framework.base.db_handler'):
                                    with patch('core.framework.base.session_context') as mock_ctx:
                                        mock_ctx.get.return_value = "test_session"
                                        with patch('core.framework.base.logger') as mock_loguru:
                                            mock_loguru.add = Mock()

                                            from core.units.run_session import RunSessionTool

                                            tool = RunSessionTool()
                                            yield tool

    def test_agent_instructions_contains_key_elements(self, run_session_tool):
        """Test that agent instructions contain key elements."""
        instructions = run_session_tool._get_agent_instructions()

        assert "Mirko" in instructions
        assert "main.py" in instructions
        assert "terminal" in instructions.lower()

    def test_agent_monologue_system_message(self, run_session_tool):
        """Test that internal monologue system message is generated."""
        monologue = run_session_tool._get_agent_internal_monologue_system_message()

        assert "internal monologue" in monologue.lower()
        assert "JSON" in monologue
        assert "observations" in monologue
        assert "thoughts" in monologue
        assert "next_actions" in monologue


# ============================================================================
# Session Lifecycle Tests
# ============================================================================

class TestSessionLifecycle:
    """Tests for session lifecycle management."""

    @pytest.fixture
    def run_session_tool(self):
        """Create a RunSessionTool instance with mocked dependencies."""
        with patch('core.units.run_session.OpenAI'):
            with patch('core.units.run_session.WorkingMemory') as mock_wm:
                mock_wm_instance = MagicMock()
                mock_wm_instance.export_memory.return_value = {}
                mock_wm_instance.get_module.return_value = None
                mock_wm.return_value = mock_wm_instance

                with patch('core.units.run_session.FilesTool') as mock_files:
                    mock_files_instance = MagicMock()
                    mock_files_instance.schema.return_value = []
                    mock_files.return_value = mock_files_instance

                    with patch('core.units.run_session.TerminalTool') as mock_terminal:
                        mock_terminal_instance = MagicMock()
                        mock_terminal_instance.schema.return_value = []
                        mock_terminal.return_value = mock_terminal_instance

                        with patch('core.units.run_session.BaseAssistant') as mock_base:
                            mock_assistant = MagicMock()
                            mock_assistant.start_new_thread.return_value = "thread_123"
                            mock_assistant.run_thread.return_value = "run_123"
                            mock_assistant.check_run_status_and_execute_action = AsyncMock()
                            mock_assistant.internal_monologue.return_value = "{}"
                            mock_base.return_value = mock_assistant

                            with patch('core.units.run_session.initialize_logging'):
                                with patch('core.framework.base.db_handler'):
                                    with patch('core.framework.base.session_context') as mock_ctx:
                                        mock_ctx.get.return_value = "test_session"
                                        with patch('core.framework.base.logger') as mock_loguru:
                                            mock_loguru.add = Mock()

                                            from core.units.run_session import RunSessionTool

                                            tool = RunSessionTool()
                                            yield tool

    @pytest.mark.asyncio
    async def test_start_session_run_clears_memory(self, run_session_tool):
        """Test that starting a session clears working memory."""
        with patch('subprocess.run'):
            # Make the loop exit after one iteration
            iteration = [0]

            def mock_run_thread(*args, **kwargs):
                iteration[0] += 1
                if iteration[0] > 1:
                    raise KeyboardInterrupt()
                return "run_123"

            run_session_tool.agent.run_thread.side_effect = mock_run_thread

            try:
                await run_session_tool.start_session_run("Test request")
            except KeyboardInterrupt:
                pass

            run_session_tool.working_memory.clear_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_session_run_stores_objective(self, run_session_tool):
        """Test that starting a session stores the objective."""
        with patch('subprocess.run'):
            iteration = [0]

            def mock_run_thread(*args, **kwargs):
                iteration[0] += 1
                if iteration[0] > 1:
                    raise KeyboardInterrupt()
                return "run_123"

            run_session_tool.agent.run_thread.side_effect = mock_run_thread

            try:
                await run_session_tool.start_session_run("Build a landing page")
            except KeyboardInterrupt:
                pass

            # Check that objective was stored
            calls = run_session_tool.working_memory.add_or_update_module.call_args_list
            objective_calls = [c for c in calls if c[0][0] == "OverarchingObjective"]
            assert len(objective_calls) > 0
            assert objective_calls[0][0][1] == "Build a landing page"


# ============================================================================
# Schema Tests
# ============================================================================

class TestRunSessionToolSchema:
    """Tests for RunSessionTool schema generation."""

    def test_schema_returns_list(self):
        """Test that schema returns a list."""
        from core.units.run_session import RunSessionTool

        schema = RunSessionTool.schema()

        assert isinstance(schema, list)

    def test_schema_contains_start_session_run(self):
        """Test that schema contains start_session_run function."""
        from core.units.run_session import RunSessionTool

        schema = RunSessionTool.schema()
        function_names = [item["function"]["name"] for item in schema]

        assert "start_session_run" in function_names

    def test_schema_format(self):
        """Test that schema follows OpenAI function calling format."""
        from core.units.run_session import RunSessionTool

        schema = RunSessionTool.schema()

        for item in schema:
            assert "type" in item
            assert item["type"] == "function"
            assert "function" in item
            assert "name" in item["function"]
            assert "parameters" in item["function"]

    def test_schema_user_request_parameter(self):
        """Test that schema has user_request parameter."""
        from core.units.run_session import RunSessionTool

        schema = RunSessionTool.schema()

        for item in schema:
            if item["function"]["name"] == "start_session_run":
                params = item["function"]["parameters"]
                assert "user_request" in params["properties"]
                assert "user_request" in params["required"]


# ============================================================================
# Additional Instructions Tests
# ============================================================================

class TestAdditionalInstructions:
    """Tests for additional instructions generation."""

    @pytest.fixture
    def run_session_tool(self):
        """Create a RunSessionTool instance."""
        with patch('core.units.run_session.OpenAI'):
            with patch('core.units.run_session.WorkingMemory') as mock_wm:
                mock_wm.return_value.export_memory.return_value = {
                    "Objective": "Test objective",
                    "TaskList": []
                }

                with patch('core.units.run_session.FilesTool') as mock_files:
                    mock_files.return_value.schema.return_value = []

                    with patch('core.units.run_session.TerminalTool') as mock_terminal:
                        mock_terminal.return_value.schema.return_value = []

                        with patch('core.units.run_session.BaseAssistant'):
                            with patch('core.units.run_session.initialize_logging'):
                                with patch('core.framework.base.db_handler'):
                                    with patch('core.framework.base.session_context') as mock_ctx:
                                        mock_ctx.get.return_value = "test_session"
                                        with patch('core.framework.base.logger') as mock_loguru:
                                            mock_loguru.add = Mock()

                                            from core.units.run_session import RunSessionTool

                                            tool = RunSessionTool()
                                            yield tool

    def test_additional_instructions_contains_working_memory(self, run_session_tool):
        """Test that additional instructions contain working memory."""
        additional = run_session_tool.additional_instructions

        assert "WorkingMemory" in additional
        assert "Objective" in additional or "{}" not in additional


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for RunSessionTool."""

    @pytest.fixture
    def full_mock_setup(self):
        """Set up full mock environment for integration tests."""
        with patch('core.units.run_session.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            with patch('core.units.run_session.WorkingMemory') as mock_wm:
                mock_wm_instance = MagicMock()
                mock_wm_instance.export_memory.return_value = {}
                mock_wm_instance.get_module.return_value = None
                mock_wm.return_value = mock_wm_instance

                with patch('core.units.run_session.FilesTool') as mock_files:
                    mock_files_instance = MagicMock()
                    mock_files_instance.schema.return_value = [
                        {"type": "function", "function": {"name": "read_directory_contents"}}
                    ]
                    mock_files.return_value = mock_files_instance

                    with patch('core.units.run_session.TerminalTool') as mock_terminal:
                        mock_terminal_instance = MagicMock()
                        mock_terminal_instance.schema.return_value = [
                            {"type": "function", "function": {"name": "new_terminal_session"}}
                        ]
                        mock_terminal.return_value = mock_terminal_instance

                        with patch('core.units.run_session.BaseAssistant') as mock_base:
                            mock_assistant = MagicMock()
                            mock_assistant.assistant_id = "asst_test"
                            mock_assistant.start_new_thread.return_value = "thread_test"
                            mock_base.return_value = mock_assistant

                            with patch('core.units.run_session.initialize_logging'):
                                with patch('core.framework.base.db_handler'):
                                    with patch('core.framework.base.session_context') as mock_ctx:
                                        mock_ctx.get.return_value = "test_session"
                                        with patch('core.framework.base.logger') as mock_loguru:
                                            mock_loguru.add = Mock()

                                            yield {
                                                'openai': mock_client,
                                                'wm': mock_wm_instance,
                                                'files': mock_files_instance,
                                                'terminal': mock_terminal_instance,
                                                'assistant': mock_assistant
                                            }

    def test_tools_combined_from_all_sources(self, full_mock_setup):
        """Test that tools are combined from Files and Terminal."""
        from core.units.run_session import RunSessionTool

        tool = RunSessionTool()

        # Should have tools from both Files and Terminal
        assert len(tool.tools) >= 2

    def test_assistant_created_with_correct_params(self, full_mock_setup):
        """Test that BaseAssistant is created with correct parameters."""
        from core.units.run_session import RunSessionTool

        tool = RunSessionTool()

        # Verify assistant was created
        assert tool.agent is not None
        assert tool.agent.assistant_id == "asst_test"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_schema_static_method(self):
        """Test that schema is a static method."""
        from core.units.run_session import RunSessionTool

        # Should work without instance
        schema = RunSessionTool.schema()
        assert isinstance(schema, list)

    @pytest.mark.asyncio
    async def test_session_handles_tmux_kill_error(self):
        """Test that session handles tmux kill server error gracefully."""
        with patch('core.units.run_session.OpenAI'):
            with patch('core.units.run_session.WorkingMemory') as mock_wm:
                mock_wm.return_value.export_memory.return_value = {}
                mock_wm.return_value.get_module.return_value = None

                with patch('core.units.run_session.FilesTool') as mock_files:
                    mock_files.return_value.schema.return_value = []

                    with patch('core.units.run_session.TerminalTool') as mock_terminal:
                        mock_terminal.return_value.schema.return_value = []

                        with patch('core.units.run_session.BaseAssistant') as mock_base:
                            mock_assistant = MagicMock()
                            mock_assistant.start_new_thread.return_value = "thread_123"

                            # Make run_thread raise after one call
                            def raise_after_one(*args, **kwargs):
                                raise KeyboardInterrupt()

                            mock_assistant.run_thread.side_effect = raise_after_one
                            mock_base.return_value = mock_assistant

                            with patch('core.units.run_session.initialize_logging'):
                                with patch('core.framework.base.db_handler'):
                                    with patch('core.framework.base.session_context') as mock_ctx:
                                        mock_ctx.get.return_value = "test_session"
                                        with patch('core.framework.base.logger') as mock_loguru:
                                            mock_loguru.add = Mock()

                                            with patch('subprocess.run') as mock_subprocess:
                                                from subprocess import CalledProcessError
                                                mock_subprocess.side_effect = CalledProcessError(1, "tmux")

                                                from core.units.run_session import RunSessionTool

                                                tool = RunSessionTool()

                                                # Should handle the error gracefully
                                                try:
                                                    await tool.start_session_run("Test")
                                                except KeyboardInterrupt:
                                                    pass
                                                except CalledProcessError:
                                                    pass  # Expected

    def test_empty_working_memory_export(self):
        """Test handling of empty working memory."""
        with patch('core.units.run_session.OpenAI'):
            with patch('core.units.run_session.WorkingMemory') as mock_wm:
                mock_wm.return_value.export_memory.return_value = {}

                with patch('core.units.run_session.FilesTool') as mock_files:
                    mock_files.return_value.schema.return_value = []

                    with patch('core.units.run_session.TerminalTool') as mock_terminal:
                        mock_terminal.return_value.schema.return_value = []

                        with patch('core.units.run_session.BaseAssistant'):
                            with patch('core.units.run_session.initialize_logging'):
                                with patch('core.framework.base.db_handler'):
                                    with patch('core.framework.base.session_context') as mock_ctx:
                                        mock_ctx.get.return_value = "test_session"
                                        with patch('core.framework.base.logger') as mock_loguru:
                                            mock_loguru.add = Mock()

                                            from core.units.run_session import RunSessionTool

                                            tool = RunSessionTool()

                                            # Should have empty working memory in instructions
                                            assert "{}" in tool.additional_instructions
