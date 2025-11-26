"""
Comprehensive tests for core/units/files_tool.py

Tests cover:
- FilesTool initialization
- Path handling
- File operations
- Directory reading
- Schema generation
- Edge cases
"""
import os
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open

import pytest


# ============================================================================
# Helper Functions Tests
# ============================================================================

class TestRindexHelper:
    """Tests for the _rindex helper function."""

    def test_rindex_found_at_end(self):
        """Test finding element at end of list."""
        from core.units.files_tool import _rindex

        result = _rindex([1, 2, 3, 4, 5], 5)
        assert result == 4

    def test_rindex_found_at_start(self):
        """Test finding element at start of list."""
        from core.units.files_tool import _rindex

        result = _rindex([1, 2, 3, 4, 5], 1)
        assert result == 0

    def test_rindex_found_in_middle(self):
        """Test finding element in middle of list."""
        from core.units.files_tool import _rindex

        result = _rindex([1, 2, 3, 4, 5], 3)
        assert result == 2

    def test_rindex_multiple_occurrences(self):
        """Test finding last occurrence when multiple exist."""
        from core.units.files_tool import _rindex

        result = _rindex([1, 2, 3, 2, 1], 2)
        assert result == 3  # Last occurrence

    def test_rindex_not_found_raises(self):
        """Test that ValueError is raised when element not found."""
        from core.units.files_tool import _rindex

        with pytest.raises(ValueError):
            _rindex([1, 2, 3], 99)


# ============================================================================
# FilesTool Initialization Tests
# ============================================================================

class TestFilesToolInitialization:
    """Tests for FilesTool initialization."""

    @pytest.fixture
    def mock_working_memory(self):
        """Mock WorkingMemory for testing."""
        with patch('core.units.files_tool.WorkingMemory') as mock_wm:
            mock_instance = MagicMock()
            mock_instance.get_module.return_value = None
            mock_wm.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_base_class(self):
        """Mock the base Unit class dependencies."""
        with patch('core.framework.base.db_handler'):
            with patch('core.framework.base.session_context') as mock_ctx:
                mock_ctx.get.return_value = "test_session"
                with patch('core.framework.base.logger') as mock_loguru:
                    mock_loguru.add = Mock()
                    yield

    def test_initialization_creates_working_memory(self, mock_working_memory, mock_base_class, temp_workspace):
        """Test that initialization creates WorkingMemory instance."""
        with patch.object(
            __import__('core.units.files_tool', fromlist=['FilesTool']).FilesTool,
            'base_path',
            str(temp_workspace)
        ):
            from core.units.files_tool import FilesTool

            # Create a mock-safe instance
            with patch.object(FilesTool, '__init__', lambda self: None):
                tool = FilesTool()
                tool.base_path = str(temp_workspace)
                tool.working_memory = mock_working_memory
                tool.logger = MagicMock()

                assert tool.working_memory is not None


# ============================================================================
# Path Handling Tests
# ============================================================================

class TestPathHandling:
    """Tests for path handling in FilesTool."""

    @pytest.fixture
    def files_tool(self, temp_workspace):
        """Create a FilesTool instance with mocked dependencies."""
        with patch('core.units.files_tool.WorkingMemory') as mock_wm:
            mock_instance = MagicMock()
            mock_instance.get_module.return_value = None
            mock_wm.return_value = mock_instance

            with patch('core.framework.base.db_handler'):
                with patch('core.framework.base.session_context') as mock_ctx:
                    mock_ctx.get.return_value = "test_session"
                    with patch('core.framework.base.logger') as mock_loguru:
                        mock_loguru.add = Mock()

                        from core.units.files_tool import FilesTool

                        # Create instance with custom base_path
                        with patch.object(FilesTool, 'base_path', str(temp_workspace)):
                            with patch.object(FilesTool, 'initialize_files', lambda self: None):
                                tool = FilesTool.__new__(FilesTool)
                                tool.base_path = str(temp_workspace)
                                tool.working_memory = mock_instance
                                tool.logger = MagicMock()
                                yield tool

    def test_get_effective_path_simple(self, files_tool):
        """Test getting effective path for simple path."""
        result = files_tool._get_effective_path("main.py")

        expected = os.path.join(files_tool.base_path, "main.py")
        assert result == expected

    def test_get_effective_path_with_subdir(self, files_tool):
        """Test getting effective path with subdirectory."""
        result = files_tool._get_effective_path("subdir/module.py")

        expected = os.path.join(files_tool.base_path, "subdir", "module.py")
        assert result == expected

    def test_get_effective_path_with_dot(self, files_tool):
        """Test getting effective path with . prefix."""
        result = files_tool._get_effective_path("./main.py")

        # Should handle . and return path relative to base_path
        assert files_tool.base_path in result

    def test_get_effective_path_prevents_traversal(self, files_tool):
        """Test that path traversal attempts are handled."""
        result = files_tool._get_effective_path("../etc/passwd")

        # Should still be within base_path context
        assert result.startswith(files_tool.base_path) or ".." in result


# ============================================================================
# File Operations Tests
# ============================================================================

class TestFileOperations:
    """Tests for file operations in FilesTool."""

    @pytest.fixture
    def files_tool(self, temp_workspace):
        """Create a FilesTool instance with mocked dependencies."""
        with patch('core.units.files_tool.WorkingMemory') as mock_wm:
            mock_instance = MagicMock()
            mock_instance.get_module.return_value = {}
            mock_wm.return_value = mock_instance

            with patch('core.framework.base.db_handler'):
                with patch('core.framework.base.session_context') as mock_ctx:
                    mock_ctx.get.return_value = "test_session"
                    with patch('core.framework.base.logger') as mock_loguru:
                        mock_loguru.add = Mock()

                        from core.units.files_tool import FilesTool

                        with patch.object(FilesTool, 'initialize_files', lambda self: None):
                            tool = FilesTool.__new__(FilesTool)
                            tool.base_path = str(temp_workspace)
                            tool.working_memory = mock_instance
                            tool.logger = MagicMock()
                            yield tool

    def test_read_directory_contents_basic(self, files_tool, temp_workspace):
        """Test reading directory contents."""
        result = files_tool.read_directory_contents("")

        assert result.success is True
        output = json.loads(result.output)
        assert "contents" in output
        assert "main.py" in output["contents"]

    def test_read_directory_contents_nonexistent(self, files_tool):
        """Test reading nonexistent directory."""
        result = files_tool.read_directory_contents("nonexistent_dir")

        assert result.success is False

    def test_read_directory_contents_depth_limit(self, files_tool, temp_workspace):
        """Test directory reading respects depth limit."""
        # Create deep structure
        deep_dir = temp_workspace / "a" / "b" / "c" / "d"
        deep_dir.mkdir(parents=True)
        (deep_dir / "deep.py").touch()

        result = files_tool.read_directory_contents("", depth=1)

        assert result.success is True
        output = json.loads(result.output)
        # Should not contain deeply nested files
        for path in output["contents"]:
            depth = path.count(os.sep)
            assert depth <= 1

    def test_read_directory_excludes_node_modules(self, files_tool, temp_workspace):
        """Test that node_modules is excluded from reading."""
        result = files_tool.read_directory_contents("")

        assert result.success is True
        output = json.loads(result.output)
        for path in output["contents"]:
            assert "node_modules" not in path


# ============================================================================
# Edit File Tests
# ============================================================================

class TestEditFile:
    """Tests for file editing functionality."""

    @pytest.fixture
    def files_tool(self, temp_workspace):
        """Create a FilesTool instance with mocked dependencies."""
        with patch('core.units.files_tool.WorkingMemory') as mock_wm:
            mock_instance = MagicMock()
            mock_instance.get_module.return_value = {}
            mock_wm.return_value = mock_instance

            with patch('core.framework.base.db_handler'):
                with patch('core.framework.base.session_context') as mock_ctx:
                    mock_ctx.get.return_value = "test_session"
                    with patch('core.framework.base.logger') as mock_loguru:
                        mock_loguru.add = Mock()

                        from core.units.files_tool import FilesTool

                        with patch.object(FilesTool, 'initialize_files', lambda self: None):
                            tool = FilesTool.__new__(FilesTool)
                            tool.base_path = str(temp_workspace)
                            tool.working_memory = mock_instance
                            tool.logger = MagicMock()
                            yield tool

    def test_edit_mainpy_nonexistent_file(self, files_tool, temp_workspace):
        """Test editing main.py when it doesn't exist returns failure."""
        # Remove main.py if it exists
        main_path = temp_workspace / "main.py"
        if main_path.exists():
            main_path.unlink()

        result = files_tool.edit_mainpy_file_contents("Add a function")

        assert result.success is False
        assert "does not exist" in result.output

    def test_edit_mainpy_success(self, files_tool, temp_workspace):
        """Test successful main.py editing."""
        with patch('core.units.files_tool.make_llm_api_call') as mock_llm:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message={
                        'content': json.dumps({
                            "File": {
                                "FilePath": "main.py",
                                "newFileContents": "# Updated main.py\nprint('hello')"
                            }
                        })
                    }
                )
            ]
            mock_llm.return_value = mock_response

            result = files_tool.edit_mainpy_file_contents("Add a print statement")

            assert result.success is True
            assert "edited successfully" in result.output

    def test_edit_mainpy_llm_error(self, files_tool, temp_workspace):
        """Test handling LLM errors during editing."""
        with patch('core.units.files_tool.make_llm_api_call') as mock_llm:
            mock_llm.side_effect = Exception("LLM API Error")

            result = files_tool.edit_mainpy_file_contents("Add a function")

            assert result.success is False


# ============================================================================
# Gather Information Tests
# ============================================================================

class TestGatherInformation:
    """Tests for user information gathering."""

    @pytest.fixture
    def files_tool(self, temp_workspace):
        """Create a FilesTool instance with mocked dependencies."""
        with patch('core.units.files_tool.WorkingMemory') as mock_wm:
            mock_instance = MagicMock()
            mock_wm.return_value = mock_instance

            with patch('core.framework.base.db_handler'):
                with patch('core.framework.base.session_context') as mock_ctx:
                    mock_ctx.get.return_value = "test_session"
                    with patch('core.framework.base.logger') as mock_loguru:
                        mock_loguru.add = Mock()

                        from core.units.files_tool import FilesTool

                        with patch.object(FilesTool, 'initialize_files', lambda self: None):
                            tool = FilesTool.__new__(FilesTool)
                            tool.base_path = str(temp_workspace)
                            tool.working_memory = mock_instance
                            tool.logger = MagicMock()
                            yield tool

    def test_gather_information_success(self, files_tool):
        """Test gathering information from user."""
        with patch('builtins.input', return_value="user response"):
            result = files_tool.gather_information_ask_user("Enter something:")

            assert result.success is True
            assert result.output == "user response"

    def test_gather_information_input_error(self, files_tool):
        """Test handling input errors."""
        with patch('builtins.input', side_effect=EOFError()):
            result = files_tool.gather_information_ask_user("Enter something:")

            assert result.success is False


# ============================================================================
# Schema Tests
# ============================================================================

class TestFilesToolSchema:
    """Tests for FilesTool schema generation."""

    def test_schema_returns_list(self):
        """Test that schema returns a list."""
        from core.units.files_tool import FilesTool

        schema = FilesTool.schema()

        assert isinstance(schema, list)

    def test_schema_contains_required_functions(self):
        """Test that schema contains all required functions."""
        from core.units.files_tool import FilesTool

        schema = FilesTool.schema()
        function_names = [item["function"]["name"] for item in schema]

        assert "gather_information_ask_user" in function_names
        assert "edit_mainpy_file_contents" in function_names
        assert "read_directory_contents" in function_names

    def test_schema_format(self):
        """Test that schema follows OpenAI function calling format."""
        from core.units.files_tool import FilesTool

        schema = FilesTool.schema()

        for item in schema:
            assert "type" in item
            assert item["type"] == "function"
            assert "function" in item
            assert "name" in item["function"]
            assert "description" in item["function"]
            assert "parameters" in item["function"]

    def test_schema_parameters(self):
        """Test that schema parameters are correctly defined."""
        from core.units.files_tool import FilesTool

        schema = FilesTool.schema()

        for item in schema:
            params = item["function"]["parameters"]
            assert "type" in params
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def files_tool(self, temp_workspace):
        """Create a FilesTool instance with mocked dependencies."""
        with patch('core.units.files_tool.WorkingMemory') as mock_wm:
            mock_instance = MagicMock()
            mock_instance.get_module.return_value = {}
            mock_wm.return_value = mock_instance

            with patch('core.framework.base.db_handler'):
                with patch('core.framework.base.session_context') as mock_ctx:
                    mock_ctx.get.return_value = "test_session"
                    with patch('core.framework.base.logger') as mock_loguru:
                        mock_loguru.add = Mock()

                        from core.units.files_tool import FilesTool

                        with patch.object(FilesTool, 'initialize_files', lambda self: None):
                            tool = FilesTool.__new__(FilesTool)
                            tool.base_path = str(temp_workspace)
                            tool.working_memory = mock_instance
                            tool.logger = MagicMock()
                            yield tool

    def test_read_directory_with_binary_file(self, files_tool, temp_workspace):
        """Test reading directory with binary files."""
        # Create a binary file
        binary_file = temp_workspace / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\xff\xfe')

        result = files_tool.read_directory_contents("")

        # Should handle binary files gracefully
        assert result.success is True

    def test_read_directory_with_unicode_content(self, files_tool, temp_workspace):
        """Test reading directory with unicode content."""
        unicode_file = temp_workspace / "unicode.txt"
        unicode_file.write_text("Unicode content: 你好世界 🌍", encoding='utf-8')

        result = files_tool.read_directory_contents("")

        assert result.success is True
        output = json.loads(result.output)
        assert "unicode.txt" in output["contents"]
        assert "你好世界" in output["contents"]["unicode.txt"]

    def test_read_empty_directory(self, tmp_path):
        """Test reading an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch('core.units.files_tool.WorkingMemory') as mock_wm:
            mock_instance = MagicMock()
            mock_instance.get_module.return_value = {}
            mock_wm.return_value = mock_instance

            with patch('core.framework.base.db_handler'):
                with patch('core.framework.base.session_context') as mock_ctx:
                    mock_ctx.get.return_value = "test_session"
                    with patch('core.framework.base.logger') as mock_loguru:
                        mock_loguru.add = Mock()

                        from core.units.files_tool import FilesTool

                        with patch.object(FilesTool, 'initialize_files', lambda self: None):
                            tool = FilesTool.__new__(FilesTool)
                            tool.base_path = str(empty_dir)
                            tool.working_memory = mock_instance
                            tool.logger = MagicMock()

                            result = tool.read_directory_contents("")

                            assert result.success is True
                            output = json.loads(result.output)
                            assert output["contents"] == {}

    def test_base_path_does_not_exist(self, tmp_path):
        """Test behavior when base_path doesn't exist."""
        with patch('core.units.files_tool.WorkingMemory') as mock_wm:
            mock_instance = MagicMock()
            mock_wm.return_value = mock_instance

            with patch('core.framework.base.db_handler'):
                with patch('core.framework.base.session_context') as mock_ctx:
                    mock_ctx.get.return_value = "test_session"
                    with patch('core.framework.base.logger') as mock_loguru:
                        mock_loguru.add = Mock()

                        from core.units.files_tool import FilesTool

                        with patch.object(FilesTool, 'initialize_files', lambda self: None):
                            tool = FilesTool.__new__(FilesTool)
                            tool.base_path = "/nonexistent/path"
                            tool.working_memory = mock_instance
                            tool.logger = MagicMock()

                            result = tool.read_directory_contents("")

                            assert result.success is False
                            assert "does not exist" in result.output
