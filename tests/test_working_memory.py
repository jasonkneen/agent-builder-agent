"""
Comprehensive tests for core/units/working_memory.py

Tests cover:
- WorkingMemory initialization
- Module CRUD operations
- Memory export
- Memory clearing
- Edge cases and error handling
"""
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ============================================================================
# WorkingMemory Initialization Tests
# ============================================================================

class TestWorkingMemoryInitialization:
    """Tests for WorkingMemory class initialization."""

    def test_initialization_creates_database(self, tmp_path):
        """Test that initialization creates the database file."""
        db_path = str(tmp_path / "test_wm.db")

        from core.units.working_memory import WorkingMemory

        wm = WorkingMemory(db_path)

        assert Path(db_path).exists()
        wm.conn.close()

    def test_initialization_creates_tables(self, tmp_path):
        """Test that initialization creates required tables."""
        db_path = str(tmp_path / "test_wm.db")

        from core.units.working_memory import WorkingMemory

        wm = WorkingMemory(db_path)

        # Check table exists
        cursor = wm.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='MemoryModules'"
        )
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "MemoryModules"
        wm.conn.close()

    def test_initialization_with_existing_database(self, tmp_path):
        """Test initialization with existing database."""
        db_path = str(tmp_path / "test_wm.db")

        from core.units.working_memory import WorkingMemory

        # Create first instance
        wm1 = WorkingMemory(db_path)
        wm1.add_or_update_module("TestModule", {"data": "value"})
        wm1.conn.close()

        # Create second instance with same db
        wm2 = WorkingMemory(db_path)
        result = wm2.get_module("TestModule")

        assert result == {"data": "value"}
        wm2.conn.close()


# ============================================================================
# Module CRUD Tests
# ============================================================================

class TestModuleCRUD:
    """Tests for module Create, Read, Update, Delete operations."""

    @pytest.fixture
    def working_memory(self, tmp_path):
        """Create a WorkingMemory instance with temp database."""
        db_path = str(tmp_path / "test_wm.db")

        from core.units.working_memory import WorkingMemory

        wm = WorkingMemory(db_path)
        yield wm
        wm.conn.close()

    def test_add_module_string_data(self, working_memory):
        """Test adding a module with string data."""
        working_memory.add_or_update_module("StringModule", "test string")

        result = working_memory.get_module("StringModule")
        assert result == "test string"

    def test_add_module_dict_data(self, working_memory):
        """Test adding a module with dictionary data."""
        data = {"key": "value", "number": 42}
        working_memory.add_or_update_module("DictModule", data)

        result = working_memory.get_module("DictModule")
        assert result == data

    def test_add_module_list_data(self, working_memory):
        """Test adding a module with list data."""
        data = [1, 2, 3, "four", {"five": 5}]
        working_memory.add_or_update_module("ListModule", data)

        result = working_memory.get_module("ListModule")
        assert result == data

    def test_add_module_nested_data(self, working_memory):
        """Test adding a module with deeply nested data."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": ["deep", "data"]
                    }
                }
            }
        }
        working_memory.add_or_update_module("NestedModule", data)

        result = working_memory.get_module("NestedModule")
        assert result == data
        assert result["level1"]["level2"]["level3"]["level4"] == ["deep", "data"]

    def test_update_existing_module(self, working_memory):
        """Test updating an existing module."""
        working_memory.add_or_update_module("UpdateModule", "initial value")
        working_memory.add_or_update_module("UpdateModule", "updated value")

        result = working_memory.get_module("UpdateModule")
        assert result == "updated value"

    def test_get_nonexistent_module(self, working_memory):
        """Test getting a module that doesn't exist."""
        result = working_memory.get_module("NonExistentModule")
        assert result is None

    def test_delete_module(self, working_memory):
        """Test deleting a module."""
        working_memory.add_or_update_module("DeleteModule", "to be deleted")
        working_memory.delete_module("DeleteModule")

        result = working_memory.get_module("DeleteModule")
        assert result is None

    def test_delete_nonexistent_module(self, working_memory):
        """Test deleting a module that doesn't exist (should not raise)."""
        # Should not raise an exception
        working_memory.delete_module("NonExistent")

    def test_multiple_modules(self, working_memory):
        """Test managing multiple modules."""
        modules = {
            "Module1": "data1",
            "Module2": {"key": "value"},
            "Module3": [1, 2, 3],
            "Module4": True,
            "Module5": 42
        }

        for name, data in modules.items():
            working_memory.add_or_update_module(name, data)

        for name, expected_data in modules.items():
            result = working_memory.get_module(name)
            assert result == expected_data


# ============================================================================
# Export Memory Tests
# ============================================================================

class TestExportMemory:
    """Tests for memory export functionality."""

    @pytest.fixture
    def working_memory(self, tmp_path):
        """Create a WorkingMemory instance with temp database."""
        db_path = str(tmp_path / "test_wm.db")

        from core.units.working_memory import WorkingMemory

        wm = WorkingMemory(db_path)
        yield wm
        wm.conn.close()

    def test_export_empty_memory(self, working_memory):
        """Test exporting empty memory."""
        result = working_memory.export_memory()
        assert result == {}

    def test_export_single_module(self, working_memory):
        """Test exporting memory with single module."""
        working_memory.add_or_update_module("OnlyModule", {"data": "value"})

        result = working_memory.export_memory()

        assert "OnlyModule" in result
        assert result["OnlyModule"] == {"data": "value"}

    def test_export_multiple_modules(self, working_memory):
        """Test exporting memory with multiple modules."""
        working_memory.add_or_update_module("Module1", "data1")
        working_memory.add_or_update_module("Module2", "data2")
        working_memory.add_or_update_module("Module3", "data3")

        result = working_memory.export_memory()

        assert len(result) == 3
        assert result["Module1"] == "data1"
        assert result["Module2"] == "data2"
        assert result["Module3"] == "data3"

    def test_export_preserves_data_types(self, working_memory):
        """Test that export preserves data types."""
        working_memory.add_or_update_module("String", "string")
        working_memory.add_or_update_module("Number", 42)
        working_memory.add_or_update_module("Float", 3.14)
        working_memory.add_or_update_module("Bool", True)
        working_memory.add_or_update_module("List", [1, 2, 3])
        working_memory.add_or_update_module("Dict", {"key": "value"})

        result = working_memory.export_memory()

        assert isinstance(result["String"], str)
        assert isinstance(result["Number"], int)
        assert isinstance(result["Float"], float)
        assert isinstance(result["Bool"], bool)
        assert isinstance(result["List"], list)
        assert isinstance(result["Dict"], dict)


# ============================================================================
# Clear Memory Tests
# ============================================================================

class TestClearMemory:
    """Tests for memory clearing functionality."""

    @pytest.fixture
    def working_memory(self, tmp_path):
        """Create a WorkingMemory instance with temp database."""
        db_path = str(tmp_path / "test_wm.db")

        from core.units.working_memory import WorkingMemory

        wm = WorkingMemory(db_path)
        yield wm
        wm.conn.close()

    def test_clear_empty_memory(self, working_memory):
        """Test clearing already empty memory."""
        working_memory.clear_memory()

        result = working_memory.export_memory()
        assert result == {}

    def test_clear_populated_memory(self, working_memory):
        """Test clearing populated memory."""
        working_memory.add_or_update_module("Module1", "data1")
        working_memory.add_or_update_module("Module2", "data2")
        working_memory.add_or_update_module("Module3", "data3")

        working_memory.clear_memory()

        result = working_memory.export_memory()
        assert result == {}

    def test_clear_then_add(self, working_memory):
        """Test adding modules after clearing."""
        working_memory.add_or_update_module("OldModule", "old data")
        working_memory.clear_memory()
        working_memory.add_or_update_module("NewModule", "new data")

        result = working_memory.export_memory()

        assert "OldModule" not in result
        assert "NewModule" in result
        assert result["NewModule"] == "new data"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def working_memory(self, tmp_path):
        """Create a WorkingMemory instance with temp database."""
        db_path = str(tmp_path / "test_wm.db")

        from core.units.working_memory import WorkingMemory

        wm = WorkingMemory(db_path)
        yield wm
        wm.conn.close()

    def test_module_name_with_spaces(self, working_memory):
        """Test module name with spaces."""
        working_memory.add_or_update_module("Module With Spaces", "data")

        result = working_memory.get_module("Module With Spaces")
        assert result == "data"

    def test_module_name_with_special_chars(self, working_memory):
        """Test module name with special characters."""
        working_memory.add_or_update_module("Module-With_Special.Chars!", "data")

        result = working_memory.get_module("Module-With_Special.Chars!")
        assert result == "data"

    def test_module_name_unicode(self, working_memory):
        """Test module name with unicode characters."""
        working_memory.add_or_update_module("模块名称", "数据")

        result = working_memory.get_module("模块名称")
        assert result == "数据"

    def test_very_large_data(self, working_memory):
        """Test storing very large data."""
        large_data = {"items": ["item_" + str(i) for i in range(10000)]}
        working_memory.add_or_update_module("LargeModule", large_data)

        result = working_memory.get_module("LargeModule")
        assert len(result["items"]) == 10000

    def test_null_data(self, working_memory):
        """Test storing null/None data."""
        working_memory.add_or_update_module("NullModule", None)

        result = working_memory.get_module("NullModule")
        assert result is None

    def test_empty_string_module_name(self, working_memory):
        """Test with empty string module name."""
        working_memory.add_or_update_module("", "data")

        result = working_memory.get_module("")
        assert result == "data"

    def test_data_with_quotes(self, working_memory):
        """Test data containing quotes."""
        data = {"message": 'He said "Hello"', "response": "She said 'Hi'"}
        working_memory.add_or_update_module("QuotesModule", data)

        result = working_memory.get_module("QuotesModule")
        assert result == data

    def test_data_with_newlines(self, working_memory):
        """Test data containing newlines."""
        data = "Line 1\nLine 2\nLine 3"
        working_memory.add_or_update_module("NewlinesModule", data)

        result = working_memory.get_module("NewlinesModule")
        assert result == data
        assert "\n" in result

    def test_concurrent_access_simulation(self, tmp_path):
        """Test simulating concurrent access."""
        db_path = str(tmp_path / "test_wm.db")

        from core.units.working_memory import WorkingMemory

        wm1 = WorkingMemory(db_path)
        wm2 = WorkingMemory(db_path)

        wm1.add_or_update_module("Module1", "from wm1")
        wm2.add_or_update_module("Module2", "from wm2")

        # Both should see both modules (after connection refresh)
        assert wm1.get_module("Module1") == "from wm1"
        assert wm2.get_module("Module2") == "from wm2"

        wm1.conn.close()
        wm2.conn.close()


# ============================================================================
# Integration with Sample Data
# ============================================================================

class TestIntegrationWithSampleData:
    """Tests using sample data patterns from the codebase."""

    @pytest.fixture
    def working_memory(self, tmp_path):
        """Create a WorkingMemory instance with temp database."""
        db_path = str(tmp_path / "test_wm.db")

        from core.units.working_memory import WorkingMemory

        wm = WorkingMemory(db_path)
        yield wm
        wm.conn.close()

    def test_objective_module(self, working_memory):
        """Test storing an objective like in the example."""
        objective = "Build a simple Landing Page for my construction company."
        working_memory.add_or_update_module("Objective", objective)

        result = working_memory.get_module("Objective")
        assert result == objective

    def test_terminal_sessions_module(self, working_memory):
        """Test storing terminal sessions like in the example."""
        terminal_sessions = [
            {
                "session_id": "1",
                "active": True,
                "terminal_log": [
                    "2023-03-15T09:00:00 New_terminal_session: Session 1 started",
                    "2023-03-15T09:01:00 Command: echo 'Hello World!', Output: Hello World!",
                ]
            },
            {
                "session_id": "2",
                "active": False,
                "terminal_log": []
            }
        ]
        working_memory.add_or_update_module("TerminalSessions", terminal_sessions)

        result = working_memory.get_module("TerminalSessions")
        assert len(result) == 2
        assert result[0]["session_id"] == "1"
        assert result[0]["active"] is True
        assert result[1]["active"] is False

    def test_task_list_module(self, working_memory):
        """Test storing a task list like in the example."""
        task_list = [
            {"task_id": "1", "instruction": "First Instruction"},
            {"task_id": "2", "instruction": "Second Instruction"},
            {"task_id": "3", "instruction": "Third Instruction"}
        ]
        working_memory.add_or_update_module("TaskList", task_list)

        result = working_memory.get_module("TaskList")
        assert len(result) == 3
        assert result[0]["task_id"] == "1"

    def test_update_task_list(self, working_memory):
        """Test updating a task list like in the example."""
        # Initial task list
        initial_tasks = [
            {"task_id": "1", "instruction": "First Instruction"},
            {"task_id": "2", "instruction": "Second Instruction"},
        ]
        working_memory.add_or_update_module("TaskList", initial_tasks)

        # Updated task list
        updated_tasks = [
            {"task_id": "1", "instruction": "First Instruction - Updated"},
            {"task_id": "2", "instruction": "Second Instruction - Updated"},
            {"task_id": "4", "instruction": "Fourth Instruction"}
        ]
        working_memory.add_or_update_module("TaskList", updated_tasks)

        result = working_memory.get_module("TaskList")
        assert len(result) == 3
        assert "Updated" in result[0]["instruction"]
        assert result[2]["task_id"] == "4"

    def test_full_workflow_simulation(self, working_memory):
        """Test simulating a full workflow with working memory."""
        # 1. Set objective
        working_memory.add_or_update_module(
            "OverarchingObjective",
            "Create an email automation tool"
        )

        # 2. Initialize terminal sessions
        working_memory.add_or_update_module("TerminalSessions", [])

        # 3. Add a terminal session
        sessions = working_memory.get_module("TerminalSessions")
        sessions.append({
            "session_id": "session_1",
            "action_history": []
        })
        working_memory.add_or_update_module("TerminalSessions", sessions)

        # 4. Store workspace contents
        working_memory.add_or_update_module(
            "WorkspaceDirectoryContents",
            {"main.py": "# Main file content"}
        )

        # 5. Export all
        memory_export = working_memory.export_memory()

        assert "OverarchingObjective" in memory_export
        assert "TerminalSessions" in memory_export
        assert "WorkspaceDirectoryContents" in memory_export
        assert len(memory_export["TerminalSessions"]) == 1
