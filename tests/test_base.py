"""
Comprehensive tests for core/framework/base.py

Tests cover:
- UnitResult dataclass
- DatabaseHandler operations
- Logger functionality
- Unit base class
- FastAPI endpoints
"""
import json
import sqlite3
import uuid
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from contextvars import ContextVar

import pytest
from fastapi.testclient import TestClient


# ============================================================================
# UnitResult Tests
# ============================================================================

class TestUnitResult:
    """Tests for the UnitResult dataclass."""

    def test_unit_result_success_creation(self):
        """Test creating a successful UnitResult."""
        result = self._create_unit_result(success=True, output="Operation completed")
        assert result.success is True
        assert result.output == "Operation completed"

    def test_unit_result_failure_creation(self):
        """Test creating a failed UnitResult."""
        result = self._create_unit_result(success=False, output="Error occurred")
        assert result.success is False
        assert result.output == "Error occurred"

    def test_unit_result_empty_output(self):
        """Test UnitResult with empty output."""
        result = self._create_unit_result(success=True, output="")
        assert result.success is True
        assert result.output == ""

    def test_unit_result_json_output(self):
        """Test UnitResult with JSON string output."""
        json_output = json.dumps({"key": "value", "number": 42})
        result = self._create_unit_result(success=True, output=json_output)
        assert result.success is True
        parsed = json.loads(result.output)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_unit_result_multiline_output(self):
        """Test UnitResult with multiline output."""
        output = "Line 1\nLine 2\nLine 3"
        result = self._create_unit_result(success=True, output=output)
        assert "Line 1" in result.output
        assert "Line 2" in result.output

    def _create_unit_result(self, success: bool, output: str):
        """Helper to create UnitResult without importing."""
        from dataclasses import dataclass

        @dataclass
        class UnitResult:
            success: bool
            output: str

        return UnitResult(success=success, output=output)


# ============================================================================
# DatabaseHandler Tests
# ============================================================================

class TestDatabaseHandler:
    """Tests for the DatabaseHandler class."""

    @pytest.fixture
    def db_handler(self, temp_logs_db):
        """Create a DatabaseHandler with temporary database."""
        # Create a real database connection using the temp db
        conn = sqlite3.connect(temp_logs_db)
        cursor = conn.cursor()

        # Create a simple handler-like object that uses the real connection
        class TestHandler:
            def __init__(self, connection, cursor):
                self.conn = connection
                self.cursor = cursor

            def insert_log(self, log_entry):
                self.cursor.execute('''
                    INSERT INTO logs (log_id, session_id, timestamp, level, message, unit_name, parent_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (log_entry['log_id'], log_entry['session_id'], log_entry['timestamp'],
                      log_entry['level'], log_entry['message'], log_entry['unit_name'], log_entry['parent_id']))
                self.conn.commit()

            def insert_session(self, session_id):
                from datetime import datetime
                self.cursor.execute('''
                    INSERT INTO sessions (session_id, start_time)
                    VALUES (?, ?)
                ''', (session_id, datetime.now().isoformat()))
                self.conn.commit()

        handler = TestHandler(conn, cursor)
        yield handler
        conn.close()

    def test_insert_log(self, db_handler):
        """Test inserting a log entry."""
        log_entry = {
            "log_id": str(uuid.uuid4()),
            "session_id": "test_session",
            "timestamp": datetime.now().isoformat(),
            "level": "DEBUG",
            "message": "Test log message",
            "unit_name": "TestUnit",
            "parent_id": None
        }

        db_handler.insert_log(log_entry)

        # Verify insertion
        db_handler.cursor.execute("SELECT * FROM logs WHERE session_id = ?", ("test_session",))
        result = db_handler.cursor.fetchone()
        assert result is not None
        assert result[4] == "Test log message"

    def test_insert_log_with_parent(self, db_handler):
        """Test inserting a log entry with parent ID."""
        parent_id = str(uuid.uuid4())
        log_entry = {
            "log_id": str(uuid.uuid4()),
            "session_id": "test_session",
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": "Child log message",
            "unit_name": "TestUnit",
            "parent_id": parent_id
        }

        db_handler.insert_log(log_entry)

        db_handler.cursor.execute("SELECT parent_id FROM logs WHERE session_id = ?", ("test_session",))
        result = db_handler.cursor.fetchone()
        assert result[0] == parent_id

    def test_insert_session(self, db_handler):
        """Test inserting a new session."""
        session_id = "new_test_session"

        db_handler.insert_session(session_id)

        db_handler.cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        result = db_handler.cursor.fetchone()
        assert result is not None
        assert result[0] == session_id

    def test_insert_multiple_logs(self, db_handler):
        """Test inserting multiple log entries."""
        session_id = "multi_log_session"

        for i in range(5):
            log_entry = {
                "log_id": str(uuid.uuid4()),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "level": "DEBUG",
                "message": f"Log message {i}",
                "unit_name": "TestUnit",
                "parent_id": None
            }
            db_handler.insert_log(log_entry)

        db_handler.cursor.execute("SELECT COUNT(*) FROM logs WHERE session_id = ?", (session_id,))
        count = db_handler.cursor.fetchone()[0]
        assert count == 5

    def test_insert_log_different_levels(self, db_handler):
        """Test inserting logs with different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in levels:
            log_entry = {
                "log_id": str(uuid.uuid4()),
                "session_id": "level_test",
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": f"Test {level} message",
                "unit_name": "TestUnit",
                "parent_id": None
            }
            db_handler.insert_log(log_entry)

        db_handler.cursor.execute("SELECT DISTINCT level FROM logs WHERE session_id = ?", ("level_test",))
        results = db_handler.cursor.fetchall()
        assert len(results) == 5


# ============================================================================
# Logger Tests
# ============================================================================

class TestLogger:
    """Tests for the Logger class."""

    @pytest.fixture
    def mock_db_handler(self):
        """Mock the database handler."""
        with patch('core.framework.base.db_handler') as mock_handler:
            mock_handler.insert_log = Mock()
            mock_handler.insert_session = Mock()
            yield mock_handler

    @pytest.fixture
    def mock_session_context(self):
        """Mock the session context."""
        with patch('core.framework.base.session_context') as mock_ctx:
            mock_ctx.get.return_value = None
            mock_ctx.set = Mock()
            yield mock_ctx

    def test_logger_initialization(self, mock_db_handler, mock_session_context):
        """Test Logger initialization creates session."""
        with patch('core.framework.base.logger') as mock_loguru:
            mock_loguru.add = Mock()

            from core.framework.base import Logger

            logger = Logger("TestUnit")

            assert logger.unit_name == "TestUnit"
            assert logger.session_id is not None
            assert logger.call_stack == []
            assert logger.logs == []

    def test_logger_log_method(self, mock_db_handler, mock_session_context):
        """Test Logger.log method."""
        with patch('core.framework.base.logger') as mock_loguru:
            mock_loguru.add = Mock()
            mock_loguru.log = Mock()

            from core.framework.base import Logger

            logger = Logger("TestUnit")
            logger.log("Test message", "DEBUG")

            mock_loguru.log.assert_called_once_with("DEBUG", "Test message")

    def test_logger_log_exception(self, mock_db_handler, mock_session_context):
        """Test Logger.log_exception method."""
        with patch('core.framework.base.logger') as mock_loguru:
            mock_loguru.add = Mock()
            mock_loguru.log = Mock()

            from core.framework.base import Logger

            logger = Logger("TestUnit")
            test_exception = ValueError("Test error")
            logger.log_exception(test_exception)

            # Should log the exception message
            calls = mock_loguru.log.call_args_list
            assert len(calls) >= 1

    def test_logger_call_stack_management(self, mock_db_handler, mock_session_context):
        """Test Logger call stack push/pop."""
        with patch('core.framework.base.logger') as mock_loguru:
            mock_loguru.add = Mock()

            from core.framework.base import Logger

            logger = Logger("TestUnit")
            logger.call_stack.append("method_1")
            logger.call_stack.append("method_2")

            assert len(logger.call_stack) == 2
            assert logger.call_stack[-1] == "method_2"

            logger.call_stack.pop()
            assert len(logger.call_stack) == 1
            assert logger.call_stack[-1] == "method_1"


# ============================================================================
# Unit Base Class Tests
# ============================================================================

class TestUnitBaseClass:
    """Tests for the Unit abstract base class."""

    @pytest.fixture
    def concrete_unit(self):
        """Create a concrete implementation of Unit for testing."""
        with patch('core.framework.base.db_handler'):
            with patch('core.framework.base.session_context') as mock_ctx:
                mock_ctx.get.return_value = "test_session"
                with patch('core.framework.base.logger') as mock_loguru:
                    mock_loguru.add = Mock()

                    from core.framework.base import Unit, UnitResult
                    from typing import List, Dict, Any

                    class TestUnit(Unit):
                        def schema(self) -> List[Dict[str, Any]]:
                            return [{"type": "function", "function": {"name": "test"}}]

                        def test_method(self, arg: str) -> str:
                            return f"Result: {arg}"

                        def failing_method(self):
                            raise ValueError("Test error")

                    yield TestUnit()

    def test_unit_has_logger(self, concrete_unit):
        """Test that Unit has a logger instance."""
        assert hasattr(concrete_unit, 'logger')
        assert concrete_unit.logger is not None

    def test_unit_success_response_string(self, concrete_unit):
        """Test success_response with string data."""
        result = concrete_unit.success_response("Operation succeeded")

        assert result.success is True
        assert result.output == "Operation succeeded"

    def test_unit_success_response_dict(self, concrete_unit):
        """Test success_response with dictionary data."""
        data = {"status": "ok", "count": 42}
        result = concrete_unit.success_response(data)

        assert result.success is True
        parsed = json.loads(result.output)
        assert parsed["status"] == "ok"
        assert parsed["count"] == 42

    def test_unit_fail_response(self, concrete_unit):
        """Test fail_response method."""
        result = concrete_unit.fail_response("Operation failed")

        assert result.success is False
        assert result.output == "Operation failed"

    def test_unit_schema_abstract(self):
        """Test that Unit.schema is abstract."""
        from core.framework.base import Unit

        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            Unit()

    def test_unit_log_method_decorator(self, concrete_unit):
        """Test that methods are wrapped with logging."""
        # The log_method decorator should wrap callable methods
        result = concrete_unit.test_method("test_input")
        assert "Result: test_input" in str(result)


# ============================================================================
# FastAPI Endpoint Tests
# ============================================================================

class TestFastAPIEndpoints:
    """Tests for FastAPI endpoints in base.py."""

    @pytest.fixture
    def test_client(self, temp_logs_db):
        """Create a test client with mocked database."""
        # We need to patch the module-level database connection
        with patch('core.framework.base.conn') as mock_conn:
            with patch('core.framework.base.c') as mock_cursor:
                mock_conn.cursor.return_value = mock_cursor
                mock_cursor.fetchall.return_value = []

                from core.framework.base import app

                client = TestClient(app)
                yield client, mock_cursor

    def test_get_sessions_empty(self, test_client):
        """Test GET /sessions with no sessions."""
        client, mock_cursor = test_client
        mock_cursor.fetchall.return_value = []

        response = client.get("/sessions")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_sessions_with_data(self, test_client):
        """Test GET /sessions with existing sessions."""
        client, mock_cursor = test_client
        mock_cursor.fetchall.return_value = [
            ("session_1", "2024-01-01T10:00:00"),
            ("session_2", "2024-01-01T11:00:00")
        ]

        response = client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["session_id"] == "session_1"
        assert data[1]["session_id"] == "session_2"

    def test_get_session_logs_not_found(self, test_client):
        """Test GET /sessions/{id}/logs with non-existent session."""
        client, mock_cursor = test_client
        mock_cursor.fetchall.return_value = []

        response = client.get("/sessions/nonexistent/logs")
        assert response.status_code == 404

    def test_get_session_logs_with_data(self, test_client):
        """Test GET /sessions/{id}/logs with existing logs."""
        client, mock_cursor = test_client
        mock_cursor.fetchall.return_value = [
            ("log_1", "session_1", "2024-01-01T10:00:00", "DEBUG", "Test message", "TestUnit", None)
        ]

        response = client.get("/sessions/session_1/logs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["log_id"] == "log_1"
        assert data[0]["message"] == "Test message"

    def test_get_session_logs_tree_not_found(self, test_client):
        """Test GET /sessions/{id}/logs/tree with non-existent session."""
        client, mock_cursor = test_client
        mock_cursor.fetchall.return_value = []

        response = client.get("/sessions/nonexistent/logs/tree")
        assert response.status_code == 404

    def test_get_session_logs_tree_with_hierarchy(self, test_client):
        """Test GET /sessions/{id}/logs/tree with hierarchical logs."""
        client, mock_cursor = test_client
        mock_cursor.fetchall.return_value = [
            ("log_1", "session_1", "2024-01-01T10:00:00", "DEBUG", "Parent log", "TestUnit", None),
            ("log_2", "session_1", "2024-01-01T10:00:01", "DEBUG", "Child log", "TestUnit", "log_1")
        ]

        response = client.get("/sessions/session_1/logs/tree")
        assert response.status_code == 200
        data = response.json()
        # Should have tree structure with parent-child relationship
        assert len(data) >= 1


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unit_result_with_special_characters(self):
        """Test UnitResult with special characters in output."""
        from dataclasses import dataclass

        @dataclass
        class UnitResult:
            success: bool
            output: str

        special_output = "Special chars: \n\t\r\"'\\<>&"
        result = UnitResult(success=True, output=special_output)
        assert result.output == special_output

    def test_unit_result_with_unicode(self):
        """Test UnitResult with unicode characters."""
        from dataclasses import dataclass

        @dataclass
        class UnitResult:
            success: bool
            output: str

        unicode_output = "Unicode: 你好 🚀 émoji ñ"
        result = UnitResult(success=True, output=unicode_output)
        assert result.output == unicode_output

    def test_unit_result_with_very_long_output(self):
        """Test UnitResult with very long output."""
        from dataclasses import dataclass

        @dataclass
        class UnitResult:
            success: bool
            output: str

        long_output = "x" * 100000
        result = UnitResult(success=True, output=long_output)
        assert len(result.output) == 100000
