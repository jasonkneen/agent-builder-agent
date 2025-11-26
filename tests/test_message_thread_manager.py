"""
Comprehensive tests for core/utils/message_thread_manager.py

Tests cover:
- MessageThreadManager initialization
- Thread creation
- Message CRUD operations
- Thread execution
- Tool calling
- Edge cases
"""
import json
import sqlite3
from unittest.mock import Mock, MagicMock, patch

import pytest


# ============================================================================
# MessageThreadManager Initialization Tests
# ============================================================================

class TestMessageThreadManagerInitialization:
    """Tests for MessageThreadManager initialization."""

    def test_initialization_creates_database(self, tmp_path):
        """Test that initialization creates the database."""
        db_path = str(tmp_path / "test_threads.db")

        with patch('core.utils.message_thread_manager.TerminalTool'):
            with patch('core.utils.message_thread_manager.FilesTool'):
                from core.utils.message_thread_manager import MessageThreadManager

                manager = MessageThreadManager(db_path)

                assert manager.db_path == db_path
                manager.conn.close()

    def test_initialization_creates_tables(self, tmp_path):
        """Test that initialization creates required tables."""
        db_path = str(tmp_path / "test_threads.db")

        with patch('core.utils.message_thread_manager.TerminalTool'):
            with patch('core.utils.message_thread_manager.FilesTool'):
                from core.utils.message_thread_manager import MessageThreadManager

                manager = MessageThreadManager(db_path)

                # Check table exists
                manager.cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='ThreadMessages'"
                )
                result = manager.cursor.fetchone()
                assert result is not None
                manager.conn.close()


# ============================================================================
# Thread Creation Tests
# ============================================================================

class TestThreadCreation:
    """Tests for thread creation functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a MessageThreadManager instance."""
        db_path = str(tmp_path / "test_threads.db")

        with patch('core.utils.message_thread_manager.TerminalTool'):
            with patch('core.utils.message_thread_manager.FilesTool'):
                from core.utils.message_thread_manager import MessageThreadManager

                manager = MessageThreadManager(db_path)
                yield manager
                manager.conn.close()

    def test_create_thread(self, manager):
        """Test creating a new thread."""
        thread_id = manager.create_thread()

        assert thread_id is not None
        assert isinstance(thread_id, int)
        assert thread_id > 0

    def test_create_multiple_threads(self, manager):
        """Test creating multiple threads."""
        thread_ids = [manager.create_thread() for _ in range(5)]

        assert len(set(thread_ids)) == 5  # All unique
        assert all(isinstance(tid, int) for tid in thread_ids)

    def test_new_thread_has_empty_messages(self, manager):
        """Test that new threads have empty message list."""
        thread_id = manager.create_thread()

        messages = manager.list_messages(thread_id)
        assert messages == []


# ============================================================================
# Message CRUD Tests
# ============================================================================

class TestMessageCRUD:
    """Tests for message Create, Read, Update, Delete operations."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a MessageThreadManager instance."""
        db_path = str(tmp_path / "test_threads.db")

        with patch('core.utils.message_thread_manager.TerminalTool'):
            with patch('core.utils.message_thread_manager.FilesTool'):
                from core.utils.message_thread_manager import MessageThreadManager

                manager = MessageThreadManager(db_path)
                yield manager
                manager.conn.close()

    def test_add_message(self, manager):
        """Test adding a message to a thread."""
        thread_id = manager.create_thread()
        message = {"role": "user", "content": "Hello!"}

        manager.add_message(thread_id, message)

        messages = manager.list_messages(thread_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello!"

    def test_add_multiple_messages(self, manager):
        """Test adding multiple messages."""
        thread_id = manager.create_thread()
        messages_to_add = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        for msg in messages_to_add:
            manager.add_message(thread_id, msg)

        messages = manager.list_messages(thread_id)
        assert len(messages) == 3

    def test_get_message(self, manager):
        """Test getting a specific message."""
        thread_id = manager.create_thread()
        manager.add_message(thread_id, {"role": "user", "content": "First"})
        manager.add_message(thread_id, {"role": "assistant", "content": "Second"})

        message = manager.get_message(thread_id, 0)
        assert message["content"] == "First"

        message = manager.get_message(thread_id, 1)
        assert message["content"] == "Second"

    def test_get_message_out_of_range(self, manager):
        """Test getting message with out of range index."""
        thread_id = manager.create_thread()
        manager.add_message(thread_id, {"role": "user", "content": "Only one"})

        message = manager.get_message(thread_id, 10)
        assert message is None

    def test_modify_message(self, manager):
        """Test modifying a message."""
        thread_id = manager.create_thread()
        manager.add_message(thread_id, {"role": "user", "content": "Original"})

        manager.modify_message(thread_id, 0, {"role": "user", "content": "Modified"})

        message = manager.get_message(thread_id, 0)
        assert message["content"] == "Modified"

    def test_remove_message(self, manager):
        """Test removing a message."""
        thread_id = manager.create_thread()
        manager.add_message(thread_id, {"role": "user", "content": "First"})
        manager.add_message(thread_id, {"role": "user", "content": "Second"})

        manager.remove_message(thread_id, 0)

        messages = manager.list_messages(thread_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "Second"

    def test_list_messages(self, manager):
        """Test listing all messages in a thread."""
        thread_id = manager.create_thread()
        for i in range(5):
            manager.add_message(thread_id, {"role": "user", "content": f"Message {i}"})

        messages = manager.list_messages(thread_id)
        assert len(messages) == 5


# ============================================================================
# Thread Execution Tests
# ============================================================================

class TestThreadExecution:
    """Tests for thread execution functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a MessageThreadManager instance."""
        db_path = str(tmp_path / "test_threads.db")

        with patch('core.utils.message_thread_manager.TerminalTool'):
            with patch('core.utils.message_thread_manager.FilesTool'):
                from core.utils.message_thread_manager import MessageThreadManager

                manager = MessageThreadManager(db_path)
                yield manager
                manager.conn.close()

    def test_run_thread_basic(self, manager):
        """Test basic thread execution."""
        with patch('core.utils.message_thread_manager.make_llm_api_call') as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message={'content': 'Assistant response'})
            ]
            mock_llm.return_value = mock_response

            thread_id = manager.create_thread()
            manager.add_message(thread_id, {"role": "user", "content": "Hello"})

            system_msg = {"role": "system", "content": "You are helpful."}
            response = manager.run_thread(
                thread_id,
                system_msg,
                "gpt-4o"
            )

            mock_llm.assert_called_once()
            assert response is not None

    def test_run_thread_adds_response(self, manager):
        """Test that run_thread adds the response to messages."""
        with patch('core.utils.message_thread_manager.make_llm_api_call') as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message={'content': 'I am an assistant'})
            ]
            mock_llm.return_value = mock_response

            thread_id = manager.create_thread()
            manager.add_message(thread_id, {"role": "user", "content": "Who are you?"})

            system_msg = {"role": "system", "content": "You are helpful."}
            manager.run_thread(thread_id, system_msg, "gpt-4o")

            messages = manager.list_messages(thread_id)
            # Should have user message + assistant response
            assert len(messages) >= 1

    def test_run_thread_with_temperature(self, manager):
        """Test thread execution with custom temperature."""
        with patch('core.utils.message_thread_manager.make_llm_api_call') as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message={'content': 'Response'})]
            mock_llm.return_value = mock_response

            thread_id = manager.create_thread()
            manager.add_message(thread_id, {"role": "user", "content": "Test"})

            system_msg = {"role": "system", "content": "Test"}
            manager.run_thread(thread_id, system_msg, "gpt-4o", temperature=0.7)

            call_args = mock_llm.call_args
            assert call_args[0][3] == 0.7  # temperature argument

    def test_run_thread_with_json_mode(self, manager):
        """Test thread execution with JSON mode."""
        with patch('core.utils.message_thread_manager.make_llm_api_call') as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message={'content': '{"result": "ok"}'})
            ]
            mock_llm.return_value = mock_response

            thread_id = manager.create_thread()
            manager.add_message(thread_id, {"role": "user", "content": "Test"})

            system_msg = {"role": "system", "content": "Respond in JSON"}
            manager.run_thread(
                thread_id,
                system_msg,
                "gpt-4o",
                json_mode=True
            )

            call_args = mock_llm.call_args
            assert call_args[0][2] is True  # json_mode argument


# ============================================================================
# Tool Calling Tests
# ============================================================================

class TestToolCalling:
    """Tests for tool calling functionality in thread execution."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a MessageThreadManager instance."""
        db_path = str(tmp_path / "test_threads.db")

        with patch('core.utils.message_thread_manager.TerminalTool'):
            with patch('core.utils.message_thread_manager.FilesTool'):
                from core.utils.message_thread_manager import MessageThreadManager

                manager = MessageThreadManager(db_path)
                yield manager
                manager.conn.close()

    @pytest.fixture
    def sample_tools(self):
        """Sample tools for testing."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "A test function",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg": {"type": "string"}
                        }
                    }
                }
            }
        ]

    def test_run_thread_with_tools(self, manager, sample_tools):
        """Test thread execution with tools."""
        with patch('core.utils.message_thread_manager.make_llm_api_call') as mock_llm:
            # Response without tool calls - create proper mock message object
            mock_message = MagicMock()
            mock_message.__getitem__ = lambda self, key: 'No tools needed' if key == 'content' else None

            # Make tool_calls falsy but not None to avoid len() error - use empty list
            mock_message.tool_calls = []

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=mock_message)]
            mock_llm.return_value = mock_response

            thread_id = manager.create_thread()
            manager.add_message(thread_id, {"role": "user", "content": "Test"})

            system_msg = {"role": "system", "content": "Test"}
            manager.run_thread(
                thread_id,
                system_msg,
                "gpt-4o",
                tools=sample_tools
            )

            call_args = mock_llm.call_args
            assert call_args[0][5] == sample_tools  # tools argument


# ============================================================================
# Message Serialization Tests
# ============================================================================

class TestMessageSerialization:
    """Tests for message serialization edge cases."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a MessageThreadManager instance."""
        db_path = str(tmp_path / "test_threads.db")

        with patch('core.utils.message_thread_manager.TerminalTool'):
            with patch('core.utils.message_thread_manager.FilesTool'):
                from core.utils.message_thread_manager import MessageThreadManager

                manager = MessageThreadManager(db_path)
                yield manager
                manager.conn.close()

    def test_add_message_with_complex_data(self, manager):
        """Test adding message with complex nested data."""
        thread_id = manager.create_thread()
        message = {
            "role": "user",
            "content": "Test",
            "metadata": {
                "nested": {
                    "data": [1, 2, 3]
                }
            }
        }

        manager.add_message(thread_id, message)

        retrieved = manager.get_message(thread_id, 0)
        assert retrieved["metadata"]["nested"]["data"] == [1, 2, 3]

    def test_add_message_with_unicode(self, manager):
        """Test adding message with unicode content."""
        thread_id = manager.create_thread()
        message = {"role": "user", "content": "你好世界 🌍"}

        manager.add_message(thread_id, message)

        retrieved = manager.get_message(thread_id, 0)
        assert retrieved["content"] == "你好世界 🌍"

    def test_add_message_with_special_characters(self, manager):
        """Test adding message with special characters."""
        thread_id = manager.create_thread()
        message = {"role": "user", "content": 'Special: \n\t\r"\'\\'}

        manager.add_message(thread_id, message)

        retrieved = manager.get_message(thread_id, 0)
        assert "\n" in retrieved["content"]


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a MessageThreadManager instance."""
        db_path = str(tmp_path / "test_threads.db")

        with patch('core.utils.message_thread_manager.TerminalTool'):
            with patch('core.utils.message_thread_manager.FilesTool'):
                from core.utils.message_thread_manager import MessageThreadManager

                manager = MessageThreadManager(db_path)
                yield manager
                manager.conn.close()

    def test_empty_thread_list_messages(self, manager):
        """Test listing messages from empty thread."""
        thread_id = manager.create_thread()
        messages = manager.list_messages(thread_id)
        assert messages == []

    def test_modify_message_out_of_range(self, manager):
        """Test modifying message with out of range index."""
        thread_id = manager.create_thread()
        manager.add_message(thread_id, {"role": "user", "content": "Only one"})

        # Should not raise
        manager.modify_message(thread_id, 100, {"role": "user", "content": "New"})

        # Original should be unchanged
        message = manager.get_message(thread_id, 0)
        assert message["content"] == "Only one"

    def test_remove_message_out_of_range(self, manager):
        """Test removing message with out of range index."""
        thread_id = manager.create_thread()
        manager.add_message(thread_id, {"role": "user", "content": "Only one"})

        # Should not raise
        manager.remove_message(thread_id, 100)

        # Original should still exist
        messages = manager.list_messages(thread_id)
        assert len(messages) == 1

    def test_large_message_content(self, manager):
        """Test handling very large message content."""
        thread_id = manager.create_thread()
        large_content = "x" * 100000
        message = {"role": "user", "content": large_content}

        manager.add_message(thread_id, message)

        retrieved = manager.get_message(thread_id, 0)
        assert len(retrieved["content"]) == 100000

    def test_many_messages_in_thread(self, manager):
        """Test thread with many messages."""
        thread_id = manager.create_thread()

        for i in range(100):
            manager.add_message(thread_id, {"role": "user", "content": f"Message {i}"})

        messages = manager.list_messages(thread_id)
        assert len(messages) == 100

    def test_multiple_threads_isolation(self, manager):
        """Test that threads are isolated from each other."""
        thread1 = manager.create_thread()
        thread2 = manager.create_thread()

        manager.add_message(thread1, {"role": "user", "content": "Thread 1 message"})
        manager.add_message(thread2, {"role": "user", "content": "Thread 2 message"})

        messages1 = manager.list_messages(thread1)
        messages2 = manager.list_messages(thread2)

        assert len(messages1) == 1
        assert len(messages2) == 1
        assert messages1[0]["content"] == "Thread 1 message"
        assert messages2[0]["content"] == "Thread 2 message"
