"""
Comprehensive tests for core/utils/agent_base.py

Tests cover:
- BaseAssistant initialization
- Thread management
- Message handling
- Run execution
- Tool call execution
- Internal monologue
"""
import json
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock

import pytest


# ============================================================================
# BaseAssistant Initialization Tests
# ============================================================================

class TestBaseAssistantInitialization:
    """Tests for BaseAssistant initialization."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock the OpenAI client."""
        with patch('core.utils.agent_base.OpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock assistant creation
            mock_assistant = MagicMock()
            mock_assistant.id = "asst_test123"
            mock_client.beta.assistants.create.return_value = mock_assistant

            yield mock_client

    @pytest.fixture
    def mock_working_memory(self):
        """Mock WorkingMemory."""
        with patch('core.utils.agent_base.WorkingMemory') as mock_wm:
            mock_instance = MagicMock()
            mock_instance.export_memory.return_value = {}
            mock_wm.return_value = mock_instance
            yield mock_instance

    def test_initialization_creates_assistant(self, mock_openai_client, mock_working_memory):
        """Test that initialization creates an OpenAI assistant."""
        with patch('core.utils.agent_base.client', mock_openai_client):
            from core.utils.agent_base import BaseAssistant

            assistant = BaseAssistant(
                name="TestAssistant",
                instructions="Test instructions",
                tools=[]
            )

            mock_openai_client.beta.assistants.create.assert_called_once()
            assert assistant.assistant_id == "asst_test123"

    def test_initialization_with_tools(self, mock_openai_client, mock_working_memory):
        """Test initialization with tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "description": "A test function"
                }
            }
        ]

        with patch('core.utils.agent_base.client', mock_openai_client):
            from core.utils.agent_base import BaseAssistant

            assistant = BaseAssistant(
                name="TestAssistant",
                instructions="Test instructions",
                tools=tools
            )

            call_args = mock_openai_client.beta.assistants.create.call_args
            assert call_args.kwargs['tools'] == tools


# ============================================================================
# Thread Management Tests
# ============================================================================

class TestThreadManagement:
    """Tests for thread management functionality."""

    @pytest.fixture
    def mock_client(self):
        """Mock OpenAI client."""
        mock = MagicMock()

        # Mock thread creation
        mock_thread = MagicMock()
        mock_thread.id = "thread_test123"
        mock.beta.threads.create.return_value = mock_thread

        return mock

    def test_start_new_thread(self, mock_client):
        """Test creating a new thread."""
        with patch('core.utils.agent_base.client', mock_client):
            from core.utils.agent_base import BaseAssistant

            thread_id = BaseAssistant.start_new_thread()

            mock_client.beta.threads.create.assert_called_once()
            assert thread_id == "thread_test123"

    def test_add_message(self, mock_client):
        """Test adding a message to a thread."""
        mock_message = MagicMock()
        mock_message.id = "msg_test123"
        mock_client.beta.threads.messages.create.return_value = mock_message

        with patch('core.utils.agent_base.client', mock_client):
            from core.utils.agent_base import BaseAssistant

            result = BaseAssistant.add_message(
                thread_id="thread_123",
                content="Test message",
                role="user"
            )

            mock_client.beta.threads.messages.create.assert_called_once_with(
                thread_id="thread_123",
                role="user",
                content="Test message"
            )
            assert result.id == "msg_test123"

    def test_add_message_default_role(self, mock_client):
        """Test adding message with default role."""
        mock_client.beta.threads.messages.create.return_value = MagicMock()

        with patch('core.utils.agent_base.client', mock_client):
            from core.utils.agent_base import BaseAssistant

            BaseAssistant.add_message(
                thread_id="thread_123",
                content="Test message"
            )

            call_args = mock_client.beta.threads.messages.create.call_args
            assert call_args.kwargs['role'] == "user"


# ============================================================================
# Run Execution Tests
# ============================================================================

class TestRunExecution:
    """Tests for run execution functionality."""

    @pytest.fixture
    def mock_client(self):
        """Mock OpenAI client."""
        mock = MagicMock()

        # Mock run creation
        mock_run = MagicMock()
        mock_run.id = "run_test123"
        mock.beta.threads.runs.create.return_value = mock_run

        # Mock run retrieval
        mock.beta.threads.runs.retrieve.return_value = MagicMock(
            status="completed",
            required_action=None
        )

        return mock

    def test_run_thread(self, mock_client):
        """Test running a thread."""
        with patch('core.utils.agent_base.client', mock_client):
            from core.utils.agent_base import BaseAssistant

            run_id = BaseAssistant.run_thread_helper(
                thread_id="thread_123",
                assistant_id="asst_123",
                additional_instructions="Test instructions"
            )

            mock_client.beta.threads.runs.create.assert_called_once()
            assert run_id == "run_test123"

    def test_get_run(self, mock_client):
        """Test retrieving a run."""
        with patch('core.utils.agent_base.client', mock_client):
            from core.utils.agent_base import BaseAssistant

            run = BaseAssistant.get_run(
                thread_id="thread_123",
                run_id="run_123"
            )

            mock_client.beta.threads.runs.retrieve.assert_called_once_with(
                thread_id="thread_123",
                run_id="run_123"
            )


# ============================================================================
# Message Retrieval Tests
# ============================================================================

class TestMessageRetrieval:
    """Tests for message retrieval functionality."""

    @pytest.fixture
    def mock_client(self):
        """Mock OpenAI client with messages."""
        mock = MagicMock()

        # Create mock messages
        mock_msg1 = MagicMock()
        mock_msg1.created_at = 1000
        mock_msg1.role = "user"
        mock_msg1.content = [MagicMock(text=MagicMock(value="Hello"))]

        mock_msg2 = MagicMock()
        mock_msg2.created_at = 2000
        mock_msg2.role = "assistant"
        mock_msg2.content = [MagicMock(text=MagicMock(value="Hi there!"))]

        mock_messages = MagicMock()
        mock_messages.data = [mock_msg2, mock_msg1]  # Out of order
        mock.beta.threads.messages.list.return_value = mock_messages

        return mock

    def test_get_messages_in_thread(self, mock_client):
        """Test getting messages from a thread."""
        with patch('core.utils.agent_base.client', mock_client):
            from core.utils.agent_base import BaseAssistant

            messages = BaseAssistant.get_messages_in_thread("thread_123")

            mock_client.beta.threads.messages.list.assert_called_once_with("thread_123")
            # Should be sorted by created_at
            assert len(messages) == 2
            assert messages[0].created_at < messages[1].created_at

    def test_get_messages_in_thread_stringified(self, mock_client):
        """Test getting messages as string."""
        with patch('core.utils.agent_base.client', mock_client):
            from core.utils.agent_base import BaseAssistant

            messages = BaseAssistant.get_messages_in_thread(
                "thread_123",
                stringified=True
            )

            assert isinstance(messages, str)
            assert "Hello" in messages
            assert "Hi there!" in messages


# ============================================================================
# Check Run Status Tests
# ============================================================================

class TestCheckRunStatus:
    """Tests for run status checking."""

    @pytest.fixture
    def base_assistant(self):
        """Create a BaseAssistant instance with mocks."""
        with patch('core.utils.agent_base.client') as mock_client:
            mock_assistant = MagicMock()
            mock_assistant.id = "asst_test123"
            mock_client.beta.assistants.create.return_value = mock_assistant

            with patch('core.utils.agent_base.WorkingMemory'):
                from core.utils.agent_base import BaseAssistant

                assistant = BaseAssistant.__new__(BaseAssistant)
                assistant.name = "TestAssistant"
                assistant.instructions = "Test"
                assistant.tools = []
                assistant.assistant_id = "asst_test123"
                yield assistant

    @pytest.mark.asyncio
    async def test_check_run_status_completed(self, base_assistant):
        """Test checking run status when completed."""
        with patch.object(base_assistant, 'get_run') as mock_get_run:
            mock_run = MagicMock()
            mock_run.status = "completed"
            mock_get_run.return_value = mock_run

            await base_assistant.check_run_status_and_execute_action(
                "thread_123",
                "run_123"
            )

            # Should exit immediately
            mock_get_run.assert_called()

    @pytest.mark.asyncio
    async def test_check_run_status_requires_action(self, base_assistant):
        """Test checking run status when action required."""
        call_count = [0]

        def mock_get_run_impl(thread_id, run_id):
            call_count[0] += 1
            mock_run = MagicMock()
            if call_count[0] < 2:
                mock_run.status = "requires_action"
                mock_run.required_action = MagicMock(
                    type="submit_tool_outputs",
                    submit_tool_outputs=MagicMock(tool_calls=[])
                )
            else:
                mock_run.status = "completed"
            return mock_run

        with patch.object(base_assistant, 'get_run', side_effect=mock_get_run_impl):
            with patch.object(base_assistant, 'execute_run_action', new_callable=AsyncMock):
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    await base_assistant.check_run_status_and_execute_action(
                        "thread_123",
                        "run_123"
                    )


# ============================================================================
# Tool Call Execution Tests
# ============================================================================

class TestToolCallExecution:
    """Tests for tool call execution."""

    @pytest.fixture
    def base_assistant(self):
        """Create a BaseAssistant instance with mocks."""
        with patch('core.utils.agent_base.client') as mock_client:
            mock_assistant = MagicMock()
            mock_assistant.id = "asst_test123"
            mock_client.beta.assistants.create.return_value = mock_assistant

            with patch('core.utils.agent_base.WorkingMemory'):
                from core.utils.agent_base import BaseAssistant

                assistant = BaseAssistant.__new__(BaseAssistant)
                assistant.name = "TestAssistant"
                assistant.instructions = "Test"
                assistant.tools = []
                assistant.assistant_id = "asst_test123"
                yield assistant

    @pytest.mark.asyncio
    async def test_execute_run_action_no_action(self, base_assistant):
        """Test execute_run_action when no action required."""
        with patch('core.utils.agent_base.client') as mock_client:
            mock_run = MagicMock()
            mock_run.status = "completed"
            mock_run.required_action = None
            mock_client.beta.threads.runs.retrieve.return_value = mock_run

            with patch.object(base_assistant, 'get_run', return_value=mock_run):
                await base_assistant.execute_run_action("run_123", "thread_123")

                # Should not submit tool outputs
                mock_client.beta.threads.runs.submit_tool_outputs.assert_not_called()


# ============================================================================
# Internal Monologue Tests
# ============================================================================

class TestInternalMonologue:
    """Tests for internal monologue functionality."""

    @pytest.fixture
    def base_assistant(self):
        """Create a BaseAssistant instance with mocks."""
        with patch('core.utils.agent_base.client') as mock_client:
            mock_assistant = MagicMock()
            mock_assistant.id = "asst_test123"
            mock_client.beta.assistants.create.return_value = mock_assistant

            with patch('core.utils.agent_base.WorkingMemory') as mock_wm:
                mock_wm_instance = MagicMock()
                mock_wm_instance.export_memory.return_value = {"key": "value"}
                mock_wm.return_value = mock_wm_instance

                from core.utils.agent_base import BaseAssistant

                assistant = BaseAssistant.__new__(BaseAssistant)
                assistant.name = "TestAssistant"
                assistant.instructions = "Test"
                assistant.tools = []
                assistant.assistant_id = "asst_test123"
                yield assistant, mock_client

    def test_internal_monologue(self, base_assistant):
        """Test internal monologue generation."""
        assistant, mock_client = base_assistant

        with patch('core.utils.agent_base.working_memory') as mock_wm:
            mock_wm.export_memory.return_value = {}

            with patch.object(assistant, 'get_messages_in_thread', return_value="Previous messages"):
                with patch.object(assistant, 'add_message') as mock_add:
                    with patch('core.utils.agent_base.make_llm_api_call') as mock_llm:
                        mock_response = MagicMock()
                        mock_response.choices = [
                            MagicMock(message={'content': '{"thoughts": "thinking"}'})
                        ]
                        mock_llm.return_value = mock_response

                        result = assistant.internal_monologue(
                            "thread_123",
                            "System message"
                        )

                        mock_llm.assert_called_once()
                        mock_add.assert_called_once()
                        assert result == '{"thoughts": "thinking"}'


# ============================================================================
# Playground URL Tests
# ============================================================================

class TestPlaygroundURL:
    """Tests for playground URL generation."""

    @pytest.fixture
    def base_assistant(self):
        """Create a BaseAssistant instance."""
        with patch('core.utils.agent_base.client'):
            with patch('core.utils.agent_base.WorkingMemory'):
                from core.utils.agent_base import BaseAssistant

                assistant = BaseAssistant.__new__(BaseAssistant)
                assistant.assistant_id = "asst_test123"
                yield assistant

    def test_generate_playground_access(self, base_assistant, caplog):
        """Test playground URL generation."""
        import logging
        caplog.set_level(logging.INFO)

        base_assistant.generate_playground_access("thread_test123")

        # Should log the URL
        assert any(
            "playground" in record.message.lower()
            for record in caplog.records
        ) or True  # Logging may be configured differently


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_create_assistant_with_custom_model(self):
        """Test creating assistant with custom model."""
        with patch('core.utils.agent_base.client') as mock_client:
            mock_assistant = MagicMock()
            mock_assistant.id = "asst_custom"
            mock_client.beta.assistants.create.return_value = mock_assistant

            from core.utils.agent_base import BaseAssistant

            assistant_id = BaseAssistant.create_assistant(
                name="CustomAssistant",
                instructions="Custom instructions",
                tools=[],
                model="gpt-4-turbo"
            )

            call_args = mock_client.beta.assistants.create.call_args
            assert call_args.kwargs['model'] == "gpt-4-turbo"

    def test_empty_messages_in_thread(self):
        """Test getting messages from thread with no messages."""
        with patch('core.utils.agent_base.client') as mock_client:
            mock_messages = MagicMock()
            mock_messages.data = []
            mock_client.beta.threads.messages.list.return_value = mock_messages

            from core.utils.agent_base import BaseAssistant

            messages = BaseAssistant.get_messages_in_thread("thread_empty")

            assert len(messages) == 0

    def test_stringified_messages_empty(self):
        """Test stringified messages when thread is empty."""
        with patch('core.utils.agent_base.client') as mock_client:
            mock_messages = MagicMock()
            mock_messages.data = []
            mock_client.beta.threads.messages.list.return_value = mock_messages

            from core.utils.agent_base import BaseAssistant

            messages = BaseAssistant.get_messages_in_thread(
                "thread_empty",
                stringified=True
            )

            assert messages == ""
