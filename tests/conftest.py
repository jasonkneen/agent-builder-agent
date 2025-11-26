"""
Shared pytest fixtures for the Kortix test suite.
"""
import os
import sys
import json
import sqlite3
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass

import pytest

# Set environment variables BEFORE importing modules that need them
os.environ.setdefault("OPENAI_API_KEY", "test-api-key-for-testing")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    db_path = tmp_path / "test.db"
    yield str(db_path)
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def temp_logs_db(tmp_path):
    """Create a temporary logs database."""
    db_path = tmp_path / "logs.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            log_id TEXT,
            session_id TEXT,
            timestamp TEXT,
            level TEXT,
            message TEXT,
            unit_name TEXT,
            parent_id TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT,
            start_time TEXT
        )
    ''')
    conn.commit()
    conn.close()
    yield str(db_path)


@pytest.fixture
def working_memory_db(tmp_path):
    """Create a temporary working memory database."""
    db_path = tmp_path / "working_memory.db"
    yield str(db_path)


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory with sample files."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create sample files
    (workspace / "main.py").write_text("# Main file\nprint('hello')")
    (workspace / "utils.py").write_text("# Utils\ndef helper(): pass")
    (workspace / "config.json").write_text('{"key": "value"}')

    # Create subdirectories
    sub_dir = workspace / "subdir"
    sub_dir.mkdir()
    (sub_dir / "module.py").write_text("# Module")

    # Create an excluded directory
    node_modules = workspace / "node_modules"
    node_modules.mkdir()
    (node_modules / "package.js").write_text("// ignored")

    yield workspace


@pytest.fixture
def temp_logs_dir(tmp_path):
    """Create a temporary logs directory."""
    logs_dir = tmp_path / "terminal_logs"
    logs_dir.mkdir()
    yield str(logs_dir)


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.OpenAI') as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Mock assistants API
        client_instance.beta.assistants.create.return_value = MagicMock(id="asst_test123")
        client_instance.beta.threads.create.return_value = MagicMock(id="thread_test123")
        client_instance.beta.threads.messages.create.return_value = MagicMock(id="msg_test123")
        client_instance.beta.threads.messages.list.return_value = MagicMock(data=[])
        client_instance.beta.threads.runs.create.return_value = MagicMock(id="run_test123")
        client_instance.beta.threads.runs.retrieve.return_value = MagicMock(
            status="completed",
            required_action=None
        )

        yield client_instance


@pytest.fixture
def mock_litellm():
    """Mock LiteLLM completion for testing."""
    with patch('litellm.completion') as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message={
                    'content': '{"result": "test"}',
                    'role': 'assistant'
                }
            )
        ]
        mock_completion.return_value = mock_response
        yield mock_completion


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for terminal operations."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        yield mock_run


@pytest.fixture
def mock_docker():
    """Mock Docker operations."""
    with patch('docker.from_env') as mock_docker_env:
        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.id = "container_test123"
        mock_container.name = "workspace_dev-env_1"
        mock_client.containers.list.return_value = [mock_container]

        yield mock_client


# ============================================================================
# Helper Fixtures
# ============================================================================

@pytest.fixture
def sample_messages():
    """Sample LLM messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"}
    ]


@pytest.fixture
def sample_tool_schema():
    """Sample tool schema for testing."""
    return [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "A test function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "First argument"}
                    },
                    "required": ["arg1"]
                }
            }
        }
    ]


@pytest.fixture
def sample_working_memory_data():
    """Sample working memory data for testing."""
    return {
        "Objective": "Build a test application",
        "TerminalSessions": [
            {
                "session_id": "session_1",
                "active": True,
                "action_history": ["echo hello"]
            }
        ],
        "TaskList": [
            {"task_id": "1", "instruction": "First task"},
            {"task_id": "2", "instruction": "Second task"}
        ]
    }


# ============================================================================
# Context Manager Fixtures
# ============================================================================

@pytest.fixture
def isolated_session_context():
    """Isolate session context for tests."""
    from contextvars import copy_context
    ctx = copy_context()
    yield ctx


@pytest.fixture
def mock_environment():
    """Mock environment variables."""
    env_vars = {
        "OPENAI_API_KEY": "test-api-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GROQ_API_KEY": "test-groq-key"
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


# ============================================================================
# Async Fixtures
# ============================================================================

@pytest.fixture
def event_loop_policy():
    """Set event loop policy for async tests."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_loguru():
    """Clean up loguru handlers after each test."""
    from loguru import logger
    yield
    logger.remove()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    yield
    # Add cleanup for any singletons if needed
