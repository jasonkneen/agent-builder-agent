"""
Comprehensive tests for core/utils/llm.py

Tests cover:
- make_llm_api_call function
- Retry logic
- JSON mode handling
- Tool calling
- Error handling
"""
import json
import time
from unittest.mock import Mock, MagicMock, patch, call

import pytest
from openai import OpenAIError


# ============================================================================
# make_llm_api_call Tests
# ============================================================================

class TestMakeLlmApiCall:
    """Tests for the make_llm_api_call function."""

    @pytest.fixture
    def mock_completion(self):
        """Mock the litellm.completion function."""
        with patch('core.utils.llm.completion') as mock:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message={
                        'content': 'Test response',
                        'role': 'assistant'
                    }
                )
            ]
            mock.return_value = mock_response
            yield mock

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]

    def test_basic_api_call(self, mock_completion, sample_messages):
        """Test basic API call without special options."""
        from core.utils.llm import make_llm_api_call

        result = make_llm_api_call(sample_messages, "gpt-4o")

        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "gpt-4o"
        assert call_args.kwargs["messages"] == sample_messages
        assert call_args.kwargs["temperature"] == 0

    def test_api_call_with_temperature(self, mock_completion, sample_messages):
        """Test API call with custom temperature."""
        from core.utils.llm import make_llm_api_call

        make_llm_api_call(sample_messages, "gpt-4o", temperature=0.7)

        call_args = mock_completion.call_args
        assert call_args.kwargs["temperature"] == 0.7

    def test_api_call_with_max_tokens(self, mock_completion, sample_messages):
        """Test API call with max_tokens."""
        from core.utils.llm import make_llm_api_call

        make_llm_api_call(sample_messages, "gpt-4o", max_tokens=1000)

        call_args = mock_completion.call_args
        assert call_args.kwargs["max_tokens"] == 1000

    def test_api_call_json_mode(self, mock_completion, sample_messages):
        """Test API call with JSON mode enabled."""
        from core.utils.llm import make_llm_api_call

        # Setup mock for JSON response
        mock_completion.return_value.choices[0].message = {
            'content': '{"key": "value"}',
            'role': 'assistant'
        }

        make_llm_api_call(sample_messages, "gpt-4o", json_mode=True)

        call_args = mock_completion.call_args
        assert call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_api_call_without_json_mode(self, mock_completion, sample_messages):
        """Test API call without JSON mode."""
        from core.utils.llm import make_llm_api_call

        make_llm_api_call(sample_messages, "gpt-4o", json_mode=False)

        call_args = mock_completion.call_args
        assert call_args.kwargs["response_format"] is None

    def test_api_call_with_tools(self, mock_completion, sample_messages, sample_tool_schema):
        """Test API call with tool definitions."""
        from core.utils.llm import make_llm_api_call

        make_llm_api_call(sample_messages, "gpt-4o", tools=sample_tool_schema)

        call_args = mock_completion.call_args
        assert call_args.kwargs["tools"] == sample_tool_schema
        assert call_args.kwargs["tool_choice"] == "auto"

    def test_api_call_with_tool_choice(self, mock_completion, sample_messages, sample_tool_schema):
        """Test API call with specific tool choice."""
        from core.utils.llm import make_llm_api_call

        make_llm_api_call(
            sample_messages,
            "gpt-4o",
            tools=sample_tool_schema,
            tool_choice="required"
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["tool_choice"] == "required"


# ============================================================================
# Retry Logic Tests
# ============================================================================

class TestRetryLogic:
    """Tests for retry logic in API calls."""

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [{"role": "user", "content": "Test"}]

    def test_retry_on_openai_error(self, sample_messages):
        """Test that API call retries on OpenAI error."""
        with patch('core.utils.llm.completion') as mock_completion:
            with patch('core.utils.llm.time.sleep') as mock_sleep:
                # Fail twice, succeed on third
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message={'content': 'success'})]

                mock_completion.side_effect = [
                    OpenAIError("Error 1"),
                    OpenAIError("Error 2"),
                    mock_response
                ]

                from core.utils.llm import make_llm_api_call

                result = make_llm_api_call(sample_messages, "gpt-4o")

                # Should have retried
                assert mock_completion.call_count == 3
                # Should have slept between retries
                assert mock_sleep.call_count == 2

    def test_max_retries_exceeded(self, sample_messages):
        """Test that exception is raised after max retries."""
        with patch('core.utils.llm.completion') as mock_completion:
            with patch('core.utils.llm.time.sleep'):
                mock_completion.side_effect = OpenAIError("Persistent error")

                from core.utils.llm import make_llm_api_call

                with pytest.raises(Exception) as exc_info:
                    make_llm_api_call(sample_messages, "gpt-4o")

                assert "Failed to make API call" in str(exc_info.value)
                assert mock_completion.call_count == 3

    def test_retry_on_json_decode_error(self, sample_messages):
        """Test retry on invalid JSON in JSON mode."""
        with patch('core.utils.llm.completion') as mock_completion:
            with patch('core.utils.llm.time.sleep'):
                # First return invalid JSON, then valid
                invalid_response = MagicMock()
                invalid_response.choices = [MagicMock(message={'content': 'not json'})]

                valid_response = MagicMock()
                valid_response.choices = [MagicMock(message={'content': '{"valid": true}'})]

                mock_completion.side_effect = [invalid_response, valid_response]

                from core.utils.llm import make_llm_api_call

                result = make_llm_api_call(sample_messages, "gpt-4o", json_mode=True)

                # Should return the valid response
                assert result == valid_response

    def test_successful_call_no_retry(self, sample_messages):
        """Test that successful calls don't trigger retry."""
        with patch('core.utils.llm.completion') as mock_completion:
            with patch('core.utils.llm.time.sleep') as mock_sleep:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message={'content': 'success'})]
                mock_completion.return_value = mock_response

                from core.utils.llm import make_llm_api_call

                make_llm_api_call(sample_messages, "gpt-4o")

                assert mock_completion.call_count == 1
                mock_sleep.assert_not_called()


# ============================================================================
# Model Configuration Tests
# ============================================================================

class TestModelConfiguration:
    """Tests for different model configurations."""

    @pytest.fixture
    def mock_completion(self):
        """Mock completion function."""
        with patch('core.utils.llm.completion') as mock:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message={'content': 'response'})]
            mock.return_value = mock_response
            yield mock

    @pytest.mark.parametrize("model_name", [
        "gpt-4o",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229"
    ])
    def test_different_models(self, mock_completion, model_name):
        """Test API call with different model names."""
        from core.utils.llm import make_llm_api_call

        messages = [{"role": "user", "content": "Test"}]
        make_llm_api_call(messages, model_name)

        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == model_name

    @pytest.mark.parametrize("temperature", [0, 0.5, 1.0, 2.0])
    def test_temperature_range(self, mock_completion, temperature):
        """Test API call with different temperature values."""
        from core.utils.llm import make_llm_api_call

        messages = [{"role": "user", "content": "Test"}]
        make_llm_api_call(messages, "gpt-4o", temperature=temperature)

        call_args = mock_completion.call_args
        assert call_args.kwargs["temperature"] == temperature


# ============================================================================
# JSON Response Tests
# ============================================================================

class TestJsonResponse:
    """Tests for JSON response handling."""

    def test_valid_json_response(self):
        """Test handling of valid JSON response."""
        with patch('core.utils.llm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message={'content': '{"status": "ok", "data": [1, 2, 3]}'})
            ]
            mock_completion.return_value = mock_response

            from core.utils.llm import make_llm_api_call

            result = make_llm_api_call(
                [{"role": "user", "content": "Test"}],
                "gpt-4o",
                json_mode=True
            )

            content = result.choices[0].message['content']
            parsed = json.loads(content)
            assert parsed["status"] == "ok"
            assert parsed["data"] == [1, 2, 3]

    def test_nested_json_response(self):
        """Test handling of nested JSON response."""
        with patch('core.utils.llm.completion') as mock_completion:
            nested_json = json.dumps({
                "level1": {
                    "level2": {
                        "level3": "deep value"
                    }
                }
            })
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message={'content': nested_json})]
            mock_completion.return_value = mock_response

            from core.utils.llm import make_llm_api_call

            result = make_llm_api_call(
                [{"role": "user", "content": "Test"}],
                "gpt-4o",
                json_mode=True
            )

            content = result.choices[0].message['content']
            parsed = json.loads(content)
            assert parsed["level1"]["level2"]["level3"] == "deep value"


# ============================================================================
# Tool Calling Tests
# ============================================================================

class TestToolCalling:
    """Tests for tool calling functionality."""

    @pytest.fixture
    def tools_schema(self):
        """Sample tools schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def test_tool_call_response(self, tools_schema):
        """Test handling of tool call response."""
        with patch('core.utils.llm.completion') as mock_completion:
            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "get_weather"
            mock_tool_call.function.arguments = '{"location": "San Francisco"}'

            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message={
                        'content': None,
                        'tool_calls': [mock_tool_call]
                    }
                )
            ]
            mock_completion.return_value = mock_response

            from core.utils.llm import make_llm_api_call

            result = make_llm_api_call(
                [{"role": "user", "content": "What's the weather?"}],
                "gpt-4o",
                tools=tools_schema
            )

            assert result.choices[0].message['tool_calls'] is not None

    def test_multiple_tools(self, tools_schema):
        """Test API call with multiple tools."""
        with patch('core.utils.llm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message={'content': 'response'})]
            mock_completion.return_value = mock_response

            from core.utils.llm import make_llm_api_call

            make_llm_api_call(
                [{"role": "user", "content": "Test"}],
                "gpt-4o",
                tools=tools_schema
            )

            call_args = mock_completion.call_args
            assert len(call_args.kwargs["tools"]) == 2


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_messages(self):
        """Test API call with empty messages list."""
        with patch('core.utils.llm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message={'content': 'response'})]
            mock_completion.return_value = mock_response

            from core.utils.llm import make_llm_api_call

            # Should not raise
            make_llm_api_call([], "gpt-4o")

    def test_very_long_message(self):
        """Test API call with very long message content."""
        with patch('core.utils.llm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message={'content': 'response'})]
            mock_completion.return_value = mock_response

            from core.utils.llm import make_llm_api_call

            long_content = "x" * 100000
            messages = [{"role": "user", "content": long_content}]

            make_llm_api_call(messages, "gpt-4o")

            call_args = mock_completion.call_args
            assert len(call_args.kwargs["messages"][0]["content"]) == 100000

    def test_unicode_in_messages(self):
        """Test API call with unicode content."""
        with patch('core.utils.llm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message={'content': '你好'})]
            mock_completion.return_value = mock_response

            from core.utils.llm import make_llm_api_call

            messages = [{"role": "user", "content": "你好世界 🌍"}]
            result = make_llm_api_call(messages, "gpt-4o")

            assert result.choices[0].message['content'] == '你好'

    def test_special_characters_in_content(self):
        """Test API call with special characters."""
        with patch('core.utils.llm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message={'content': 'response'})]
            mock_completion.return_value = mock_response

            from core.utils.llm import make_llm_api_call

            messages = [{"role": "user", "content": "Special: \n\t\r\"'\\<>&"}]
            make_llm_api_call(messages, "gpt-4o")

            # Should complete without error

    def test_none_max_tokens(self):
        """Test API call with None max_tokens."""
        with patch('core.utils.llm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message={'content': 'response'})]
            mock_completion.return_value = mock_response

            from core.utils.llm import make_llm_api_call

            make_llm_api_call(
                [{"role": "user", "content": "Test"}],
                "gpt-4o",
                max_tokens=None
            )

            call_args = mock_completion.call_args
            # max_tokens should not be in kwargs when None
            assert "max_tokens" not in call_args.kwargs or call_args.kwargs.get("max_tokens") is None
