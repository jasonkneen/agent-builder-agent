"""Tests for middleware architecture."""
import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

import os
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")


class TestToolContext:
    """Tests for ToolContext."""

    def test_context_creation(self):
        """Test creating a tool context."""
        from core.middleware import ToolContext

        context = ToolContext(
            tool_name="TestTool",
            method_name="test_method",
            args=(1, 2),
            kwargs={"key": "value"}
        )

        assert context.tool_name == "TestTool"
        assert context.method_name == "test_method"
        assert context.args == (1, 2)
        assert context.kwargs == {"key": "value"}
        assert context.request_id is not None

    def test_context_with_user_session(self):
        """Test context with user and session IDs."""
        from core.middleware import ToolContext

        context = ToolContext(
            tool_name="TestTool",
            method_name="test_method",
            args=(),
            kwargs={},
            user_id="user123",
            session_id="session456"
        )

        assert context.user_id == "user123"
        assert context.session_id == "session456"


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        """Test creating a success result."""
        from core.middleware import ToolResult

        result = ToolResult(success=True, value="test_value", duration_ms=100.5)

        assert result.success is True
        assert result.value == "test_value"
        assert result.error is None
        assert result.duration_ms == 100.5

    def test_error_result(self):
        """Test creating an error result."""
        from core.middleware import ToolResult

        error = ValueError("Test error")
        result = ToolResult(success=False, value=None, error=error)

        assert result.success is False
        assert result.value is None
        assert result.error == error


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create logging middleware with mock logger."""
        from core.middleware import LoggingMiddleware
        mock_logger = MagicMock()
        return LoggingMiddleware(logger=mock_logger), mock_logger

    @pytest.mark.asyncio
    async def test_logs_before_execution(self, middleware):
        """Test logging before execution."""
        from core.middleware import ToolContext

        mw, mock_logger = middleware
        context = ToolContext(
            tool_name="TestTool",
            method_name="test",
            args=(),
            kwargs={}
        )

        await mw.before(context)
        mock_logger.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_after_success(self, middleware):
        """Test logging after successful execution."""
        from core.middleware import ToolContext, ToolResult

        mw, mock_logger = middleware
        context = ToolContext(
            tool_name="TestTool",
            method_name="test",
            args=(),
            kwargs={}
        )
        result = ToolResult(success=True, value="ok", duration_ms=50.0)

        await mw.after(context, result)
        mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_after_failure(self, middleware):
        """Test logging after failed execution."""
        from core.middleware import ToolContext, ToolResult

        mw, mock_logger = middleware
        context = ToolContext(
            tool_name="TestTool",
            method_name="test",
            args=(),
            kwargs={}
        )
        result = ToolResult(
            success=False,
            value=None,
            error=ValueError("Test error"),
            duration_ms=50.0
        )

        await mw.after(context, result)
        mock_logger.info.assert_called()
        mock_logger.error.assert_called()


class TestValidationMiddleware:
    """Tests for ValidationMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create validation middleware with schema."""
        from core.middleware import ValidationMiddleware

        mw = ValidationMiddleware()
        mw.register_schema("TestTool", "test_method", {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["name"]
        })
        return mw

    @pytest.mark.asyncio
    async def test_valid_input_passes(self, middleware):
        """Test valid input passes validation."""
        from core.middleware import ToolContext

        context = ToolContext(
            tool_name="TestTool",
            method_name="test_method",
            args=(),
            kwargs={"name": "test", "count": 5}
        )

        result = await middleware.before(context)
        assert result == context

    @pytest.mark.asyncio
    async def test_missing_required_field_fails(self, middleware):
        """Test missing required field raises error."""
        from core.middleware import ToolContext

        context = ToolContext(
            tool_name="TestTool",
            method_name="test_method",
            args=(),
            kwargs={"count": 5}  # missing 'name'
        )

        with pytest.raises(ValueError, match="Missing required field"):
            await middleware.before(context)

    @pytest.mark.asyncio
    async def test_wrong_type_fails(self, middleware):
        """Test wrong type raises error."""
        from core.middleware import ToolContext

        context = ToolContext(
            tool_name="TestTool",
            method_name="test_method",
            args=(),
            kwargs={"name": "test", "count": "not an int"}
        )

        with pytest.raises(TypeError, match="expected integer"):
            await middleware.before(context)


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        """Test requests within limit are allowed."""
        from core.middleware import RateLimitMiddleware, ToolContext

        mw = RateLimitMiddleware(requests_per_window=5, window_seconds=60)
        context = ToolContext(
            tool_name="Test",
            method_name="test",
            args=(),
            kwargs={}
        )

        # Should allow 5 requests
        for _ in range(5):
            result = await mw.before(context)
            assert result == context

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Test requests over limit are blocked."""
        from core.middleware import RateLimitMiddleware, RateLimitExceeded, ToolContext

        mw = RateLimitMiddleware(requests_per_window=3, window_seconds=60)
        context = ToolContext(
            tool_name="Test",
            method_name="test",
            args=(),
            kwargs={}
        )

        # Use up the limit
        for _ in range(3):
            await mw.before(context)

        # Next request should fail
        with pytest.raises(RateLimitExceeded):
            await mw.before(context)

    @pytest.mark.asyncio
    async def test_per_user_rate_limiting(self):
        """Test rate limiting is per-user."""
        from core.middleware import RateLimitMiddleware, ToolContext

        mw = RateLimitMiddleware(requests_per_window=2, window_seconds=60)

        context_user1 = ToolContext(
            tool_name="Test",
            method_name="test",
            args=(),
            kwargs={},
            user_id="user1"
        )
        context_user2 = ToolContext(
            tool_name="Test",
            method_name="test",
            args=(),
            kwargs={},
            user_id="user2"
        )

        # User 1 uses their limit
        await mw.before(context_user1)
        await mw.before(context_user1)

        # User 2 should still be able to make requests
        result = await mw.before(context_user2)
        assert result == context_user2


class TestRetryMiddleware:
    """Tests for RetryMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create retry middleware."""
        from core.middleware import RetryMiddleware
        return RetryMiddleware(
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0,
            jitter=False
        )

    def test_calculate_delay(self, middleware):
        """Test delay calculation with exponential backoff."""
        delays = [middleware.calculate_delay(i) for i in range(4)]

        assert delays[0] == 0.1  # base_delay
        assert delays[1] == 0.2  # base_delay * 2
        assert delays[2] == 0.4  # base_delay * 4
        assert delays[3] == 0.8  # base_delay * 8

    def test_delay_capped_at_max(self):
        """Test delay is capped at max_delay."""
        from core.middleware import RetryMiddleware

        mw = RetryMiddleware(base_delay=1.0, max_delay=5.0, jitter=False)
        delay = mw.calculate_delay(10)  # Would be 1024 without cap
        assert delay == 5.0

    @pytest.mark.asyncio
    async def test_on_error_returns_retry(self, middleware):
        """Test on_error returns retry signal."""
        from core.middleware import ToolContext

        context = ToolContext(
            tool_name="Test",
            method_name="test",
            args=(),
            kwargs={},
            metadata={"retry_count": 0}
        )

        result = await middleware.on_error(context, ValueError("Test"))

        assert result is not None
        assert result.metadata["should_retry"] is True
        assert result.metadata["delay"] > 0

    @pytest.mark.asyncio
    async def test_on_error_stops_after_max_retries(self, middleware):
        """Test on_error stops after max retries."""
        from core.middleware import ToolContext

        context = ToolContext(
            tool_name="Test",
            method_name="test",
            args=(),
            kwargs={},
            metadata={"retry_count": 3}  # Already at max
        )

        result = await middleware.on_error(context, ValueError("Test"))
        assert result is None


class TestCircuitBreakerMiddleware:
    """Tests for CircuitBreakerMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create circuit breaker middleware."""
        from core.middleware import CircuitBreakerMiddleware
        return CircuitBreakerMiddleware(
            failure_threshold=3,
            recovery_timeout=0.1,
            half_open_requests=1
        )

    @pytest.mark.asyncio
    async def test_starts_closed(self, middleware):
        """Test circuit breaker starts in closed state."""
        assert middleware.state == middleware.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failures(self, middleware):
        """Test circuit opens after threshold failures."""
        from core.middleware import ToolContext, ToolResult

        context = ToolContext(
            tool_name="Test",
            method_name="test",
            args=(),
            kwargs={}
        )

        # Record failures
        for _ in range(3):
            await middleware.on_error(context, ValueError("Test"))

        assert middleware.state == middleware.OPEN

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_requests(self, middleware):
        """Test open circuit blocks new requests."""
        from core.middleware import ToolContext, CircuitBreakerOpen

        context = ToolContext(
            tool_name="Test",
            method_name="test",
            args=(),
            kwargs={}
        )

        # Force open
        for _ in range(3):
            await middleware.on_error(context, ValueError("Test"))

        with pytest.raises(CircuitBreakerOpen):
            await middleware.before(context)

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, middleware):
        """Test circuit goes half-open after recovery timeout."""
        from core.middleware import ToolContext

        context = ToolContext(
            tool_name="Test",
            method_name="test",
            args=(),
            kwargs={}
        )

        # Force open
        for _ in range(3):
            await middleware.on_error(context, ValueError("Test"))

        assert middleware.state == middleware.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        assert middleware.state == middleware.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_success_in_half_open(self, middleware):
        """Test circuit closes after success in half-open state."""
        from core.middleware import ToolContext, ToolResult

        context = ToolContext(
            tool_name="Test",
            method_name="test",
            args=(),
            kwargs={}
        )

        # Force open then half-open
        for _ in range(3):
            await middleware.on_error(context, ValueError("Test"))
        await asyncio.sleep(0.15)

        # Access state to trigger OPEN -> HALF_OPEN transition
        assert middleware.state == middleware.HALF_OPEN

        # Success in half-open state
        result = ToolResult(success=True, value="ok")
        await middleware.after(context, result)

        assert middleware.state == middleware.CLOSED


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create metrics middleware."""
        from core.middleware import MetricsMiddleware
        return MetricsMiddleware()

    @pytest.mark.asyncio
    async def test_tracks_calls(self, middleware):
        """Test middleware tracks call counts."""
        from core.middleware import ToolContext, ToolResult

        context = ToolContext(
            tool_name="TestTool",
            method_name="test",
            args=(),
            kwargs={}
        )

        for _ in range(3):
            result = ToolResult(success=True, value="ok", duration_ms=10.0)
            await middleware.after(context, result)

        metrics = middleware.get_metrics()
        assert metrics["TestTool.test"]["total_calls"] == 3
        assert metrics["TestTool.test"]["successful_calls"] == 3

    @pytest.mark.asyncio
    async def test_tracks_failures(self, middleware):
        """Test middleware tracks failure counts."""
        from core.middleware import ToolContext, ToolResult

        context = ToolContext(
            tool_name="TestTool",
            method_name="test",
            args=(),
            kwargs={}
        )

        result = ToolResult(success=False, value=None, error=ValueError("Test"))
        await middleware.after(context, result)

        metrics = middleware.get_metrics()
        assert metrics["TestTool.test"]["failed_calls"] == 1

    @pytest.mark.asyncio
    async def test_tracks_durations(self, middleware):
        """Test middleware tracks durations."""
        from core.middleware import ToolContext, ToolResult

        context = ToolContext(
            tool_name="TestTool",
            method_name="test",
            args=(),
            kwargs={}
        )

        await middleware.after(context, ToolResult(success=True, value="ok", duration_ms=10.0))
        await middleware.after(context, ToolResult(success=True, value="ok", duration_ms=20.0))
        await middleware.after(context, ToolResult(success=True, value="ok", duration_ms=30.0))

        metrics = middleware.get_metrics()
        assert metrics["TestTool.test"]["min_duration_ms"] == 10.0
        assert metrics["TestTool.test"]["max_duration_ms"] == 30.0
        assert metrics["TestTool.test"]["avg_duration_ms"] == 20.0


class TestMiddlewareChain:
    """Tests for MiddlewareChain."""

    @pytest.mark.asyncio
    async def test_executes_function(self):
        """Test chain executes the target function."""
        from core.middleware import MiddlewareChain

        chain = MiddlewareChain()

        def my_func(x, y):
            return x + y

        result = await chain.execute(
            tool_name="Test",
            method_name="my_func",
            func=my_func,
            args=(1, 2),
        )

        assert result.success is True
        assert result.value == 3

    @pytest.mark.asyncio
    async def test_executes_async_function(self):
        """Test chain executes async functions."""
        from core.middleware import MiddlewareChain

        chain = MiddlewareChain()

        async def my_async_func(x):
            await asyncio.sleep(0.01)
            return x * 2

        result = await chain.execute(
            tool_name="Test",
            method_name="my_async_func",
            func=my_async_func,
            kwargs={"x": 5},
        )

        assert result.success is True
        assert result.value == 10

    @pytest.mark.asyncio
    async def test_middleware_order(self):
        """Test middleware executes in correct order."""
        from core.middleware import MiddlewareChain, Middleware, ToolContext, ToolResult

        calls = []

        class TrackingMiddleware(Middleware):
            def __init__(self, name):
                self.name = name

            async def before(self, context):
                calls.append(f"{self.name}_before")
                return context

            async def after(self, context, result):
                calls.append(f"{self.name}_after")
                return result

        chain = MiddlewareChain()
        chain.add(TrackingMiddleware("A"))
        chain.add(TrackingMiddleware("B"))

        await chain.execute(
            tool_name="Test",
            method_name="test",
            func=lambda: "ok",
        )

        # Before should be A then B, after should be B then A (reverse)
        assert calls == ["A_before", "B_before", "B_after", "A_after"]

    @pytest.mark.asyncio
    async def test_handles_errors(self):
        """Test chain handles function errors."""
        from core.middleware import MiddlewareChain

        chain = MiddlewareChain()

        def failing_func():
            raise ValueError("Test error")

        result = await chain.execute(
            tool_name="Test",
            method_name="failing_func",
            func=failing_func,
        )

        assert result.success is False
        assert isinstance(result.error, ValueError)
