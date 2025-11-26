"""
Middleware Architecture for Kortix Agent Framework

Provides a flexible plugin system for:
- Pre/post execution hooks
- Logging and observability
- Input/output validation
- Rate limiting
- Retry with exponential backoff
- Error handling
"""
import asyncio
import functools
import time
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)
from collections import defaultdict
import random
import threading

from .config import get_config, get_llm_config


T = TypeVar("T")


@dataclass
class ToolContext:
    """Context object passed through middleware chain."""
    tool_name: str
    method_name: str
    args: tuple
    kwargs: Dict[str, Any]
    start_time: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: hashlib.md5(
        f"{time.time()}{random.random()}".encode()
    ).hexdigest()[:12])
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class ToolResult:
    """Result wrapper with metadata."""
    success: bool
    value: Any
    error: Optional[Exception] = None
    duration_ms: float = 0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Middleware(ABC):
    """Base class for all middleware."""

    @abstractmethod
    async def before(self, context: ToolContext) -> ToolContext:
        """Called before tool execution. Can modify context."""
        pass

    @abstractmethod
    async def after(self, context: ToolContext, result: ToolResult) -> ToolResult:
        """Called after tool execution. Can modify result."""
        pass

    async def on_error(self, context: ToolContext, error: Exception) -> Optional[ToolResult]:
        """Called when an error occurs. Return ToolResult to handle, None to propagate."""
        return None


class LoggingMiddleware(Middleware):
    """Logs all tool executions with timing."""

    def __init__(self, logger: Optional[Any] = None):
        from loguru import logger as default_logger
        self.logger = logger or default_logger

    async def before(self, context: ToolContext) -> ToolContext:
        self.logger.debug(
            f"[{context.request_id}] Executing {context.tool_name}.{context.method_name} "
            f"args={context.args} kwargs={context.kwargs}"
        )
        return context

    async def after(self, context: ToolContext, result: ToolResult) -> ToolResult:
        status = "SUCCESS" if result.success else "FAILED"
        self.logger.info(
            f"[{context.request_id}] {context.tool_name}.{context.method_name} "
            f"{status} in {result.duration_ms:.2f}ms"
        )
        if not result.success and result.error:
            self.logger.error(f"[{context.request_id}] Error: {result.error}")
        return result

    async def on_error(self, context: ToolContext, error: Exception) -> Optional[ToolResult]:
        self.logger.error(
            f"[{context.request_id}] Unhandled error in {context.tool_name}.{context.method_name}: {error}"
        )
        return None


class ValidationMiddleware(Middleware):
    """Validates tool inputs against JSON schemas."""

    def __init__(self, schemas: Optional[Dict[str, Dict]] = None):
        self.schemas = schemas or {}

    def register_schema(self, tool_name: str, method_name: str, schema: Dict) -> None:
        """Register a JSON schema for a tool method."""
        key = f"{tool_name}.{method_name}"
        self.schemas[key] = schema

    async def before(self, context: ToolContext) -> ToolContext:
        key = f"{context.tool_name}.{context.method_name}"
        schema = self.schemas.get(key)

        if schema and context.kwargs:
            self._validate(context.kwargs, schema)

        return context

    async def after(self, context: ToolContext, result: ToolResult) -> ToolResult:
        return result

    def _validate(self, data: Dict, schema: Dict) -> None:
        """Simple JSON schema validation."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Check types
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    raise TypeError(
                        f"Field '{field}' expected {expected_type}, got {type(value).__name__}"
                    )

    def _check_type(self, value: Any, expected: str) -> bool:
        """Check if value matches expected JSON schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_types = type_map.get(expected)
        if expected_types:
            return isinstance(value, expected_types)
        return True


class RateLimitMiddleware(Middleware):
    """Rate limits tool executions."""

    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
    ):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    async def before(self, context: ToolContext) -> ToolContext:
        key = context.user_id or "global"
        current_time = time.time()

        with self._lock:
            # Clean old entries
            self._requests[key] = [
                t for t in self._requests[key]
                if current_time - t < self.window_seconds
            ]

            if len(self._requests[key]) >= self.requests_per_window:
                raise RateLimitExceeded(
                    f"Rate limit exceeded: {self.requests_per_window} requests "
                    f"per {self.window_seconds} seconds"
                )

            self._requests[key].append(current_time)

        return context

    async def after(self, context: ToolContext, result: ToolResult) -> ToolResult:
        return result


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


class RetryMiddleware(Middleware):
    """
    Retries failed operations with exponential backoff.

    Uses jitter to prevent thundering herd problem.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add jitter: random value between 0 and delay
            delay = delay * (0.5 + random.random())

        return delay

    async def before(self, context: ToolContext) -> ToolContext:
        context.metadata["retry_count"] = 0
        return context

    async def after(self, context: ToolContext, result: ToolResult) -> ToolResult:
        result.retries = context.metadata.get("retry_count", 0)
        return result

    async def on_error(self, context: ToolContext, error: Exception) -> Optional[ToolResult]:
        if not isinstance(error, self.retryable_exceptions):
            return None

        retry_count = context.metadata.get("retry_count", 0)
        if retry_count >= self.max_retries:
            return None

        context.metadata["retry_count"] = retry_count + 1
        delay = self.calculate_delay(retry_count)

        # Return special result indicating retry needed
        return ToolResult(
            success=False,
            value=None,
            error=error,
            metadata={"should_retry": True, "delay": delay}
        )


class CircuitBreakerMiddleware(Middleware):
    """
    Implements circuit breaker pattern for external service calls.

    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Circuit tripped, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._state = self.CLOSED
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_successes = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                # Check if recovery timeout has passed
                if (
                    self._last_failure_time
                    and time.time() - self._last_failure_time >= self.recovery_timeout
                ):
                    self._state = self.HALF_OPEN
                    self._half_open_successes = 0
            return self._state

    async def before(self, context: ToolContext) -> ToolContext:
        state = self.state

        if state == self.OPEN:
            raise CircuitBreakerOpen(
                f"Circuit breaker is open. Recovery in "
                f"{self.recovery_timeout - (time.time() - self._last_failure_time):.1f}s"
            )

        return context

    async def after(self, context: ToolContext, result: ToolResult) -> ToolResult:
        with self._lock:
            if result.success:
                if self._state == self.HALF_OPEN:
                    self._half_open_successes += 1
                    if self._half_open_successes >= self.half_open_requests:
                        self._state = self.CLOSED
                        self._failures = 0
                elif self._state == self.CLOSED:
                    self._failures = 0
            else:
                self._record_failure()

        return result

    async def on_error(self, context: ToolContext, error: Exception) -> Optional[ToolResult]:
        self._record_failure()
        return None

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._failures >= self.failure_threshold:
                self._state = self.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = self.CLOSED
            self._failures = 0
            self._last_failure_time = None


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


class MetricsMiddleware(Middleware):
    """Collects metrics about tool executions."""

    def __init__(self):
        self._metrics: Dict[str, Dict] = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration_ms": 0,
            "min_duration_ms": float("inf"),
            "max_duration_ms": 0,
        })
        self._lock = threading.Lock()

    async def before(self, context: ToolContext) -> ToolContext:
        return context

    async def after(self, context: ToolContext, result: ToolResult) -> ToolResult:
        key = f"{context.tool_name}.{context.method_name}"

        with self._lock:
            self._metrics[key]["total_calls"] += 1
            self._metrics[key]["total_duration_ms"] += result.duration_ms

            if result.success:
                self._metrics[key]["successful_calls"] += 1
            else:
                self._metrics[key]["failed_calls"] += 1

            self._metrics[key]["min_duration_ms"] = min(
                self._metrics[key]["min_duration_ms"],
                result.duration_ms
            )
            self._metrics[key]["max_duration_ms"] = max(
                self._metrics[key]["max_duration_ms"],
                result.duration_ms
            )

        return result

    def get_metrics(self) -> Dict[str, Dict]:
        """Get a copy of all metrics."""
        with self._lock:
            result = {}
            for key, metrics in self._metrics.items():
                result[key] = {
                    **metrics,
                    "avg_duration_ms": (
                        metrics["total_duration_ms"] / metrics["total_calls"]
                        if metrics["total_calls"] > 0 else 0
                    ),
                    "success_rate": (
                        metrics["successful_calls"] / metrics["total_calls"]
                        if metrics["total_calls"] > 0 else 0
                    ),
                }
            return result

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()


class MiddlewareChain:
    """
    Executes a chain of middleware around tool calls.

    Usage:
        chain = MiddlewareChain()
        chain.add(LoggingMiddleware())
        chain.add(ValidationMiddleware())
        chain.add(RetryMiddleware())

        result = await chain.execute(
            tool_name="FilesTool",
            method_name="read_file",
            func=file_tool.read_file,
            args=(),
            kwargs={"path": "/tmp/test.txt"}
        )
    """

    def __init__(self):
        self.middlewares: List[Middleware] = []

    def add(self, middleware: Middleware) -> "MiddlewareChain":
        """Add middleware to the chain."""
        self.middlewares.append(middleware)
        return self

    def remove(self, middleware_type: type) -> "MiddlewareChain":
        """Remove all middleware of a given type."""
        self.middlewares = [
            m for m in self.middlewares
            if not isinstance(m, middleware_type)
        ]
        return self

    async def execute(
        self,
        tool_name: str,
        method_name: str,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ToolResult:
        """Execute function with middleware chain."""
        kwargs = kwargs or {}

        context = ToolContext(
            tool_name=tool_name,
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            user_id=user_id,
            session_id=session_id,
        )

        # Run before hooks
        for middleware in self.middlewares:
            context = await middleware.before(context)

        # Execute function with retry support
        max_attempts = 10  # Safety limit
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            start_time = time.time()

            try:
                if asyncio.iscoroutinefunction(func):
                    value = await func(*context.args, **context.kwargs)
                else:
                    value = await asyncio.to_thread(func, *context.args, **context.kwargs)

                duration_ms = (time.time() - start_time) * 1000

                result = ToolResult(
                    success=True,
                    value=value,
                    duration_ms=duration_ms,
                )

                # Run after hooks
                for middleware in reversed(self.middlewares):
                    result = await middleware.after(context, result)

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # Try error handlers
                for middleware in reversed(self.middlewares):
                    error_result = await middleware.on_error(context, e)
                    if error_result:
                        # Check if we should retry
                        if error_result.metadata.get("should_retry"):
                            delay = error_result.metadata.get("delay", 1.0)
                            await asyncio.sleep(delay)
                            break
                        return error_result
                else:
                    # No middleware handled the error
                    result = ToolResult(
                        success=False,
                        value=None,
                        error=e,
                        duration_ms=duration_ms,
                    )

                    # Run after hooks even on error
                    for middleware in reversed(self.middlewares):
                        result = await middleware.after(context, result)

                    return result

        # Should not reach here
        return ToolResult(
            success=False,
            value=None,
            error=Exception("Max retry attempts exceeded"),
        )


def with_middleware(chain: MiddlewareChain, tool_name: str = ""):
    """
    Decorator to wrap a method with middleware chain.

    Usage:
        chain = MiddlewareChain().add(LoggingMiddleware())

        class MyTool:
            @with_middleware(chain, "MyTool")
            def my_method(self, arg):
                return arg
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await chain.execute(
                tool_name=tool_name,
                method_name=func.__name__,
                func=func,
                args=args,
                kwargs=kwargs,
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(chain.execute(
                tool_name=tool_name,
                method_name=func.__name__,
                func=func,
                args=args,
                kwargs=kwargs,
            ))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Default middleware chain factory
def create_default_chain() -> MiddlewareChain:
    """Create a middleware chain with sensible defaults."""
    config = get_config()
    llm_config = get_llm_config()

    chain = MiddlewareChain()

    # Always add logging
    chain.add(LoggingMiddleware())

    # Add metrics
    chain.add(MetricsMiddleware())

    # Add validation if enabled
    if "validation" in config.middleware.enabled_middleware:
        chain.add(ValidationMiddleware())

    # Add rate limiting if enabled
    if "rate_limit" in config.middleware.enabled_middleware:
        chain.add(RateLimitMiddleware(
            requests_per_window=config.middleware.rate_limit_requests,
            window_seconds=config.middleware.rate_limit_window_seconds,
        ))

    # Add retry if enabled
    if "retry" in config.middleware.enabled_middleware:
        chain.add(RetryMiddleware(
            max_retries=llm_config.max_retries,
            base_delay=llm_config.retry_base_delay,
            max_delay=llm_config.retry_max_delay,
        ))

    return chain
