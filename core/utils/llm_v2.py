"""
Enhanced LLM API Module for Kortix

Features:
- Async-first design with sync compatibility
- Streaming support for real-time responses
- Exponential backoff with jitter
- Circuit breaker for service resilience
- Multiple provider support (OpenAI, Anthropic, etc.)
- Request/response logging and metrics
"""
import asyncio
import json
import time
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)
from enum import Enum
import logging

import litellm
from litellm import acompletion, completion
from openai import OpenAIError

from ..config import get_llm_config

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    AZURE = "azure"
    LOCAL = "local"


@dataclass
class LLMRequest:
    """Structured LLM request."""
    messages: List[Dict[str, Any]]
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    json_mode: bool = False
    tools: Optional[List[Dict]] = None
    tool_choice: str = "auto"
    stream: bool = False
    timeout: float = 120.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    tool_calls: Optional[List[Dict]] = None
    finish_reason: str = "stop"
    request_id: Optional[str] = None
    latency_ms: float = 0
    raw_response: Optional[Any] = None


@dataclass
class StreamChunk:
    """Single chunk from streaming response."""
    content: str
    is_final: bool = False
    tool_calls: Optional[List[Dict]] = None
    finish_reason: Optional[str] = None


class RetryStrategy:
    """Exponential backoff retry strategy with jitter."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Full jitter: random between 0 and calculated delay
            delay = random.uniform(0, delay)

        return delay

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if we should retry based on attempt and error type."""
        if attempt >= self.max_retries:
            return False

        # Retry on rate limits and server errors
        retryable_errors = (
            OpenAIError,
            ConnectionError,
            TimeoutError,
        )
        return isinstance(error, retryable_errors)


class CircuitBreaker:
    """Circuit breaker for LLM API calls."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def check(self) -> bool:
        """Check if request should be allowed."""
        async with self._lock:
            if self._state == self.CLOSED:
                return True

            if self._state == self.OPEN:
                if (
                    self._last_failure_time
                    and time.time() - self._last_failure_time >= self.recovery_timeout
                ):
                    self._state = self.HALF_OPEN
                    return True
                return False

            # HALF_OPEN
            return True

    async def record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._state = self.CLOSED
            self._failures = 0

    async def record_failure(self) -> None:
        """Record failed call."""
        async with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._failures >= self.failure_threshold:
                self._state = self.OPEN


class LLMClient:
    """
    Enhanced LLM client with async support, streaming, and resilience.

    Usage:
        client = LLMClient()

        # Simple call
        response = await client.chat(messages, model="gpt-4o")

        # Streaming
        async for chunk in client.chat_stream(messages, model="gpt-4o"):
            print(chunk.content, end="")

        # With tools
        response = await client.chat(messages, model="gpt-4o", tools=tools)
    """

    def __init__(
        self,
        retry_strategy: Optional[RetryStrategy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        config = get_llm_config()

        self.retry_strategy = retry_strategy or RetryStrategy(
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
        )
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.default_model = config.default_model
        self.default_temperature = config.default_temperature
        self.default_max_tokens = config.default_max_tokens
        self.request_timeout = config.request_timeout

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """
        Make an async LLM API call with automatic retries.

        Args:
            messages: List of message dicts with role and content
            model: Model name (uses default if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            json_mode: Whether to request JSON output
            tools: List of tool definitions for function calling
            tool_choice: Tool selection mode
            timeout: Request timeout in seconds

        Returns:
            LLMResponse with content, usage, and metadata
        """
        request = LLMRequest(
            messages=messages,
            model=model or self.default_model,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            json_mode=json_mode,
            tools=tools,
            tool_choice=tool_choice,
            timeout=timeout or self.request_timeout,
        )

        return await self._execute_with_retry(request)

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream LLM response chunks.

        Args:
            messages: List of message dicts
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Tool definitions
            tool_choice: Tool selection mode

        Yields:
            StreamChunk objects with incremental content
        """
        # Check circuit breaker
        if not await self.circuit_breaker.check():
            raise CircuitBreakerOpenError("Circuit breaker is open")

        params = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "stream": True,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        try:
            response = await acompletion(**params)

            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta:
                    content = delta.content or ""
                    finish_reason = chunk.choices[0].finish_reason

                    yield StreamChunk(
                        content=content,
                        is_final=finish_reason is not None,
                        finish_reason=finish_reason,
                    )

            await self.circuit_breaker.record_success()

        except Exception as e:
            await self.circuit_breaker.record_failure()
            raise

    async def _execute_with_retry(self, request: LLMRequest) -> LLMResponse:
        """Execute request with retry logic."""
        last_error: Optional[Exception] = None

        for attempt in range(self.retry_strategy.max_retries + 1):
            # Check circuit breaker
            if not await self.circuit_breaker.check():
                raise CircuitBreakerOpenError("Circuit breaker is open")

            try:
                return await self._make_request(request)

            except Exception as e:
                last_error = e
                await self.circuit_breaker.record_failure()

                if not self.retry_strategy.should_retry(attempt, e):
                    break

                delay = self.retry_strategy.get_delay(attempt)
                logger.warning(
                    f"LLM API call failed (attempt {attempt + 1}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)

        raise LLMError(f"LLM API call failed after {attempt + 1} attempts: {last_error}")

    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        """Make a single LLM API request."""
        start_time = time.time()

        params = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }

        if request.max_tokens:
            params["max_tokens"] = request.max_tokens

        if request.json_mode:
            params["response_format"] = {"type": "json_object"}

        if request.tools:
            params["tools"] = request.tools
            params["tool_choice"] = request.tool_choice

        logger.debug(f"LLM request: {json.dumps(params, indent=2)}")

        response = await asyncio.wait_for(
            acompletion(**params),
            timeout=request.timeout
        )

        latency_ms = (time.time() - start_time) * 1000

        await self.circuit_breaker.record_success()

        # Parse response
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls if present
        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]

        return LLMResponse(
            content=message.content or "",
            model=response.model,
            provider=self._get_provider(request.model),
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            request_id=response.id,
            latency_ms=latency_ms,
            raw_response=response,
        )

    def _get_provider(self, model: str) -> str:
        """Determine provider from model name."""
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower:
            return LLMProvider.OPENAI.value
        if "claude" in model_lower:
            return LLMProvider.ANTHROPIC.value
        if "groq" in model_lower or "llama" in model_lower or "mixtral" in model_lower:
            return LLMProvider.GROQ.value
        return "unknown"

    def chat_sync(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Synchronous wrapper for chat method."""
        return asyncio.run(self.chat(messages, model, **kwargs))


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class CircuitBreakerOpenError(LLMError):
    """Raised when circuit breaker is open."""
    pass


# Global client instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get global LLM client instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


# Convenience functions for backwards compatibility
async def async_llm_call(
    messages: List[Dict[str, Any]],
    model: str,
    json_mode: bool = False,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict]] = None,
    tool_choice: str = "auto",
) -> LLMResponse:
    """Async LLM API call (convenience function)."""
    client = get_llm_client()
    return await client.chat(
        messages=messages,
        model=model,
        json_mode=json_mode,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )


def make_llm_api_call(
    messages: List[Dict[str, Any]],
    model_name: str,
    json_mode: bool = False,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict]] = None,
    tool_choice: str = "auto",
):
    """
    Sync LLM API call (backwards compatible with original function).

    Returns the raw LiteLLM response for compatibility.
    """
    client = get_llm_client()
    response = client.chat_sync(
        messages=messages,
        model=model_name,
        json_mode=json_mode,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
    )
    return response.raw_response
