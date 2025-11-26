"""
Streaming and Event System for Kortix

Provides:
- Server-Sent Events (SSE) for real-time updates
- WebSocket support for bidirectional communication
- Event bus for internal pub/sub
- Streaming generators for tool output
"""
import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)
from enum import Enum
import weakref
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EventType(str, Enum):
    """Standard event types in the system."""
    # Tool events
    TOOL_STARTED = "tool.started"
    TOOL_PROGRESS = "tool.progress"
    TOOL_COMPLETED = "tool.completed"
    TOOL_ERROR = "tool.error"

    # Agent events
    AGENT_THINKING = "agent.thinking"
    AGENT_RESPONSE = "agent.response"
    AGENT_TOOL_CALL = "agent.tool_call"

    # Session events
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"

    # Terminal events
    TERMINAL_OUTPUT = "terminal.output"
    TERMINAL_COMMAND = "terminal.command"

    # File events
    FILE_CHANGED = "file.changed"
    FILE_CREATED = "file.created"
    FILE_DELETED = "file.deleted"

    # System events
    HEARTBEAT = "system.heartbeat"
    ERROR = "system.error"


@dataclass
class Event:
    """Base event structure."""
    type: str
    data: Any
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    event_id: str = field(default_factory=lambda: f"{time.time_ns()}")
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
            "eventId": self.event_id,
            "sessionId": self.session_id,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        lines = [
            f"event: {self.type}",
            f"id: {self.event_id}",
            f"data: {self.to_json()}",
            "",  # Empty line to end event
        ]
        return "\n".join(lines) + "\n"


class EventHandler(ABC):
    """Abstract event handler."""

    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle an event."""
        pass


class EventBus:
    """
    Async event bus for pub/sub messaging.

    Usage:
        bus = EventBus()

        # Subscribe to events
        async def handler(event):
            print(f"Received: {event.type}")

        bus.subscribe("tool.*", handler)

        # Publish events
        await bus.publish(Event(type="tool.started", data={"name": "FilesTool"}))
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()

    def subscribe(
        self,
        pattern: str,
        handler: Callable[[Event], Any],
    ) -> Callable[[], None]:
        """
        Subscribe to events matching pattern.

        Patterns support wildcards:
        - "tool.*" matches "tool.started", "tool.completed", etc.
        - "*" matches all events

        Returns unsubscribe function.
        """
        if pattern not in self._handlers:
            self._handlers[pattern] = []
        self._handlers[pattern].append(handler)

        def unsubscribe():
            if pattern in self._handlers:
                self._handlers[pattern].remove(handler)

        return unsubscribe

    async def publish(self, event: Event) -> None:
        """Publish an event to all matching subscribers."""
        handlers = self._get_matching_handlers(event.type)

        # Execute handlers concurrently
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(event))
            else:
                tasks.append(asyncio.to_thread(handler, event))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _get_matching_handlers(self, event_type: str) -> List[Callable]:
        """Get all handlers matching an event type."""
        handlers = []

        for pattern, pattern_handlers in self._handlers.items():
            if self._matches_pattern(pattern, event_type):
                handlers.extend(pattern_handlers)

        return handlers

    def _matches_pattern(self, pattern: str, event_type: str) -> bool:
        """Check if event type matches pattern."""
        if pattern == "*":
            return True

        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix + ".")

        return pattern == event_type


class StreamBuffer(Generic[T]):
    """
    Async buffer for streaming data.

    Supports multiple consumers reading from the same stream.
    """

    def __init__(self, max_size: int = 1000):
        self._buffer: List[T] = []
        self._max_size = max_size
        self._closed = False
        self._event = asyncio.Event()
        self._consumers: Set[int] = set()
        self._consumer_positions: Dict[int, int] = {}
        self._next_consumer_id = 0

    def write(self, item: T) -> None:
        """Write item to buffer."""
        if self._closed:
            raise RuntimeError("Buffer is closed")

        self._buffer.append(item)

        # Trim old items if buffer is too large
        min_position = min(self._consumer_positions.values()) if self._consumer_positions else 0
        if len(self._buffer) > self._max_size:
            trim_count = len(self._buffer) - self._max_size
            trim_count = min(trim_count, min_position)
            if trim_count > 0:
                self._buffer = self._buffer[trim_count:]
                for consumer_id in self._consumer_positions:
                    self._consumer_positions[consumer_id] -= trim_count

        self._event.set()

    def close(self) -> None:
        """Close the buffer."""
        self._closed = True
        self._event.set()

    async def read(self) -> AsyncIterator[T]:
        """Read items from buffer as an async iterator."""
        consumer_id = self._next_consumer_id
        self._next_consumer_id += 1
        self._consumers.add(consumer_id)
        self._consumer_positions[consumer_id] = 0

        try:
            while True:
                # Get next item
                position = self._consumer_positions[consumer_id]

                if position < len(self._buffer):
                    item = self._buffer[position]
                    self._consumer_positions[consumer_id] = position + 1
                    yield item
                elif self._closed:
                    break
                else:
                    # Wait for new data
                    self._event.clear()
                    await self._event.wait()
        finally:
            self._consumers.discard(consumer_id)
            self._consumer_positions.pop(consumer_id, None)


class SSEStream:
    """
    Server-Sent Events stream for HTTP responses.

    Usage with FastAPI:
        @app.get("/stream")
        async def stream():
            stream = SSEStream()

            async def generate():
                async for event in stream.events():
                    yield event

            # Start background task to push events
            asyncio.create_task(push_events(stream))

            return StreamingResponse(generate(), media_type="text/event-stream")
    """

    def __init__(self, heartbeat_interval: float = 15.0):
        self._buffer: StreamBuffer[Event] = StreamBuffer()
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None

    def send(self, event: Event) -> None:
        """Send an event to all connected clients."""
        self._buffer.write(event)

    def send_data(
        self,
        event_type: str,
        data: Any,
        session_id: Optional[str] = None,
    ) -> None:
        """Convenience method to send data as an event."""
        event = Event(type=event_type, data=data, session_id=session_id)
        self.send(event)

    def close(self) -> None:
        """Close the stream."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        self._buffer.close()

    async def events(self) -> AsyncIterator[str]:
        """Iterate over SSE-formatted events."""
        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat())

        try:
            async for event in self._buffer.read():
                yield event.to_sse()
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()

    async def _heartbeat(self) -> None:
        """Send periodic heartbeat events."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            self.send(Event(type=EventType.HEARTBEAT.value, data={"time": time.time()}))


class ProgressTracker:
    """
    Track and stream progress of long-running operations.

    Usage:
        tracker = ProgressTracker(total=100, event_bus=bus)
        for i in range(100):
            # Do work
            await tracker.update(i + 1, message=f"Processing item {i}")
        await tracker.complete()
    """

    def __init__(
        self,
        total: int,
        operation_id: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
        sse_stream: Optional[SSEStream] = None,
    ):
        self.total = total
        self.operation_id = operation_id or f"op_{time.time_ns()}"
        self.current = 0
        self.message = ""
        self.started_at = time.time()
        self._event_bus = event_bus
        self._sse_stream = sse_stream

    async def update(
        self,
        current: int,
        message: str = "",
    ) -> None:
        """Update progress."""
        self.current = current
        self.message = message

        event = Event(
            type=EventType.TOOL_PROGRESS.value,
            data={
                "operationId": self.operation_id,
                "current": self.current,
                "total": self.total,
                "percentage": (self.current / self.total * 100) if self.total > 0 else 0,
                "message": message,
                "elapsedSeconds": time.time() - self.started_at,
            },
        )

        await self._emit(event)

    async def complete(self, message: str = "Completed") -> None:
        """Mark operation as complete."""
        event = Event(
            type=EventType.TOOL_COMPLETED.value,
            data={
                "operationId": self.operation_id,
                "total": self.total,
                "message": message,
                "durationSeconds": time.time() - self.started_at,
            },
        )

        await self._emit(event)

    async def error(self, error: str) -> None:
        """Report an error."""
        event = Event(
            type=EventType.TOOL_ERROR.value,
            data={
                "operationId": self.operation_id,
                "error": error,
                "current": self.current,
                "total": self.total,
            },
        )

        await self._emit(event)

    async def _emit(self, event: Event) -> None:
        """Emit event to all connected outputs."""
        if self._event_bus:
            await self._event_bus.publish(event)
        if self._sse_stream:
            self._sse_stream.send(event)


class TerminalStreamReader:
    """
    Stream terminal output in real-time.

    Instead of reading entire log files, this watches for changes
    and streams new content as it appears.
    """

    def __init__(
        self,
        log_file: str,
        poll_interval: float = 0.1,
    ):
        self.log_file = log_file
        self.poll_interval = poll_interval
        self._position = 0
        self._running = False

    async def stream(self) -> AsyncIterator[str]:
        """Stream new content from log file."""
        import aiofiles
        import os

        self._running = True
        self._position = 0

        # Start from end of file
        if os.path.exists(self.log_file):
            self._position = os.path.getsize(self.log_file)

        while self._running:
            try:
                if not os.path.exists(self.log_file):
                    await asyncio.sleep(self.poll_interval)
                    continue

                current_size = os.path.getsize(self.log_file)

                if current_size > self._position:
                    async with aiofiles.open(self.log_file, "r") as f:
                        await f.seek(self._position)
                        content = await f.read()
                        if content:
                            self._position = current_size
                            yield content

                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error reading terminal log: {e}")
                await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        """Stop streaming."""
        self._running = False


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


# FastAPI integration helpers
def create_sse_response(stream: SSEStream):
    """Create FastAPI StreamingResponse for SSE."""
    from fastapi.responses import StreamingResponse

    async def generate():
        async for event in stream.events():
            yield event

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
