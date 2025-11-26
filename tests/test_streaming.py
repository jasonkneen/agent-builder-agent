"""Tests for streaming and event system."""
import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch

import os
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")


class TestEvent:
    """Tests for Event class."""

    def test_event_creation(self):
        """Test creating an event."""
        from core.streaming import Event

        event = Event(type="test.event", data={"key": "value"})

        assert event.type == "test.event"
        assert event.data == {"key": "value"}
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        from core.streaming import Event

        event = Event(
            type="test.event",
            data={"key": "value"},
            session_id="session123"
        )
        event_dict = event.to_dict()

        assert event_dict["type"] == "test.event"
        assert event_dict["data"] == {"key": "value"}
        assert event_dict["sessionId"] == "session123"

    def test_event_to_json(self):
        """Test converting event to JSON."""
        from core.streaming import Event

        event = Event(type="test.event", data={"key": "value"})
        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "test.event"
        assert parsed["data"] == {"key": "value"}

    def test_event_to_sse(self):
        """Test formatting event as SSE."""
        from core.streaming import Event

        event = Event(type="test.event", data={"key": "value"})
        sse = event.to_sse()

        assert "event: test.event" in sse
        assert "id:" in sse
        assert "data:" in sse
        assert sse.endswith("\n\n")


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_exist(self):
        """Test standard event types are defined."""
        from core.streaming import EventType

        assert EventType.TOOL_STARTED.value == "tool.started"
        assert EventType.TOOL_COMPLETED.value == "tool.completed"
        assert EventType.AGENT_THINKING.value == "agent.thinking"
        assert EventType.TERMINAL_OUTPUT.value == "terminal.output"


class TestEventBus:
    """Tests for EventBus."""

    @pytest.fixture
    def bus(self):
        """Create an event bus."""
        from core.streaming import EventBus
        return EventBus()

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, bus):
        """Test subscribing to and publishing events."""
        from core.streaming import Event

        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("test.event", handler)

        event = Event(type="test.event", data={"key": "value"})
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].type == "test.event"

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, bus):
        """Test wildcard pattern subscription."""
        from core.streaming import Event

        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("test.*", handler)

        await bus.publish(Event(type="test.started", data={}))
        await bus.publish(Event(type="test.completed", data={}))
        await bus.publish(Event(type="other.event", data={}))

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_global_wildcard(self, bus):
        """Test global wildcard subscription."""
        from core.streaming import Event

        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("*", handler)

        await bus.publish(Event(type="test.event", data={}))
        await bus.publish(Event(type="other.event", data={}))

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        """Test unsubscribing from events."""
        from core.streaming import Event

        received = []

        async def handler(event):
            received.append(event)

        unsubscribe = bus.subscribe("test.event", handler)

        await bus.publish(Event(type="test.event", data={}))
        assert len(received) == 1

        unsubscribe()

        await bus.publish(Event(type="test.event", data={}))
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_sync_handler(self, bus):
        """Test synchronous handlers work."""
        from core.streaming import Event

        received = []

        def sync_handler(event):
            received.append(event)

        bus.subscribe("test.event", sync_handler)

        await bus.publish(Event(type="test.event", data={}))

        assert len(received) == 1


class TestStreamBuffer:
    """Tests for StreamBuffer."""

    @pytest.fixture
    def buffer(self):
        """Create a stream buffer."""
        from core.streaming import StreamBuffer
        return StreamBuffer(max_size=100)

    @pytest.mark.asyncio
    async def test_write_and_read(self, buffer):
        """Test writing to and reading from buffer."""
        buffer.write("item1")
        buffer.write("item2")
        buffer.write("item3")
        buffer.close()

        items = []
        async for item in buffer.read():
            items.append(item)

        assert items == ["item1", "item2", "item3"]

    @pytest.mark.asyncio
    async def test_multiple_consumers(self, buffer):
        """Test multiple consumers reading same stream."""
        buffer.write("item1")
        buffer.write("item2")
        buffer.close()

        async def consumer():
            items = []
            async for item in buffer.read():
                items.append(item)
            return items

        # Run two consumers concurrently
        results = await asyncio.gather(consumer(), consumer())

        assert results[0] == ["item1", "item2"]
        assert results[1] == ["item1", "item2"]

    @pytest.mark.asyncio
    async def test_waits_for_new_data(self):
        """Test reader waits for new data."""
        from core.streaming import StreamBuffer

        buffer = StreamBuffer()
        items = []

        async def reader():
            async for item in buffer.read():
                items.append(item)
                if len(items) >= 3:
                    break

        async def writer():
            await asyncio.sleep(0.01)
            buffer.write("item1")
            await asyncio.sleep(0.01)
            buffer.write("item2")
            await asyncio.sleep(0.01)
            buffer.write("item3")

        await asyncio.gather(reader(), writer())

        assert items == ["item1", "item2", "item3"]

    def test_write_to_closed_buffer_raises(self, buffer):
        """Test writing to closed buffer raises error."""
        buffer.close()

        with pytest.raises(RuntimeError):
            buffer.write("item")


class TestSSEStream:
    """Tests for SSEStream."""

    @pytest.fixture
    def stream(self):
        """Create an SSE stream."""
        from core.streaming import SSEStream
        return SSEStream(heartbeat_interval=1.0)

    def test_send_event(self, stream):
        """Test sending an event."""
        from core.streaming import Event

        event = Event(type="test.event", data={"key": "value"})
        stream.send(event)

        # Event should be in buffer
        assert stream._buffer._buffer[-1] == event

    def test_send_data(self, stream):
        """Test sending data as event."""
        stream.send_data("test.type", {"key": "value"}, session_id="session123")

        event = stream._buffer._buffer[-1]
        assert event.type == "test.type"
        assert event.data == {"key": "value"}
        assert event.session_id == "session123"

    @pytest.mark.asyncio
    async def test_events_generator(self, stream):
        """Test events generator yields SSE-formatted strings."""
        from core.streaming import Event

        stream.send(Event(type="test.event", data={}))
        stream.close()

        events = []
        async for sse_event in stream.events():
            events.append(sse_event)
            break  # Just get one event

        assert len(events) == 1
        assert "event: test.event" in events[0]


class TestProgressTracker:
    """Tests for ProgressTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a progress tracker."""
        from core.streaming import ProgressTracker
        return ProgressTracker(total=100)

    @pytest.mark.asyncio
    async def test_update_progress(self, tracker):
        """Test updating progress."""
        await tracker.update(50, message="Halfway there")

        assert tracker.current == 50
        assert tracker.message == "Halfway there"

    @pytest.mark.asyncio
    async def test_complete(self, tracker):
        """Test marking complete."""
        await tracker.update(100)
        await tracker.complete("Done!")

        # Should not raise

    @pytest.mark.asyncio
    async def test_error(self, tracker):
        """Test reporting error."""
        await tracker.error("Something went wrong")

        # Should not raise

    @pytest.mark.asyncio
    async def test_emits_to_event_bus(self):
        """Test tracker emits to event bus."""
        from core.streaming import ProgressTracker, EventBus, EventType

        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("tool.*", handler)

        tracker = ProgressTracker(total=100, event_bus=bus)
        await tracker.update(50)
        await tracker.complete()

        assert len(received) == 2
        assert received[0].type == EventType.TOOL_PROGRESS.value
        assert received[1].type == EventType.TOOL_COMPLETED.value

    @pytest.mark.asyncio
    async def test_emits_to_sse_stream(self):
        """Test tracker emits to SSE stream."""
        from core.streaming import ProgressTracker, SSEStream

        stream = SSEStream()
        tracker = ProgressTracker(total=100, sse_stream=stream)

        await tracker.update(50)

        assert len(stream._buffer._buffer) == 1


class TestGetEventBus:
    """Tests for global event bus instance."""

    def test_returns_singleton(self):
        """Test get_event_bus returns singleton."""
        from core.streaming import get_event_bus

        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2

    @pytest.mark.asyncio
    async def test_global_bus_works(self):
        """Test global bus can subscribe and publish."""
        from core.streaming import get_event_bus, Event

        bus = get_event_bus()
        received = []

        async def handler(event):
            received.append(event)

        # Use a unique event type to avoid interference
        unsubscribe = bus.subscribe("test.global.event", handler)

        try:
            await bus.publish(Event(type="test.global.event", data={}))
            assert len(received) == 1
        finally:
            unsubscribe()
