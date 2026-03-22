"""Test that SSE stream works with the gevent gunicorn worker.

This verifies that the announcer's queue.Queue is gevent-cooperative,
so the SSE stream can send heartbeats and events without blocking.
"""
import time
import threading

import pytest


def test_sse_heartbeat(test_context):
    """The SSE stream should send a heartbeat within 6 seconds."""
    # Simulate what the gevent worker does: monkey-patch, then import.
    import gevent.monkey
    gevent.monkey.patch_all()

    from web_server.backend.stream import app
    from web_server.backend.announcer import start_event_loop

    start_event_loop()

    client = app.test_client()
    resp = client.get("/stream")

    # Read from the stream with a timeout
    data = b""
    deadline = time.time() + 8
    for chunk in resp.response:
        data += chunk
        if b"heartbeat" in data:
            break
        if time.time() > deadline:
            break

    assert b"heartbeat" in data, f"No heartbeat received. Got: {data[:200]}"


def test_sse_receives_announced_event(test_context):
    """Events sent via announce() should appear in the SSE stream."""
    import gevent.monkey
    gevent.monkey.patch_all()

    from web_server.backend.stream import app
    from web_server.backend.announcer import announcer, start_event_loop

    start_event_loop()

    client = app.test_client()
    resp = client.get("/stream")

    # Announce an event after a short delay
    def send_event():
        time.sleep(1)
        announcer.announce("test_event", "hello")

    t = threading.Thread(target=send_event)
    t.start()

    # Read from the stream
    data = b""
    deadline = time.time() + 5
    for chunk in resp.response:
        data += chunk
        if b"test_event" in data:
            break
        if time.time() > deadline:
            break

    t.join()
    assert b"test_event" in data, f"Event not received. Got: {data[:200]}"
    assert b"hello" in data
