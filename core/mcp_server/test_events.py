"""The events tool: SSE parsing, the ring buffer, and waiting for a named
event against a tiny local stream server."""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcp_server import events


def test_parse_sse_frames_events_with_optional_data():
    lines = ["event: heartbeat", "data: connected", "", "event: reload_data", "",
             "event: ray_status", "data: {\"x\": 1}", ""]
    assert list(events.parse_sse(lines)) == [
        ("heartbeat", "connected"), ("reload_data", None), ("ray_status", '{"x": 1}')]


def test_parse_sse_flushes_a_trailing_event_without_blank_line():
    assert list(events.parse_sse(["event: reload_data"])) == [("reload_data", None)]


class _Stream(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for frame in ("event: heartbeat\ndata: connected\n\n",
                      "event: compute_graph_status\n\n",
                      "event: reload_data\ndata: 1\n\n"):
            self.wfile.write(frame.encode())
            self.wfile.flush()
            time.sleep(0.05)
        time.sleep(2.0)  # hold the connection open like the real stream

    def log_message(self, *a):  # quiet
        pass


def test_listener_buffers_events_and_wait_for_returns_on_arrival():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        events.start_listener(f"http://127.0.0.1:{server.server_port}/stream")
        out = events.events(wait_for="reload_data", timeout_s=5)
        assert out["arrived"] and out["connected"]
        names = [e["event"] for e in out["events"]]
        assert "reload_data" in names and "compute_graph_status" in names
        assert "heartbeat" not in names and out["last_heartbeat_at"] is not None
        assert events.last_seen("compute_graph_status") is not None
        only = events.events(event="reload_data")["events"]
        assert {e["event"] for e in only} == {"reload_data"}
    finally:
        server.shutdown()
