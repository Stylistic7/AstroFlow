"""`python -m mcp_server --port 8010` — the head's MCP sidecar."""
import argparse

from mcp.server.transport_security import TransportSecuritySettings

from mcp_server import MCP_PORT, events
from mcp_server import decisions, entities, graph, images  # noqa: F401 — registers the tools
from mcp_server.server import mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="AstroFlow MCP sidecar")
    parser.add_argument("--port", type=int, default=MCP_PORT)
    parser.add_argument("--stream-url", default=events.DEFAULT_STREAM_URL,
                        help="the stream server's SSE endpoint to mirror")
    args = parser.parse_args()
    events.start_listener(args.stream_url)
    mcp.run(
        transport="streamable-http", host="0.0.0.0", port=args.port,
        stateless_http=True, json_response=True,
        # Same trust model as the web API on :8000 — LAN-reachable, any Host.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )


if __name__ == "__main__":
    main()
