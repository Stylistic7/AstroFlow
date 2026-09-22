"""AstroFlow's MCP sidecar — see `mcp_server.server`."""

# The sidecar's port on the head. Alongside the web servers (8000-8005), not
# among them: a client addresses the MCP endpoint at http://<head>:8010/mcp.
MCP_PORT = 8010
