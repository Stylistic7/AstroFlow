"""The assembled server, in process: every tool registered, a
not-available snapshot surfaces as an error, and a preview comes back as
an image block."""
from pathlib import Path

import anyio
import pytest

pytestmark = pytest.mark.testdata  # reads core/testdata

TESTDATA = Path(__file__).resolve().parent.parent / "testdata"
EXPECTED_TOOLS = {
    "list_objects", "get_object", "finishes_for_object", "resolve_finish",
    "image_metadata", "graph_status", "node_status", "running_tasks",
    "preview", "read_region", "image_stats", "events", "list_decisions",
    "set_decision", "clear_decision", "drain_and_restart",
}


def _server():
    from mcp_server import decisions, entities, events, graph, images  # noqa: F401
    from mcp_server.server import mcp
    return mcp


def test_every_tool_is_registered():
    names = {t.name for t in anyio.run(_server().list_tools)}
    assert EXPECTED_TOOLS <= names


def test_a_missing_snapshot_is_an_error_naming_what_is_missing(test_context):
    from mcp.server.mcpserver.exceptions import UnexpectedToolError
    from mcp_server.server import NotAvailableYet
    with pytest.raises(UnexpectedToolError) as info:
        anyio.run(_server().call_tool, "graph_status", {})
    assert isinstance(info.value.__cause__, NotAvailableYet)
    assert "compute_graph_status" in str(info.value.__cause__)


def test_preview_returns_an_image_block(test_context):
    from context import local_context as lc
    lc.set("input_dirs", [str(TESTDATA / "input")])
    fits = sorted((TESTDATA / "input").glob("*.fit"))[0]
    result = anyio.run(_server().call_tool, "preview", {"path": str(fits), "min_px": 96})
    assert result.content[0].type == "image"
    assert result.content[1].type == "text" and str(fits) in result.content[1].text
