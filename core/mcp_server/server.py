"""AstroFlow's MCP server: typed tools for an LLM operator, served alongside
the web backend on the head node.

Same read layer as the web workers: `local_context` reads the head's
diskcache directly, ContextRefs resolve locally, and nothing here traces a
pipeline on a request — a snapshot that is not built yet is answered with
`NotAvailableYet`, never by computing it. Ray is touched only by confirmed
writes, lazily, through `state.actor.state`.

Where a tool would otherwise copy a web route's body, the body lives in a
shared function both call (`catalog.queries.product_matrix`,
`decisions.helpers.with_override`, `previews.linear_stats.band_stats`).
"""
from __future__ import annotations

import datetime as dt
import enum
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.mcpserver import MCPServer

from cache.context_ref import use_local_context_for_refs
from context import local_context as context

use_local_context_for_refs()

INSTRUCTIONS = """AstroFlow is a content-addressed astrophotography pipeline on a Ray cluster.
Identity is by stable hash, never by name: Mosaic > Target (panel) > Stack
(one camera+filter+exposure of a target) > Run (one night's contiguous
frames, keyed by its content hash) > frame. Start with list_objects, drill
with get_object, find files with products_for_object / resolve_product, look
at pixels with preview / read_region / image_stats, and watch the cluster
with graph_status / running_tasks / events. Writes (set_decision,
clear_decision, drain_and_restart) preview their effect unless
confirm=true; every confirmed write re-plans the whole compute graph."""

mcp = MCPServer("astroflow", instructions=INSTRUCTIONS)


class NotAvailableYet(RuntimeError):
    """The snapshot a tool reads is not built yet this boot (or was evicted)."""


def tool(**kwargs):
    """Every tool returns plain JSON text (or image blocks) — no output
    schema, so the shapes documented in each docstring are the contract."""
    return mcp.tool(structured_output=False, **kwargs)


def _leaf(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, enum.Enum):
        return v.value
    if hasattr(v, "points"):  # spherical_geometry polygons
        return np.asarray(v.points).tolist()
    return str(v)


def to_json(obj: Any) -> Any:
    """JSON-safe copy of a dict / list / dataclass tree. Unlike the web's
    `convert_to_dict`, None values are KEPT: to a typed consumer a null
    and a missing key mean different things."""
    import dataclasses
    from collections.abc import Mapping, Sequence
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [to_json(v) for v in obj]
    return _leaf(obj)
