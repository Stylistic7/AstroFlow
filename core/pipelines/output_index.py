"""Per-context index of pipeline tap-point outputs.

When a pipeline traces during compute-graph build, the dispatcher
walks the resulting Graph and records — for every @makes_image Node
whose task has a `display=` label — a `(pipeline_name, object_type,
object_id, display, filter)` → file_path mapping. View-side endpoints
(the mosaic dashboard, tile / thumbnail handlers) read from this
index to answer "what tap points does this pipeline produce, and
where on disk does each live."

Each tap is what a pipeline author has explicitly marked as
user-visible. Intermediate ops (per-strip merge_images, per-panel
place_image) leave no `_display` and don't appear here. User-defined
pipelines drop a `.py` file in `~/AstroFlow/pipelines/` and their
taps surface automatically the first time they trace.

The index is stored in the per-context diskcache (under key
`_pipeline_output_index`) so it survives restarts. Entries are
upserted on each trace; nothing prunes them, so a pipeline that
stops producing some (display, filter) leaves a stale entry until
the user clears it.
"""
from typing import Any

_KEY = "_pipeline_output_index"


def record(
    pipeline_name: str, object_type: str, object_id: str,
    display: str, filter: str | None, path: str,
) -> None:
    """Record one (pipeline, object, display, filter) → path mapping."""
    from context import context
    idx = context.get(_KEY) or {}
    idx[(pipeline_name, object_type, object_id, display, str(filter))] = str(path)
    context[_KEY] = idx


def for_object(
    object_type: str, object_id: str,
) -> dict[str, dict[str, dict[str, str]]]:
    """Return `{pipeline_name: {display: {filter: path}}}` for one
    object. Two-level nesting so consumers can render one row per
    (pipeline, display) with filter columns."""
    from context import context
    idx = context.get(_KEY) or {}
    result: dict[str, dict[str, dict[str, str]]] = {}
    for (pn, ot, oid, display, f), p in idx.items():
        if ot != object_type or oid != object_id:
            continue
        result.setdefault(pn, {}).setdefault(display, {})[f] = p
    return result


def tap_path(
    pipeline_name: str, object_type: str, object_id: str,
    display: str, filter: str | None,
) -> str | None:
    """Look up a specific tap point's path. Returns None on miss."""
    from context import context
    idx = context.get(_KEY) or {}
    return idx.get((pipeline_name, object_type, object_id, display, str(filter)))


def clear() -> None:
    """Wipe the index. Used by tests; not exposed to UI."""
    from context import context
    if _KEY in context:
        del context[_KEY]
