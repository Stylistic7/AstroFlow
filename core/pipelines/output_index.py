"""Per-context index of pipeline output paths.

When a pipeline traces during compute-graph build, it records the
on-disk path of every sink it produced — keyed by `(pipeline_name,
object_type, object_id, filter)` — into a context-backed dict.
View-side endpoints (the mosaic dashboard, file-download routes)
read from this index to list available outputs without statically
mapping pipeline names to (implementation, quality) tuples.

For the three built-in pipelines (PixInsight Master / Downscaled /
Subframe), this duplicates information that lives in the
`_PIPELINE_FINAL_OUTPUT` mapping in `web_server/backend/metadata.py`.
The index's real value is forward-looking: a user-registered pipeline
in `~/AstroFlow/pipelines/MyPipeline.py` can't be reached through a
static table; recording sinks at trace time is the only general
mechanism that scales.

The index is stored in the per-context diskcache (under key
`_pipeline_output_index`) so it survives restarts. Entries are added
incrementally; nothing prunes them, so a pipeline that stops producing
some filter leaves a stale entry until the user clears it.
"""
from typing import Any

_KEY = "_pipeline_output_index"


def record(
    pipeline_name: str, object_type: str, object_id: str,
    filter: str | None, path: str,
) -> None:
    """Record one (pipeline, object, filter) → path mapping."""
    from context import context
    idx = context.get(_KEY) or {}
    idx[(pipeline_name, object_type, object_id, str(filter))] = str(path)
    context[_KEY] = idx


def for_object(object_type: str, object_id: str) -> dict[str, dict[str, str]]:
    """Return `{pipeline_name: {filter: path}}` for one object."""
    from context import context
    idx = context.get(_KEY) or {}
    result: dict[str, dict[str, str]] = {}
    for (pn, ot, oid, f), p in idx.items():
        if ot != object_type or oid != object_id:
            continue
        result.setdefault(pn, {})[f] = p
    return result


def clear() -> None:
    """Wipe the index. Used by tests; not exposed to UI."""
    from context import context
    if _KEY in context:
        del context[_KEY]
