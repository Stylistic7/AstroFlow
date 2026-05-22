"""Per-context index of pipeline tap-point outputs.

When a pipeline traces, the dispatcher walks the resulting Graph and
records — for every @makes_image Node whose task has a `display=`
label — a `(pipeline_name, object_type, object_id, display, filter)`
→ file_path mapping. View-side endpoints (the mosaic dashboard, tile
/ thumbnail handlers) read from this index to answer "what tap
points does this pipeline produce, and where on disk does each
live."

Storage uses two diskcache keys per tap:

* `('_pipeline_output', pipeline, object_type, object_id, display,
  filter) -> path` — the actual tap path.
* `('_pipeline_output_keys_for', object_type, object_id) -> set` —
  a per-object index of every tap-key for that object.

The secondary index matters because the alternative — `get_matching`
with the tap-key prefix — does a full diskcache scan (the diskcache
library has no key-prefix index). On a months-old cache that's
seconds per call. Keeping a per-object key-set lets `for_object`
read O(taps for this object) instead of O(whole cache).

Each tap is what a pipeline author explicitly marked as user-visible.
Intermediate ops (per-strip merge_images, per-panel place_image)
leave no `_display` and don't appear here. User-defined pipelines
drop a `.py` file in `~/AstroFlow/pipelines/` and their taps surface
automatically the first time they trace.
"""

_PREFIX = "_pipeline_output"
_INDEX_PREFIX = "_pipeline_output_keys_for"


def _tap_key(pipeline_name, object_type, object_id, display, filter):
    return (_PREFIX, pipeline_name, object_type, object_id, display, str(filter))


def _index_key(object_type, object_id):
    return (_INDEX_PREFIX, object_type, object_id)


def record(
    pipeline_name: str, object_type: str, object_id: str,
    display: str, filter: str | None, path: str,
) -> None:
    """Record one (pipeline, object, display, filter) → path mapping
    plus an entry in the per-object secondary index so `for_object`
    can find this tap without a full cache scan."""
    from context import context
    key = _tap_key(pipeline_name, object_type, object_id, display, filter)
    context[key] = str(path)

    idx_key = _index_key(object_type, object_id)
    keys = context.get(idx_key) or set()
    if key not in keys:
        keys = set(keys)  # copy in case set is shared
        keys.add(key)
        context[idx_key] = keys


def for_object(
    object_type: str, object_id: str,
) -> dict[str, dict[str, dict[str, str]]]:
    """Return `{pipeline_name: {display: {filter: path}}}` for one
    object. Reads the per-object secondary index, then fetches each
    tap-key directly — O(taps for this object), not O(whole cache).
    """
    from context import context
    idx_key = _index_key(object_type, object_id)
    keys = context.get(idx_key) or set()
    result: dict[str, dict[str, dict[str, str]]] = {}
    for key in keys:
        if not isinstance(key, tuple) or len(key) != 6:
            continue
        path = context.get(key)
        if path is None:
            continue
        _, pn, _ot, _oid, display, f = key
        result.setdefault(pn, {}).setdefault(display, {})[f] = path
    return result


def tap_path(
    pipeline_name: str, object_type: str, object_id: str,
    display: str, filter: str | None,
) -> str | None:
    """Look up a specific tap point's path. Returns None on miss."""
    from context import context
    return context.get(
        _tap_key(pipeline_name, object_type, object_id, display, filter)
    )


def clear() -> None:
    """Wipe every tap entry + secondary index. Used by tests; not
    exposed to UI."""
    from context import context
    # Scan via get_matching once (slow but only used by tests).
    for k in list((context.get_matching((_PREFIX,)) or {}).keys()):
        if k in context:
            del context[k]
    for k in list((context.get_matching((_INDEX_PREFIX,)) or {}).keys()):
        if k in context:
            del context[k]
