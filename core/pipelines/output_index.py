"""Per-context index of pipeline tap-point outputs.

Two-phase population. The tap-EXISTENCE entries are written when a
pipeline traces — every @makes_image-decorated Node whose task carries
a `display=...` label registers `(pipeline, type, id, display, filter)
→ "pending"` so the UI can render the tap even before any compute
starts. The actual on-disk PATH is written at runtime by the @node
wrapper after the op body finishes producing its output. The two
phases use the same key shape.

Storage uses two diskcache keys per tap:

* `('_pipeline_output', pipeline, object_type, object_id, display,
  filter) -> path | "pending"` — trace-time existence + runtime
  path overwrites it.
* `('_pipeline_output_keys_for', object_type, object_id) -> set` —
  per-object index of every tap-key for that object, so
  `for_object` reads O(taps for this object) instead of scanning
  the full diskcache.

Each tap is what a pipeline author explicitly marked as user-visible.
Intermediate ops (per-strip merge_images, per-panel place_image)
leave no `_display` and don't appear here.
"""

_PREFIX = "_pipeline_output"
_INDEX_PREFIX = "_pipeline_output_keys_for"
PENDING = "pending"


def _tap_key(pipeline_name, object_type, object_id, display, filter):
    return (_PREFIX, pipeline_name, object_type, object_id, display, str(filter))


def _index_key(object_type, object_id):
    return (_INDEX_PREFIX, object_type, object_id)


def _register(pipeline_name, object_type, object_id, display, filter, value):
    """Internal: write `value` to the tap key + ensure the per-object
    secondary index has this tap. Used by both phases.

    The secondary index is a LIST (not a set) so taps preserve the
    order they were registered in. The recorder visits Nodes in
    trace-insertion order (`graph.keys()` on a dict subclass), and
    its suffix counter ("Merged" → "Merged 2" → ...) increments
    per visit — so the list naturally orders taps in graph-reach
    order, which is also numerical order for the suffix series."""
    from context import context
    key = _tap_key(pipeline_name, object_type, object_id, display, filter)
    context[key] = value

    idx_key = _index_key(object_type, object_id)
    keys = context.get(idx_key)
    # Tolerate legacy set-typed entries left in the cache from
    # before this index became a list.
    if not isinstance(keys, list):
        keys = list(keys) if keys else []
    if key not in keys:
        keys = list(keys) + [key]
        context[idx_key] = keys


def record_existence(
    pipeline_name: str, object_type: str, object_id: str,
    display: str, filter: str | None,
) -> None:
    """Trace-time: register that this tap WILL exist. If a real path
    is already in the index (the op ran in a previous session), leave
    it; otherwise mark the tap pending so the UI can show a
    placeholder."""
    from context import context
    key = _tap_key(pipeline_name, object_type, object_id, display, filter)
    if context.get(key) is None:
        _register(pipeline_name, object_type, object_id, display, filter, PENDING)
    else:
        # Even when the path is already there, make sure the secondary
        # index has this key — handles caches that lost it.
        _register(
            pipeline_name, object_type, object_id, display, filter,
            context.get(key),
        )


def record_path(
    pipeline_name: str, object_type: str, object_id: str,
    display: str, filter: str | None, path: str,
) -> None:
    """Runtime: an @makes_image @node body just wrote its output.
    Overwrite the pending placeholder (or any stale path) with the
    real one."""
    _register(pipeline_name, object_type, object_id, display, filter, str(path))


# Back-compat alias for callers that don't yet distinguish existence
# vs path. Treated as a path-write since older callers always had the
# real path in hand.
record = record_path


def for_object(
    object_type: str, object_id: str,
) -> dict[str, dict[str, dict[str, str]]]:
    """Return `{pipeline_name: {display: {filter: path_or_PENDING}}}`
    for one object. Pending entries are returned as the string
    `"pending"` — callers distinguish by string comparison."""
    from context import context
    idx_key = _index_key(object_type, object_id)
    keys = context.get(idx_key) or []
    # Legacy set-typed entries from older versions still iterate
    # fine — the first re-trace re-registers via `_register`,
    # which upgrades the entry to a list (in graph-reach order).
    result: dict[str, dict[str, dict[str, str]]] = {}
    for key in keys:
        if not isinstance(key, tuple) or len(key) != 6:
            continue
        value = context.get(key)
        if value is None:
            continue
        _, pn, _ot, _oid, display, f = key
        result.setdefault(pn, {}).setdefault(display, {})[f] = value
    return result


def tap_path(
    pipeline_name: str, object_type: str, object_id: str,
    display: str, filter: str | None,
) -> str | None:
    """Look up a specific tap point's path. Returns None on miss.
    Returns `"pending"` for taps whose path hasn't been recorded yet."""
    from context import context
    return context.get(
        _tap_key(pipeline_name, object_type, object_id, display, filter)
    )


def clear() -> None:
    """Wipe every tap entry + secondary index. Used by tests."""
    from context import context
    for k in list((context.get_matching((_PREFIX,)) or {}).keys()):
        if k in context:
            del context[k]
    for k in list((context.get_matching((_INDEX_PREFIX,)) or {}).keys()):
        if k in context:
            del context[k]
