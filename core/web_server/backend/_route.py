"""Canonical tap-keyed route helper. Every URL that addresses a
`(pipeline, display, filter)` tap on an object must use this — see
`[[project_url_grammar]]`."""


def tap_route(blueprint, prefix: str, *, hierarchy: tuple[str, ...], suffix: str = ""):
    """`suffix` lets a tap-keyed route address sub-resources of the tap
    (e.g. a tile pyramid's `/<z>/<x>/<y>.png`) while keeping the view
    triple in its canonical global→local position."""
    pattern = (
        f"{prefix}/"
        + "/".join(f"<{h}>" for h in hierarchy)
        + "/<pipeline>/<display>/<filter>"
        + suffix
    )
    def decorator(fn):
        return blueprint.route(pattern)(fn)
    return decorator


def parse_ancestors(raw: "str | None") -> tuple[tuple[str, str], ...]:
    """Parse the `ancestors` query param — `type:id,type:id` in root→leaf order —
    into a tuple of `(object_type, object_id)`. Empty / missing → `()`.

    A read addressed at a bare leaf (a child stack) misses its own tap snapshot,
    because only the cart item (an ancestor) has one. The frontend knows the full
    hierarchy from the URL, so it passes the ancestors here; the backend reads the
    ancestor's snapshot and filters it to the leaf (`_routes_to_item`). This keeps
    the leaf in the path (the canonical view triple stays put) while naming which
    snapshot holds the data — no scan, no index (see feedback_avoid_derived_state)."""
    if not raw:
        return ()
    out = []
    for seg in raw.split(","):
        object_type, _, object_id = seg.partition(":")
        if object_type and object_id:
            out.append((object_type, object_id))
    return tuple(out)
