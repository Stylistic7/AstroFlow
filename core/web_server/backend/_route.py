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
