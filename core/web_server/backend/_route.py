"""Canonical tap-keyed route helper. Every URL that addresses a
`(pipeline, display, filter)` tap on an object must use this — see
`[[project_url_grammar]]`."""


def tap_route(blueprint, prefix: str, *, hierarchy: tuple[str, ...]):
    pattern = (
        f"{prefix}/"
        + "/".join(f"<{h}>" for h in hierarchy)
        + "/<pipeline>/<display>/<filter>"
    )
    def decorator(fn):
        return blueprint.route(pattern)(fn)
    return decorator
