"""Canonical finish-keyed route helper. Every URL that addresses a
`(pipeline, finish)` view of an object must use this — see
`[[project_url_grammar]]`."""


def finish_route(blueprint, prefix: str, *, hierarchy: tuple[str, ...],
                 suffix: str = "", methods=None):
    """`suffix` lets a view route address sub-resources of the view (a tile
    pyramid's `/<i>/<z>/<x>/<y>.f32`) while keeping the view pair in its
    canonical global→local position."""
    pattern = (
        f"{prefix}/"
        + "/".join(f"<{h}>" for h in hierarchy)
        + "/<pipeline>/<finish>"
        + suffix
    )
    def decorator(fn):
        return blueprint.route(pattern, methods=methods)(fn)
    return decorator
