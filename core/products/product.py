"""A `Product`: the identity of one addressable output image.

A Product names a single point in the catalog's lazy product
`(object × pipeline × tap/display × filter)` — one image AstroFlow knows how to
make. It carries no path and no pixels: those are *derived* from it by the
catalog (`catalog.predict_path` / `.viewable_path` / `.exists`). Identity only,
so it's frozen, hashable, and safe to put in sets and dict keys.

This is the read-side counterpart to the rest of `products/`: `registry.py`
defines how producers *make and name* products; this defines how you *refer to*
one.

`filter` is `None` for a filterless entity (a calibration_stack), and may be a
trace-derived *sentinel* (e.g. a colour composite's `"RGB"`, which isn't one of
the object's real filters) — the catalog treats it as just another filter value.
"""
from dataclasses import dataclass


def object_class(object_type: str) -> "type | None":
    """The entity class for a URL/cart object-type string, or None if unknown.

    The single home for the `{stack: Stack, target: Target, mosaic: Mosaic}`
    map that was redeclared across `trace_paths`, `compute_graph`, and
    `catalog.queries`. Lives here — a leaf module — so every one of those can
    import it without an import cycle (`catalog` imports `trace_paths`, so the
    map could not live in `catalog`). Imports are local for the same reason:
    the entity modules never import this one, but keeping them lazy makes that
    invariant impossible to break by accident."""
    from stacks.composite_stack import Stack
    from targets.types import Target
    from mosaics.types import Mosaic
    return {"stack": Stack, "target": Target, "mosaic": Mosaic}.get(object_type)


@dataclass(frozen=True)
class Product:
    object_type: str          # "stack" | "calibration_stack" | "target" | "mosaic"
    object_id: str            # the entity's stable_hash
    pipeline: str             # a registered pipeline display name
    display: str              # a @node(display=...) tap name
    filter: str | None        # a filter, a sentinel like "RGB", or None

    def as_view(self) -> dict:
        """The `{pipeline, display, filter}` triple the URL grammar uses."""
        return {"pipeline": self.pipeline, "display": self.display,
                "filter": self.filter}
