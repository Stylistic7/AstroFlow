"""A `Product`: the identity of one addressable output image.

A Product names a single point in the catalog's lazy product
`(object × pipeline × tap/display × filter)` — one image AstroFlow knows how to
make. It carries no path and no pixels: those are *derived* from it by the
catalog (`catalog.predict_path` / `.resolve_product`). Identity only,
so it's frozen, hashable, and safe to put in sets and dict keys.

This is the read-side counterpart to the rest of `products/`: `registry.py`
defines how producers *make and name* products; this defines how you *refer to*
one.

`filter` is `None` for a filterless entity (a calibration_stack), and may be a
trace-derived *sentinel* (e.g. a colour composite's `"RGB"`, which isn't one of
the object's real filters) — the catalog treats it as just another filter value.

The ONE spelling of an object type is the URL/cart word: `"stack"`,
`"calibration_stack"`, `"target"`, `"mosaic"`. `url_type_of` maps a live
entity to it (a `StarFlatStack` is a `"calibration_stack"`), `object_class`
maps it back to the class the pipeline registry dispatches on, and
`resolve_entity` maps `(type, stable_hash)` to the in-context object. This is
a leaf module so every layer — trace, catalog, compute graph, web — can import
it without a cycle; the entity imports are local for the same reason.
"""
from dataclasses import dataclass


def object_class(object_type: str) -> "type | None":
    """The class the pipeline registry dispatches on for a URL/cart object
    type, or None. `calibration_stack` has none: no pipeline processes a
    calibration stack — its master is built INSIDE a light stack's trace."""
    from stacks.composite_stack import Stack
    from targets.types import Target
    from mosaics.types import Mosaic
    return {"stack": Stack, "target": Target, "mosaic": Mosaic}.get(object_type)


def url_type_of(entity) -> str:
    """The URL/cart object type of a live entity — by CLASS, walking the MRO so
    a subclass routes as its base: `StarFlatStack` → `"calibration_stack"`.
    A tap is stamped with this at trace time, so a reader never has to know
    which subclass produced it. Unknown classes fall back to the lowercased
    class name."""
    from stacks.composite_stack import CalibrationStack, Stack
    from targets.types import Target
    from mosaics.types import Mosaic
    words = {Stack: "stack", CalibrationStack: "calibration_stack",
             Target: "target", Mosaic: "mosaic"}
    for cls in type(entity).__mro__:
        if cls in words:
            return words[cls]
    return type(entity).__name__.lower()


def resolve_entity(object_type: str, object_id: str):
    """The in-context entity (Stack / CalibrationStack / Target / Mosaic) for
    an `(object_type, stable_hash)`, or None if not present."""
    from context import local_context as context
    if object_type == "calibration_stack":
        return (context.calibration_stacks or {}).get(object_id)
    table = {"stack": context.stacks, "target": context.targets,
             "mosaic": context.mosaics}.get(object_type)
    ref = (table or {}).get(object_id)
    return ref.get() if ref is not None else None


@dataclass(frozen=True)
class Product:
    object_type: str          # "stack" | "calibration_stack" | "target" | "mosaic"
    object_id: str            # the entity's stable_hash
    pipeline: str             # a registered pipeline display name
    display: str              # a @node(display=...) tap name
    filter: str | None        # a filter, a sentinel like "RGB", or None
