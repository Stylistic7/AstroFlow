"""Two-tier cart helpers: live `context.shopping_cart` + archived
`context.archive`.

When the user removes an item from the cart, they pick what to do
with the outputs that pipeline already produced:

  * `"all"` — every path the pipeline writes (per a fresh trace of
    the cart item's pipeline) goes into the archive entry. Disk is
    preserved unconditionally.
  * `"finals"` — only paths produced by `@makes_image(final=True)`
    ops are kept.
  * `"none"` — no archive entry; every output for this item is
    orphan-eligible.

Paths are computed by tracing the pipeline against the current
context (see `dependency_graph.trace_paths.paths_for_cart_item`).
Tracing is deterministic and statically complete under the pure-
Merkle `content_hash := producer.call_hash` convention — no per-
run snapshot needed.

Archive entries are `(object_type, object_id, pipeline, paths)`
4-tuples — the same shape as cart entries plus an inline frozen
list of preserved paths. The sweep CLI reads them as part of the
live-set union.
"""
from __future__ import annotations

from context import local_context as context
from dependency_graph.trace_paths import paths_for_cart_item


CartItem = tuple[str, str, str]
ArchiveEntry = tuple[str, str, str, list[str]]


def apply_remove_from_cart(item: CartItem, preserve: str = "none") -> None:
    """Remove `item` from `context.shopping_cart` and, depending on
    `preserve` (`"all" | "finals" | "none"`), append a preservation
    entry to `context.archive`. Paths are obtained by tracing the
    pipeline at archive time.

    Idempotent on the cart side: removing an item that isn't in
    the cart is a no-op. If the entity is no longer in context (a
    stale cart entry), no archive entry is created — the trace
    returns `[]`.
    """
    if preserve not in ("all", "finals", "none"):
        raise ValueError(
            f"preserve must be one of 'all'|'finals'|'none', got {preserve!r}"
        )

    cart = list(context.shopping_cart or [])
    if tuple(item) in cart:
        cart.remove(tuple(item))
        context.shopping_cart = cart

    if preserve != "none":
        paths = [
            str(p) for p in paths_for_cart_item(
                tuple(item), finals_only=(preserve == "finals"),
            )
        ]
        if paths:
            archive = list(context.archive or [])
            archive = [a for a in archive if tuple(a[:3]) != tuple(item)]
            archive.append((item[0], item[1], item[2], paths))
            context.archive = archive


def apply_remove_from_archive(item: CartItem) -> None:
    """Drop the archive entry matching `item`. The paths it was
    preserving are no longer in the live set, so the next sweep
    will treat them as orphans."""
    archive = list(context.archive or [])
    archive = [a for a in archive if tuple(a[:3]) != tuple(item)]
    context.archive = archive


def apply_move_archive_to_cart(item: CartItem) -> None:
    """Re-activate an archived item: drop its archive entry and
    add it back to the cart. The pipeline regenerates outputs
    on its next run."""
    apply_remove_from_archive(item)
    cart = list(context.shopping_cart or [])
    if tuple(item) not in cart:
        cart.append(tuple(item))
        context.shopping_cart = cart


def get_archive() -> list[ArchiveEntry]:
    """Read-only snapshot of the archive list."""
    return list(context.archive or [])
