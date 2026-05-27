"""Two-tier cart helpers: live `context.shopping_cart` + archived
`context.archive`.

When the user removes an item from the cart, they pick what to do
with the outputs that pipeline already produced:

  * `"all"` — every path the pipeline trace would write goes into
    the archive entry. Disk is preserved unconditionally.
  * `"finals"` — only paths produced by `@makes_image(final=True)`
    ops are kept. Intermediates (per-tile placements, partial
    merges, per-Stack masters) go on the next sweep.
  * `"none"` — no archive entry; every output for this item is
    orphan-eligible.

Archive entries are `(object_type, object_id, pipeline, paths)`
4-tuples — the same shape as cart entries plus an inline frozen
list of preserved paths. The sweep CLI reads them as part of the
live-set union.
"""
from __future__ import annotations

from typing import Iterable

from context import local_context as context


CartItem = tuple[str, str, str]
ArchiveEntry = tuple[str, str, str, list[str]]


def apply_remove_from_cart(item: CartItem, preserve: str = "none") -> None:
    """Remove `item` from `context.shopping_cart` and, depending on
    `preserve` (`"all" | "finals" | "none"`), append a preservation
    entry to `context.archive`.

    Idempotent on the cart side: removing an item that isn't in the
    cart is a no-op. Archive append still happens for "all"/"finals"
    if the trace can resolve paths — useful for "I forgot to archive
    this when I removed it; let me re-archive now".
    """
    cart = list(context.shopping_cart or [])
    if tuple(item) in cart:
        cart.remove(tuple(item))
        context.shopping_cart = cart

    if preserve == "none":
        return

    if preserve not in ("all", "finals"):
        raise ValueError(
            f"preserve must be one of 'all'|'finals'|'none', got {preserve!r}"
        )

    paths = _resolve_paths(item, finals_only=(preserve == "finals"))
    if not paths:
        return

    archive = list(context.archive or [])
    entry: ArchiveEntry = (
        item[0], item[1], item[2], [str(p) for p in paths],
    )
    archive = [a for a in archive if tuple(a[:3]) != tuple(item)]
    archive.append(entry)
    context.archive = archive


def apply_remove_from_archive(item: CartItem) -> None:
    """Drop the archive entry matching `item`. The paths it was
    preserving are no longer in the live set, so the next sweep
    will treat them as orphans."""
    archive = list(context.archive or [])
    archive = [a for a in archive if tuple(a[:3]) != tuple(item)]
    context.archive = archive


def apply_move_archive_to_cart(item: CartItem) -> None:
    """Re-activate an archived item: drop its archive entry and add
    it back to the cart. The preserved paths become regular
    trace-derived live paths once it's in the cart."""
    apply_remove_from_archive(item)
    cart = list(context.shopping_cart or [])
    if tuple(item) not in cart:
        cart.append(tuple(item))
        context.shopping_cart = cart


def get_archive() -> list[ArchiveEntry]:
    """Read-only snapshot of the archive list."""
    return list(context.archive or [])


def _resolve_paths(item: CartItem, *, finals_only: bool) -> Iterable:
    """Trace the cart item and return the paths it would write.
    Localized so tests can patch this without importing the heavy
    trace machinery."""
    from dependency_graph.trace_paths import paths_for_item
    return paths_for_item(tuple(item), finals_only=finals_only)
