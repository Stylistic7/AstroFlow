"""Per-display-product version history — the one piece of state graceful
N/N-1 versioning needs.

For each display product `(object_type, object_id, pipeline, display, filter)`
we keep an ordered log of the last `_KEEP` output *paths* (newest last). The
path's stem is the output call_hash, which already folds in BOTH the object's
`content_hash` and the pipeline source hash — so a new entry appears iff the
actual output would differ (a data *or* code change); an unchanged rebuild logs
nothing.

This is the ONE thing here that cannot be recomputed on demand: once the old
output files are swept, their call_hashes are unrecoverable. So the log lives in
the never-evict durable store — `"product_versions"` is registered in
`context.cache._DURABLE_TUPLE_NS`. Three readers consume it: the resolver
(`catalog.queries.resolve_product`, last-good fallback), the per-build prune
(delete N-1 once N lands), and the sweep (`scripts.sweep_orphans`, spare N-1
while N is building). Writes go through the actor-backed `context` (like
`asiair.guide_event_stream._push_to_buffer`) so the Flask workers see them.

Leaf module: depends only on `context` + stdlib, so every reader/writer can
import it without a cycle. Keys are built from primitive fields (no `Product`
import needed).
"""
import time
from pathlib import Path

_NS = "product_versions"
_KEEP = 2  # N + N-1. Bump to 3 for extra trickle-in robustness (see plan).


def key(object_type: str, object_id: str, pipeline: str, display: str,
        filter):
    """The durable log key for one display product — the full Product identity
    (`object_type` included: `object_id`/stable_hash collides across a
    panel/target/stack, per `trace_paths._routes_to_item`).

    `filter` is normalized to `str(filter)` so the write side (raw trace
    `filter`, possibly `None`) and every read side (the tile URL string, the
    dashboard's `str(filter)` columns, a `Product.filter`) all resolve to the
    SAME key — including the filterless case (`None` → `"None"`)."""
    return (_NS, object_type, object_id, pipeline, display, str(filter))


def key_for(product) -> tuple:
    """The log key for a `products.product.Product` (duck-typed)."""
    return key(product.object_type, product.object_id, product.pipeline,
               product.display, product.filter)


def call_hash_of(path) -> str:
    """The output call_hash a path encodes — its stem (`<call_hash>.<ext>`)."""
    return Path(str(path)).stem


def describe(k: tuple) -> str:
    """A compact human label for a product-version key, for log lines:
    `mosaic:abcd1234 [Pipeline / Display / Filter]`."""
    _ns, object_type, object_id, pipeline, display, filt = k
    return f"{object_type}:{str(object_id)[:8]} [{pipeline} / {display} / {filt}]"


def _trimmed_append(log: list, entry: dict) -> list:
    """Pure dedup + trim: append `entry` to `log`, but no-op when the newest
    entry already carries this call_hash (an unchanged rebuild — appending a
    duplicate would trim the real N-1 away), then keep the last `_KEEP`."""
    if log and log[-1].get("call_hash") == entry["call_hash"]:
        return log
    return (log + [entry])[-_KEEP:]


def record(k: tuple, path) -> None:
    """Append `path` as this product's newest version (dedup + trim to `_KEEP`).

    Read-check-write through the actor-backed `context` (mirrors
    `_push_to_buffer`) so the Flask workers see it; durable because
    `k[0] == "product_versions"`.
    """
    from context import context
    log = context.get(k)
    if not isinstance(log, list):
        log = []
    entry = {"path": str(path), "call_hash": call_hash_of(path), "ts": time.time()}
    new = _trimmed_append(log, entry)
    if new is not log:
        context[k] = new
        # The one live signal for the invalidation side (data or pipeline-code
        # change → new output hash). Until this new version finishes building,
        # the UI serves the prior (N-1) file with an out-of-date badge. Fires
        # ONLY on a genuine change — an unchanged rebuild dedups and prints
        # nothing, so a code edit's full blast radius shows as one line/product.
        prev = log[-1]["call_hash"][:8] if log else "none"
        print(f"### version_log {describe(k)}: NEW {entry['call_hash'][:8]} "
              f"(prev {prev}) → serving last-good until rebuilt")


def versions(k: tuple) -> list:
    """The recorded versions, oldest-first (`[{path, call_hash, ts}, ...]`)."""
    from context import context
    log = context.get(k)
    return log if isinstance(log, list) else []


def set_versions(k: tuple, entries: list) -> None:
    """Overwrite the log (used by the prune to keep only the current version)."""
    from context import context
    context[k] = list(entries)


def drop(k: tuple) -> None:
    """Forget a product's version history entirely (deliberate discard/evict)."""
    from context import context
    if context.get(k) is not None:
        context[k] = []
