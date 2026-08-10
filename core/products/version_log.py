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
`context.cache._DURABLE_TUPLE_NS`.

Lifecycle (deterministic, write-site-anchored — no global reconciler):
  * `record` appends the newly PREDICTED output at trace/snapshot time; from
    that moment the UI serves the prior (N-1) file with an out-of-date badge.
  * `supersede` fires from the graph runtime AT THE MOMENT the new version
    lands on disk (node completion): it deletes every other logged version's
    files and trims the log to the landed one. Steady state = exactly ONE
    version per product on disk; the transient 2× lasts only while a rebuild
    is in flight.
Readers: the resolver (`catalog.queries.resolve_product`, last-good fallback)
and the manual sweep (`scripts.sweep_orphans`, which pins every logged path).
Writes go through the actor-backed `context` (like
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
    SAME key — including the filterless case (`None` → `"None"`).

    `object_type` is normalized to the CLASS-NAME case ("Mosaic") the write
    side uses (extract_pipeline_taps emits `type(entity).__name__` verbatim):
    legacy routes pass URL-lowercase ("mosaic"), and the mismatch silently
    disabled the endpoint-side N-1 fallback — every reader missed every
    stored entry (2026-08-10, the "not yet built" dashboard). First-letter
    upper maps both spellings onto the form already durably stored, so no
    migration."""
    object_type = str(object_type)[:1].upper() + str(object_type)[1:]
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


def _files_of(path) -> list:
    """A version's on-disk footprint: primary + `.json` sidecar + display
    pyramid. Lazy imports keep this module a leaf."""
    from file_io.sidecar import sidecar_path_for
    from previews.create_tile_pyramid import pyramid_path_for
    p = Path(str(path))
    return [p, sidecar_path_for(p), pyramid_path_for(p)]


def supersede(k: tuple, landed_path) -> int:
    """The new version just LANDED on disk — delete every other logged
    version's files (primary + sidecar + pyramid) and trim the log to the
    landed entry. This is the ONLY thing that deletes a display product's
    previous version, and it runs at the one moment supersession is a local
    fact rather than an inference: node completion, head-side, single-writer.

    Guards: no file at `landed_path` → delete nothing (a completion that
    produced no file must never orphan the last-good the UI is showing); an
    entry with the landed call_hash is kept, so an unchanged rebuild is a
    no-op. Returns the number of files unlinked; never raises past OSError
    swallowing (a failed unlink is retried implicitly next supersede).
    """
    import os
    landed = str(landed_path)
    if not os.path.exists(landed):
        return 0
    landed_hash = call_hash_of(landed)
    removed = 0
    for e in versions(k):
        if e.get("call_hash") == landed_hash:
            continue
        for p in _files_of(e.get("path", "")):
            try:
                os.unlink(p)
                removed += 1
            except OSError:
                pass  # already gone / never built — fine
    set_versions(k, [{"path": landed, "call_hash": landed_hash,
                      "ts": time.time()}])
    if removed:
        print(f"### version supersede {describe(k)}: {landed_hash[:8]} landed, "
              f"removed {removed} stale file(s)")
    return removed


def versions(k: tuple) -> list:
    """The recorded versions, oldest-first (`[{path, call_hash, ts}, ...]`).

    Reads the DURABLE FILE directly, not the actor-backed context: every
    reader (resolve_product in the Ray-free gunicorn workers, supersede in
    the StateActor, the manual sweep) runs on the head, where the durable
    diskcache lives. Reading via `context` silently returned nothing in the
    web workers — no Ray actor there — so the out-of-date fallback NEVER
    functioned in the UI ("absent" where last-good should have served).
    Writes stay actor-routed (`record`/`set_versions`): run_pipeline tasks
    execute on any node and must land the write on the head."""
    from context.cache import get_durable_cache
    log = get_durable_cache().get(k)
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
