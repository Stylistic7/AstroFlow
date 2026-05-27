"""Per-cart-item path snapshots, written live as a pipeline runs.

The runtime already populates `GraphRuntime.filenames[node]` with
the resolved on-disk path for every Node whose task has a
`filename_fn` — using the *real* runtime args (the upstream
Node's actual result, not a substituted path). That dict is the
truth of "what paths this cart item has written / will write."

We persist it to diskcache on every node completion via a
callback registered alongside `store_workflow_graph`'s
`graph_update_fn`. Keyed by the cart item shape
`(object_type, object_id, pipeline_name)`. The sweep CLI and
the archive flow read these snapshots — no pipeline tracing,
no type-substitution call_hash bugs.

Shape: `{"all": [path_str, ...], "finals": [path_str, ...]}`.
`finals` is the subset whose producer Node's task carries
`_final == True` (via `@makes_image(final=True)`). Separating
them at write time means "preserve finals only" at archive
time is a trivial lookup, no need to map paths back to ops.
"""
from __future__ import annotations

from context import context
from dependency_graph.graph_runtime import GraphRuntime


SNAPSHOT_KEY_PREFIX = "cart_snapshot"


def store_cart_snapshot(key: tuple, rt: GraphRuntime) -> None:
    """`graph_update_fn` callback: write the cart item's path
    snapshot. Called on every node completion (same firing
    cadence as `store_workflow_graph`), so even interrupted
    runs leave a useful partial snapshot.

    `key` is the `set_workflow_key` tuple
    `(object_type, object_id, pipeline_name)`.
    """
    all_paths: list[str] = []
    final_paths: list[str] = []
    for node, path in rt.filenames.items():
        path_str = str(path)
        all_paths.append(path_str)
        if getattr(node.task, "_final", False):
            final_paths.append(path_str)
    context[(SNAPSHOT_KEY_PREFIX,) + tuple(key)] = {
        "all": all_paths, "finals": final_paths,
    }


def load_cart_snapshot(key: tuple) -> dict | None:
    """Read the snapshot for a cart item. Returns `None` if the
    item has never run (no snapshot was ever written)."""
    return context.get((SNAPSHOT_KEY_PREFIX,) + tuple(key))


def clear_cart_snapshot(key: tuple) -> None:
    """Drop the snapshot for a cart item. Called when the item
    is removed from the cart and (optionally) consumed into the
    archive — at which point the archive entry's inline path
    list takes over as the live-set contribution."""
    context.delete((SNAPSHOT_KEY_PREFIX,) + tuple(key))
