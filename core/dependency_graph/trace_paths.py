"""Trace a cart item's pipeline and return every expected output path.

Under the pure-Merkle `content_hash := producer.call_hash` convention
(stamped into file headers by `@cached.remote`), filenames are deterministic
and statically computable from the trace alone — no file I/O at trace time
beyond ImageMetadata already in context.

The trace itself can be expensive (real measurement on a "PixInsight
Downscaled" mosaic with 730 nodes: ~17 s — dominated by
`resolve_calibration`'s filesystem-walking calibration matching). Since
the trace result is a deterministic pure function of (pipeline source,
entity, ProcessingConfig, metadata), it's cached as a `@node`:
`_traced_paths`. First call pays the cost; every subsequent call with the
same inputs is a sub-ms diskcache hit, durably persisted across restarts.

**Developer note on cache invalidation.** The trace cache key invalidates
when ANY of these change:
  - pipeline body source (`pipeline_source_hash` arg)
  - entity (via stable_hash)
  - ProcessingConfig (decisions, dev settings, cal_stacks ref)
  - metadata_ref content_hash (subframe set + external master set)
It does NOT invalidate when helper functions change (`resolve_calibration`,
`find_external_master`, `_stack_files_narrow`, etc.) — consistent with
the codebase-wide policy that `@node`'s fn_hash covers only the wrapped
function, not its transitive closure. When you edit one of those helpers
and need the trace to re-fire, bust the relevant cache entries explicitly.

Used by `scripts.sweep_orphans` to compute the live set and by
`state.archive` to populate archive entries at cart→archive transition.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any

from context import local_context as context
from decorators import node


@dataclass(frozen=True)
class _TracedOutput:
    """In-memory proxy for an upstream Node's result at trace time.

    Carries only `content_hash`, which equals the producer's `call_hash`
    under the pure-Merkle convention. Deliberately NOT a `ContentHashable`
    subclass — `hashing._maybe_use_content_hash` then skips the
    `get_content_hash(file)` branch (which would read disk) and takes
    the `.content_hash` attribute branch instead. Bit-for-bit identical
    to what the runtime produces when it reads the stamped file header.
    """
    content_hash: str


def paths_for_cart_item(
    item: tuple[str, str, str], *, finals_only: bool = False,
) -> list[Path]:
    """Trace `item`'s pipeline against the current context and return
    every `filename_fn`-resolvable output path.

    `item` is the canonical cart-entry tuple `(object_type, object_id,
    pipeline_name)`. Returns `[]` if the entity isn't in context (stale
    cart entry) or the pipeline isn't registered.

    With `finals_only=True`, restricts to outputs of ops decorated
    `@node(... final=True)` — used by archive transition with
    `preserve="finals"`.

    Dispatches via the cached `_traced_paths` `@node`: first call per
    (pipeline, entity, config, metadata, source) pays the trace cost
    (~17 s for a large mosaic); every subsequent call with identical
    inputs is a diskcache hit.
    """
    import ray
    from hashing import hash_source_code
    from metadata.image_metadata import load_metadata

    fn, entity, config = _resolve_pipeline_call(item)
    if fn is None:
        return []

    metadata_ref = ray.get(load_metadata.remote())
    pipeline_source_hash = hash_source_code(fn)
    _, _, pipeline_name = item

    try:
        tagged: list[tuple[str, bool]] = ray.get(_traced_paths.remote(
            pipeline_name, entity, config, metadata_ref, pipeline_source_hash,
        ))
    except Exception:
        # Trace failures (half-built context, decorator bug, etc.)
        # degrade to "no live paths" rather than crashing the caller.
        return []

    return [Path(p) for p, is_final in tagged
            if not finals_only or is_final]


@node
def _traced_paths(
    pipeline_name: str,
    entity,
    config,
    metadata_ref,
    pipeline_source_hash: str,
) -> list[tuple[str, bool]]:
    """Cacheable core of `paths_for_cart_item`.

    Returns `(path_str, is_final)` tuples for every filename_fn-resolvable
    node in the pipeline's trace. Tagging both modes (finals-only and
    everything) in one cache entry means a single trace serves both
    sweep_orphans and archive transitions.

    `pipeline_source_hash` is an invalidation key only — looked up at
    runtime via `pipeline_name` (the source hash is captured here purely
    to participate in `canonical_call_hash`). `metadata_ref` similarly:
    its content_hash enters the cache key; the trace body doesn't
    consume it directly (the pipeline's downstream tasks read it via
    `config.calibration_stacks` and `context` lookups).
    """
    from dependency_graph.tracing import trace

    fn = _lookup_pipeline_fn(pipeline_name, entity)
    if fn is None:
        return []
    try:
        graph = trace(fn, entity, config)
    except Exception:
        return []
    if graph is None:
        return []
    return _walk_and_resolve_filenames_tagged(graph)


def _lookup_pipeline_fn(pipeline_name: str, entity):
    from pipelines.registry import get_pipeline
    try:
        return get_pipeline(pipeline_name, processes=type(entity))._function
    except KeyError:
        return None


def _resolve_pipeline_call(
    item: tuple[str, str, str],
) -> tuple[Any, Any, Any]:
    """Return `(pipeline_body_fn, entity, ProcessingConfig)` ready to
    trace, or `(None, None, None)` when the cart entry is unresolvable.

    Bypasses the `@pipeline` runner so we don't trigger
    `_snapshot_workflow_graph` / tap recording / `set_workflow_key`
    side effects at sweep time.
    """
    object_type, object_id, pipeline_name = item

    from pipelines.registry import get_pipeline
    from stacks.composite_stack import Stack
    from targets.types import Target
    from mosaics.types import Mosaic
    from cache.context_ref import ContextRef
    from decisions.settings import ProcessingConfig

    type_to_class = {"stack": Stack, "target": Target, "mosaic": Mosaic}
    type_to_dict = {
        "stack": context.stacks or {},
        "target": context.targets or {},
        "mosaic": context.mosaics or {},
    }

    cls = type_to_class.get(object_type)
    if cls is None:
        return None, None, None

    entity_ref = type_to_dict[object_type].get(object_id)
    if entity_ref is None:
        return None, None, None
    entity = entity_ref.get() if hasattr(entity_ref, "get") else entity_ref

    try:
        runner = get_pipeline(pipeline_name, processes=cls)
    except KeyError:
        return None, None, None
    fn = runner._function

    # Mirror build_main_processing_graph's config construction so the
    # trace's call_hashes match the live graph's. `decisions` is the
    # full overrides set here; narrow_stack_config filters it to per-
    # stack relevant entries inside stack_files's narrow hook.
    settings = context.settings
    cal_stacks_ref = ContextRef.store_large(context.calibration_stacks or {})
    config = ProcessingConfig(
        calibration_stacks=cal_stacks_ref,
        development_settings=settings.development_settings,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
        decisions=settings.decision_overrides,
    )
    return fn, entity, config


def _walk_and_resolve_filenames_tagged(graph) -> list[tuple[str, bool]]:
    """Topologically walk `graph`, computing each filename_fn-bearing
    node's filename and threading `_TracedOutput` proxies through to
    downstream nodes' filename_fn calls.

    Returns `(path_str, is_final)` for every resolved filename. The
    `is_final` flag mirrors the producer's `_final` attribute (set by
    `@node(final=True)`). Returning both modes lets callers filter for
    `finals_only` without re-tracing.

    Non-filename_fn nodes (`compute_image_metadata`,
    `get_stack_subframe_image`, helpers like `calibrate`) don't write
    files but DO appear as deps of filename_fn nodes. We propagate a
    content_hash through them: a node with a single dep inherits that
    dep's content_hash (compute_image_metadata is a pass-through —
    `content_hash(ImageMetadata) == content_hash(source file)` under
    pure Merkle). A leaf without a dep tries to resolve from context.
    """
    order = _topological_order(graph)
    node_hash: dict[Any, str] = {}
    out: list[tuple[str, bool]] = []

    for node in order:
        if hasattr(node.task, "filename_fn"):
            args = _resolve_args(node, graph, node_hash)
            try:
                path = node.task.filename_fn(*args)
            except (AttributeError, KeyError, TypeError):
                # Missing upstream proxy or unhashable arg — skip the
                # subtree from here (downstream nodes that depend on us
                # will silently lack a hash for us).
                continue
            # Filename is `<output_dir>/.../<call_hash>.<ext>`. The
            # stem is exactly the call_hash.
            node_hash[node] = Path(path).stem
            is_final = bool(getattr(node.task, "_final", False))
            out.append((str(path), is_final))
        else:
            # Pass-through propagation for helpers without filename_fn.
            deps = list(graph.get(node, []) or [])
            inherited = next(
                (node_hash[d] for d in deps if d in node_hash), None,
            )
            if inherited is not None:
                node_hash[node] = inherited
            else:
                # Leaf-level resolver (best-effort). E.g.
                # `get_stack_subframe_image(stack)` returns a raw frame
                # path; its content_hash lives on the corresponding
                # ImageMetadata in `stack.all_metadata()`.
                leaf_hash = _resolve_leaf_content_hash(node)
                if leaf_hash is not None:
                    node_hash[node] = leaf_hash

    return out


def _resolve_args(
    node, graph, node_hash: dict[Any, str],
) -> list[Any]:
    """Build the argument list for `node.task.filename_fn`, mirroring
    `GraphRuntime.make_fn_args`: literal `node.args` followed by
    dep results. Dep results are replaced with `_TracedOutput` proxies
    carrying the dep's call_hash (or the propagated content_hash for
    pass-through helpers)."""
    sig = inspect.signature(node.task._function)
    args = list(node.args)
    if len(sig.parameters) > len(node.args):
        for dep in graph.get(node, []) or []:
            if dep in node_hash:
                args.append(_TracedOutput(content_hash=node_hash[dep]))
            else:
                # Unresolved dep — append a sentinel that will cause
                # filename_fn to raise (caught upstream → skip subtree).
                args.append(None)
    return args


def _resolve_leaf_content_hash(node) -> str | None:
    """Try to resolve a leaf helper's effective content_hash without
    actually running heavy code. Handles `get_stack_subframe_image`
    specifically — its arg is a `Stack`, and the picked frame's
    content_hash is already on the stack's per-frame ImageMetadata."""
    fn_name = getattr(node.task._function, "__name__", "")
    if fn_name == "get_stack_subframe_image" and node.args:
        stack = node.args[0]
        try:
            from stacks.compute_graph import get_light_frames_sorted_by_stars
            best = get_light_frames_sorted_by_stars(stack)[0]
            meta = stack.all_metadata()[best]
            return meta.content_hash
        except Exception:
            return None
    return None


def _topological_order(graph) -> list:
    """Stable topological sort of `graph` (a dict {Node: [dep, ...]}).
    Same semantics as `GraphRuntime`'s ready/done scheduling, just
    flattened for offline walking."""
    sorter: TopologicalSorter = TopologicalSorter()
    for node, deps in graph.items():
        sorter.add(node, *(deps or ()))
    return list(sorter.static_order())
