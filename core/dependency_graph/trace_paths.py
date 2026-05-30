"""Trace a cart item's pipeline and return every expected output path.

Under the pure-Merkle `content_hash := producer.call_hash` convention
(stamped into file headers by `@cached.remote`), filenames are deterministic
and statically computable from the trace alone — no file I/O at trace time
beyond ImageMetadata already in context.

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
    `@makes_image(... final=True)` — used by archive transition with
    `preserve="finals"`.
    """
    from dependency_graph.tracing import trace

    fn, entity, config = _resolve_pipeline_call(item)
    if fn is None:
        return []

    try:
        graph = trace(fn, entity, config)
    except Exception:
        # Tracing should never fail under healthy code, but if it does
        # (decorator bug, half-built context, etc.) the sweep degrades
        # to "this item contributes no live paths" rather than crashing.
        return []
    if graph is None:
        return []

    return _walk_and_resolve_filenames(graph, finals_only=finals_only)


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


def _walk_and_resolve_filenames(
    graph, *, finals_only: bool,
) -> list[Path]:
    """Topologically walk `graph`, computing each `@makes_image` node's
    filename and threading `_TracedOutput` proxies through to downstream
    nodes' filename_fn calls.

    Non-`@makes_image` nodes (`compute_image_metadata`,
    `get_stack_subframe_image`, helpers like `calibrate`) don't write
    files but DO appear as deps of `@makes_image` nodes. We propagate a
    content_hash through them: a node with a single dep inherits that
    dep's content_hash (compute_image_metadata is a pass-through —
    `content_hash(ImageMetadata) == content_hash(source file)` under
    pure Merkle). A leaf without a dep tries to resolve from context.
    """
    order = _topological_order(graph)
    node_hash: dict[Any, str] = {}
    paths: list[Path] = []

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
            if finals_only and not getattr(node.task, "_final", False):
                continue
            paths.append(Path(path))
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

    return paths


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
