"""Contract tests for the type-keyed producer registry.

Producers are keyed by the runtime type of their first parameter. The
URL-side resolution uses a small string -> class map at the route layer.

Pure + fast (no Ray, no graph, no exhaustive fixture).
"""
import pytest


# --- per-type producer enumeration via MRO walk -----------------------

def test_producers_for_returns_decoration_order():
    """`producers_for(type)` returns each op whose first-parameter
    type accepts an instance of `type`. Walks MRO so an op annotated
    with a base class (`BaseStack`) surfaces for both subclasses
    (Stack, CalibrationStack)."""
    from products import ensure_producers_registered
    from products.registry import producers_for
    from stacks.composite_stack import Stack, CalibrationStack

    ensure_producers_registered()

    stack_ops = {p.__name__ for p in producers_for(Stack)}
    assert "stack_files" in stack_ops, (
        f"stack_files (annotated stack: BaseStack) must surface for "
        f"producers_for(Stack); got {stack_ops}"
    )
    assert "post_process_stack" in stack_ops

    cal_ops = {p.__name__ for p in producers_for(CalibrationStack)}
    # Same op surfaces for both subclasses via MRO walk.
    assert "stack_files" in cal_ops


# --- final=True on @node marks the user's deliverable --------------

def test_final_kwarg_propagates_to_wrapped_op():
    """`@node(final=True)` sets `_final` on the wrapped op so
    `dependency_graph.trace_paths.paths_for_cart_item(...,
    finals_only=True)` can filter to the user's preserve-on-archive
    set. Unmarked ops should lack `_final` (or have it False)."""
    from mosaics.final_storage import (
        store_mosaic_color_channel_xisf,
        store_mosaic_color_channel_subframe,
        store_mosaic_color_channel_oss,
    )
    from stacks.post_processing.pixinsight import post_process_stack

    for op in (
        store_mosaic_color_channel_xisf,
        store_mosaic_color_channel_subframe,
        store_mosaic_color_channel_oss,
    ):
        assert getattr(op, "_final", False) is True, (
            f"{op.__name__} should be marked final=True"
        )

    # post_process_stack is a per-Stack intermediate from the mosaic
    # user's perspective — should NOT be marked final.
    assert not getattr(post_process_stack, "_final", False), (
        "post_process_stack must not be final (intermediate from "
        "the mosaic user's perspective)"
    )
