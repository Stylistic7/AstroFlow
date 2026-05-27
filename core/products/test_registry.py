"""Contract tests for the identity-keyed producer registry.

These pin the TARGET behavior of the object_type->identity refactor:
the producer registry is keyed by the ObjectId identity class (not the
lossy object_type), each ObjectId carries its own object_type, and the
Run/SUBFRAME collision that 404'd every calibrated /thumbnail/run/.../
.../... is gone.

Written test-first: several are RED on pre-refactor code and turn green
as Phases 1-3 land. Imports are deferred into each test so a missing
symbol fails that test precisely instead of erroring collection.

Pure + fast (no Ray, no graph, no exhaustive fixture).
"""
import pytest


# --- ObjectId.object_type: every identity is self-describing ----------

def test_objectid_subclasses_declare_object_type():
    from stacks.composite_stack import Stack, CalibrationStack
    from stacks.run import Run
    from mosaics.types import Mosaic
    from targets.types import Target
    from products import identity as I

    expected = {
        "StackId": Stack,
        "CalibrationStackId": CalibrationStack,
        "CalibratedStackId": Run,
        "UncorrectedRunId": Run,
        "MosaicId": Mosaic,
        "MosaicChannelId": Mosaic,
        "TargetId": Target,
    }
    for name, obj_type in expected.items():
        cls = getattr(I, name)
        assert cls.object_type is obj_type, (
            f"{name}.object_type should be {obj_type.__name__}, "
            f"got {getattr(cls, 'object_type', None)!r}"
        )


def test_stackid_split_is_total_and_distinct():
    """StackId previously mapped to BOTH Stack and CalibrationStack
    (the polymorphism that made a single object_type impossible).
    After the split each names exactly one type."""
    from stacks.composite_stack import Stack, CalibrationStack
    from products.identity import StackId, CalibrationStackId

    assert StackId is not CalibrationStackId
    assert StackId.object_type is Stack
    assert CalibrationStackId.object_type is CalibrationStack


def test_new_identities_exist_for_formerly_identityless_producers():
    from mosaics.merging import MergedImage
    from products import identity as I

    assert hasattr(I, "MergedImageId"), (
        "merge_images had no identity; object_type cannot be removed "
        "while an identity-less producer exists"
    )
    assert I.MergedImageId.object_type is MergedImage


# --- presentation_cache_key no longer takes object_type ---------------

def test_presentation_cache_key_drops_object_type_arg():
    import inspect
    from products.identity import ObjectId

    params = list(
        inspect.signature(ObjectId.presentation_cache_key).parameters
    )
    assert "object_type" not in params, (
        "presentation_cache_key must derive type from self.object_type, "
        f"not take it as a param; got {params}"
    )


def test_presentation_cache_key_distinguishes_object_kind():
    """Two identities with identical field values but different
    object_type must produce different presentation keys (the key must
    still embed the object_type name — validity and on-disk path layout
    depend on it — just sourced from self now)."""
    from dataclasses import dataclass
    from products.identity import ObjectId, PresentationType

    @dataclass(frozen=True)
    class _A(ObjectId):
        object_type = type("KindA", (), {})
        ref: str

    @dataclass(frozen=True)
    class _B(ObjectId):
        object_type = type("KindB", (), {})
        ref: str

    a = _A(ref="x").presentation_cache_key(
        PresentationType.THUMBNAIL, "fake_op",
    )
    b = _B(ref="x").presentation_cache_key(
        PresentationType.THUMBNAIL, "fake_op",
    )
    assert a != b, "distinct object kinds must not alias in presentation cache"
    assert a == _A(ref="x").presentation_cache_key(
        PresentationType.THUMBNAIL, "fake_op",
    ), "presentation key must be stable for the same identity"


# --- per-identity producer enumeration ---

def test_producers_for_returns_decoration_order():
    """`producers_for(identity)` returns each op registered for that
    identity, in decoration order. Polymorphic ops (one body
    registered for multiple identities) appear in each identity's
    list."""
    from products import ensure_producers_registered
    from products.registry import producers_for
    from products.identity import StackId, CalibrationStackId

    ensure_producers_registered()

    stack_ops = {p.__name__ for p in producers_for(StackId)}
    assert "stack_files" in stack_ops, (
        f"stack_files must register for StackId; got {stack_ops}"
    )
    assert "post_process_stack" in stack_ops

    cal_ops = {p.__name__ for p in producers_for(CalibrationStackId)}
    # stack_files registers for BOTH StackId and CalibrationStackId; the
    # polymorphic decoration appends to each identity's producer list.
    assert "stack_files" in cal_ops


def test_real_run_subframe_producers_resolve_distinctly():
    """The two Run/SUBFRAME producers — `flat_corrected_stack` and
    `uncorrected_run_thumbnail` — register under distinct identities
    (CalibratedStackId vs UncorrectedRunId). Neither's
    `filename_fn` can resolve the other's identity args, so they
    can't be confused at view-side dispatch time."""
    from products.registry import producers_for
    from products.identity import CalibratedStackId, UncorrectedRunId
    from products import ensure_producers_registered

    ensure_producers_registered()

    cal = {p.__name__ for p in producers_for(CalibratedStackId)}
    unc = {p.__name__ for p in producers_for(UncorrectedRunId)}

    assert "flat_corrected_stack" in cal
    assert "uncorrected_run_thumbnail" in unc
    assert cal != unc, (
        f"CalibratedStackId producers ({cal}) and UncorrectedRunId "
        f"producers ({unc}) must be distinct sets"
    )


# --- final=True on @makes_image marks the user's deliverable --------

def test_final_kwarg_propagates_to_wrapped_op():
    """`@makes_image(final=True)` sets `_final` on the wrapped op so
    the sweep CLI's `paths_for_item(finals_only=True)` filter can
    identify keepers. Unmarked ops should lack `_final` (or have it
    False)."""
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
