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
    from previews.types import PIXINSIGHT, SUBFRAME, MASTER

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


# --- the core regression: identity-keyed resolution, no last-wins -----

def test_registry_no_last_wins_on_shared_level_impl():
    """Two producers sharing (level, implementation) but declaring
    different identities must BOTH be retrievable by their identity.
    Pre-refactor the second clobbered the first."""
    from products import ensure_producers_registered, registry
    from products.registry import register_producer, get_producer, MediaType
    from previews.types import PIXINSIGHT, SUBFRAME, MASTER
    from products.identity import ObjectId

    # Load the real producers FIRST so the lazy _ensure_registered()
    # import inside get_producer can't clobber our entries mid-test.
    ensure_producers_registered()

    # Throwaway identities: distinct registry keys that the real
    # producers never register, so this test mutates only its own two
    # keys and cleans them up precisely — no global snapshot/restore,
    # no production-side replay machinery.
    class _IdA(ObjectId):
        object_type = type("KindA", (), {})

    class _IdB(ObjectId):
        object_type = type("KindB", (), {})

    def prod_a():
        return "a"

    def prod_b():
        return "b"

    lvl = SUBFRAME
    impl = PIXINSIGHT
    try:
        register_producer(
            MediaType.IMAGE, lvl, prod_a, implementation=impl, identity=_IdA,
        )
        register_producer(
            MediaType.IMAGE, lvl, prod_b, implementation=impl, identity=_IdB,
        )
        # Same (level, impl), different identity -> both retrievable.
        assert get_producer(MediaType.IMAGE, _IdA, lvl, impl) is prod_a
        assert get_producer(MediaType.IMAGE, _IdB, lvl, impl) is prod_b
    finally:
        registry._registry[MediaType.IMAGE].pop((_IdA, lvl, impl), None)
        registry._registry[MediaType.IMAGE].pop((_IdB, lvl, impl), None)


def test_real_run_subframe_producers_resolve_distinctly():
    """End of the actual bug: the two real Run/SUBFRAME producers must
    resolve to different functions by identity (calibrated route used
    to wrongly get the 2-arg uncorrected producer -> TypeError -> 404)."""
    from products.registry import get_producer, MediaType
    from previews.types import PIXINSIGHT, SUBFRAME, MASTER
    from products.identity import CalibratedStackId, UncorrectedRunId
    from products import ensure_producers_registered

    ensure_producers_registered()
    lvl = SUBFRAME
    impl = PIXINSIGHT

    cal = get_producer(MediaType.IMAGE, CalibratedStackId, lvl, impl)
    unc = get_producer(MediaType.IMAGE, UncorrectedRunId, lvl, impl)

    assert cal is not None and unc is not None
    assert cal is not unc
    assert cal.__name__ == "flat_corrected_stack"
    assert unc.__name__ == "uncorrected_run_thumbnail"


def test_get_producer_signature_is_identity_keyed():
    """get_producer must take an identity class, not object_type."""
    import inspect
    from products.registry import get_producer

    params = list(inspect.signature(get_producer).parameters)
    assert "object_type" not in params, (
        f"get_producer must be identity-keyed; got params {params}"
    )
    assert "identity" in params, (
        f"get_producer must accept identity; got params {params}"
    )
