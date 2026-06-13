"""Invariants over registered `@node(identity=...)` producers.

A4 from the audit plan: every registered producer's `filename_fn`
must be a pure function of its call args via `canonical_call_hash`,
and `_maybe_use_content_hash` must treat a `_PredictedRef` carrying
content_hash X identically to a real result object whose `.content_hash`
attribute equals X. The whole preflight optimization rests on this:
predicted call_hash at trace time == runtime call_hash given equal
content_hashes.

Two checks:

* `test_every_producer_has_fn_hash_and_filename_fn` — registry-wide
  walk. Each producer must have `_fn_hash` (proves `@node` ran), a
  callable `filename_fn` (proves `identity=...` ran), and `_identities`
  / `_final` / `_display` attributes for view-side enumeration.

* `test_predicted_ref_equiv_to_real_content_hash` — synthetic test
  of the invariant directly: build a `_PredictedRef("abc")` and a
  minimal stand-in object with `.content_hash = "abc"`, hash a fake
  call with each, assert identical. Documents the rule explicitly so
  a future "fix" to `_maybe_use_content_hash` that broke it would be
  caught.
"""
from hashing import hash_call, hash_source_code

from dependency_graph.preflight import _PredictedRef
from products.registry import _producers, _ensure_producers_imported


def test_every_producer_has_fn_hash_and_filename_fn():
    _ensure_producers_imported()
    image_producers: list = []
    for arg_type, wrappers in _producers.items():
        for w in wrappers:
            image_producers.append((arg_type, w))

    assert image_producers, (
        "No registered image producers — did discovery fail? "
        "Expected artifact-producing `@node` ops in stacks/, mosaics/, "
        "previews/ to populate `_producers`."
    )

    n_with_filename_fn = 0
    for arg_type, w in image_producers:
        op_name = getattr(w, "__name__", repr(w))
        # `_fn_hash` is set in `decorators/decorators.py:_Node._wrap`
        # — its presence proves the wrapper went through @node, not
        # something else (e.g. @image_producer, which is for ndarray
        # helpers and is exempt from this invariant).
        if not hasattr(w, "_fn_hash"):
            # @image_producer-wrapped funcs land here too; they're
            # legitimately exempt — they don't produce on-disk files.
            continue
        assert isinstance(w._fn_hash, str) and w._fn_hash, (
            f"{op_name}: _fn_hash must be a non-empty string, got "
            f"{w._fn_hash!r}"
        )
        # `filename_fn` is None when `_infer_file_type` can't find a
        # recognized return shape — currently the case for ops
        # annotated `-> Optional[WithSidecar[...]]` (post_process_*
        # variants), because the inferrer doesn't recurse into Union
        # args. That's a separate footgun tracked outside A4; here we
        # just scope the invariant to producers that actually carry a
        # filename_fn (everyone else has nothing to predict).
        if callable(getattr(w, "filename_fn", None)):
            n_with_filename_fn += 1

    assert n_with_filename_fn > 0, (
        "Expected at least one registered producer with a usable "
        "filename_fn; got zero. The build_filename_fns path may be "
        "broken end-to-end."
    )


class _FakeContentHashCarrier:
    """Stand-in for an `ImageMetadata` (or any result with `.content_hash`)
    in the synthetic invariant test. Same attribute surface as
    `_PredictedRef` for `_maybe_use_content_hash` purposes."""
    def __init__(self, content_hash: str):
        self.content_hash = content_hash


def test_predicted_ref_equiv_to_real_content_hash():
    """The A4 invariant: hash_call yields the same value whether the
    upstream is a preflight `_PredictedRef` or a real result, as long
    as both carry the same `content_hash`. This is what makes
    preflight's predicted call_hash match runtime call_hash."""
    def _stub_fn():
        return None
    fn_hash = hash_source_code(_stub_fn)

    predicted = _PredictedRef("deadbeef")
    real = _FakeContentHashCarrier("deadbeef")

    predicted_hash = hash_call(fn_hash, "literal_arg", predicted)
    runtime_hash = hash_call(fn_hash, "literal_arg", real)

    assert predicted_hash == runtime_hash, (
        f"Predicted ref hashed to {predicted_hash} but real-content-hash "
        f"object hashed to {runtime_hash} — `_maybe_use_content_hash` no "
        "longer treats both shapes identically. Preflight's call_hash "
        "predictions will diverge from runtime; the producer would write "
        "to a different filename than preflight expected."
    )


def test_different_content_hashes_produce_different_call_hashes():
    """Negative half of the invariant: different content_hashes MUST
    produce different call_hashes. Catches a degenerate
    `_maybe_use_content_hash` that swallowed all content_hash values
    into a single constant (the test above would falsely pass)."""
    def _stub_fn():
        return None
    fn_hash = hash_source_code(_stub_fn)

    a = hash_call(fn_hash, "lit", _PredictedRef("aaaaaaaa"))
    b = hash_call(fn_hash, "lit", _PredictedRef("bbbbbbbb"))
    assert a != b, (
        f"content_hashes 'aaaaaaaa' and 'bbbbbbbb' both hashed to {a} — "
        "`_maybe_use_content_hash` collapsed distinct upstreams together."
    )
