"""The thumbnail dispatcher must only invoke a producer whose arity matches the
URL's positional-arg count. A Run registers both a 2-arg uncorrected producer
and a 3-arg flat-corrected producer; invoking the mismatched one raised a noisy
(caught) `TypeError: flat_corrected_stack() missing 1 required positional
argument: 'flat_stack'` for every plain Run-thumbnail request.
"""
from web_server.backend.thumbnails import _producer_accepts
from previews.flat_correction import flat_corrected_stack, uncorrected_run_thumbnail


def test_matches_real_producer_arities():
    # uncorrected_run_thumbnail(stack, substack) → 2
    assert _producer_accepts(uncorrected_run_thumbnail, 2)
    assert not _producer_accepts(uncorrected_run_thumbnail, 3)
    # flat_corrected_stack(stack, substack, flat_stack) → 3
    assert _producer_accepts(flat_corrected_stack, 3)
    assert not _producer_accepts(flat_corrected_stack, 2)


def test_variadic_and_uninspectable_accept_any_count():
    def variadic(*args): ...
    assert _producer_accepts(variadic, 0)
    assert _producer_accepts(variadic, 7)
    # builtins / C-callables whose signature can't be introspected: don't exclude.
    assert _producer_accepts(len, 1)


def test_optional_args_allow_a_range():
    def with_default(a, b, c=None): ...
    assert not _producer_accepts(with_default, 1)
    assert _producer_accepts(with_default, 2)   # c defaulted
    assert _producer_accepts(with_default, 3)   # c supplied
    assert not _producer_accepts(with_default, 4)
