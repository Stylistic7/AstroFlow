"""The thumbnail dispatcher must only invoke a producer whose arity matches the
URL's positional-arg count. Producers registered under `Stack` have several
different arities, and invoking a mismatched one raises a noisy (caught)
`TypeError` on every request.
"""
from web_server.backend.thumbnails import _producer_accepts
from previews.flat_correction import calibrated_run_preview


def test_matches_real_producer_arities():
    # calibrated_run_preview(stack, run, flat_id, dark_id, bias_id) → 5.
    # Every calibration input is REQUIRED (pass None to skip one), so the
    # arity is exact and no shorter Stack producer can be invoked by a
    # Run-preview URL, nor this one by a bare /thumbnail/stack/<id>.
    assert _producer_accepts(calibrated_run_preview, 5)
    assert not _producer_accepts(calibrated_run_preview, 4)
    assert not _producer_accepts(calibrated_run_preview, 2)


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
