"""Decoration-time contracts on `@node`.

These fail at IMPORT of the offending module rather than hours later in a run,
which is the whole point: a mis-declared op's symptom (a missing tap, a
deliverable that deleted itself) surfaces far from its cause.
"""
def test_final_and_ephemeral_are_contradictory():
    """A deliverable that deletes itself is a contradiction, and the failure
    would be silent and remote (a missing tap, hours later). Catch it at
    decoration time."""
    import pytest
    from decorators import node

    with pytest.raises(ValueError, match="contradictory"):
        @node(display="Deliverable", final=True, ephemeral=True)
        def _bad(x) -> str:
            return "x"


def test_ephemeral_defaults_off_and_is_readable():
    """The runtime reads `_ephemeral` off the wrapper; default False keeps an
    op's output — the safe direction for anything that never opted in."""
    from decorators import node

    @node(plate_solve="none")
    def _keeps(x) -> str:
        return "x"

    @node(plate_solve="none", ephemeral=True)
    def _reclaims(x) -> str:
        return "x"

    assert _keeps._ephemeral is False
    assert _reclaims._ephemeral is True


def test_plate_solve_booleans_are_retired():
    """`plate_solve` is the enum "solve" | "none" | "inherit" (2026-09-02).
    A stale boolean call site must fail AT DECORATION with the migration
    message, on every op — not only artifact producers."""
    import pytest
    from decorators import node

    for stale in (True, False):
        with pytest.raises(ValueError, match="booleans retired"):
            @node(plate_solve=stale)
            def _stale(x) -> str:
                return "x"

    with pytest.raises(ValueError, match="'solve', 'none' or 'inherit'"):
        @node(plate_solve="never")
        def _typo(x) -> str:
            return "x"


def test_plate_solve_inherit_requires_an_image_metadata_arg():
    """"inherit" declares geometric identity with the op's first
    `ImageMetadata` argument; an op with no such argument has nothing to
    inherit from — fail at decoration, not mid-run."""
    import pytest
    from decorators import node
    from file_io.types import TiffFilePath, WithSidecar

    with pytest.raises(ValueError, match="no ImageMetadata argument"):
        @node(plate_solve="inherit")
        def _orphan(path: str) -> WithSidecar[TiffFilePath]:
            return WithSidecar(TiffFilePath(path))

    from metadata.classes import ImageMetadata

    @node(plate_solve="inherit")
    def _ok(source: ImageMetadata) -> WithSidecar[TiffFilePath]:
        return WithSidecar(TiffFilePath(str(source.filename)))
