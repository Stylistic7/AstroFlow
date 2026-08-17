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

    @node(plate_solve=False)
    def _keeps(x) -> str:
        return "x"

    @node(plate_solve=False, ephemeral=True)
    def _reclaims(x) -> str:
        return "x"

    assert _keeps._ephemeral is False
    assert _reclaims._ephemeral is True
