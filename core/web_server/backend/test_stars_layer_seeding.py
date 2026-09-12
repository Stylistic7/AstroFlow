"""A stars layer is seeded from the COMPOSITE, like its starless twin.

Stylistic7's 2026-09-03 rule: the stretch is a property of the IMAGE, not of the
layer you happen to be viewing. That was applied one way round — a starless
tap is sampled as `starless + stars`, because its razor-thin sigma otherwise
made the autostretch amplify ~1e-6 residuals into paint.

The reverse was missing. A stars layer's background is identically ZERO by
construction (`channel - starless`, clamped; `polish_background` skips it for
that reason), so seeding from it alone gives the autostretch a median that is
a zero floor rather than a sky level. Measured live 2026-09-11 on mosaic
75a1da64 "Mosaic Stars"/H:

    median=3.02e-06  p95=1.23e-04  p99=6.00e-04

which solves to a midtone of ~1.7e-5 and maps p95 -> 0.88, p99 -> 0.97: the
image renders white.
"""
import pytest

from web_server.backend.tiles import _STARS_SIBLING

_PAIRS = [
    ("Mosaic Starless", "Mosaic Stars"),
    ("Linear RGB Starless", "Linear RGB Stars"),
    ("Linear L Starless", "Linear L Stars"),
]


@pytest.mark.parametrize("starless,stars", _PAIRS)
def test_layer_pairs_are_symmetric(starless, stars):
    """Either layer must name the other, so the seed is composite-derived
    whichever one the viewer is pointed at."""
    assert _STARS_SIBLING[starless] == stars
    assert _STARS_SIBLING[stars] == starless


@pytest.mark.parametrize("stars", [s for _, s in _PAIRS])
def test_every_stars_layer_has_a_sibling(stars):
    assert stars in _STARS_SIBLING, (
        f"{stars} would be seeded from its own zero background and render white"
    )


def test_per_band_entries_are_untouched():
    """The 3-band narrowband taps map each S/H/O slot to that filter's mono
    stars tap; they have no single reverse partner and must stay as they were."""
    assert _STARS_SIBLING["Linear SHO"] == "per-band"
    assert _STARS_SIBLING["Linear Narrowband Starless"] == "per-band"


def test_no_entry_points_at_itself():
    """A self-pairing would double the primary instead of completing it."""
    for k, v in _STARS_SIBLING.items():
        assert v != k, f"{k} pairs with itself"
