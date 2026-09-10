"""calibrate_image — preview-grade partial calibration.

Regression for 2026-09-04: the unified CALIBRATION_TOPOLOGY gives lights
`("dark", "flat")` and never a bias, but calibrate_image required bias AND
dark AND flat, silently returning the raw light otherwise. Every
flat-corrected Run preview therefore served the raw frame — byte-identical
carousel entries for every candidate flat, a flat-picker that picked
nothing. These tests pin the partial-calibration semantics.
"""
import numpy as np
from types import SimpleNamespace

from previews.calibration import calibrate_image


def _md(exposure):
    return SimpleNamespace(header={"EXPOSURE": exposure}, filename="test")


def _fixtures():
    rng = np.random.default_rng(3)
    H, W = 120, 160
    light = (0.2 + rng.random((H, W)) * 0.1).astype(np.float32)
    dark = (0.01 + rng.random((H, W)) * 0.002).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1.0 - 0.4 * (((yy - H / 2) / H) ** 2 + ((xx - W / 2) / W) ** 2)
    flat_a = (vig * 0.5).astype(np.float32)
    flat_b = flat_a.copy()
    flat_b[40:60, 60:80] *= 0.7          # dust patch — the thing the picker exists to show
    return light, dark, flat_a, flat_b


def test_no_bias_still_applies_dark_and_flat():
    """The production topology: lights carry dark+flat, no bias. The flat
    MUST be applied — this exact case used to return the raw light."""
    light, dark, flat_a, _ = _fixtures()
    out = calibrate_image(_md(300), light, None, None,
                          _md(300), dark, _md(3), flat_a, False)
    assert np.abs(out - light).max() > 0.01, "calibration was a no-op"


def test_different_flats_produce_different_previews():
    """The carousel's whole purpose: candidate flats must render
    distinguishably (byte-identical entries = the 2026-09-04 bug)."""
    light, dark, flat_a, flat_b = _fixtures()
    args = (_md(300), light, None, None, _md(300), dark)
    out_a = calibrate_image(*args, _md(3), flat_a, False)
    out_b = calibrate_image(*args, _md(3), flat_b, False)
    assert np.abs(out_a - out_b).max() > 0.01
    # and the dust patch is where they differ
    diff = np.abs(out_a - out_b)
    assert diff[40:60, 60:80].mean() > 10 * diff[:20, :20].mean()


def test_no_flat_returns_uncorrected():
    """The 2-arg "uncorrected" preview path: no flat → the light as-is."""
    light, dark, *_ = _fixtures()
    out = calibrate_image(_md(300), light, None, None,
                          None, None, None, None, False)
    assert np.allclose(out, light)


def test_flat_only_no_dark():
    """A run with a flat but no dark still gets the flat division."""
    light, _, flat_a, _ = _fixtures()
    out = calibrate_image(_md(300), light, None, None,
                          None, None, _md(3), flat_a, False)
    assert np.abs(out - light).max() > 0.01


def test_rotation_still_applies():
    light, *_ = _fixtures()
    out = calibrate_image(_md(300), light, None, None,
                          None, None, None, None, True)
    assert np.allclose(out, np.rot90(light, 2))


def _star_flat_fixtures():
    """Real scales: a light and dark in ADU, a star flat in RESPONSE units.

    A star flat is not an exposed frame — `response_image` writes a
    multiplicative response around 0.5 — so it is three orders of magnitude
    below the dark it would otherwise be corrected against.
    """
    rng = np.random.default_rng(7)
    H, W = 120, 160
    light = (1200.0 + rng.random((H, W)) * 60.0).astype(np.float32)
    dark = (500.0 + rng.random((H, W)) * 5.0).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1.0 - 0.4 * (((yy - H / 2) / H) ** 2 + ((xx - W / 2) / W) ** 2)
    return light, dark, (vig * 0.5).astype(np.float32)


def _star_md():
    # SFLATCEL is written only by the star-flat writer, so it is what marks
    # a flat as synthetic. EXPOSURE=1.0 is nominal and meaningless here.
    return SimpleNamespace(
        header={"EXPOSURE": 1.0, "SFLATCEL": 256}, filename="star_flat.xisf",
    )


def test_a_star_flat_is_not_bias_or_dark_corrected():
    """A star flat carries no bias and no dark current, so neither may be
    subtracted from it.

    Doing so was catastrophic rather than merely inaccurate: subtracting a
    ~500 ADU dark term from a ~0.5 response drives the whole flat NEGATIVE,
    the positive-pixel median falls back to 1.0, and the light is divided by
    a near-constant negative — an inverted, contrast-crushed preview. That
    is the washed-out star-flat preview (2026-09-08).
    """
    light, dark, star = _star_flat_fixtures()
    out = calibrate_image(_md(300), light, None, None,
                          _md(300), dark, _star_md(), star, False)

    expected = (light - dark) / (star / np.median(star))
    assert np.allclose(out, expected, rtol=1e-5)
    assert out.min() > 0.0, "divided by a negative flat"
    # Contrast survives: vignetting correction lifts the corners, and the
    # frame keeps the light's own signal range rather than collapsing.
    assert out.max() / out.min() > 1.2


def test_a_real_flat_still_gets_its_bias_and_dark_term():
    """The synthetic exemption must not leak to panel/sky flats, which ARE
    exposed frames and do carry bias + dark current."""
    light, dark, flat = _star_flat_fixtures()
    flat = (flat * 40000.0).astype(np.float32)          # a real flat's ADU
    real_md = SimpleNamespace(header={"EXPOSURE": 3.0}, filename="flat.fit")

    out = calibrate_image(_md(300), light, None, None,
                          _md(300), dark, real_md, flat, False)

    signal = flat - (3.0 / 300.0) * dark
    expected = (light - dark) / (signal / np.median(signal))
    assert np.allclose(out, expected, rtol=1e-5)


def test_star_flat_thumbnail_resolves_its_built_master(tmp_path, monkeypatch):
    """A star flat's picture is its built master, not a member frame.

    Regression for 2026-09-09: `get_first_frame` did `stack.all_frames()[0]`,
    but a star flat's members are light subs read for photometry, so `contents`
    is empty and that raised IndexError. `get_thumbnail` swallowed it, no other
    registered producer matched a 1-arg call (`build_star_flat` needs three, to
    carry the solve settings in its cache key), and every star flat detail page
    404'd.
    """
    from types import SimpleNamespace
    import previews.stack_thumbnails as st
    import stacks.star_flat as sfm

    master = tmp_path / "flat.xisf"
    master.write_bytes(b"not really xisf")
    stack = SimpleNamespace(kind="star", stable_hash="sf1",
                            all_frames=lambda: [], all_metadata=lambda: {})

    monkeypatch.setattr(sfm, "star_flat_master_path", lambda s: master)
    assert str(st.get_first_frame(stack).path) == str(master)

    # An unbuilt flat has no thumbnail yet -- and must never trigger a build.
    monkeypatch.setattr(sfm, "star_flat_master_path", lambda s: None)
    try:
        st.get_first_frame(stack)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for an unbuilt star flat")


def test_ordinary_calibration_stack_still_uses_its_first_frame(monkeypatch):
    """The star-flat branch must not change panel/sky flats, which do have
    frames to integrate."""
    from types import SimpleNamespace
    import previews.stack_thumbnails as st

    md = {"h1": SimpleNamespace(filename="/raw/flat_0001.fit")}
    stack = SimpleNamespace(kind=None, stable_hash="cf1",
                            all_frames=lambda: ["h1"], all_metadata=lambda: md)
    assert str(st.get_first_frame(stack).path) == "/raw/flat_0001.fit"
