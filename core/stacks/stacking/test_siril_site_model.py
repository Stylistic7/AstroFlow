"""Pure — no Ray, no Siril: the gather step between Siril's calibration and
its registration, where each calibrated frame has its site's predicted glow
subtracted (light_pollution). Synthetic calibrated frames in a scratch
`proc/`, a stack of one run, one site model."""
import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from light_pollution.model import NightAmplitudes, SiteModel
from light_pollution.sites import Site
from light_pollution.test_model import CITY, MOON, THETA
from light_pollution.time import night_of
from metadata.classes import ImageMetadata
from plate_solving import PlateSolution
from stacks.stacking.siril import SIRIL_ADU_PER_UNIT, _gather
from testing.shared_masters import tan_wcs_header
from units import ArcsecondsPerPixel, Degrees, Radians

NX, NY = 384, 256
SF, OAKLAND = (37.75, -122.42), (37.78, -122.20)
T0 = datetime.datetime(2026, 10, 1, 6, 0, 0)
FULL_MOON = datetime.datetime(2026, 9, 27, 7, 0, 0)       # local midnight after the full Moon of 2026-09-26


def _frame(k: int, position, t0=T0) -> ImageMetadata:
    header = {"DATE-OBS": (t0 + datetime.timedelta(minutes=5 * k)).isoformat(), "EXPTIME": 300.0, "FILTER": "H",
              "SITELAT": position[0], "SITELONG": position[1]}
    solution = PlateSolution(header=tan_wcs_header(305.0, 40.0, 20.0, NX, NY), num_stars=300,
                             center=(Degrees(305.0), Degrees(40.0)), image_scale=ArcsecondsPerPixel(20.0), rotation_angle=Radians(0.0))
    return ImageMetadata(content_hash=f"frame{k}", stable_hash=f"frame{k}", filename=Path(f"/nowhere/sub{k}.fit"),
                         header=header, width_pixels=NX, height_pixels=NY, channels=1, dtype=np.dtype(np.uint16),
                         plate_solution=solution)


def _calibrated(tmp_path, frames, sky_adu=1000.0):
    """Siril's calibrated frames in `proc/`: a 16-bit light's ADU over 65535, as Siril stores them."""
    proc = tmp_path / "proc"
    proc.mkdir()
    rng = np.random.default_rng(0)
    sources = {}
    for k, m in enumerate(frames, 1):
        data = ((sky_adu + rng.normal(0, 2.0, (NY, NX))) / SIRIL_ADU_PER_UNIT).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header["DATE-OBS"] = m.header["DATE-OBS"]
        hdu.writeto(proc / f"pp_r0_{k:05d}.fit")
        sources[k] = data
    return sources


def test_each_calibrated_frame_loses_its_own_glow_and_the_record_says_what_happened(tmp_path):
    frames = [_frame(k, SF) for k in range(3)] + [_frame(3, OAKLAND)]
    stack = SimpleNamespace(contents={"run": SimpleNamespace(metadata={m.content_hash: m for m in frames})})
    sources = _calibrated(tmp_path, frames)
    model = SiteModel.of(Site("37.75,-122.42", *SF), THETA * 4, (("H", CITY, MOON),))

    all_dir, record = _gather(stack, tmp_path, [model])

    assert record["models"] == [model.content_hash] and record["nights"] == [] and record["frames"] == 4
    assert record["no-site"] == 1 and record["unmatched"] == 0 and record["uncalibrated"] == 0 and record["overshoot"] == 0
    corrected = record["moonless"] + record["moonlit-pooled"]
    assert corrected + record["gated"] + record["no-amplitude"] == 3 and corrected >= 1 and record["moonlit"] == 0
    for k in range(1, 5):
        out = np.asarray(fits.getdata(all_dir / f"{k:05d}.fit"), np.float32)
        diff = sources[k] - out
        if (all_dir / f"{k:05d}.fit").is_symlink():
            assert np.array_equal(diff, np.zeros_like(diff))            # linked: untouched
        else:
            assert abs(float(diff.mean())) < 0.05 * float(np.abs(diff).max())   # a mean-zero surface came off
            assert 0 < float(np.abs(diff).max()) < float(np.median(sources[k]))  # in the frame's units: less than its sky
    assert (all_dir / "00004.fit").is_symlink()                          # the Oakland frame: no model, untouched


def test_a_moonlit_frame_takes_its_night_s_own_amplitudes_when_the_integration_carries_them(tmp_path):
    frames = [_frame(k, SF, t0=FULL_MOON) for k in range(3)]
    stack = SimpleNamespace(contents={"run": SimpleNamespace(metadata={m.content_hash: m for m in frames})})
    sources = _calibrated(tmp_path, frames)
    model = SiteModel.of(Site("37.75,-122.42", *SF), THETA * 4, (("H", CITY, MOON),))
    night = NightAmplitudes.of(model.site.key, night_of(frames[0].header["DATE-OBS"], SF[1]), "H", CITY, 3.0 * MOON)
    other = NightAmplitudes.of(model.site.key, "2026-01-01", "H", CITY, 3.0 * MOON)
    all_dir, record = _gather(stack, tmp_path, [model], [other, night])
    assert record["nights"] == sorted([other.content_hash, night.content_hash])
    assert record["moonlit"] == 3 and record["moonlit-pooled"] == 0
    own = sources[1] - np.asarray(fits.getdata(all_dir / "00001.fit"), np.float32)
    (tmp_path / "all").rename(tmp_path / "all-own")
    _, record = _gather(stack, tmp_path, [model], [other])
    assert record["moonlit-pooled"] == 3 and record["moonlit"] == 0
    pooled = sources[1] - np.asarray(fits.getdata(all_dir / "00001.fit"), np.float32)
    assert not np.allclose(own, pooled)


def test_a_glow_that_would_swing_by_more_than_the_sky_is_not_subtracted(tmp_path):
    frames = [_frame(k, SF) for k in range(3)]
    stack = SimpleNamespace(contents={"run": SimpleNamespace(metadata={m.content_hash: m for m in frames})})
    _calibrated(tmp_path, frames)
    absurd = SiteModel.of(Site("37.75,-122.42", *SF), THETA * 400, (("H", 1e6, 1e6),))
    all_dir, record = _gather(stack, tmp_path, [absurd])
    assert record["overshoot"] == 3 and record["moonless"] + record["moonlit"] + record["moonlit-pooled"] == 0
    assert all((all_dir / f"{k:05d}.fit").is_symlink() for k in (1, 2, 3))


def test_without_models_every_frame_is_linked(tmp_path):
    frames = [_frame(k, SF) for k in range(2)]
    stack = SimpleNamespace(contents={"run": SimpleNamespace(metadata={m.content_hash: m for m in frames})})
    proc = tmp_path / "proc"
    proc.mkdir()
    for k, m in enumerate(frames, 1):
        hdu = fits.PrimaryHDU(np.zeros((NY, NX), np.float32))
        hdu.header["DATE-OBS"] = m.header["DATE-OBS"]
        hdu.writeto(proc / f"pp_r0_{k:05d}.fit")
    all_dir, record = _gather(stack, tmp_path)
    assert record["frames"] == 2 and record["models"] == [] and all((all_dir / f"{k:05d}.fit").is_symlink() for k in (1, 2))
