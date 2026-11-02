"""Pure — no Ray, no Siril: the gather step between Siril's calibration and
its registration, where each calibrated frame has its site's predicted glow
subtracted (light_pollution). Synthetic calibrated frames in a scratch
`proc/`, a stack of one run, one site model."""
import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from light_pollution.model import SiteModel
from light_pollution.sites import Site
from light_pollution.test_model import CITY, MOON, THETA
from metadata.classes import ImageMetadata
from plate_solving import PlateSolution
from stacks.stacking.siril import _gather
from testing.shared_masters import tan_wcs_header
from units import ArcsecondsPerPixel, Degrees, Radians

NX, NY = 384, 256
SF, OAKLAND = (37.75, -122.42), (37.78, -122.20)
T0 = datetime.datetime(2026, 10, 1, 6, 0, 0)


def _frame(k: int, position) -> ImageMetadata:
    header = {"DATE-OBS": (T0 + datetime.timedelta(minutes=5 * k)).isoformat(), "EXPTIME": 300.0, "FILTER": "H",
              "SITELAT": position[0], "SITELONG": position[1]}
    solution = PlateSolution(header=tan_wcs_header(305.0, 40.0, 20.0, NX, NY), num_stars=300,
                             center=(Degrees(305.0), Degrees(40.0)), image_scale=ArcsecondsPerPixel(20.0), rotation_angle=Radians(0.0))
    return ImageMetadata(content_hash=f"frame{k}", stable_hash=f"frame{k}", filename=Path(f"/nowhere/sub{k}.fit"),
                         header=header, width_pixels=NX, height_pixels=NY, channels=1, dtype=np.dtype(np.float32),
                         plate_solution=solution)


def test_each_calibrated_frame_loses_its_own_glow_and_the_record_says_what_happened(tmp_path):
    frames = [_frame(k, SF) for k in range(3)] + [_frame(3, OAKLAND)]
    run = SimpleNamespace(metadata={m.content_hash: m for m in frames})
    stack = SimpleNamespace(contents={"run": run})
    proc = tmp_path / "proc"
    proc.mkdir()
    rng = np.random.default_rng(0)
    sources = {}
    for k, m in enumerate(frames, 1):
        data = (1000.0 + rng.normal(0, 2.0, (NY, NX))).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header["DATE-OBS"] = m.header["DATE-OBS"]
        hdu.writeto(proc / f"pp_r0_{k:05d}.fit")
        sources[k] = data
    model = SiteModel.of(Site("37.75,-122.42", *SF), THETA * 4, (("H", CITY, MOON),))

    all_dir, record = _gather(stack, tmp_path, [model])

    assert record["models"] == [model.content_hash] and record["frames"] == 4
    assert record["no-site"] == 1 and record["unmatched"] == 0 and record["uncalibrated"] == 0
    corrected = record["moonless"] + record["moonlit"]
    assert corrected + record["gated"] + record["no-amplitude"] == 3 and corrected >= 1
    for k in range(1, 5):
        out = np.asarray(fits.getdata(all_dir / f"{k:05d}.fit"), np.float32)
        diff = sources[k] - out
        if k == 4 or not (all_dir / f"{k:05d}.fit").is_symlink() is False:
            pass
        if (all_dir / f"{k:05d}.fit").is_symlink():
            assert np.array_equal(diff, np.zeros_like(diff))            # linked: untouched
        else:
            assert abs(float(diff.mean())) < 0.05 * float(np.abs(diff).max())   # a mean-zero surface came off
            assert float(diff.std()) > 0.5
    assert (all_dir / "00004.fit").is_symlink()                          # the Oakland frame: no model, untouched


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
