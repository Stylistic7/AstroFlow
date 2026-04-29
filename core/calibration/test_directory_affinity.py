"""Unit tests for directory-affinity calibration matching.

These tests don't need the full Ray/context machinery — they construct a
temporary FITS tree on disk and exercise `find_external_master` directly.
"""
from __future__ import annotations

from pathlib import Path

import astropy.io.fits as pyfits
import numpy as np
import pytest

from calibration.directory_affinity import (
    _dir_master_cache,
    find_external_master,
    masters_in_dir,
)


@pytest.fixture(autouse=True)
def _clear_master_cache():
    """The directory cache survives across tests by design; clear between tests."""
    _dir_master_cache.clear()
    yield
    _dir_master_cache.clear()


def _write_fits(path: Path, header: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = pyfits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32))
    for k, v in header.items():
        hdu.header[k] = v
    hdu.writeto(path, overwrite=True)


def _master_header(image_type: str, *, exposure: float | None = None,
                   filter: str | None = None, temperature: float | None = -20.0,
                   camera: str = "ZWO ASI6200MM Pro") -> dict:
    h = {"IMAGETYP": f"Master {image_type.capitalize()}", "INSTRUME": camera}
    if exposure is not None:
        h["EXPOSURE"] = exposure
    if filter is not None:
        h["FILTER"] = filter
    if temperature is not None:
        h["CCD-TEMP"] = temperature
    return h


def test_finds_master_in_sibling_directory(tmp_path):
    """Light frames in `Lights/` find a master in a sibling `Calibration Masters/`."""
    target_dir = tmp_path / "Cygnus 4"
    light_dir = target_dir / "Lights"
    cal_dir = target_dir / "Calibration Masters"
    _write_fits(light_dir / "light_001.fit", {"IMAGETYP": "Light"})
    _write_fits(light_dir / "light_002.fit", {"IMAGETYP": "Light"})
    master_path = cal_dir / "MasterDark_-20c_120s.fit"
    _write_fits(master_path, _master_header(
        "dark", exposure=120.0, temperature=-20.0,
    ))

    found = find_external_master(
        [str(light_dir / "light_001.fit"), str(light_dir / "light_002.fit")],
        kind="dark",
        camera="ZWO ASI6200MM Pro",
        temperature=-20.0,
        filter=None,
        exposure=120.0,
    )
    assert found == str(master_path.resolve())


def test_walks_up_to_grandparent(tmp_path):
    """Lights in `<target>/Lights/2025-10-21/` find a master in `<target>/Calibration Masters/`."""
    target_dir = tmp_path / "M42"
    light_dir = target_dir / "Lights" / "2025-10-21"
    cal_dir = target_dir / "Calibration Masters"
    _write_fits(light_dir / "light_001.fit", {"IMAGETYP": "Light"})
    master_path = cal_dir / "MasterBias_-20c.fit"
    _write_fits(master_path, _master_header("bias", temperature=-20.0))

    found = find_external_master(
        [str(light_dir / "light_001.fit")],
        kind="bias", camera="ZWO ASI6200MM Pro", temperature=-20.0,
        filter=None, exposure=None,
    )
    assert found == str(master_path.resolve())


def test_skips_master_with_wrong_temperature(tmp_path):
    """A 0°C master in a sibling dir is too far from a -20°C light → no match."""
    target_dir = tmp_path / "ngc7000"
    light_dir = target_dir / "Lights"
    cal_dir = target_dir / "Calibration Masters"
    _write_fits(light_dir / "light_001.fit", {"IMAGETYP": "Light"})
    _write_fits(cal_dir / "MasterBias_0c.fit",
                _master_header("bias", temperature=0.0))

    found = find_external_master(
        [str(light_dir / "light_001.fit")],
        kind="bias", camera="ZWO ASI6200MM Pro", temperature=-20.0,
        filter=None, exposure=None,
    )
    assert found is None


def test_flat_requires_filter_match(tmp_path):
    """A G-filter flat doesn't match a Ha-filter light."""
    target_dir = tmp_path / "veil"
    light_dir = target_dir / "Lights"
    cal_dir = target_dir / "Calibration Masters"
    _write_fits(light_dir / "light_001.fit", {"IMAGETYP": "Light", "FILTER": "Ha"})
    _write_fits(cal_dir / "MasterFlat_G.fit",
                _master_header("flat", filter="G", temperature=None))

    found = find_external_master(
        [str(light_dir / "light_001.fit")],
        kind="flat", camera="ZWO ASI6200MM Pro", temperature=None,
        filter="Ha", exposure=None,
    )
    assert found is None


def test_only_master_files_picked_up(tmp_path):
    """A FITS file that isn't a master (raw bias frame) is ignored even if its
    IMAGETYP matches the kind."""
    cal_dir = tmp_path / "obs" / "Calibration Masters"
    light_dir = tmp_path / "obs" / "Lights"
    _write_fits(light_dir / "light_001.fit", {"IMAGETYP": "Light"})
    # A regular bias frame, NOT a master — basename doesn't start with "master"
    # AND IMAGETYP doesn't say "master".
    _write_fits(cal_dir / "bias_001.fit", {"IMAGETYP": "Bias", "INSTRUME": "ZWO ASI6200MM Pro"})

    found = find_external_master(
        [str(light_dir / "light_001.fit")],
        kind="bias", camera="ZWO ASI6200MM Pro", temperature=None,
        filter=None, exposure=None,
    )
    assert found is None


def test_dark_exposure_must_be_at_least_light(tmp_path):
    """A 60s master dark cannot calibrate a 120s light — needs equal-or-longer."""
    target = tmp_path / "ic443"
    light_dir = target / "Lights"
    cal_dir = target / "Calibration Masters"
    _write_fits(light_dir / "light_001.fit", {"IMAGETYP": "Light"})
    _write_fits(cal_dir / "MasterDark_-20c_60s.fit",
                _master_header("dark", exposure=60.0, temperature=-20.0))

    found = find_external_master(
        [str(light_dir / "light_001.fit")],
        kind="dark", camera="ZWO ASI6200MM Pro", temperature=-20.0,
        filter=None, exposure=120.0,
    )
    assert found is None


def test_masters_in_dir_caches(tmp_path):
    """masters_in_dir is memoized — second call doesn't re-read the FITS files."""
    cal = tmp_path / "Calibration Masters"
    _write_fits(cal / "MasterBias_-20c.fit", _master_header("bias", temperature=-20.0))
    first = masters_in_dir(cal)
    # Add a new master after first call; cached call shouldn't see it.
    _write_fits(cal / "MasterDark_-20c_120s.fit",
                _master_header("dark", exposure=120, temperature=-20.0))
    second = masters_in_dir(cal)
    assert first == second
