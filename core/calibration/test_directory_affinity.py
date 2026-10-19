"""Unit tests for directory-affinity calibration matching.

These tests don't need the full Ray/context machinery — they construct a
temporary FITS tree on disk and exercise `find_external_master` directly.
"""
from __future__ import annotations

import os
from pathlib import Path

import astropy.io.fits as pyfits
import numpy as np
import pytest

from calibration.directory_affinity import (
    _ancestor_dirs,
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


def _write_fits(path: Path, header: dict, shape: tuple = (4, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = pyfits.PrimaryHDU(np.zeros(shape, dtype=np.float32))
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
        shape=(4, 4),
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
        kind="bias", shape=(4, 4), temperature=-20.0,
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
        kind="bias", shape=(4, 4), temperature=-20.0,
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
        kind="flat", shape=(4, 4), temperature=None,
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
        kind="bias", shape=(4, 4), temperature=None,
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
        kind="dark", shape=(4, 4), temperature=-20.0,
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


def test_renamed_camera_still_matches(tmp_path):
    """THE REGRESSION. A master whose INSTRUME disagrees with the light's still
    calibrates it.

    On 2026-02-16 a reconnecting driver renamed one ZWO body mid-session, so
    the afternoon's lights carried "ZWO ASI2600MM Duo" while the flats sitting
    in their own `Calibration Masters/` folder still said "ASI Camera (1)".
    The old string-equality test rejected them and four Starbase stacks were
    integrated with no flat at all. Geometry matches, so the name may differ.
    """
    target = tmp_path / "Starbase"
    light_dir = target / "Lights"
    cal_dir = target / "Calibration Masters"
    _write_fits(light_dir / "light_001.fit",
                {"IMAGETYP": "Light", "FILTER": "B",
                 "INSTRUME": "ZWO ASI2600MM Duo"})
    master = cal_dir / "MasterFlat_B.fit"
    _write_fits(master, _master_header(
        "flat", filter="B", temperature=None, camera="ASI Camera (1)",
    ))

    found = find_external_master(
        [str(light_dir / "light_001.fit")],
        kind="flat", shape=(4, 4), temperature=None,
        filter="B", exposure=None,
    )
    assert found == str(master.resolve())


def test_geometry_mismatch_is_rejected(tmp_path):
    """A master of different pixel dimensions cannot calibrate the frame.

    This is the constraint that replaced camera-string equality, so it has to
    actually bite — otherwise dropping the string would have removed the only
    check that a master belongs to this sensor at all.
    """
    target = tmp_path / "mismatch"
    light_dir = target / "Lights"
    cal_dir = target / "Calibration Masters"
    _write_fits(light_dir / "light_001.fit", {"IMAGETYP": "Light", "FILTER": "B"})
    _write_fits(cal_dir / "MasterFlat_B.fit",
                _master_header("flat", filter="B", temperature=None),
                shape=(8, 8))

    found = find_external_master(
        [str(light_dir / "light_001.fit")],
        kind="flat", shape=(4, 4), temperature=None,
        filter="B", exposure=None,
    )
    assert found is None


def test_calibration_named_directory_wins_over_unnamed_sibling(tmp_path):
    """Two siblings both hold a valid master; the one NAMED for calibration wins.

    A folder called `Calibration Masters/` beside the lights is the clearest
    statement of intent the filesystem can carry. `Archive/` sorts first
    alphabetically, so a plain `sorted()` walk would have picked it.
    """
    target = tmp_path / "intent"
    light_dir = target / "Lights"
    _write_fits(light_dir / "light_001.fit", {"IMAGETYP": "Light", "FILTER": "B"})
    intended = target / "Calibration Masters" / "MasterFlat_B.fit"
    other = target / "Archive" / "MasterFlat_B.fit"
    for path in (intended, other):
        _write_fits(path, _master_header("flat", filter="B", temperature=None))

    found = find_external_master(
        [str(light_dir / "light_001.fit")],
        kind="flat", shape=(4, 4), temperature=None,
        filter="B", exposure=None,
    )
    assert found == str(intended.resolve())


def test_the_walk_stops_at_the_home_folder_or_the_volume_never_a_system_root(tmp_path):
    """Masters live beside the library. Climbing to `/` probed every other
    volume and every automount, and `iterdir()` on macOS's `/home` automount
    hung a clean Mac for good (2026-10-19)."""
    home = Path.home().resolve()
    walk = list(_ancestor_dirs(home / "Pictures" / "AstroFlow" / "Raw" / "night"))
    assert walk[0] == home / "Pictures" / "AstroFlow" / "Raw" / "night" and walk[-1] == home
    walk = list(_ancestor_dirs(tmp_path / "lights"))           # under a system volume, outside home
    assert walk[0] == (tmp_path / "lights").resolve()
    assert Path("/") not in walk and Path("/System/Volumes/Data") not in walk
    assert all(not os.path.ismount(p) or p == walk[-1] for p in walk)
