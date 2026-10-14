"""Baking a finish into a 16-bit sRGB TIFF for Photoshop (2026-10-04):
exact (strip-wise bake == whole bake), streaming, Photoshop's 4 GB limit
respected, the clipping overlay never exported, and the job reports done or
the error through its status file."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import pyvips
import tifffile

from finishes import export
from finishes.registry import get_finish


def _pyramid(path, arr, fmt="float"):
    a = np.ascontiguousarray(arr)
    pyvips.Image.new_from_memory(a.data, a.shape[1], a.shape[0], a.shape[2], fmt).tiffsave(
        str(path), tile=True, pyramid=True, tile_width=256, tile_height=256, bigtiff=True)
    return str(path)


@pytest.fixture
def scene(tmp_path):
    rng = np.random.default_rng(2)
    nb = (0.005 + rng.normal(0, 1e-5, (600, 520, 3))).astype(np.float32)
    nb[100:300, 100:300] += 3e-4
    return _pyramid(tmp_path / "nb.tiff", nb), nb


def _values(f):
    v = dict(f.defaults(), star_s=0.0, show_clipping=1.0)
    for r in "SHO":
        v.update({f"sky{r}": 0.005, f"sig{r}": 1e-5, f"top{r}": 0.0053})
    return v


def test_strip_bake_is_exact_profiled_and_unclipped(scene, tmp_path):
    path, nb = scene
    f = get_finish("Hubble", primary="Linear SHO")
    v = _values(f)
    r = export.bake_to_tiff(NS(finish=f, inputs=[NS(path=path), NS(path=None)]), v, tmp_path / "o.tif")
    assert (r["level"], r["width"], r["height"]) == (0, 520, 600)
    with tifffile.TiffFile(r["path"]) as t:
        page = t.pages[0]
        assert page.dtype == np.uint16 and page.compression.name == "NONE" and 34675 in page.tags
        got = page.asarray()
    want = np.round(np.clip(f.bake([nb, np.zeros_like(nb)], dict(v, show_clipping=0.0)), 0, 1) * 65535)
    assert np.array_equal(got, want.astype(np.uint16))


def test_photoshop_size_limit_picks_the_finest_fitting_level():
    assert export.export_level(9554, 6345, 7) == 0
    assert export.export_level(49023, 46388, 9) == 1


def test_eight_bit_pyramids_are_rescaled(tmp_path):
    f = get_finish("Master", primary="Master")
    p = _pyramid(tmp_path / "u8.tiff", np.full((40, 50, 1), 255, np.uint8), fmt="uchar")
    v = dict(f.defaults(), blackpoint=0.0, whitepoint=1.0, midtones=0.5)
    r = export.bake_to_tiff(NS(finish=f, inputs=[NS(path=p)]), v, tmp_path / "m.tif")
    assert tifffile.imread(r["path"]).min() == 65535


def test_job_opens_the_file_or_reports_the_error(scene, tmp_path, monkeypatch):
    path, _ = scene
    f = get_finish("Hubble", primary="Linear SHO")
    calls = []
    monkeypatch.setattr(export.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    job = {"finish": "Hubble", "primary": "Linear SHO", "paths": [path, None],
           "params": _values(f), "dest": str(tmp_path / "x.tif"), "status": str(tmp_path / "s.json"),
           "app": "/Applications/Adobe Photoshop 2026/Adobe Photoshop 2026.app"}
    (tmp_path / "j.job").write_text(json.dumps(job))
    export.run_job(tmp_path / "j.job")
    status = json.loads((tmp_path / "s.json").read_text())
    assert status["state"] == "done" and Path(status["path"]).exists()
    assert calls == [["open", "-a", job["app"], job["dest"]]]
    job["paths"] = [str(tmp_path / "missing.tiff"), None]
    (tmp_path / "j.job").write_text(json.dumps(job))
    export.run_job(tmp_path / "j.job")
    status = json.loads((tmp_path / "s.json").read_text())
    assert status["state"] == "error" and status["error"]
