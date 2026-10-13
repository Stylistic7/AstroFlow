"""Dust runs (`stacks/dust_runs.py`): the grouping of a library into
(optical system, night, pointing) runs, and the measurement of one run's
map from synthetic subs on disk."""
from __future__ import annotations

import datetime as _dt
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits

from metadata.classes import FrameHash, ImageMetadata, MetadataDict
from plate_solving import PlateSolution
from stacks.dust_runs import (
    BLOCK, PEDESTAL_PER_OFFSET, DustEpochs, combine_dust_map, dust_runs,
)
from stacks.dust_epochs import MoteModel

T0 = _dt.datetime(2026, 10, 1, 4, 0, 0)


def _meta(i, *, centre=(10.0, 20.0), focal=351.0, when=None, w=960, h=640, solved=True, image_type="Light"):
    hdr = {"IMAGETYP": image_type, "FILTER": "H", "FOCALLEN": focal, "INSTRUME": "ZWO ASI6200MM Pro",
           "NAXIS1": w, "NAXIS2": h, "BITPIX": 16, "EXPOSURE": 300.0, "OFFSET": 50,
           "DATE-OBS": when if when is not None else T0 + _dt.timedelta(minutes=5 * i)}
    ps = PlateSolution(header={}, num_stars=100, center=centre, image_scale=2.2, rotation_angle=0.0) if solved else None
    return ImageMetadata(content_hash=f"f{i:04d}", stable_hash=f"f{i:04d}", filename=Path(f"/raw/L{i:04d}.fit"),
                         header=hdr, width_pixels=w, height_pixels=h, channels=1, dtype=np.dtype("uint16"),
                         plate_solution=ps)


def _library(metas) -> MetadataDict:
    return MetadataDict({FrameHash(m.content_hash): m for m in metas})


def test_runs_split_by_system_night_and_pointing():
    metas = [_meta(i) for i in range(6)]                                            # night 1, pointing A
    metas += [_meta(10 + i, centre=(12.0, 20.0)) for i in range(4)]                 # night 1, pointing B
    metas += [_meta(20 + i, when=T0 + _dt.timedelta(days=1, minutes=5 * i)) for i in range(5)]   # night 2, A
    metas += [_meta(30 + i, focal=1230.0) for i in range(4)]                        # another train
    metas += [_meta(40 + i, centre=(10.01, 20.01)) for i in range(2)]               # dithered A, too few alone
    metas += [_meta(50, solved=False), _meta(51, image_type="Flat")]                # never
    runs = dust_runs(_library(metas), {})
    by = sorted((r.system.split("@")[1], len(r.frames), r.date.day) for r in runs.values())
    assert by == [("1230", 4, 1), ("351", 4, 1), ("351", 5, 2), ("351", 8, 1)]
    a = next(r for r in runs.values() if len(r.frames) == 8)
    assert set(a.frames) == {f"f{i:04d}" for i in range(6)} | {"f0040", "f0041"}   # the dithers join A
    assert a.centre[0] == np.mean([10.0] * 6 + [10.01] * 2)


def test_identity_is_the_frames_not_the_paths():
    a = dust_runs(_library([_meta(i) for i in range(4)]), {})
    moved = [replace(_meta(i), filename=Path("/elsewhere") / f"L{i:04d}.fit") for i in range(4)]
    b = dust_runs(_library(moved), {})
    assert next(iter(b.values())).files[0].startswith("/elsewhere/")
    assert set(a) == set(b)
    assert hash(next(iter(a.values()))) == hash(next(iter(b.values())))


def _write_sub(path: Path, mote, seed, sky=2000.0, offset=50):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:640, :960]
    vignette = 1.0 - 0.15 * ((xx - 480) ** 2 + (yy - 320) ** 2) / (480 ** 2 + 320 ** 2)
    img = sky * vignette
    x, y, r, depth = mote
    img *= 1.0 - depth / (1.0 + np.exp((np.sqrt((xx - x) ** 2 + (yy - y) ** 2) - r) / 3.0))
    for _ in range(40):                                              # stars
        sx, sy = rng.integers(20, 940), rng.integers(20, 620)
        img += 20000.0 * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / 8.0)
    img += rng.normal(0, np.sqrt(sky), img.shape)
    data = np.clip(np.rint(img + PEDESTAL_PER_OFFSET * offset), 0, 65535).astype(np.uint16)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["OFFSET"] = offset
    hdu.header["IMAGETYP"] = "Light"
    hdu.writeto(path, overwrite=True)


def test_the_map_of_a_run_is_its_dust_at_true_depth(tmp_path):
    mote = (300.0, 200.0, 24.0, 0.08)
    files = []
    for i in range(6):
        f = tmp_path / f"sub{i}.fit"
        _write_sub(f, mote, seed=i)
        files.append(str(f))
    out, n = combine_dust_map(files)
    assert n == 6 and out.shape == (640 // BLOCK, 960 // BLOCK)
    yy, xx = np.ogrid[:out.shape[0], :out.shape[1]]
    d2 = (xx - 300 / BLOCK) ** 2 + (yy - 200 / BLOCK) ** 2
    inside, ring = d2 <= 1.5 ** 2, (d2 > 6 ** 2) & (d2 < 10 ** 2)
    depth = 1.0 - float(np.median(out[inside])) / float(np.median(out[ring]))
    assert abs(depth - 0.08) < 0.012, depth                      # the pedestal is right: depth is not inflated
    assert float(np.std(out[ring])) < 0.01                        # stars are masked, the vignetting flattened


def test_model_lookup_is_by_majority_run_of_the_frames():
    m1, m2 = MoteModel.of(()), MoteModel.of((), epoch="x")
    table = DustEpochs.of({"r1": m1, "r2": m2}, {"a": "r1", "b": "r1", "c": "r2"}, ())
    assert table.model_for(("a", "b", "c")) is m1
    assert table.model_for(("c",)) is m2
    assert table.model_for(("zzz",)) is None
    assert DustEpochs.empty().model_for(("a",)) is None


def test_a_scaled_flat_variant_carries_the_epoch_s_motes(tmp_path):
    """`with_scaled_flats` writes, per gained flat and mote model, the scaled
    flat times the model's mote map under "<flat id>:<model hash>" — the
    file a run of that epoch is calibrated with — and the plain scaled flat
    for every other run."""
    from file_io.generic import read_image
    from stacks.dust_epochs import Mote, render_motes
    from stacks.star_flat_gain import FlatGain, apply_gain, with_scaled_flats
    from stacks.starflat import write_master_flat

    rng = np.random.default_rng(0)
    flat = (1.0 + 0.02 * rng.standard_normal((192, 256))).astype(np.float32)
    src = tmp_path / "flat.xisf"
    write_master_flat(flat, src, {"FILTER": "O", "STBLHASH": "fid", "IMAGETYP": "Master Flat"})
    model = MoteModel.of((Mote(100.0, 80.0, 40.0, (0.1, 0.1, 0.08, 0.05, 0.02, 0.0)),), epoch="e")
    gain = FlatGain(k=1.5, flat_id="fid", fit_json="{}")
    out = with_scaled_flats({"fid": src}, [gain], tmp_path, [model])
    assert set(out) == {"fid", f"fid:{model.content_hash}"}
    # the writer stores a flat at its own scale; a stacker divides by the
    # flat normalised to its median, so compare normalised
    norm = lambda a: a / np.median(a)   # noqa: E731
    plain = norm(np.asarray(read_image(out["fid"]), np.float32).squeeze())
    variant = norm(np.asarray(read_image(out[f"fid:{model.content_hash}"]), np.float32).squeeze())
    scaled = apply_gain(flat, 1.5)
    assert np.allclose(plain, norm(scaled), atol=5e-4)
    assert np.allclose(variant, norm(scaled * render_motes(model, flat.shape, BLOCK)), atol=5e-4)
    assert variant[80, 100] < plain[80, 100] * 0.92 and np.isclose(variant[10, 10], plain[10, 10], atol=2e-3)
