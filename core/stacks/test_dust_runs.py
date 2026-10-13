"""Dust runs (`stacks/dust_runs.py`): the grouping of the identified
stacks' runs into rigs and filters, and the measurement of one run's map
from synthetic subs on disk."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from stacks.dust_runs import BLOCK, PEDESTAL_PER_OFFSET, DustEpochs, Rig, combine_dust_map, dust_graph, rigs
from stacks.dust_epochs import MoteModel

T0 = _dt.datetime(2026, 10, 1, 4, 0, 0)


class _R:
    """A Run stand-in that is hashable (SimpleNamespace defines __eq__ and
    is not), as a task key must be."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _stack(name, *, cam="3a15ec99ffff", focal=351.0, filt="H", runs=2, image_type="light", w=9576, h=6388):
    contents = {}
    for i in range(runs):
        meta = {f"{name}f{i}{j}": SimpleNamespace(header={"FOCALLEN": focal}) for j in range(3)}
        contents[f"{name}r{i}"] = _R(content_hash=f"{name}r{i}", frames=tuple(meta), metadata=meta,
                                     oldest_date=T0 + _dt.timedelta(days=i), newest_date=T0 + _dt.timedelta(days=i, hours=1),
                                     center=(10.0 + i, 20.0))
    return SimpleNamespace(camera_id_hash=cam, camera="ZWO ASI6200MM Pro", filter=filt, image_type=image_type,
                           width_pixels=w, height_pixels=h, contents=contents)


def test_rigs_are_camera_and_focal_cluster_per_filter():
    stacks = [_stack("a"), _stack("b", focal=349.0), _stack("c", filt="O"), _stack("d", focal=1230.0),
              _stack("e", cam="19ea9805aaaa"), _stack("f", image_type="dark"), _stack("g", cam=None)]
    out = rigs(stacks)
    assert sorted(out) == ["19ea9805@351/H", "3a15ec99@1230/H", "3a15ec99@351/H", "3a15ec99@351/O", "9576x638@351/H"]
    h = out["3a15ec99@351/H"]
    assert sorted(r.content_hash for r in h["runs"]) == ["ar0", "ar1", "br0", "br1"]     # 351 and 349 are one rig
    assert h["rig"] == Rig("3a15ec99", "ZWO ASI6200MM Pro", 351, 9576, 6388)
    assert out["3a15ec99@351/O"]["filter"] == "O"


def test_the_dust_graph_keys_are_hashable_and_runs_are_in_date_order():
    """Every task key holds a Run and the rig (boot184/185: an unhashable
    field-wise Run hash, then a dict rig, each killed the builder)."""
    from stacks.dust_runs import dust_epochs, measure_dust_run, merge_dust_epochs
    graph = dust_graph([_stack("a"), _stack("b", filt="O")])
    keys = list(graph)
    assert all(hash(k) is not None for k in keys)
    folds = [k for k in keys if isinstance(k, tuple) and k[0] is dust_epochs]
    assert sorted(k[1] for k in folds) == ["3a15ec99@351/H", "3a15ec99@351/O"]
    runs = next(k for k in folds if k[1] == "3a15ec99@351/H")[4]
    assert [r.content_hash for r in runs] == ["ar0", "ar1"]
    assert [k[1].content_hash for k in graph[next(k for k in folds if k[1] == "3a15ec99@351/H")]] == ["ar0", "ar1"]
    assert sum(1 for k in keys if isinstance(k, tuple) and k[0] is measure_dust_run) == 4
    assert merge_dust_epochs in keys


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


def test_the_measurement_node_is_a_file_producer():
    """The decorator must see the real return annotation: as a string (a
    postponed annotation) the node had no `filename_fn` and every
    measurement failed on the first deploy (boot181)."""
    from stacks.dust_runs import measure_dust_run
    assert hasattr(measure_dust_run, "filename_fn")


def test_a_run_hashes_by_its_content_hash():
    """A measurement task is keyed by its Run; the dataclass must not have
    replaced BaseRun's content-hash __hash__ with a field-wise one (boot184:
    "unhashable type: 'dict'" from the run's frozendict fields)."""
    from stacks.run import BaseRun, Run
    assert Run.__hash__ is BaseRun.__hash__


def test_the_map_s_image_type_is_one_the_header_normaliser_admits():
    """boot186: an IMAGETYP outside bias/dark/flat/light wordings made the
    metadata wrap mark every freshly written map as broken."""
    from file_io.normalize_header import normalize_header
    from stacks.dust_runs import IMAGE_TYPE
    assert normalize_header({"IMAGETYP": IMAGE_TYPE})["IMAGETYP"] == "flat"
