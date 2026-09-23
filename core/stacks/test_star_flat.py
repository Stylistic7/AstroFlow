"""Star flats as calibration masters: identity, grouping, the adaptive solve
and quality gate, the master written for WBPP, the calibrate dispatch, and
the decision default. Pure — no Ray, no context."""
import dataclasses
import datetime
from pathlib import Path
from types import SimpleNamespace

import astropy.io.fits as pf
import numpy as np
import pytest
from frozendict import frozendict

import stacks.calibration_selection as sel
import stacks.star_flat as sfm
from calibration.calibration_frames import _build_cal_index, focal_length_of
from decisions.resolve import NO_FLAT, _star_flat_for
from file_io.generic import read_header, read_image
from file_io.sidecar import get_section
from hashing import hash_list
from metadata.classes import FrameHash, ImageMetadata, MetadataDict
from metadata.star_photometry import SCHEMA as PHOTOMETRY_SCHEMA, read_star_photometry, write_star_photometry
from plate_solving import PlateSolution
from stacks.run import RunsDict
from stacks.star_flat import (
    CELL_COVERAGE_MIN,
    CELLS, GATE_UNCERTAINTY_MAX, MAG_WINDOW, MIN_SPLIT_HALF, MIN_SUBS, StarFlatCandidates, StarFlatStack,
    StarFlats, _focal_clusters, gate, passes_gate, solve_adaptive, solve_and_write, star_flat_candidates,
)
from stacks.test_starflat import H, W, _synth, _truth
from stacks.types import StackHash, StackIdDependency
from units import Celsius, Seconds


# ---------------------------------------------------------------- fixtures
def _meta(i, *, filt="H", focal=351.0, instrume="ZWO ASI6200MM Pro", solved=True,
          image_type="Light", w=960, h=640, day=0):
    hdr = {"IMAGETYP": image_type, "FILTER": filt, "FOCALLEN": focal, "INSTRUME": instrume,
           "CCD-TEMP": -10.0, "NAXIS1": w, "NAXIS2": h, "BITPIX": 16, "EXPOSURE": 300.0,
           "DATE-OBS": datetime.datetime(2026, 1, 1) + datetime.timedelta(days=day, minutes=i)}
    ps = PlateSolution(header={}, num_stars=100, center=(0.0, 0.0), image_scale=1.5, rotation_angle=0.0) if solved else None
    return ImageMetadata(content_hash=f"c{i:04d}", stable_hash=f"c{i:04d}", filename=Path(f"/raw/L{i:04d}.fit"),
                         header=hdr, width_pixels=w, height_pixels=h, channels=1, dtype=np.dtype("uint16"),
                         plate_solution=ps)


def _library(n=30, start=0, **kw) -> MetadataDict:
    # keyed by content hash, exactly like the real library
    return MetadataDict({FrameHash(f"c{i:04d}"): _meta(i, **kw) for i in range(start, start + n)})


def _star_flat(members=("a", "b", "c"), **kw):
    fields = dict(content_hash=hash_list(sorted(members)), image_type="flat", camera="cam", filter="H",
                  mean_temperature=Celsius(-10.0), exposure=Seconds(1.0), integration=Seconds(0.0),
                  width_pixels=960, height_pixels=640, stable_hash="a", oldest_date=datetime.datetime(2026, 1, 1),
                  newest_date=datetime.datetime(2026, 1, 2), state=frozendict(), contents=RunsDict(),
                  camera_id_hash=None, focal_length=351.0, member_files=tuple(f"/raw/{m}.fit" for m in members),
                  photometry_schema=PHOTOMETRY_SCHEMA)
    fields.update(kw)
    return StarFlatStack(**fields)


# ---------------------------------------------------------------- identity
def test_identity_follows_the_stack_conventions():
    cands = star_flat_candidates(_library(30), {})
    assert len(cands) == 1
    sf = next(iter(cands.values()))
    assert sf.stable_hash == "c0000"                                  # the oldest member's content hash
    # the house convention — membership — plus the one term a Stack does not
    # need: the photometry schema, because the flat's pixels come from the
    # members' MEASUREMENTS, not their pixels (cf. DownscaledStack's _fn_hash)
    from metadata.star_photometry import SCHEMA as PHOTOMETRY_SCHEMA
    assert sf.content_hash == hash_list(sorted(f"c{i:04d}" for i in range(30)) + [PHOTOMETRY_SCHEMA])
    # re-solving a member must NOT move it — that would re-stack the whole train
    lib = _library(30)
    lib[FrameHash("c0007")] = dataclasses.replace(lib[FrameHash("c0007")], plate_solution=dataclasses.replace(
        lib[FrameHash("c0007")].plate_solution, header={**lib[FrameHash("c0007")].plate_solution.header, "CRVAL1": 99.0}))
    assert next(iter(star_flat_candidates(lib, {}).values())).content_hash == sf.content_hash
    assert sf.kind == "star" and sf.image_type == "flat" and sf.n_subs == 30
    assert sf.contents == RunsDict() and sf.focal_length == 351.0 and sf.filter == "H"
    assert sf.photometry_schema == PHOTOMETRY_SCHEMA                  # an engine change re-keys
    # a new sub re-keys (content) but keeps identity (stable)
    more = star_flat_candidates(_library(31), {})
    sf2 = next(iter(more.values()))
    assert sf2.stable_hash == sf.stable_hash and sf2.content_hash != sf.content_hash
    assert hash(sf) != hash(sf2) and sf != sf2
    assert " at 0x" not in repr(sf)                                   # determinism guard


def test_focal_clusters_use_the_matchers_two_percent():
    assert _focal_clusters([350, 351, 353, 1230, 1234, 2060]) == [[0, 1, 2], [3, 4], [5]]
    assert _focal_clusters([]) == []


def test_candidates_group_by_camera_focal_and_filter_and_skip_unusable():
    lib = MetadataDict()
    lib.update(_library(30, filt="H", focal=350.0))
    lib.update(_library(30, start=100, filt="S", focal=351.0))         # other filter
    lib.update(_library(30, start=200, filt="H", focal=1230.0))        # other train
    lib.update(_library(30, start=300, solved=False))                  # unsolved
    lib.update(_library(30, start=400, filt=""))                       # no filter
    lib.update(_library(30, start=500, image_type="Master Light"))     # masters
    lib.update(_library(MIN_SUBS - 1, start=600, filt="O"))            # too few
    cands = star_flat_candidates(lib, {})
    keys = sorted((v.filter, round(v.focal_length)) for v in cands.values())
    assert keys == [("H", 350), ("H", 1230), ("S", 351)]
    assert all(v.n_subs == 30 for v in cands.values())


def test_candidates_split_by_camera_fingerprint():
    lib = MetadataDict()
    lib.update(_library(30)); lib.update(_library(30, start=100))
    fingerprints = ({FrameHash(f"c{i:04d}"): (1, 2, 3) for i in range(30)}
                    | {FrameHash(f"c{i:04d}"): (7, 8, 9) for i in range(100, 130)})
    cands = star_flat_candidates(lib, fingerprints)
    assert len(cands) == 2 and {v.camera_id_hash is not None for v in cands.values()} == {True}
    assert len(star_flat_candidates(lib, {})) == 1               # no fingerprints: one sensor bucket


# ---------------------------------------------------------------- solve + gate
def _write_synthetic_subs(tmp_path, n_subs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    stars, _ = _synth(seed=3, n_subs=n_subs, stars_per_sub=1200, contaminated=())
    paths = []
    for s in range(n_subs):
        p = tmp_path / f"L{s:03d}.fit"
        pf.PrimaryHDU(np.zeros((4, 4), np.uint16)).writeto(p)
        rows = stars[stars[:, 10] == s][:, :10]
        write_star_photometry(p, rows, {"engine": "test"})
        paths.append(str(p))
    return paths


def test_solve_adaptive_picks_the_finest_cell_every_cell_can_fill():
    """Sufficiency is judged per CELL, and the finest cell size at which every
    cell qualifies wins. A group with the same stars spread thinner drops to a
    coarser cell rather than fitting a dozen stars per cell."""
    stars, _ = _synth(seed=0, n_subs=24, stars_per_sub=1200, contaminated=())
    sol, cell, r_raw, r_blob, cov = solve_adaptive(stars, (H, W), cells=(64, 128, 256))
    assert cell == 64 and r_raw > 0.8 and cov["covered_fraction"] >= CELL_COVERAGE_MIN
    sparse, _ = _synth(seed=0, n_subs=24, stars_per_sub=40, contaminated=())
    sol2, cell2, _, _, cov2 = solve_adaptive(sparse, (H, W), cells=(128, 256))
    assert cell2 == 256 and sol2 is not None                      # too thin for anything finer
    # nothing at all when even the coarsest cell is starved
    thin, _ = _synth(seed=0, n_subs=3, stars_per_sub=5, contaminated=())
    sol3, cell3, _, _, cov3 = solve_adaptive(thin, (H, W), cells=(128, 256))
    assert sol3 is None and cell3 is None and cov3["covered_fraction"] < CELL_COVERAGE_MIN
    assert CELLS == (256, 512, 1024)                              # production: fine -> coarse


def test_solve_and_write_makes_a_wbpp_master_and_passes_the_gate(tmp_path):
    paths = _write_synthetic_subs(tmp_path, 30)
    sf = _star_flat(members=[Path(p).stem for p in paths], member_files=tuple(paths), width_pixels=W, height_pixels=H)
    dest = tmp_path / "out" / "flat.xisf"
    # the synthetic detector holds only 12 cells at 256 px — too few for the split-half
    q = solve_and_write(sf, dest, cells=(128,))
    assert q["good"] and q["n_subs"] == 30 and q["cell"] == 128 and q["per_cell_uncertainty"] < GATE_UNCERTAINTY_MAX
    hdr = read_header(dest)
    assert str(hdr["STBLHASH"]) == sf.stable_hash and "master flat" in str(hdr["IMAGETYP"]).lower() and hdr["FILTER"] == "H"
    assert get_section(dest, sfm.SECTION)["good"] is True
    img = np.asarray(read_image(dest), dtype=np.float64).squeeze()
    assert img.shape == (H, W) and abs(np.median(img) - 0.5) < 0.01     # level inside [0, 1]
    yy, xx = np.mgrid[:H, :W]
    truth = _truth(xx, yy)
    r = np.corrcoef((img / np.median(img)).ravel(), (truth / np.median(truth)).ravel())[0, 1]
    # the flat is the truth's response — 128 px cells under-resolve the
    # synthetic ~280 px ripple, so 0.92 on this small detector is the ceiling
    assert r > 0.9


def test_solve_and_write_rejects_a_sparse_group(tmp_path):
    paths = _write_synthetic_subs(tmp_path, 5)
    sf = _star_flat(members=[Path(p).stem for p in paths], member_files=tuple(paths), width_pixels=W, height_pixels=H)
    q = solve_and_write(sf, tmp_path / "flat.xisf")
    assert q["good"] is False and q["n_subs"] == 5


def test_gate_keeps_only_good_masters(tmp_path):
    good = _write_synthetic_subs(tmp_path / "g", 30); bad = _write_synthetic_subs(tmp_path / "b", 5)
    sf_good = _star_flat(members=[Path(p).stem for p in good], member_files=tuple(good), width_pixels=W, height_pixels=H, stable_hash="good")
    sf_bad = _star_flat(members=[Path(p).stem for p in bad], member_files=tuple(bad), width_pixels=W, height_pixels=H, stable_hash="bad")
    solve_and_write(sf_good, tmp_path / "good.xisf", cells=(128,)); solve_and_write(sf_bad, tmp_path / "bad.xisf")
    cands = StarFlatCandidates({StackHash("good"): sf_good, StackHash("bad"): sf_bad})
    rule = dict(min_subs=MIN_SUBS, min_split_half=MIN_SPLIT_HALF, max_uncertainty=GATE_UNCERTAINTY_MAX)
    kept = gate(cands, [tmp_path / "good.xisf", tmp_path / "bad.xisf"], **rule)
    assert isinstance(kept, StarFlats) and list(kept) == ["good"]
    assert list(gate(cands, [tmp_path / "good.xisf"], **{**rule, "max_uncertainty": 0.0001})) == []
    # the graph's collector receives each built master as its ImageMetadata
    from stacks.star_flat import mint_star_flats
    masters = [dataclasses.replace(_meta(0, image_type="Master Flat"), filename=tmp_path / n) for n in ("good.xisf", "bad.xisf")]
    minted = mint_star_flats._function(cands, MIN_SUBS, MIN_SPLIT_HALF, GATE_UNCERTAINTY_MAX, *masters)
    assert list(minted) == ["good"]


def test_gate_rule_needs_subs_agreement_and_low_noise():
    ok = dict(n_subs=30, split_half_r=0.9, per_cell_uncertainty=0.001)
    rule = dict(min_subs=25, min_split_half=0.8, max_uncertainty=0.005)
    assert passes_gate(ok, **rule)
    assert not passes_gate({**ok, "n_subs": 3}, **rule)                     # a 3-sub flat slipped through on 2026-09-07
    assert not passes_gate({**ok, "split_half_r": float("nan")}, **rule)    # too few cells to judge
    assert not passes_gate({**ok, "per_cell_uncertainty": 0.011}, **rule)   # long-FL groups: 0.6-1.1%
    assert not passes_gate({"n_subs": 0, "reason": "too few subs with photometry"}, **rule)


# ---------------------------------------------------------------- integration seams
def test_calibrate_dispatches_the_star_flat_producer(monkeypatch):
    """A star flat is solved (never WBPP-integrated) and then RE-SCALED to the
    light stack that divides by it, with the stack's dark master handed to
    the fit; the dark itself is integrated and passed through as built."""
    sf = _star_flat(stable_hash="sf1")
    light = SimpleNamespace(stable_hash="light1", contents={"run0": SimpleNamespace(
        dependencies={"dark": StackIdDependency(stack_id="dark1"), "flat": StackIdDependency(stack_id="sf1")})})
    dark = SimpleNamespace(stable_hash="dark1", image_type="dark", contents={"run0": SimpleNamespace(dependencies={})})
    calls = []
    monkeypatch.setattr(sel, "resolve_calibration", lambda s, c, d: s)
    monkeypatch.setattr(sel, "stack_files", lambda s, config, *m: calls.append(("wbpp", s.stable_hash)) or f"master:{s.stable_hash}")
    monkeypatch.setattr(sfm, "build_star_flat", lambda s, cells, window: calls.append(("star", s.stable_hash, cells)) or f"master:{s.stable_hash}")
    monkeypatch.setattr(sfm, "scale_star_flat",
                        lambda stack, master, dark_, *settings: calls.append(("scale", stack.stable_hash, master, dark_)) or f"scaled:{master}")
    config = SimpleNamespace(get_calibration_stacks=lambda: {"dark1": dark, "sf1": sf}, decisions={})
    masters = sel.calibrate(light, config)
    assert masters == ["master:dark1", "scaled:master:sf1"]
    assert ("star", "sf1", CELLS) in calls and ("wbpp", "dark1") in calls and ("wbpp", "sf1") not in calls
    assert ("scale", "light1", "master:sf1", "master:dark1") in calls


def test_calibrate_scales_without_a_dark_when_the_stack_has_none(monkeypatch):
    sf = _star_flat(stable_hash="sf1")
    light = SimpleNamespace(stable_hash="light1", contents={"run0": SimpleNamespace(
        dependencies={"flat": StackIdDependency(stack_id="sf1")})})
    calls = []
    monkeypatch.setattr(sel, "resolve_calibration", lambda s, c, d: s)
    monkeypatch.setattr(sfm, "build_star_flat", lambda s, cells, window: f"master:{s.stable_hash}")
    monkeypatch.setattr(sfm, "scale_star_flat",
                        lambda stack, master, dark_, *settings: calls.append(dark_) or f"scaled:{master}")
    config = SimpleNamespace(get_calibration_stacks=lambda: {"sf1": sf}, decisions={})
    assert sel.calibrate(light, config) == ["scaled:master:sf1"]
    assert calls == [None]


def test_focal_length_and_index_treat_star_flats_specially():
    sf = _star_flat()
    assert focal_length_of(sf) == 351.0
    bias_by, dark_by, flat_by = _build_cal_index({StackHash("a"): sf})
    assert flat_by == {}                                          # never a chronological candidate


def _run(filt="H", focal=351.0, w=960, h=640):
    return SimpleNamespace(filter=filt, width_pixels=w, height_pixels=h,
                           metadata={"f": SimpleNamespace(header={"FOCALLEN": focal})})


def test_default_flat_is_the_groups_star_flat():
    cal = {StackHash("sf350"): _star_flat(stable_hash="sf350", focal_length=350.0, members=("a", "b")),
           StackHash("sf1230"): _star_flat(stable_hash="sf1230", focal_length=1230.0, members=("c", "d", "e"))}
    assert _star_flat_for(_run(focal=351.0), cal, None) == "sf350"      # 2% train tolerance
    assert _star_flat_for(_run(focal=1234.0), cal, None) == "sf1230"
    assert _star_flat_for(_run(focal=540.0), cal, None) is None          # no star flat for this train
    assert _star_flat_for(_run(filt="S"), cal, None) is None             # filter must match
    assert _star_flat_for(_run(w=6248), cal, None) is None               # sensor geometry must match


def test_default_prefers_the_same_camera_then_the_deeper_flat():
    cal = {StackHash("cam1"): _star_flat(stable_hash="cam1", camera_id_hash="fp1", members=("a", "b")),
           StackHash("cam2"): _star_flat(stable_hash="cam2", camera_id_hash="fp2", members=("c", "d", "e", "f"))}
    assert _star_flat_for(_run(), cal, "fp1") == "cam1"                  # same fingerprint wins
    assert _star_flat_for(_run(), cal, None) == "cam2"                   # unknown camera: most subs


# ---------------------------------------------------------------- UI, previews, graph
def test_ui_dict_describes_a_star_flat():
    from stacks.compute_graph import process_calibration_stacks_for_ui
    sf = _star_flat(stable_hash="sf1", focal_length=350.6)
    d = process_calibration_stacks_for_ui._function({StackHash("sf1"): sf})["sf1"]   # the node's body
    assert d["kind"] == "star" and d["n_subs"] == 3 and d["focal_length"] == 351
    assert d["filenames"] == list(sf.member_files) and "member_files" not in d
    assert d["stable_hash"] == "sf1" and d["image_type"] == "flat"


def test_preview_of_a_star_flat_is_its_master_and_never_builds_one(tmp_path, monkeypatch):
    import previews.calibration as pc
    paths = _write_synthetic_subs(tmp_path, 30)
    sf = _star_flat(members=[Path(p).stem for p in paths], member_files=tuple(paths), width_pixels=W, height_pixels=H, stable_hash="sf1")
    monkeypatch.setattr(pc, "context", SimpleNamespace(calibration_stacks={StackHash("sf1"): sf}))
    preview = pc.get_subframe_for_calibration_stack.__wrapped__          # past the functools cache
    monkeypatch.setattr(sfm, "star_flat_master_path", lambda s: None)
    assert preview(StackHash("sf1"), 200) == (None, None)               # not built yet: no flat, no build
    dest = tmp_path / "flat.xisf"
    solve_and_write(sf, dest)
    monkeypatch.setattr(sfm, "star_flat_master_path", lambda s: dest)
    _, image = preview(StackHash("sf1"), 200)
    assert image.dtype == np.float32 and image.shape[1] == 200 and 0.3 < np.median(image) < 0.7


def test_scatter_builds_one_master_per_candidate_then_gates_in_order():
    from stacks.star_flat import build_star_flat, mint_star_flats, scatter_star_flats
    a, b = _star_flat(stable_hash="a"), _star_flat(stable_hash="b", members=("x", "y"))
    cands = StarFlatCandidates({StackHash("b"): b, StackHash("a"): a})
    graph = scatter_star_flats._function(cands)
    builds = [(build_star_flat, a, CELLS, MAG_WINDOW), (build_star_flat, b, CELLS, MAG_WINDOW)]
    assert len(graph) == 3 and all(graph[t] == {} for t in builds)
    # ordered: a value-consuming collector; the rule's thresholds sit in its key
    assert graph[(mint_star_flats, cands, MIN_SUBS, MIN_SPLIT_HALF, GATE_UNCERTAINTY_MAX)] == builds


def test_merge_adds_the_gated_star_flats_to_the_calibration_stacks():
    from stacks.composite_stack import CalibrationStacksDict
    from stacks.star_flat import merge_star_flats
    sf, dark = _star_flat(stable_hash="sf"), _star_flat(stable_hash="d", image_type="dark")   # any CalibrationStack
    merged = merge_star_flats._function(StarFlats({StackHash("sf"): sf}), CalibrationStacksDict({StackHash("d"): dark}))
    assert isinstance(merged, CalibrationStacksDict) and dict(merged) == {"d": dark, "sf": sf}


def test_override_serializer_passes_the_pinned_none_through():
    from decisions.decisions import BinaryChoice
    from web_server.backend.decisions import _serialize_choice
    assert _serialize_choice(NO_FLAT) == "none" and _serialize_choice("sf1") == "sf1"
    assert _serialize_choice(BinaryChoice.EXCLUDE) == BinaryChoice.EXCLUDE.value


def test_build_refuses_unreadable_members_instead_of_minting_a_blank(tmp_path, monkeypatch):
    """A worker whose share is still cold sees the member subs as missing;
    that must fail the task (retried elsewhere), never build a 0-sub master."""
    from stacks.star_flat import build_star_flat
    sf = _star_flat(member_files=(str(tmp_path / "gone.fit"),))
    with pytest.raises(FileNotFoundError, match="unreadable"):
        build_star_flat._function(sf, CELLS, MAG_WINDOW)


def test_master_path_reads_the_written_files_metadata(tmp_path, monkeypatch):
    """`star_flat_master_path` unwraps a CACHED build.

    A producer node's cached value is the written file's `ImageMetadata`,
    not the `WithSidecar` the body returned — so the path is `.filename`.
    Reading `.path` raised AttributeError on every cache HIT while a MISS
    quietly returned None, so a star-flat preview either 500'd or served an
    uncorrected frame with no sign that anything went wrong (2026-09-08).
    """
    dest = tmp_path / "flat.xisf"
    sf = _star_flat()
    # Exactly the shape a producer node hands its consumers: the written
    # file's ImageMetadata, whose path lives in `filename`.
    written = ImageMetadata(
        content_hash="f0", stable_hash="f0", filename=dest, header={},
        width_pixels=W, height_pixels=H, channels=1, dtype=np.dtype("float32"),
    )

    monkeypatch.setattr(sfm, "build_star_flat",
                        SimpleNamespace(cached_value=lambda *a: (True, written)))
    assert sfm.star_flat_master_path(sf) == Path(dest)

    # A miss is "no master yet" — never a build triggered from a preview.
    monkeypatch.setattr(sfm, "build_star_flat",
                        SimpleNamespace(cached_value=lambda *a: (False, None)))
    assert sfm.star_flat_master_path(sf) is None


def test_star_flat_master_is_a_display_product_like_every_other_master():
    """`display=` is what makes an op build its tile-pyramid sibling.

    Regression for 2026-09-09: `build_star_flat` declared no `display=`, so
    `makes_tiff` was False, `_publish_with_presentations` never ran, and the
    master shipped without a pyramid. The thumbnail path then had nothing
    cheap to serve and refused the 61 MP primary on the budget gate, so every
    star flat's detail page 404'd. Ordinary calibration masters come from
    `stack_files`, which declares `display="Master"` — star flats were the
    odd one out.
    """
    from stacks.star_flat import build_star_flat
    from stacks.stacking.pixinsight import stack_files
    assert getattr(build_star_flat, "_display", None) == "Star Flat"
    # the analogue: ordinary masters are display products too
    assert getattr(stack_files, "_display", None) is not None


def test_scale_star_flat_body_runs_on_the_runtime_types(tmp_path, monkeypatch):
    """The node's body, driven the way the runtime drives it: the group's flat
    and the dark arrive as `ImageMetadata` (a producer's result reaches its
    consumer through the metadata wrap, like `stack_files`' `*cal_masters`),
    and `@node` enforces the hints — boot 81 failed all 286 calls on a
    `WithSidecar` hint. Real files in, a WBPP-ready scaled master out, with
    the fit in its sidecar and the group's STBLHASH carried over."""
    from stacks.star_flat import GAIN_SECTION, scale_star_flat
    from stacks.starflat import write_master_flat
    from stacks.star_flat_gain import GAIN_BAND_PX, GAIN_MAX_SUBS, GAIN_METHOD, GAIN_SCALE

    h, w = 2000, 2800
    rng = np.random.default_rng(7)
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(rng.standard_normal((h, w)), 60); f = 1 + 0.012 * (f - f.mean()) / f.std()
    flat_path = tmp_path / "group_flat.xisf"
    write_master_flat(f.astype(np.float32), flat_path, {"IMAGETYP": "Master Flat", "FILTER": "S",
                                                         "INSTRUME": "ZWO ASI6200MM Pro", "STBLHASH": "1234567890123456"})
    dark_path = tmp_path / "dark.fit"
    pf.PrimaryHDU(np.full((h, w), 800, np.uint16)).writeto(dark_path)
    metas = {}
    for i in range(4):
        p = tmp_path / f"L{i}.fit"
        img = 3000.0 * (1 + 1.4 * (f - 1)) * (1 + 0.1 * i) + 800 + rng.normal(0, 30, (h, w))
        pf.PrimaryHDU(np.clip(img, 0, 65535).astype(np.uint16)).writeto(p)
        metas[f"c{i}"] = dataclasses.replace(_meta(i, filt="S", w=w, h=h), filename=p)
    stack = SimpleNamespace(stable_hash="light1", content_hash="light1c", all_metadata=lambda: metas)
    star_flat = dataclasses.replace(_meta(90, image_type="Master Flat", w=w, h=h), filename=flat_path,
                                    content_hash="flat-c", stable_hash="1234567890123456")
    dark = dataclasses.replace(_meta(91, image_type="Master Dark", w=w, h=h), filename=dark_path,
                               content_hash="dark-c", stable_hash="dark1")
    out = tmp_path / "out" / "scaled.xisf"
    monkeypatch.setattr(scale_star_flat, "filename_fn", lambda *a: out)

    # The hints are the contract the runtime enforces: the masters are ImageMetadata.
    import inspect, typing
    hints = typing.get_type_hints(inspect.unwrap(scale_star_flat._function))
    assert hints["star_flat"] is ImageMetadata and hints["dark"] == typing.Optional[ImageMetadata]
    # ...and the body, unwrapped (the fake light stack is not a `Stack`).
    body = inspect.unwrap(scale_star_flat._function)
    result = body(stack, star_flat, dark, GAIN_BAND_PX, GAIN_SCALE, GAIN_MAX_SUBS, GAIN_METHOD)

    assert Path(str(result)) == out and out.exists()
    fit = get_section(out, GAIN_SECTION)
    assert abs(fit["k"] - 1.4) < 0.08 and fit["n_used"] == 4 and fit["dark"] is True
    hdr = read_header(out)
    assert str(hdr["STBLHASH"]) == "1234567890123456" and abs(float(hdr["SFLATK"]) - fit["k"]) < 1e-3
    scaled = read_image(out); scaled = scaled / np.median(scaled)
    np.testing.assert_allclose(scaled[h // 2, ::400] - 1, fit["k"] * (f[h // 2, ::400] - 1), atol=2e-3)


# ------------------------------------------------- declared content identity
def test_a_rebuilt_flat_keeps_its_identity_until_it_really_changes(tmp_path):
    """The star flat declares its own identity (CONTHASH) so that rebuilding
    it — which happens whenever ANY stack in its optical train gains subs —
    does not re-key every master built with it. Only a materially different
    flat publishes a new identity."""
    from file_io.generic import read_declared_content_hash, read_call_hash
    from stacks.star_flat import STAR_FLAT_CHANGE

    paths = _write_synthetic_subs(tmp_path / "subs", 30)
    sf = _star_flat(members=[Path(p).stem for p in paths], member_files=tuple(paths),
                    width_pixels=W, height_pixels=H)
    out = tmp_path / "flats"
    q1 = solve_and_write(sf, out / "a.xisf", cells=(128,))
    first = read_declared_content_hash(out / "a.xisf")
    assert first and q1["good"] and "carried_forward" not in q1
    assert read_call_hash(out / "a.xisf") is None           # the FRAMEWORK stamps that, not the body

    # the same data again: same pixels, so the same declared identity
    q2 = solve_and_write(sf, out / "b.xisf", cells=(128,))
    assert q2["carried_forward"] is True and q2["change_vs_standing"] <= STAR_FLAT_CHANGE
    assert read_declared_content_hash(out / "b.xisf") == first
    a = np.asarray(read_image(out / "a.xisf"), np.float64).squeeze()
    b = np.asarray(read_image(out / "b.xisf"), np.float64).squeeze()
    np.testing.assert_array_equal(a, b)                     # bit-identical: it wrote the standing pixels

    # a genuinely different flat publishes a new identity
    other = _write_synthetic_subs(tmp_path / "subs2", 30)
    for p in other:                                          # shift every scale by 5%
        t = read_star_photometry(Path(p))
        t[:, 5] *= 1.05
        write_star_photometry(Path(p), np.column_stack(
            [t[:, 0], t[:, 1], np.zeros(len(t)), np.zeros(len(t)), t[:, 2], t[:, 3],
             t[:, 4], t[:, 5], t[:, 6], t[:, 7]]), {"engine": "test"})
    moved = _star_flat(members=[Path(p).stem for p in other], member_files=tuple(other),
                       width_pixels=W, height_pixels=H)
    q3 = solve_and_write(moved, out / "c.xisf", cells=(128,))
    assert read_declared_content_hash(out / "c.xisf") is not None


def test_a_group_with_no_photometry_builds_a_flat_of_ones(tmp_path):
    """Five groups whose members yielded no photometry crashed the whole
    rebuild on boot 97: `solve_star_flat` normalises its weights by their own
    max, and an empty star table reaches numpy as max() of a zero-size array.
    The failure propagated up through mint_star_flats to build_regroup_graph
    and ended every pass with a FAILED node. A group with nothing to solve
    must build a flat of ones and say why."""
    import astropy.io.fits as pf
    paths = []
    for i in range(3):
        p = tmp_path / f"empty{i}.fit"
        pf.PrimaryHDU(np.zeros((4, 4), np.uint16)).writeto(p)
        paths.append(str(p))                       # no star_photometry section at all
    sf = _star_flat(members=[Path(p).stem for p in paths], member_files=tuple(paths),
                    width_pixels=W, height_pixels=H)
    q = solve_and_write(sf, tmp_path / "flat.xisf")
    assert q["good"] is False and q["n_subs"] == 0 and "no member sub" in q["reason"]
    img = np.asarray(read_image(tmp_path / "flat.xisf"), np.float64).squeeze()
    assert img.shape == (H, W) and np.allclose(img, np.median(img))   # featureless
