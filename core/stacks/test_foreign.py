"""Foreign stacks (`stacks.foreign`): a shared master becomes a Stack with
no subframes, admitted by its pipeline stamp; one per stable hash; it joins
the spatial group of its pointing; a pipeline never merges one it does not
admit."""
import datetime

from frozendict import frozendict

from cache.context_ref import ContextRef

from metadata.classes import MetadataDict
from stacks.foreign import (
    ForeignStack, assign_to_spatial_groups, choose_foreign_masters, is_foreign, shared_dir_of,
)
from targets.target_channel import group_stacks_by_filter
from testing.shared_masters import solved_master_metadata, write_shared_master


def test_a_foreign_stack_carries_what_the_master_says(tmp_path):
    meta = write_shared_master(tmp_path / "m.fits", observer="Stylistic7", filter="Ha", exptime=600.0)
    from file_io.sidecar import set_section
    set_section(tmp_path / "m.fits", "shared", {
        "oldest_date": "2026-09-01T03:00:00", "newest_date": "2026-10-02T05:00:00",
        "integration_s": 7200.0, "n_frames": 12, "pipeline": "Hybrid: Siril + GraXpert + OSS Mosaic",
    })
    s = ForeignStack.from_master(meta, same_pipeline=True, shared_dir=str(tmp_path))
    assert is_foreign(s)
    assert s.stable_hash == meta.header["STBLHASH"]
    assert s.filter == "Ha" and float(s.exposure) == 600.0 and float(s.integration) == 7200.0
    utc = datetime.timezone.utc
    assert s.oldest_date == datetime.datetime(2026, 9, 1, 3, tzinfo=utc) and s.newest_date == datetime.datetime(2026, 10, 2, 5, tzinfo=utc)
    assert s.oldest_date.tzinfo is not None, "comparable with local stacks' aware dates"
    assert (s.width_pixels, s.height_pixels) == (400, 300)
    assert abs(s.center[0] - 84.0) < 1e-6 and abs(s.center[1] - 34.0) < 1e-6
    assert s.healpix_cells and s.footprint is not None and s.projected_footprints
    assert s.observer == "Stylistic7" and s.same_pipeline and s.master_path == str(tmp_path / "m.fits")
    assert s.pipeline_stamp == meta.header["PIPEHASH"]
    assert not s.contents and s.state == frozendict()


def test_the_pipeline_label_is_identity(tmp_path):
    meta = solved_master_metadata(tmp_path / "m.fits")
    yes = ForeignStack.from_master(meta, same_pipeline=True, shared_dir="")
    no = ForeignStack.from_master(meta, same_pipeline=False, shared_dir="")
    assert yes.stable_hash == no.stable_hash
    assert yes.content_hash != no.content_hash


def test_one_master_per_stable_hash_ours_first(tmp_path):
    local_stamps = ("feed000000000001",)
    stale = solved_master_metadata(tmp_path / "old.fits", call_hash="a" * 16, stable_hash="s" * 16, stamp="dead000000000000")
    fresh = solved_master_metadata(tmp_path / "new.fits", call_hash="b" * 16, stable_hash="s" * 16, stamp="feed000000000001")
    other = solved_master_metadata(tmp_path / "o.fits", call_hash="c" * 16, stable_hash="t" * 16, stamp="dead000000000000")
    masters = {m.content_hash: m for m in (stale, fresh, other)}
    chosen = choose_foreign_masters(masters, tuple(sorted(masters)), local_stamps)
    assert [(m.content_hash, a) for m, a in chosen] == [(fresh.content_hash, True), (other.content_hash, False)]


def test_a_foreign_stack_joins_the_group_of_its_pointing_or_stands_alone(tmp_path):
    frame_a = solved_master_metadata(tmp_path / "fa.fits", ra=84.0, dec=34.0, call_hash="fa" * 8)
    frame_b = solved_master_metadata(tmp_path / "fb.fits", ra=120.0, dec=-20.0, call_hash="fb" * 8)
    metadata = MetadataDict({frame_a.content_hash: frame_a, frame_b.content_hash: frame_b})
    groups = [(frame_a.content_hash,), (frame_b.content_hash,)]
    near_b = solved_master_metadata(tmp_path / "nb.fits", ra=120.01, dec=-20.01, call_hash="nb" * 8)
    far = solved_master_metadata(tmp_path / "far.fits", ra=200.0, dec=10.0, call_hash="fr" * 8)
    far2 = solved_master_metadata(tmp_path / "far2.fits", ra=200.005, dec=10.0, call_hash="f2" * 8)
    pane = solved_master_metadata(tmp_path / "pane.fits", ra=200.0, dec=12.7, call_hash="pn" * 8)
    masters = {m.content_hash: m for m in (near_b, far, far2, pane)}
    by_group, own = assign_to_spatial_groups(masters, metadata, groups, overlap_fraction=0.5)
    assert by_group == {1: [near_b.content_hash]}
    assert own == [sorted([far.content_hash, far2.content_hash]), [pane.content_hash]], \
        "the two far masters share a pointing; the other pane is a pointing of its own"


def test_every_foreign_stack_reaches_the_channels(tmp_path):
    meta = solved_master_metadata(tmp_path / "m.fits")
    ours = ForeignStack.from_master(meta, same_pipeline=True, shared_dir="")
    other = ForeignStack.from_master(
        solved_master_metadata(tmp_path / "p.fits", stable_hash="p" * 16, call_hash="p" * 16),
        same_pipeline=False, shared_dir="")
    assert group_stacks_by_filter([other, ours]) == {"H": sorted([ours, other], key=lambda s: s.stable_hash)}


def test_a_master_from_another_program_is_named_by_its_pixels(tmp_path):
    meta = solved_master_metadata(tmp_path / "wbpp.fits", call_hash="p1x3l" + "0" * 11, stable_hash="unused")
    for k in ("PIPEHASH", "CALLHASH", "STBLHASH"):
        del meta.header[k]
    import dataclasses
    meta = dataclasses.replace(meta, stable_hash=meta.content_hash)   # as compute_image_metadata leaves it
    s = ForeignStack.from_master(meta, same_pipeline=False, shared_dir="")
    assert s.stable_hash == meta.content_hash and s.pipeline_stamp == "" and not s.same_pipeline
    chosen = choose_foreign_masters({meta.content_hash: meta}, (meta.content_hash,), ("x",))
    assert chosen == [(meta, False)]


def test_shared_dir_of_finds_the_configured_folder():
    assert shared_dir_of("/Users/w/Dropbox/Orion/abc.xisf", ("/Users/w/Dropbox/Orion", "/x")) == "/Users/w/Dropbox/Orion"
    assert shared_dir_of("/elsewhere/abc.xisf", ("/Users/w/Dropbox/Orion",)) == ""


class _Ref(ContextRef):
    """A ContextRef holding its value in-process: no actor, no cache."""
    def __init__(self, value, content_hash=None):
        super().__init__(cache_key=("test", id(value)),
                         content_hash=content_hash or getattr(value, "content_hash", "ref"))
        self._value = value

    def get(self):
        return self._value


def test_identify_foreign_stacks_stores_one_stack_per_stable_hash(tmp_path, monkeypatch):
    import stacks.foreign as foreign
    stored = {}
    monkeypatch.setattr(foreign.ContextRef, "store",
                        classmethod(lambda cls, key, obj, content_hash=None: stored.setdefault(key, _Ref(obj))))
    stale = solved_master_metadata(tmp_path / "a.fits", call_hash="a" * 16, stable_hash="s" * 16, stamp="dead000000000000")
    fresh = solved_master_metadata(tmp_path / "b.fits", call_hash="b" * 16, stable_hash="s" * 16, stamp="feed000000000001")
    masters = {m.content_hash: m for m in (stale, fresh)}
    out = foreign.identify_foreign_stacks._function(
        _Ref(masters), tuple(sorted(masters)), ("feed000000000001",), (str(tmp_path),))
    assert list(out) == ["s" * 16]
    stack = out["s" * 16].get()
    assert isinstance(stack, ForeignStack) and stack.same_pipeline and stack.master_path == str(tmp_path / "b.fits")
    assert stack.shared_dir == str(tmp_path)
    assert set(stored) == {("stacks", "s" * 16)}


def test_build_shared_library_reads_sidecars_only(tmp_path, monkeypatch):
    import metadata.shared_masters as sm
    monkeypatch.setattr(sm.ContextRef, "store_large",
                        classmethod(lambda cls, obj, identity=None: _Ref(obj, str(identity))))
    good = write_shared_master(tmp_path / "g.fits", call_hash="a" * 16, stable_hash="1" * 16)
    before = sorted(p.name for p in tmp_path.iterdir())
    record = sm.scan_shared_dir(tmp_path)
    ref = sm.build_shared_library._function(str(tmp_path), record.fingerprint, record.listing)
    assert set(ref.get()) == {good.content_hash}
    assert sorted(p.name for p in tmp_path.iterdir()) == before, "nothing written into the shared folder"
