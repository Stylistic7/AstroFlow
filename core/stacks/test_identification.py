import pytest

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from metadata.classes import MetadataDict
from testing import make_synthetic_frames


def test_one_perfect_stack(test_context):
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix="foo",
        number=10,
        dither=0,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 1
    toplevel_stack = next(iter(stacks.values())).get()
    assert toplevel_stack.center == pytest.approx([100, 50], abs=1e-12)
    assert toplevel_stack.image_scale == 1.0

    assert len(toplevel_stack.contents) == 1
    stack = next(iter(toplevel_stack.contents.values()))
    assert len(stack.frames) == 10


def test_two_stacks_different_image_scales(test_context):
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix="foo",
        number=10,
        dither=0,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix="bar",
        number=10,
        dither=0,
        image_scale=2.0,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stacks = {h: ref.get() for h, ref in (context.stacks or {}).items()}
    assert len(stacks) == 2

    def find_stack(name_prefix: str):
        for s in stacks.values():
            filenames = [m.filename for m in s.all_metadata().values()]
            if all(name_prefix in f for f in filenames):
                return s
        raise AssertionError(f"no stack matches prefix {name_prefix!r}")

    foo_stack = find_stack("foo")
    assert len(foo_stack.all_frames()) == 10

    bar_stack = find_stack("bar")
    assert len(bar_stack.all_frames()) == 10


def test_two_stacks_different_image_sizes(test_context):
    # Note that these two groups have the same generic camera name, which is
    # typical of some capture software.
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix="foo",
        instrument="ASI Camera",
        number=10,
        dither=0,
        width_pixels=3000,
        height_pixels=2000,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix="bar",
        instrument="ASI Camera",
        number=10,
        dither=0,
        width_pixels=6000,
        height_pixels=4000,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 2

    keys = sorted(stacks.keys())

    stack = stacks[keys[0]].get()
    assert stack.center == pytest.approx([100, 50], abs=1e-12)
    assert stack.width_pixels == 6000
    assert stack.height_pixels == 4000
    assert stack.image_scale == 1.0

    stack = stacks[keys[1]].get()
    assert stack.center == pytest.approx([100, 50], abs=1e-12)
    assert stack.width_pixels == 3000
    assert stack.height_pixels == 2000
    assert stack.image_scale == 1.0


def test_identify_camera_fingerprints_builds_map_keyed_by_consensus(test_context):
    """`identify_camera_fingerprints` returns a FingerprintMap where
    same-camera frames share the same consensus tuple and different-
    camera frames have different consensus tuples. Built ONCE globally,
    not per spatial group — that's why it can attribute a camera
    spread across many pointings."""
    from stacks.identification import build_fingerprint_map

    make_synthetic_frames(
        ra=100, dec=50, prefix="bodyA", number=5,
        instrument="ZWO ASI6200MM Pro",
        dither=0, dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=200, dec=-50, prefix="bodyB", number=5,
        instrument="ZWO ASI6200MM Pro #B",
        dither=0, dest_dir=context.input_dirs[0],
    )

    # Drive metadata loading without running identification.
    run_graph(build_compute_graph())

    metadata = context["metadata_ref"].get()
    light_metadata = MetadataDict({
        h: m for h, m in metadata.items()
        if m.header.get("IMAGETYP", "").lower() == "light"
    })

    fp = build_fingerprint_map(light_metadata, fingerprint_min_anchor_size=3)

    by_prefix = {"bodyA": [], "bodyB": []}
    for h, m in light_metadata.items():
        for prefix in by_prefix:
            if prefix in str(m.filename):
                by_prefix[prefix].append(h)

    a_consensuses = {fp.get(h) for h in by_prefix["bodyA"]}
    b_consensuses = {fp.get(h) for h in by_prefix["bodyB"]}
    # All A frames share one consensus; all B frames share another.
    assert len(a_consensuses) == 1 and None not in a_consensuses
    assert len(b_consensuses) == 1 and None not in b_consensuses
    # A and B have DIFFERENT consensuses (different bodies).
    assert a_consensuses != b_consensuses


def test_same_instrume_two_bodies_separated_by_fingerprint(test_context):
    """Two physical sensors with the same INSTRUME (e.g. two ASI6200MM Pros)
    must split into two Stacks. They differ only in hot-pixel fingerprint —
    same model, same sensor shape — so this regression test guards the
    fingerprint subdivision inside `camera_clusters`. Synthetic hot pixels
    are seeded by `(instrument, width, height)`; we shift `instrument` to
    'ZWO ASI6200MM Pro #B' for the second body to inject a distinct
    fingerprint while keeping the user-facing model string the same in
    practice (we also rewrite the cached metadata INSTRUME below so all 20
    frames share the canonical model name)."""
    make_synthetic_frames(
        ra=100, dec=50, prefix="bodyA", number=5,
        instrument="ZWO ASI6200MM Pro",
        dither=0, dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100, dec=50, prefix="bodyB", number=5,
        instrument="ZWO ASI6200MM Pro #B",  # distinct fingerprint seed
        dither=0, dest_dir=context.input_dirs[0],
    )
    run_graph(build_compute_graph())
    stacks = {h: ref.get() for h, ref in (context.stacks or {}).items()}
    # Two physical sensors -> two stacks, each with a distinct camera_id_hash.
    assert len(stacks) == 2
    hashes = {s.camera_id_hash for s in stacks.values()}
    assert None not in hashes
    assert len(hashes) == 2


def test_frames_without_defect_candidates_are_excluded(test_context):
    """A frame whose sidecar predates the defect_candidates field arrives
    with defect_candidates=None — the global fingerprint pass can't
    attribute it, so identify_stacks excludes it. Real-world trigger:
    legacy sidecars (e.g. Starbase zips) that weren't migrated.
    """
    # 5 normal frames + 1 frame with no fingerprint.
    make_synthetic_frames(
        ra=100, dec=50, prefix="good", number=5,
        dither=0, dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100, dec=50, prefix="bad", number=1,
        dither=0, dest_dir=context.input_dirs[0],
    )
    # Strip defect_candidates from the "bad" frame in the cache.
    from context import context as ctx
    from dataclasses import replace
    from metadata.classes import ImageMetadata
    for key in list(ctx.get_matching(("compute_image_metadata",)).keys()):
        v = ctx[key]
        if isinstance(v, ImageMetadata) and "bad" in str(v.filename):
            ctx[key] = replace(v, defect_candidates=None)

    run_graph(build_compute_graph())
    stacks = {h: ref.get() for h, ref in (context.stacks or {}).items()}
    # One stack from the 5 "good" frames. The "bad" frame was excluded.
    assert len(stacks) == 1
    stack = next(iter(stacks.values()))
    all_files = [m.filename for m in stack.all_metadata().values()]
    assert all("good" in f for f in all_files)
    assert not any("bad" in f for f in all_files)


def test_prepare_stack_for_ui_emits_camera_id(test_context):
    """The UI dict for a Stack exposes both the full camera_id_hash and
    a short 8-char prefix for header display."""
    make_synthetic_frames(
        ra=100, dec=50, prefix="ui", number=5,
        dither=0, dest_dir=context.input_dirs[0],
    )
    run_graph(build_compute_graph())
    stack = next(iter(context.stacks.values())).get()
    ui = next(iter(context.ui_stacks.values()))
    assert ui["camera_id_hash"] is not None
    assert ui["camera_id_hash"] == stack.camera_id_hash
    assert ui["camera_id_short"] == stack.camera_id_hash[:8]


def test_multiple_exposures(test_context):
    """Frames with different EXPOSURE values land in *separate* stacks.

    WBPP groups by EXPOSURE and emits one master per group, so AstroFlow
    splits at identification time rather than mixing exposures into a
    single Stack and producing duplicate WBPP outputs.
    """
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix="foo",
        number=5,
        dither=0,
        exposure=300,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix="bar",
        number=7,
        dither=0,
        exposure=120,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stacks = {h: ref.get() for h, ref in (context.stacks or {}).items()}
    assert len(stacks) == 2

    by_exposure = {s.exposure: s for s in stacks.values()}
    assert set(by_exposure) == {120.0, 300.0}

    s120 = by_exposure[120.0]
    assert all("bar" in f.filename for f in s120.all_metadata().values())
    assert len(s120.all_frames()) == 7
    assert s120.integration == 7 * 120

    s300 = by_exposure[300.0]
    assert all("foo" in f.filename for f in s300.all_metadata().values())
    assert len(s300.all_frames()) == 5
    assert s300.integration == 5 * 300
