"""Publishing into a shared master folder (`stacks.publish_shared`): which
targets, and the folder brought to hold exactly our current masters —
copied with the producer's sidecar plus a `shared` section, retired when
superseded, and never anyone else's file."""
from pathlib import Path

from frozendict import frozendict

from decisions.decisions import DecisionKey, DecisionOverride, ObjectType
from file_io.sidecar import get_section
from stacks.publish_shared import SHARED, reconcile_folder, sharing_choices, stacks_shared_into
from testing.shared_masters import write_shared_master


def test_sharing_choices_come_from_the_decisions():
    overrides = frozendict({
        DecisionKey.sharing(ObjectType.TARGET, "t1", "/share"): DecisionOverride(choice=SHARED),
        DecisionKey.sharing(ObjectType.TARGET, "t1", "/other"): DecisionOverride(choice=SHARED),
        DecisionKey.sharing(ObjectType.STACK, "s9", "/share"): DecisionOverride(choice=SHARED),
        DecisionKey.mosaic_membership("t3"): DecisionOverride(choice="standalone"),
    })
    assert sharing_choices(overrides) == {("target", "t1"): {"/share", "/other"}, ("stack", "s9"): {"/share"}}


def test_a_marked_mosaic_target_or_stack_sends_its_local_stacks_only():
    stacks = {
        "mine1": {"foreign": False}, "mine2": {"foreign": False}, "mine3": {"foreign": False},
        "mine4": {"foreign": False}, "theirs": {"foreign": True, "shared_dir": "/share"},
    }
    targets = {
        "t1": {"stack_ids": ["mine1", "theirs"]},
        "t2": {"stack_ids": ["mine2"]},
        "t3": {"stack_ids": ["mine3"]},
        "t4": {"stack_ids": ["mine4"]},
    }
    mosaics = {"m1": {"target_ids": ["t1", "t2"]}}
    choices = {("mosaic", "m1"): {"/share"}, ("target", "t3"): {"/other"}, ("stack", "mine4"): {"/share", "/other"}}
    assert stacks_shared_into("/share", mosaics, targets, stacks, choices) == {"mine1", "mine2", "mine4"}
    assert stacks_shared_into("/other", mosaics, targets, stacks, choices) == {"mine3", "mine4"}
    assert stacks_shared_into("/share", mosaics, targets, stacks, {}) == set()


def test_reconcile_copies_new_masters_with_their_sidecars_and_retires_superseded_ones(tmp_path):
    out, share = tmp_path / "out", tmp_path / "share"
    share.mkdir()
    m1 = write_shared_master(out / "post_stack_calibrate" / ("1" * 16 + ".fits"),
                             call_hash="1" * 16, stable_hash="s" * 16, observer="Stylistic7")
    ui_stacks = {"s" * 16: {"oldest_date": "2026-09-01 03:00:00", "newest_date": "2026-10-02 05:00:00",
                            "integration": 7200.0, "filenames": ["a", "b", "c"]}}
    published = reconcile_folder(share, [("s" * 16, "Siril", Path(m1.filename))], {}, ui_stacks)
    assert published == {"1" * 16: "1" * 16 + ".fits"}
    copy = share / ("1" * 16 + ".fits")
    assert copy.exists() and copy.stat().st_size == Path(m1.filename).stat().st_size
    assert get_section(copy, "plate_solution")["data"] is not None
    assert get_section(copy, "image_metadata")["content_hash"] == "1" * 16
    shared = get_section(copy, "shared")
    assert shared["observer"] == "Stylistic7" and shared["integration_s"] == 7200.0 and shared["n_frames"] == 3
    assert shared["pipeline"] == "Siril" and shared["oldest_date"] == "2026-09-01 03:00:00"
    from file_io.sidecar import get_file_size
    assert get_file_size(copy) == copy.stat().st_size

    # A second pass with the same master is a no-op; a re-stack retires the old file.
    assert reconcile_folder(share, [("s" * 16, "Siril", Path(m1.filename))], published, ui_stacks) == published
    m2 = write_shared_master(out / "post_stack_calibrate" / ("2" * 16 + ".fits"),
                             call_hash="2" * 16, stable_hash="s" * 16)
    published = reconcile_folder(share, [("s" * 16, "Siril", Path(m2.filename))], published, ui_stacks)
    assert published == {"2" * 16: "2" * 16 + ".fits"}
    assert not copy.exists() and not (share / ("1" * 16 + ".fits.json")).exists()
    assert (share / ("2" * 16 + ".fits")).exists()


def test_reconcile_never_touches_files_it_did_not_publish(tmp_path):
    share = tmp_path / "share"
    theirs = write_shared_master(share / ("t" * 16 + ".fits"), call_hash="t" * 16, stable_hash="s" * 16)
    published = reconcile_folder(share, [], {}, {})
    assert published == {}
    assert Path(theirs.filename).exists()


def test_an_unstamped_or_unsolved_master_is_not_published(tmp_path):
    out, share = tmp_path / "out", tmp_path / "share"
    share.mkdir()
    from testing.shared_masters import solved_master_metadata
    meta = solved_master_metadata(out / "u.fits", call_hash="u" * 16, stable_hash="s" * 16)
    del meta.header["PIPEHASH"]
    write_shared_master(out / "u.fits", meta)
    assert reconcile_folder(share, [("s" * 16, "Siril", out / "u.fits")], {}, {}) == {}
    assert not (share / "u.fits").exists()
