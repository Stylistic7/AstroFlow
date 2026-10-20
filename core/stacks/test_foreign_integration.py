"""End to end on a Ray test context: a master in a shared master folder, of
the pointing our frames cover, is identified as a foreign stack in OUR
target, admitted when its pipeline stamp is one our own masters carry, and
only the admitted one reaches a pipeline's channel; our own master is
published into the folder on request, and the folder is then current."""
from dataclasses import replace
from pathlib import Path

from frozendict import frozendict

from compute_graph import build_compute_graph
from context import context
from decisions.decisions import DecisionKey, DecisionOverride, ObjectType
from dependency_graph import run_graph
from file_io.sidecar import get_section
from stacks.foreign import ForeignStack, local_pipeline_stamps
from stacks.publish_shared import SHARED, publish_shared_masters, shared_dirs_changed
from testing import make_synthetic_frames
from testing.shared_masters import write_shared_master

OURS, THEIRS, OTHER = "5" * 16, "e" * 16, "c" * 16


def test_a_shared_master_of_our_pointing_joins_our_target_and_is_admitted_by_its_stamp(test_context, tmp_path):
    import math
    make_synthetic_frames(ra=100, dec=50, prefix="foo", number=10, dither=0, dest_dir=context.input_dirs[0])
    share = tmp_path / "share"
    share.mkdir()
    context.shared_dirs = [share]

    # Pass 1: our own stack, from the synthetic frames. Its solved geometry is
    # what the foreign masters must match to share its target.
    run_graph(build_compute_graph())
    ((local_id, local_ref),) = context.stacks.items()
    mine = local_ref.get()
    geometry = dict(ra=mine.center[0] + 0.0005, dec=mine.center[1] + 0.0005, scale=float(mine.image_scale),
                    nx=mine.width_pixels, ny=mine.height_pixels, rotation_deg=math.degrees(mine.rotation_angle))

    # A post-processed master of ours, in the version log, carrying the stamp
    # our pipeline writes: the admission reference.
    ours = write_shared_master(Path(context.output_dir) / "post_stack_calibrate" / ("0" * 16 + ".fits"),
                               **geometry, call_hash="0" * 16, stable_hash="1" * 16, stamp=OURS)
    context[("product_versions", "Stack", "1" * 16, "P", "Post-Processed", "H")] = [
        {"path": str(ours.filename), "call_hash": "0" * 16, "ts": 0.0}]
    assert local_pipeline_stamps() == (OURS,)

    # Two of theirs on our pointing: one by our pipeline, one by another.
    write_shared_master(share / ("f" * 16 + ".fits"), **geometry,
                        call_hash="f" * 16, stable_hash=THEIRS, stamp=OURS, observer="A friend")
    write_shared_master(share / ("d" * 16 + ".fits"), **geometry,
                        call_hash="d" * 16, stable_hash=OTHER, stamp="unknown" + "0" * 9)

    # Pass 2: the folder changed, so a pass picks them up.
    run_graph(build_compute_graph())

    stacks = {h: ref.get() for h, ref in context.stacks.items()}
    foreign = {h: s for h, s in stacks.items() if isinstance(s, ForeignStack)}
    local = {h: s for h, s in stacks.items() if not isinstance(s, ForeignStack)}
    assert len(local) == 1 and set(foreign) == {THEIRS, OTHER}
    assert foreign[THEIRS].admitted and not foreign[OTHER].admitted
    assert foreign[THEIRS].observer == "A friend" and foreign[THEIRS].shared_dir == str(share)
    assert not foreign[THEIRS].contents

    targets = {h: ref.get() for h, ref in context.targets.items()}
    assert len(targets) == 1, "same pointing, sensor, scale and rotation: one target, not " + str(
        [(t.stack_ids, t.sensor, round(float(t.image_scale), 4), round(math.degrees(t.rotation_angle), 2))
         for t in targets.values()])
    (target,) = targets.values()
    assert set(target.stack_ids) == set(local) | set(foreign)
    assert {s.stable_hash for s in target.stacks_for("H")} == set(local) | {THEIRS}, \
        "only the admitted foreign stack reaches a pipeline's channel"

    ui = context.ui_stacks
    assert ui[THEIRS]["foreign"] and ui[THEIRS]["admitted"] and ui[THEIRS]["shared_dir"] == str(share)
    assert ui[OTHER]["foreign"] and not ui[OTHER]["admitted"]
    assert not ui[local_id]["foreign"]

    # Publishing is opt-in: nothing of ours is in the folder until we say so.
    publish_shared_masters(context)
    assert not (share / ("0" * 16 + ".fits")).exists()
    context[("product_versions", "Stack", local_id, "P", "Post-Processed", "H")] = [
        {"path": str(ours.filename), "call_hash": "0" * 16, "ts": 0.0}]
    context.settings = replace(context.settings, decision_overrides=frozendict({
        DecisionKey.sharing(ObjectType.STACK, local_id, str(share)): DecisionOverride(choice=SHARED)}))
    publish_shared_masters(context)
    copy = share / ("0" * 16 + ".fits")
    assert copy.exists() and get_section(copy, "shared")["pipeline"] == "P"
    assert get_section(copy, "plate_solution")["data"] is not None
    assert not shared_dirs_changed(context), "our own publish is not a change worth a drain"

    # Withdrawing retires the copy.
    context.settings = replace(context.settings, decision_overrides=frozendict())
    publish_shared_masters(context)
    assert not copy.exists() and not (share / ("0" * 16 + ".fits.json")).exists()
    assert (share / ("f" * 16 + ".fits")).exists(), "theirs is not ours to touch"
