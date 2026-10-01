"""A run-level quarantine is keyed as RUN, so the run that carries the stack's
own id (the one holding its oldest subframe) can be quarantined without the
key colliding with the stack-level key -- which used to quarantine the whole
stack, every run included (2026-10-01, Lion Nebula H)."""
import datetime

from frozendict import frozendict

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_synthetic_frames
from decisions.decisions import DecisionKey, DecisionType, ObjectType
from decisions.helpers import exclude
from decisions.compute_graph import _is_quarantined
from decisions.resolve import is_run_included, resolve_decisions


def test_run_key_is_distinct_from_the_stack_key_even_for_the_identity_run():
    stack_id = "f6af072a4bac1b8d"
    run_key = DecisionKey.quarantine(ObjectType.STACK, stack_id, parent_stack_id=stack_id)
    stack_key = DecisionKey.quarantine(ObjectType.STACK, stack_id)
    assert run_key.object_type is ObjectType.RUN
    assert run_key.parent_stack_id == stack_id and run_key.subject_id == stack_id
    assert run_key != stack_key
    assert not _is_quarantined(frozendict(exclude(run_key)), ObjectType.STACK, stack_id)


def test_quarantining_the_identity_run_keeps_the_other_run(test_context):
    for day in (1, 5):   # two sessions -> two runs in one stack
        make_synthetic_frames(
            ra=100, dec=50, prefix=f"d{day}", number=5, dither=0,
            start_datetime=datetime.datetime(2023, 12, day, 20, 0, 0),
            dest_dir=context.input_dirs[0],
        )
    run_graph(build_compute_graph())
    assert len(context.stacks) == 1
    stack_id = next(iter(context.stacks))
    stack = context.stacks[stack_id].get()
    assert stack_id in stack.contents, "the stack's id is its oldest run's id"
    assert len(stack.contents) == 2
    other = next(r for r in stack.contents if r != stack_id)

    overrides = frozendict(exclude(
        DecisionKey.quarantine(ObjectType.STACK, stack_id, parent_stack_id=stack_id)
    ))
    assert not _is_quarantined(overrides, ObjectType.STACK, stack_id)
    assert not is_run_included(stack_id, stack, overrides)
    assert is_run_included(other, stack, overrides)
    resolved = resolve_decisions(stack, (DecisionType.QUARANTINE,), overrides, min_frames_in_stack=1)
    assert resolved is not None and set(resolved.contents) == {other}

    # The trash collector sees only the quarantined run's frames.
    from dataclasses import replace
    from web_server.backend.settings import _quarantined_frame_files
    context.settings = replace(context.settings, decision_overrides=overrides)
    trashed = {p.name for p in _quarantined_frame_files()}
    from pathlib import Path
    assert trashed == {
        Path(fm.filename).name for fm in stack.contents[stack_id].metadata.values()
    }
