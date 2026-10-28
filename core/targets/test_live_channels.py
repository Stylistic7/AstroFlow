"""Quarantine settled at trace time (`targets/target_channel.live_channels`)
and all the way down at integration: a stack with every run out is a member
of no channel, so the panel simply lacks that filter instead of promising a
master the integration can only refuse — which held every finish of e308 at
the previous generation (2026-10-27). A STACK key takes every run with it
(before 2026-10-28 it took only the run carrying the stack's id)."""
import datetime

from frozendict import frozendict

from compute_graph import build_compute_graph
from context import context
from decisions.decisions import DecisionKey, DecisionType, ObjectType
from decisions.helpers import exclude
from decisions.resolve import resolve_decisions
from dependency_graph import run_graph
from testing import make_synthetic_frames


def _frames(prefix, filter, day):
    make_synthetic_frames(
        ra=100, dec=50, prefix=prefix, number=5, dither=0, filter=filter,
        start_datetime=datetime.datetime(2023, 12, day, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )


def _run_key(stack, run):
    return DecisionKey.quarantine(ObjectType.STACK, run, parent_stack_id=stack.stable_hash)


def test_quarantine_drops_runs_stacks_and_channels(test_context):
    _frames("h1", "H", 1)
    _frames("h2", "H", 5)     # a second session → a second run in the H stack
    _frames("o1", "O", 1)
    run_graph(build_compute_graph())
    assert len(context.targets) == 1
    target = context.targets[next(iter(context.targets))].get()
    assert target.filters == ["H", "O"]
    (h_stack,), (o_stack,) = target.stacks_for("H"), target.stacks_for("O")
    assert len(h_stack.contents) == 2 and h_stack.stable_hash in h_stack.contents
    other = next(r for r in h_stack.contents if r != h_stack.stable_hash)

    # A STACK key takes every run with it at integration.
    stack_key = frozendict(exclude(DecisionKey.quarantine(ObjectType.STACK, h_stack.stable_hash)))
    assert resolve_decisions(h_stack, (DecisionType.QUARANTINE,), stack_key, 1) is None

    # One run keyed: the stack keeps its channel with the other run.
    one = frozendict(exclude(_run_key(h_stack, h_stack.stable_hash)))
    left = resolve_decisions(h_stack, (DecisionType.QUARANTINE,), one, 1)
    assert set(left.contents) == {other}
    assert target.channels(one) == {"H": (h_stack,), "O": (o_stack,)}

    # Every run keyed: the channel is gone; the target keeps O; identity untouched.
    both = frozendict(exclude(_run_key(h_stack, h_stack.stable_hash)) | exclude(_run_key(h_stack, other)))
    assert target.channels(both) == {"O": (o_stack,)}
    assert h_stack.stable_hash in target.stack_ids

    # Every stack out: the target has no channels, and a key of its own says the same.
    all_out = frozendict(dict(both) | exclude(DecisionKey.quarantine(ObjectType.STACK, o_stack.stable_hash)))
    assert target.channels(all_out) == {}
    assert target.channels(frozendict(exclude(
        DecisionKey.quarantine(ObjectType.TARGET, target.stable_hash)))) == {}
