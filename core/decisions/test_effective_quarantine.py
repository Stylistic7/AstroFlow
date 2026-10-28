"""`decisions/quarantine.py`: a key of the object's own, or every child's,
run → stack → target → mosaic — and nothing fetched that no key names."""
from types import SimpleNamespace

from frozendict import frozendict

from decisions.decisions import DecisionKey, ObjectType
from decisions.helpers import exclude
from decisions.quarantine import Quarantine


def _stack(stack_id, *run_ids):
    return SimpleNamespace(stable_hash=stack_id, contents={r: None for r in (stack_id, *run_ids)})


S1 = _stack("s1", "s1b")          # two runs; "s1" carries the stack's id
S2 = _stack("s2")                  # one run
STACKS = {s.stable_hash: s for s in (S1, S2)}
T1 = SimpleNamespace(stable_hash="t1", stack_ids=("s1", "s2"))
T2 = SimpleNamespace(stable_hash="t2", stack_ids=("s2",))
PANELS = tuple((t.stable_hash, t.stack_ids) for t in (T1, T2))


def _run_key(stack_id, run_id):
    return DecisionKey.quarantine(ObjectType.STACK, run_id, parent_stack_id=stack_id)


def _quarantine(*keys, fetched=None):
    overrides = {}
    for k in keys:
        overrides |= exclude(k)

    def stack_of(sid):
        if fetched is not None:
            fetched.append(sid)
        return STACKS[sid]
    return Quarantine(frozendict(overrides), stack_of)


def test_a_stack_key_puts_every_run_out():
    q = _quarantine(DecisionKey.quarantine(ObjectType.STACK, "s1"))
    assert q.runs_of(S1) == {"s1", "s1b"}
    assert q.stack("s1") and not q.stack("s2")


def test_run_keys_put_only_those_runs_out_until_none_is_left():
    one = _quarantine(_run_key("s1", "s1"))
    assert one.runs_of(S1) == {"s1"}
    assert not one.stack("s1"), "the other run still integrates"
    both = _quarantine(_run_key("s1", "s1"), _run_key("s1", "s1b"))
    assert both.stack("s1"), "no run left: the stack is quarantined"


def test_a_target_is_out_by_its_key_or_by_every_stack():
    assert _quarantine(DecisionKey.quarantine(ObjectType.TARGET, "t1")).target("t1", T1.stack_ids)
    only_s2 = _quarantine(DecisionKey.quarantine(ObjectType.STACK, "s2"))
    assert not only_s2.target("t1", T1.stack_ids)
    assert only_s2.target("t2", T2.stack_ids)
    both = _quarantine(DecisionKey.quarantine(ObjectType.STACK, "s2"),
                       _run_key("s1", "s1"), _run_key("s1", "s1b"))
    assert both.target("t1", T1.stack_ids)


def test_a_mosaic_is_out_by_its_key_or_by_every_target():
    assert _quarantine(DecisionKey.quarantine(ObjectType.MOSAIC, "m")).mosaic("m", PANELS)
    assert not _quarantine(DecisionKey.quarantine(ObjectType.TARGET, "t1")).mosaic("m", PANELS)
    assert _quarantine(DecisionKey.quarantine(ObjectType.TARGET, "t1"),
                       DecisionKey.quarantine(ObjectType.STACK, "s2")).mosaic("m", PANELS)


def test_only_stacks_that_run_keys_name_are_fetched():
    fetched = []
    q = _quarantine(DecisionKey.quarantine(ObjectType.STACK, "s2"), _run_key("s1", "s1"),
                    fetched=fetched)
    assert q.mosaic("m", PANELS) is False
    assert fetched == ["s1"], "s2 is answered by its key; s1 needs its runs"


def test_nothing_is_quarantined_without_a_key():
    q = _quarantine()
    assert q.runs_of(S1) == frozenset() and not q.stack("s1")
    assert not q.target("t1", T1.stack_ids) and not q.mosaic("m", PANELS)
