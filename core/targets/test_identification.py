import unittest

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import copy_testdata_frames
from stacks.composite_stack import CompositeStack


def test_orion_nebula(test_context):
    copy_testdata_frames(target_name="M42", dest_dir=context.input_dirs[0])

    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.stacks) == 1
    stack = next(
        iter([s for s in context.stacks.values() if isinstance(s, CompositeStack)])
    )
    assert stack.stable_hash != None
    assert stack.content_hash != None

    assert len(context.targets) == 1
    target = next(iter(context.targets.values()))
    assert target.stable_hash == stack.stable_hash
