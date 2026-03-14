import ray

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import copy_testdata_frames
from stacks.composite_stack import CompositeStack
from testing import copy_testdata_into_cache
from targets.identification import identify_targets


def test_orion_nebula(test_context):
    copy_testdata_frames(target_name="M42", dest_dir=context.input_dirs[0])

    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.stacks) == 1
    stack = next(
        iter(
            [
                s
                for s in context.stacks.values()
                if isinstance(s, CompositeStack)
            ]
        )
    )
    assert stack.stable_hash != None
    assert stack.content_hash != None

    assert len(context.targets) == 1
    target = next(iter(context.targets.values()))
    assert target.stable_hash == stack.stable_hash


# def test_temperature_difference(test_context) -> None:
#     copy_testdata_into_cache()

#     stack_a = context.stacks["bedcbfed642ec2b9"]
#     stack_b = context.stacks["24059224549fd764"]

#     print(
#         f"stack_a {stack_a.camera} {stack_a.width_pixels} {stack_a.height_pixels} {stack_a.image_scale}"
#     )
#     print(
#         f"stack_b {stack_b.camera} {stack_b.width_pixels} {stack_b.height_pixels} {stack_b.image_scale}"
#     )

#     targets = ray.get(identify_targets.remote(context.stacks))

#     for t in targets.values():
#         print(f"### test_temperature_difference target {t.stable_hash}")
