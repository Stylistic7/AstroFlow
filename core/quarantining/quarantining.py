from dataclasses import replace, asdict
from collections.abc import Sequence
from frozendict import frozendict

from context import context
from stacks.composite_stack import Stack
from stacks.simple_stack import RunsDict
from attributes import ImagingAttributes
from stacks.types import StackHash
from beartype import beartype


# TODO: Ensure that all of the actions in this file are idempotent. The goal
# is to be able to apply these functions very quickly when the user quarantines
# something in the UI.
@beartype
def apply_quarantine(
    stack: Stack, quarantined_substacks: Sequence[StackHash] | set[StackHash]
) -> Stack | None:
    new_contents = RunsDict(
        {
            h: s
            for h, s in stack.contents.items()
            if h not in quarantined_substacks
        }
    )

    if len(new_contents) == 0:
        return None

    for substack_id, substack in new_contents.items():
        new_dependencies = frozendict(
            {
                t: s
                for t, s in substack.dependencies.items()
                if s not in quarantined_substacks
            }
        )
        new_contents[substack_id] = replace(
            substack, dependencies=new_dependencies
        )

    all_frames = []
    for substack in new_contents.values():
        all_frames.extend(substack.frames)

    if not all_frames:
        return None

    # TODO: The content_hash needs to be re-computed.
    # TODO: Should other attributes be re-computed?
    attributes = asdict(
        ImagingAttributes.from_frames(all_frames, stack.all_metadata())
    )
    return replace(stack, contents=new_contents, **attributes)
