from typing import NewType, Optional, Any
from dataclasses import dataclass, asdict, fields

from hashing import hash_dataclass, hash_list_int
from stacks.types import StackHash
from stacks.composite_stack import Stack
from attributes import (
    AstrometricAttributes,
    ChronologyAttributes,
)
from attributes.astrometric_attributes import covering_polygon
from cache.content_hashed import ContentHashed
from cache.context_ref import ContextRef
from beartype import beartype

TargetHash = NewType("TargetHash", str)


# Which channel frames the target's canvas. H for narrowband, L for LRGB, G
# for RGB — the channel carrying the structure, so it is the one that comes
# out axis-aligned and unrotated.
#
# This MUST be a fixed rule, not a data-dependent one. It feeds
# `compute_viewbox`, so anything that can flip as data arrives — "most
# integration time", say — would silently re-key the target and rebuild every
# placement and colour product the night a deeper panel lands. Unknown filters
# fall back to alphabetical, which cannot flip either.
_FILTER_PRIORITY = ("H", "L", "G", "S", "O", "R", "B")


def reference_stack(stacks: list[Stack]) -> Stack:
    """The stack whose rotation and image scale frame the target."""
    def rank(stack: Stack) -> tuple[int, str]:
        filter = str(stack.filter or "")
        known = (_FILTER_PRIORITY.index(filter)
                 if filter in _FILTER_PRIORITY else len(_FILTER_PRIORITY))
        return (known, filter)

    return min(stacks, key=rank)


@beartype
@dataclass(order=True, frozen=True)
class Target(
    ContentHashed, AstrometricAttributes, ChronologyAttributes[TargetHash]
):
    stack_ids: tuple[StackHash, ...]

    @classmethod
    def from_stacks(cls, stacks: list[Stack]) -> Optional["Target"]:
        chronology_attributes = ChronologyAttributes.from_objects(stacks)
        astrometric_attributes = AstrometricAttributes.from_objects(
            stacks, reference_stack(stacks), covering_polygon)

        from hashing import hash_list
        stable_hash = sorted(stacks, key=lambda s: s.oldest_date)[0].stable_hash
        stack_ids = tuple(s.stable_hash for s in stacks)
        # content_hash composes the children's CONTENT hashes (mirror
        # Stack.content_hash), NOT their stable/identity hashes — so a Target's
        # content_hash changes whenever any member stack's *content* changes, not
        # only when the set of member identities changes. (stack_ids stays the
        # identity list.)
        content_hash = hash_list(sorted(s.content_hash for s in stacks))

        attributes: dict[str, Any] = (
            asdict(chronology_attributes)
            | asdict(astrometric_attributes)
            | {"stable_hash": stable_hash, "stack_ids": stack_ids, "content_hash": content_hash}
        )

        return cls(**attributes)

    # User-pipeline conveniences (Phase B of the pipelines-as-Python work).
    # Delegate to existing helpers; no new behaviour, just friendly names.

    @property
    def id(self) -> TargetHash:
        """Display alias for `stable_hash`."""
        return self.stable_hash

    @property
    def filters(self) -> list[str]:
        """Filters this target was imaged in, deterministic order."""
        from targets.target_channel import filters_for_target
        return filters_for_target(self)

    def stacks_for(self, filter: str) -> tuple:
        """The constituent `Stack`s for one `(target, filter)` channel."""
        from targets.target_channel import channel_stacks_for
        return channel_stacks_for(self, filter)

    # A Target IS a single-panel mosaic. These two members let the entire
    # mosaic colour tail (place → merge → color_post_process, incl. the
    # panels-as-tiles machinery) run on a Target unchanged: `viewbox` frames
    # the target's own footprint, and `targets` makes panel enumeration
    # yield exactly one panel — itself.

    def viewbox(self, mode):
        """Pixel-space viewbox over this target's own footprint (same
        contract as `Mosaic.viewbox`)."""
        from mosaics.viewbox import compute_viewbox
        return compute_viewbox(mode, self)

    @property
    def targets(self) -> tuple:
        """Panel list of the single-panel mosaic this target is."""
        return (self,)

    @property
    def strips(self) -> tuple:
        """Strip layout of the single-panel mosaic this target is: one
        strip, one panel (same contract as `Mosaic.strips`)."""
        return ((self.id,),)

    def __str__(self) -> str:
        return (
            f"Target(name={self.name}, stable_hash={self.stable_hash}, "
            f"content_hash={self.content_hash}, {len(self.stack_ids)} stacks)"
        )

    def __hash__(self) -> int:
        assert self.content_hash, "Target has empty content_hash"
        # content_hash already captures every field; hash_dataclass
        # would re-walk all of them (stack_ids, polygons, footprint,
        # healpix_cells, ...) every time, blowing up trace cost by
        # O(target size). Matches the pattern Stack already uses.
        return hash_list_int([self.content_hash])

    def __eq__(self, other) -> bool:
        if not isinstance(other, Target):
            return NotImplemented

        return (
            self.stable_hash == other.stable_hash
            and self.content_hash == other.content_hash
        )

    # TODO: Add a valid() method, and assert on it.


class TargetsDict(dict[TargetHash, ContextRef["Target"]]):
    def __hash__(self):
        from cache.context_ref import dict_content_hash
        return hash_list_int([dict_content_hash(self)])

    def __str__(self):
        return f"TargetsDict({len(self)})"
