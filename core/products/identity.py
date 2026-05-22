"""Identity dataclasses for the artifacts pipelines produce.

An `ObjectId` declares the field shape that disambiguates one
artifact from another within an `object_type`. The dataclass is
used at two layers:

* **Tap recording.** The pipeline-trace tap collector reads
  `identity.object_type.__name__.lower()` and the first arg's
  `stable_hash` to record `(pipeline, object_type, object_id,
  display, filter) → path` in the PipelineOutputIndex.
* **Identity-field validation.** `@makes_image(identity=X)`
  asserts at decoration time that every field name in `X` is also
  a parameter of the producer function — catches typos before
  Ray dispatch.

Identities no longer carry human-readable `filename_stem()`
methods: filenames are now `<op_name>/<call_hash>.<ext>` (see
`products.registry._make_call_hash_filename_fn`). Browsing on
disk by hand is replaced by the AstroFlow UI's pipeline-keyed
viewers.
"""
import dataclasses
import importlib
from dataclasses import dataclass
from enum import StrEnum

from metadata.classes import ImageMetadata
from mosaics.types import Mosaic, Viewbox
from stacks.composite_stack import BaseStack, Stack, CalibrationStack
from stacks.run import Run
from targets.types import Target


class PresentationType(StrEnum):
    IMAGE = "Image"
    THUMBNAIL = "Thumbnail"
    TILE_PYRAMID = "TilePyramid"
    VIDEO = "Video"


class _LazyObjectType:
    """Descriptor for an `ObjectId.object_type` that lives in a module
    which (transitively) imports products — importing it eagerly here
    would be circular. Resolved + cached on first access; `is`
    identity holds because the underlying class object is a singleton.
    """

    def __init__(self, module: str, attr: str):
        self._module = module
        self._attr = attr
        self._resolved = None

    def __get__(self, obj, owner):
        if self._resolved is None:
            mod = importlib.import_module(self._module)
            self._resolved = getattr(mod, self._attr)
        return self._resolved


@dataclass(frozen=True)
class ObjectId:
    # Each concrete subclass declares the object class it produces as a
    # plain (un-annotated) class attribute so dataclass does NOT treat
    # it as a field — it must not enter cache_key / __init__.
    object_type = None

    def cache_key(self) -> str:
        parts = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if hasattr(v, 'stable_hash'):
                parts.append(v.stable_hash)
            else:
                parts.append(str(v))
        return "_".join(parts)

    def presentation_cache_key(
        self,
        presentation: PresentationType,
        op_name: str,
    ) -> tuple:
        """Cache key for a presentation (tile pyramid / thumbnail) of
        an object produced by op `op_name`."""
        return (
            str(presentation),
            type(self).object_type.__name__,
            op_name,
            self.cache_key(),
        )


@dataclass(frozen=True)
class StackId(ObjectId):
    object_type = Stack
    stack: BaseStack


@dataclass(frozen=True)
class CalibrationStackId(ObjectId):
    object_type = CalibrationStack
    stack: BaseStack


@dataclass(frozen=True)
class TargetId(ObjectId):
    object_type = Target
    target: Target


@dataclass(frozen=True)
class TargetChannelId(ObjectId):
    object_type = _LazyObjectType("targets.target_channel", "TargetChannel")
    target: Target
    filter: str


@dataclass(frozen=True)
class MosaicChannelId(ObjectId):
    object_type = Mosaic
    mosaic: Mosaic
    filter: str


@dataclass(frozen=True)
class MosaicId(ObjectId):
    object_type = Mosaic
    mosaic: Mosaic


@dataclass(frozen=True)
class PlacedImageId(ObjectId):
    object_type = _LazyObjectType("mosaics.placement.astroflow", "PlacedImage")
    mosaic: Mosaic
    target: Target
    filter: str
    viewbox: Viewbox


@dataclass(frozen=True)
class MergedImageId(ObjectId):
    object_type = _LazyObjectType("mosaics.merging.pixinsight", "MergedImage")
    mosaic: Mosaic
    filter: str


@dataclass(frozen=True)
class CalibratedStackId(ObjectId):
    object_type = Run
    stack: Stack
    substack: Run
    flat_stack: Run


@dataclass(frozen=True)
class UncorrectedRunId(ObjectId):
    object_type = Run
    stack: Stack
    substack: Run


def identity_for_stack(stack_or_cls) -> type[ObjectId]:
    """Map a stack instance or class to its identity class. Light
    stacks key on StackId; calibration stacks on CalibrationStackId."""
    cls = stack_or_cls if isinstance(stack_or_cls, type) else type(stack_or_cls)
    return StackId if issubclass(cls, Stack) else CalibrationStackId
