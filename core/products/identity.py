import dataclasses
import importlib
import math
from dataclasses import dataclass
from enum import StrEnum

from catalogs import position_string
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
    """Descriptor for an ObjectId.object_type that lives in a module
    which (transitively) imports products — importing it eagerly here
    would be circular. Resolved + cached on first access; ``is``
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
        level: "QualityLevel",
        implementation: "Implementation",
    ) -> tuple:
        return (
            str(presentation),
            type(self).object_type.__name__,
            str(level),
            str(implementation),
            self.cache_key(),
        )


@dataclass(frozen=True)
class StackId(ObjectId):
    object_type = Stack
    stack: BaseStack

    def filename_stem(self) -> str:
        stack = self.stack
        if isinstance(stack, Stack) and "light" in stack.image_type:
            ra_dec = position_string.ra_dec_string(*stack.center)
            return (
                f"light_{stack.stable_hash}_{stack.name}_{ra_dec}_"
                f"{stack.camera}_{stack.image_scale:.02f}aspp_"
                f"{math.degrees(stack.rotation_angle):.2f}deg_{stack.filter}"
            )
        raise ValueError(f'unknown image_type "{stack.image_type}"')


@dataclass(frozen=True)
class CalibrationStackId(ObjectId):
    object_type = CalibrationStack
    stack: BaseStack

    def filename_stem(self) -> str:
        stack = self.stack
        if "bias" in stack.image_type:
            return (
                f"bias_{stack.stable_hash}_{stack.camera}_"
                f"{stack.mean_temperature:.0f}C_{stack.stable_hash}"
            )
        elif "dark" in stack.image_type:
            return (
                f"dark_{stack.stable_hash}_{stack.camera}_"
                f"{stack.mean_temperature:.0f}C_{stack.exposure:.0f}s_"
                f"{stack.stable_hash}"
            )
        elif "flat" in stack.image_type:
            return (
                f"flat_{stack.stable_hash}_{stack.camera}_{stack.filter}_"
                f"{stack.stable_hash}"
            )
        raise ValueError(f'unknown image_type "{stack.image_type}"')


@dataclass(frozen=True)
class TargetId(ObjectId):
    object_type = Target
    target: Target


@dataclass(frozen=True)
class TargetChannelId(ObjectId):
    object_type = _LazyObjectType("targets.target_channel", "TargetChannel")
    target: Target
    filter: str

    def filename_stem(self) -> str:
        return f"target_{self.target.stable_hash}_{self.filter}"


@dataclass(frozen=True)
class MosaicChannelId(ObjectId):
    object_type = Mosaic
    mosaic: Mosaic
    filter: str

    def filename_stem(self) -> str:
        return f"final_{self.mosaic.stable_hash}_{self.filter}"


@dataclass(frozen=True)
class MosaicId(ObjectId):
    object_type = Mosaic
    mosaic: Mosaic

    def filename_stem(self) -> str:
        return f"final_{self.mosaic.stable_hash}_None"


@dataclass(frozen=True)
class PlacedImageId(ObjectId):
    object_type = _LazyObjectType("mosaics.placement", "PlacedImage")
    mosaic: Mosaic
    target: Target
    filter: str
    viewbox: Viewbox

    def filename_stem(self) -> str:
        return f"placed_{self.mosaic.stable_hash}_{self.target.stable_hash}_{self.filter}"


@dataclass(frozen=True)
class MergedImageId(ObjectId):
    object_type = _LazyObjectType("mosaics.merge_images", "MergedImage")
    mosaic: Mosaic
    filter: str
    input_0: ImageMetadata
    input_1: ImageMetadata


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
